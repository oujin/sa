from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time
import numpy as np
import matplotlib.pyplot as plt


def round_figures(x, n):
    """返回n位长的数值"""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """返回HHHH:MM:SS格式的时间字符串"""
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return '{:4d}:{:2d}:{:2d}'.format(h, m, s)


class Annealer(object):
    """
    可通过调用计算能量和状态转移的函数实现模拟退火。温度等初始参数的设定
    可以手动提供，也可以选择自动估计、
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    updates = 100
    copy_strategy = 'deepcopy'
    user_exit = False
    save_state_on_exit = False

    # placeholders
    best_state = None
    best_energy = None
    start = None
    energies = []
    best_energies = []
    visual = None

    def __init__(self, initial_state=None, load_state=None):
        # 初始解被子类继承后初始化
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError('无效的初始状态和加载状态！')
        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        """保存状态"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = f'{date}_energy_{self.energy()}.state'
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        """加载状态"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        """改变状态"""
        pass

    @abc.abstractmethod
    def energy(self):
        """计算状态对应的能量"""
        pass

    def set_user_exit(self, signum, frame):
        """
        改变user_exit，是退火迭代过程停止
        """
        self.user_exit = True

    def set_schedule(self, schedule):
        """
        从`auto`方法的返回值schedule中提取设置属性
        """
        self.Tmax, self.Tmin = schedule['tmax'], schedule['tmin']
        self.steps = int(schedule['steps'])
        self.updates = int(schedule['updates'])

    def copy_state(self, state):
        """根据复制策略返回一个复制的状态

        * deepcopy : 用 copy.deepcopy (慢但可靠)
        * slice: 使用列表切片 (快但只适用于当状态为列表时的情况)
        * method: 使用列表/字典的浅复制，只拷贝不可变部分，可变部分指向相同内容
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError(f'不存在名为"{self.copy_strategy}"的复制策略')

    def update(self, *args, **kwargs):
        """
        状态更新
        """
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        """
        打印显示当前的温度，能量，接受率，提升率，累计耗时，剩余时间。

        接受率是指最后一次更新之后，可被Metropolis准则接受的转移次数占总步数的比率，
        该转移可能是能降低能量或保持能量不变的转移，也可能是热激励下能达到的会提高能
        量的转移。

        提高率是指最后一次更新之后能够严格降低能量的转移次数占总步数的比率，该转移在
        高温下包括能提升全部状态的转移，和破坏前面被接受的热激励下提高能量的状态的转
        移；低温下，随着能降低能量的转移的耗尽，会提高能量的转移不再可达，提高率将趋
        于0。
        """

        elapsed = time.time() - self.start
        if step == 0:
            print(
                '        温度          能量    接受率    提升率    消耗时间    剩余时间',
                file=sys.stderr)
            print(
                '\r{:12.5f}  {:12.2f}                      {}            '.
                format(T, E, time_string(elapsed)),
                file=sys.stderr,
                end="\r")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print(
                '\r{:12.5f}  {:12.2f}  {:7.2f}%  {:7.2f}%  {}  {}\r'.format(
                    T, E, 100.0 * acceptance, 100.0 * improvement,
                    time_string(elapsed), time_string(remain)),
                file=sys.stderr,
                end="\r")
            sys.stderr.flush()

    def anneal(self):
        """
        通过SA算法最小化系统的能量。
        """

        step = 0
        self.start = time.time()

        # 计算从Tmax到Tmin的指数冷却因子
        if self.Tmin <= 0.0:
            raise Exception('温度不能低于0！')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # 初始化状态
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            # 设置
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)
        # 状态转移
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            self.move()
            E = self.energy()
            dE = E - prevEnergy
            trials += 1
            if self.visual:
                self.energies.append(prevEnergy)
                self.best_energies.append(self.best_energy)
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # 不被接受，保存之前的状态
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # 接受新状态，并保存当前最优状态
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                res1 = step // updateWavelength
                res2 = (step - 1) // updateWavelength
                if res1 > res2:
                    self.update(step, T, E, accepts / trials,
                                improves / trials)
                    trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        if self.visual:
            n = np.linspace(1, step, step)
            plt.plot(n, self.best_energies)
            plt.show()
            plt.plot(n, self.energies)
            plt.show()
        return self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        """
        估计最优的温度设置。返回一个可供`set_schedule`方法使用的字典。
        """

        def run(T, steps):
            """对给定温度进行退火，但不保存最优状态，用于预处理"""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for _ in range(steps):
                self.move()
                E = self.energy()
                dE = E - prevEnergy
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # 通过SA预处理猜测合适的温度设置
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        # 随机生产一个温度
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.energy() - E)

        # 寻找一个能产生接近但大于98%接受率的温度Tmax
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # 寻找一个能产生0%接受率的温度Tmin
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # 根据预计退货时间设置退火步数
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        return {
            'tmax': Tmax,
            'tmin': Tmin,
            'steps': duration,
            'updates': self.updates
        }

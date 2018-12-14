import abc
import copy
import math
import random
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def time_string(seconds):
    """返回HHHH:MM:SS格式的时间字符串"""
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return '{:4d}:{:2d}:{:2d}'.format(h, m, s)


class Annealer(object):
    """可通过调用计算能量和状态转移的函数实现模拟退火。"""

    __metaclass__ = abc.ABCMeta

    Tmax = 25000.
    Tmin = 2.5
    steps = 10000
    updates = 100
    copy_strategy = 'deepcopy'
    exit = False

    best_state = None
    best_energy = None
    start = None
    energies = []
    best_energies = []
    visual = None

    def __init__(self, initial_state=None):
        # 初始解被子类继承后初始化
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        else:
            raise ValueError('无效的初始状态和加载状态！')
        signal.signal(signal.SIGINT, self.set_exit)

    @abc.abstractmethod
    def move(self):
        """状态转移"""
        pass

    @abc.abstractmethod
    def energy(self):
        """计算状态对应的能量"""
        pass

    def set_exit(self, signum, frame):
        """改变exit，使退火迭代过程停止"""
        self.exit = True

    def copy_state(self, state):
        """根据复制策略返回一个复制的状态

        * deepcopy : 用 copy.deepcopy (慢但可靠)
        * slice: 使用列表切片 (快但只适用于当状态为列表时的情况)
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        else:
            raise RuntimeError(f'不存在名为"{self.copy_strategy}"的复制策略')

    def update(self, step, T, E, acceptance, improvement):
        """打印显示当前的温度，能量，接受率，提升率，累计耗时，剩余时间。
        接受率是指最后一次更新之后，可被Metropolis准则接受的转移次数占总步数的比率。
        提高率是指最后一次更新之后能够严格降低能量的转移次数占总步数的比率。
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
        """通过SA算法最小化系统的能量。"""

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
        while step < self.steps and not self.exit:
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

        if self.visual:
            n = np.linspace(1, step, step)
            plt.plot(n, self.best_energies)
            plt.show()
            plt.plot(n, self.energies)
            plt.show()

        return self.best_state, self.best_energy

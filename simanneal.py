import abc
import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np


class Annealer(object):
    """可通过调用计算能量和状态转移的函数实现模拟退火。"""

    def __init__(self, initial_state):
        self.state = self.copy(initial_state)

        self.__metaclass__ = abc.ABCMeta

        self.Tmax = 20000
        self.Tmin = 2
        self.steps = 10000

        self.best_state = None
        self.best_energy = None
        self.energies = []
        self.best_energies = []
        self.visual = None

    @abc.abstractmethod
    def move(self):
        """状态转移"""
        pass

    @abc.abstractmethod
    def energy(self):
        """计算状态对应的能量"""
        pass

    def copy(self, state):
        """复制状态"""
        if isinstance(state, list):
            return state[:]
        else:
            return copy.deepcopy(state)

    def anneal(self):
        """通过SA算法最小化系统的能量。"""

        step = 0

        if self.Tmin <= 0.0:
            raise Exception('温度不能低于0！')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # 初始化状态
        T = self.Tmax
        E = self.energy()
        prevState = self.copy(self.state)
        prevEnergy = E
        self.best_state = self.copy(self.state)
        self.best_energy = E
        # 状态转移
        while step < self.steps:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            self.move()
            E = self.energy()
            dE = E - prevEnergy
            if self.visual:
                self.energies.append(prevEnergy)
                self.best_energies.append(self.best_energy)
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # 不被接受，保存之前的状态
                self.state = self.copy(prevState)
                E = prevEnergy
            else:
                # 接受新状态，并保存当前最优状态
                prevState = self.copy(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy(self.state)
                    self.best_energy = E

        if self.visual:
            n = np.linspace(1, step, step)
            plt.plot(n, self.best_energies)
            plt.show()
            plt.plot(n, self.energies)
            plt.show()

        return self.best_state, self.best_energy

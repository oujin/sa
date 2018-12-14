from __future__ import print_function
import math
import random
from simanneal import Annealer
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """计算多极小函数值，f=[(x-1)^2+1]/[sin(x)+2]"""
    return ((x - 1) * (x - 1) + 1) / (math.sin(x) + 2)


class MultiMinimalProblem(Annealer):
    """
    用SA算法模拟多极小问题问题
    """

    def __init__(self, state, visual=None):
        super(MultiMinimalProblem, self).__init__(state)
        self.visual = visual

    def move(self):
        """邻域中随机取值"""
        # self.state += (random.random() - 0.5) * 10
        self.state += np.random.standard_cauchy()

    def energy(self):
        """计算函数值"""
        return function(self.state)


if __name__ == '__main__':
    # 画函数曲线
    x = np.linspace(0, 20, 10000)
    y = list(map(function, x))

    # 随机初始化状态
    init_state = random.random() * 10
    mmp = MultiMinimalProblem(init_state, visual=True)
    mmp.steps = 10000
    mmp.copy_strategy = "deepcopy"
    state, e = mmp.anneal()
    plt.scatter(state, e, s=30, marker='o')
    plt.plot(x, y)
    plt.show()

    states, energies = [], []
    for i in range(20):
        print()
        print(f'Epoch [{i+1}/20]')
        init_state = random.random() * 10
        mmp = MultiMinimalProblem(init_state)
        mmp.steps = 10000
        mmp.copy_strategy = "deepcopy"
        state, e = mmp.anneal()
        states.append(state)
        energies.append(e)

    with open('results.txt', 'w', encoding='utf-8') as f:
        print('|状态(x)|性能(y)', file=f)
        print('---|---|---', file=f)
        for i, (s, e) in enumerate(zip(states, energies)):
            print('{:d}|{:.6f}|{:.6f}'.format(i + 1, s, e), file=f)
    i_min = energies.index(min(energies))
    i_max = energies.index(max(energies))
    print()
    print(f'Min ----> state: {states[i_min]}, energy: {energies[i_min]}')
    print(f'Max ----> state: {states[i_max]}, energy: {energies[i_max]}')
    print(f'Aver----> energy: {np.mean(energies)}')
    print(f'Var ----> {np.var(energies)}')

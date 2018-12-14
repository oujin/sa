import math
import random

import matplotlib.pyplot as plt
import numpy as np

from simanneal import Annealer


class TravellingSalesmanProblem(Annealer):
    """用SA算法模拟旅行商问题"""

    # 导入初始状态
    def __init__(self, state, distance_matrix, visual=None):
        self.distance_matrix = distance_matrix
        super(TravellingSalesmanProblem, self).__init__(state)
        self.visual = visual

    def move(self):
        """随机交换路径上的两个城市"""
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """计算总路径距离"""
        e = 0
        for i in range(len(self.state)):
            e += self.distance_matrix[self.state[i - 1]][self.state[i]]
        return e


def generate_cities():
    n_cities = random.randint(10, 20)
    cities = np.random.uniform(0, 500, (n_cities, 2))
    name = [i for i in range(n_cities)]
    start = name[0]
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        re_coord = cities - cities[i]
        distance_matrix[i, :] = np.sqrt(np.sum(re_coord * re_coord, 1))
    return name, cities, start, distance_matrix


if __name__ == '__main__':

    cities, local, start, distance_matrix = generate_cities()

    with open('cities.txt', 'w', encoding='utf-8') as f:
        print('name|x|y', file=f)
        print('---|---|---', file=f)
        for i, city in enumerate(local):
            print('{:d}|{:.4f}|{:.4f}'.format(i, city[0], city[1]), file=f)

    # 随机初始化状态序列
    init_state = list(cities)
    random.shuffle(init_state)

    tsp = TravellingSalesmanProblem(init_state, distance_matrix, visual=True)
    tsp.steps = 100000
    state, e = tsp.anneal()

    ind = state.index(start)
    state = state[ind:] + state[:ind]
    seq = local[state]
    seq = np.vstack((seq, [seq[0]]))
    plt.plot(np.array(seq[:, 0]), np.array(seq[:, 1]), 'b-', marker='*', linewidth=1)
    plt.show()

    states, energies = [], []
    for i in range(20):
        init_state = list(cities)
        random.shuffle(init_state)

        tsp = TravellingSalesmanProblem(init_state, distance_matrix)
        tsp.steps = 100000
        state, e = tsp.anneal()
        ind = state.index(start)
        state = state[ind:] + state[:ind]
        states.append(state)
        energies.append(e)

    with open('results.txt', 'w', encoding='utf-8') as f:
        print('||状态|性能', file=f)
        print('---|---|---', file=f)
        for i, (s, e) in enumerate(zip(states, energies)):
            print('{:d}|{}|{:.6f}'.format(i + 1, s, e), file=f)
    i_min = energies.index(min(energies))
    i_max = energies.index(max(energies))
    print(f'Min ----> state: {states[i_min]}, energy: {energies[i_min]}')
    print(f'Max ----> state: {states[i_max]}, energy: {energies[i_max]}')
    print(f'Aver----> energy: {np.mean(energies)}')
    print(f'Var ----> {np.var(energies)}')

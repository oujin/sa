import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from simanneal import Annealer


class GraphMatch(Annealer):
    """用SA算法模拟二分图匹配问题"""

    # 导入初始状态
    def __init__(self, state, all_codes, distance_matrix, visual=None):
        self.distance_matrix = distance_matrix
        super(GraphMatch, self).__init__(state)
        self.visual = visual
        self.all_codes = all_codes
        self.p = len(self.state) / len(self.all_codes)

    def move(self):
        """随机交换路径上的两点或从剩余点中替代"""
        # print(self.state, remainder)
        if random.random() > self.p:
            remainder = list(set(self.all_codes) - set(self.state))
            a = random.randint(0, len(self.state) - 1)
            b = random.randint(0, len(remainder) - 1)
            self.state[a] = remainder[b]
        else:
            a = random.randint(0, len(self.state) - 1)
            b = random.randint(0, len(self.state) - 1)
            self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        """计算总距离"""
        e = 0
        for i in range(len(self.state)):
            e += self.distance_matrix[i][self.state[i]]
        return e


def distance(distance_matrix, indices):
    d = 0
    for i, j in indices:
        d += distance_matrix[i][j]
    return d


def init(distance_matrix):
    dm = distance_matrix.copy()
    state = [i for i in range(len(distance_matrix))]
    d_min = np.min(dm)
    while d_min < np.inf:
        indices = np.where(distance_matrix == d_min)
        state[int(indices[0])] = int(indices[1])
        dm[indices[0], :] = np.inf
        dm[:, indices[1]] = np.inf
        d_min = np.min(dm)
    return state


def generate_points():
    n_points = random.randint(10, 15)
    points1 = np.random.uniform(0, 500, (n_points, 2))
    points2 = np.random.uniform(0, 500, (n_points + 2, 2))
    order = [i for i in range(n_points)]
    loss = random.randint(0, n_points - 1)
    order.pop(loss)
    points2[:n_points - 1, :] = points1[order]
    noise = np.random.standard_cauchy((n_points + 2, 2))
    points2 += noise * 10
    points2[points2 < 0] = 0
    points2[points2 > 500] = 500
    distance_matrix = np.zeros((n_points - 1, n_points + 2))
    for i in range(n_points - 1):
        re_coord = points2 - points1[i]
        distance_matrix[i, :] = np.sqrt(np.sum(re_coord * re_coord, 1))

    def save_file(filename, points):
        with open(filename, 'w', encoding='utf-8') as f:
            print('name|x|y', file=f)
            print('---|---|---', file=f)
            for i, point in enumerate(points):
                print(
                    '{:d}|{:.4f}|{:.4f}'.format(i, point[0], point[1]), file=f)

    save_file('points1.txt', points1)
    save_file('points2.txt', points2)

    return points1, points2, distance_matrix


def read_points(fn1=None, fn2=None):
    if fn1 is None or fn2 is None:
        return generate_points()

    def read_file(filename):
        points = []
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline()
            line = f.readline()
            while line:
                line = f.readline()
                items = line.split('|')
                if len(items) >= 3:
                    points.append([float(items[1]), float(items[2])])
        return np.asarray(points)

    points1 = read_file(fn1)
    points2 = read_file(fn2)

    n1, n2 = points1.shape[0], points2.shape[0]
    if n2 < n1:
        points1, points2 = points2, points1
        n1, n2 = n2, n1
    distance_matrix = np.zeros((n1, n2))
    for i in range(n1):
        re_coord = points2 - points1[i]
        distance_matrix[i, :] = np.sqrt(np.sum(re_coord * re_coord, 1))
    return points1, points2, distance_matrix


if __name__ == '__main__':

    # points1, points2, distance_matrix = read_points()

    points1, points2, distance_matrix = read_points('points1.txt',
                                                    'points2.txt')
    plt.plot(points1[:, 0], points1[:, 1], 'b*')
    plt.plot(points2[:, 0], points2[:, 1], 'ro')
    plt.show()
    order = [i for i in range(len(points2))]

    # 初始化状态序列
    init_state = init(distance_matrix)

    gm = GraphMatch(init_state, order, distance_matrix, visual=True)

    gm.steps = 1000
    gm.Tmax = 10
    state, e = gm.anneal()
    print(state, e)
    plt.figure(1)
    plt.subplot(121)
    plt.plot(points1[:, 0], points1[:, 1], 'b*')
    plt.plot(points2[:, 0], points2[:, 1], 'ro')
    for i in range(len(state)):
        x = (points1[i, 0], points2[state[i], 0])
        y = (points1[i, 1], points2[state[i], 1])
        plt.plot(x, y, 'g-', marker='*', linewidth=1)
    # plt.show()
    indices = linear_assignment(distance_matrix)
    print(list(indices[:, 1]), distance(distance_matrix, indices))
    plt.subplot(122)
    plt.plot(points1[:, 0], points1[:, 1], 'b*')
    plt.plot(points2[:, 0], points2[:, 1], 'ro')
    for i, j in indices:
        x = (points1[i, 0], points2[j, 0])
        y = (points1[i, 1], points2[j, 1])
        plt.plot(x, y, 'g-', marker='*', linewidth=1)
    plt.show()
    # 测试SA算法
    states, energies = [], []
    t1 = time.time()
    for i in range(20):
        init_state = init(distance_matrix)
        gm = GraphMatch(init_state, order, distance_matrix)
        gm.steps = 1000
        gm.Tmax = 10
        state, e = gm.anneal()
        states.append(state)
        energies.append(e)

    t2 = time.time()
    i_min = energies.index(min(energies))
    i_max = energies.index(max(energies))
    print(f'Min ----> state: {states[i_min]}, energy: {energies[i_min]}')
    print(f'Max ----> state: {states[i_max]}, energy: {energies[i_max]}')
    print(f'Aver----> energy: {np.mean(energies)}')
    print(f'Var ----> {np.var(energies)}')
    print(f'Time---->{t2 - t1}')
    # 测试匈牙利算法
    states, energies = [], []
    t1 = time.time()
    for i in range(20):
        indices = linear_assignment(distance_matrix)
        states.append(list(indices[:, 1]))
        energies.append(distance(distance_matrix, indices))

    t2 = time.time()
    i_min = energies.index(min(energies))
    i_max = energies.index(max(energies))
    print(f'Min ----> state: {states[i_min]}, energy: {energies[i_min]}')
    print(f'Max ----> state: {states[i_max]}, energy: {energies[i_max]}')
    print(f'Aver----> energy: {np.mean(energies)}')
    print(f'Var ----> {np.var(energies)}')
    print(f'Time---->{t2 - t1}')

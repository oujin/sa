from __future__ import print_function
import math
import random
from simanneal import Annealer


def distance(a, b):
    """计算球面最短距离"""
    R = 3963  # 地球半径，单位：miles
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(
        math.sin(lat1) * math.sin(lat2) +
        math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


class TravellingSalesmanProblem(Annealer):
    """
    用SA算法模拟旅行商问题
    """

    # 导入初始状态
    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super(TravellingSalesmanProblem, self).__init__(state)

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


if __name__ == '__main__':

    # 美国20个城市的经纬度
    cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62)
    }

    # 随机初始化状态序列
    init_state = list(cities.keys())
    random.shuffle(init_state)

    # 创建距离矩阵
    distance_matrix = {}
    for ka, va in cities.items():
        distance_matrix[ka] = {}
        for kb, vb in cities.items():
            if kb == ka:
                distance_matrix[ka][kb] = 0.0
            else:
                distance_matrix[ka][kb] = distance(va, vb)

    tsp = TravellingSalesmanProblem(init_state, distance_matrix)
    tsp.steps = 100000
    # 状态列表使用切片方式复制最快
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()

    while state[0] != 'New York City':
        state = state[1:] + state[:1]  # 旋转列表至从纽约出发

    print()
    print(f'{e} mile route:')
    for city in state:
        print("\t", city)

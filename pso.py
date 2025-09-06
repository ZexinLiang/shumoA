import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
#是否输出调试信息
MYDEBUG = False
#增加了变异，提升了搜索的广度
#增添了初始有预设值的选定，使得第一次可以广度优先搜索，第二轮可以根据第一轮精细结果进行细致推理

def fit_fun(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

class Particle:
    def __init__(self, x_max, x_min, max_vel, dim, fitness, init_pos=None):
        if init_pos is not None:
            self.__pos = np.array(init_pos, dtype=float)
        else:
            self.__pos = np.random.uniform(x_min, x_max, dim)
        # self.__pos = np.random.uniform(x_min, x_max, dim)
        self.__vel = np.random.uniform(-max_vel, max_vel, dim)  
        # 个体历史最优
        self.__bestPos = self.__pos.copy()
        self.__fitnessValue = fitness(self.__pos)
        self.fitness = fitness
    def set_pos(self, value):
        self.__pos = value
    def get_pos(self):
        return self.__pos
    def set_best_pos(self, value):
        self.__bestPos = value
    def get_best_pos(self):
        return self.__bestPos
    def set_vel(self, value):
        self.__vel = value
    def get_vel(self):
        return self.__vel
    def set_fitness_value(self, value):
        self.__fitnessValue = value
    def get_fitness_value(self):
        return self.__fitnessValue

class PSO:
    def __init__(self, dim, size, iter_num, x_max, x_min, max_vel, tol,
                 fitness, best_fitness_value=float('Inf'),
                 C1=5, C2=5, W=0.8,
                 init_positions=None):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim
        self.size = size
        self.iter_num = iter_num
        self.x_max = x_max
        self.x_min = x_min
        self.max_vel = max_vel
        self.tol = tol
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros(dim)
        self.fitness_val_list = []
        self.fitness = fitness
        #修改以适应第二阶段最优
        self.Particle_list = []
        for i in range(self.size):
            if init_positions is not None and i < len(init_positions):
                part = Particle(self.x_max, self.x_min, self.max_vel,
                                self.dim, self.fitness, init_pos=init_positions[i])
            else:
                part = Particle(self.x_max, self.x_min, self.max_vel,
                                self.dim, self.fitness)
            self.Particle_list.append(part)
        # 初始化全局最优
        for part in self.Particle_list:
            value = part.get_fitness_value()
            if value < self.best_fitness_value:
                self.best_fitness_value = value
                self.best_position = part.get_pos().copy()
        # # 初始化粒子群
        # self.Particle_list = [
        #     Particle(self.x_max, self.x_min, self.max_vel, self.dim, self.fitness)
        #     for _ in range(self.size)
        # ]
        # for part in self.Particle_list:
        #     value = part.get_fitness_value()
        #     if value < self.best_fitness_value:
        #         self.best_fitness_value = value
        #         self.best_position = part.get_pos().copy()

        # 初始化全局最优
        # for part in self.Particle_list:
        #     if part.get_fitness_value() < self.best_fitness_value:
        #         self.best_fitness_value = part.get_fitness_value()
        #         self.best_position = part.get_pos().copy()
    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value
    def get_bestFitnessValue(self):
        return self.best_fitness_value
    def set_bestPosition(self, value):
        self.best_position = value
    def get_bestPosition(self):
        return self.best_position
    def update_vel(self, part):
        vel_value = (self.W * part.get_vel()
                     + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos())
                     + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos()))
        vel_value = np.clip(vel_value, -self.max_vel, self.max_vel)
        part.set_vel(vel_value)
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        pos_value = np.clip(pos_value, self.x_min, self.x_max)
        part.set_pos(pos_value)
        value = self.fitness(pos_value)
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)
    def update_ndim(self, mutation_rate=0.1):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)
                self.update_pos(part)
                # 变异：以概率 mutation_rate 扰动
                if np.random.rand() < mutation_rate:
                    noise = np.random.normal(0, 0.1, self.dim)  # 高斯扰动
                    new_pos = part.get_pos() + noise
                    new_pos = np.clip(new_pos, self.x_min, self.x_max)
                    part.set_pos(new_pos)
            self.fitness_val_list.append(self.get_bestFitnessValue())
            if MYDEBUG:
                print(f'第{i+1}次最佳适应值为 {self.get_bestFitnessValue():.6f}')
            if self.get_bestFitnessValue() < self.tol:
                break
        return self.fitness_val_list, self.get_bestPosition()

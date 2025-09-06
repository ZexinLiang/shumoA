import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# 测试用适应度函数：Rosenbrock 函数
def fit_fun(x):
    # # x 是形状 (1, dim) 的数组
    # x = x[0]
    # return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


class Particle:
    def __init__(self, x_max, x_min, max_vel, dim, fitness):
        # # 初始化位置
        # self.__pos = np.random.uniform(x_min, x_max, (1, dim))
        # # 初始化速度
        # self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))
        self.__pos = np.random.uniform(x_min, x_max, dim)   # 一维向量
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
                 C1=5, C2=5, W=0.8):
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

        # 初始化粒子群
        self.Particle_list = [
            Particle(self.x_max, self.x_min, self.max_vel, self.dim, self.fitness)
            for _ in range(self.size)
        ]
        for part in self.Particle_list:
            value = part.get_fitness_value()
            if value < self.best_fitness_value:
                self.best_fitness_value = value
                self.best_position = part.get_pos().copy()
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

    def update_ndim(self, mutation_rate=0.6):
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
            print(f'第{i+1}次最佳适应值为 {self.get_bestFitnessValue():.6f}')
            if self.get_bestFitnessValue() < self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition()

    # def update_ndim(self):
    #     for i in range(self.iter_num):
    #         for part in self.Particle_list:
    #             self.update_vel(part)
    #             self.update_pos(part)
    #         self.fitness_val_list.append(self.get_bestFitnessValue())
    #         print(f'第{i+1}次最佳适应值为 {self.get_bestFitnessValue():.6f}')
    #         if self.get_bestFitnessValue() < self.tol:
    #             break

    #     return self.fitness_val_list, self.get_bestPosition()

if __name__ == '__main__':
    # 参数配置
    pso = PSO(
        dim=4,              # 维度
        size=20,            # 粒子数量
        iter_num=2000,      # 最大迭代次数
        x_max=5,            # 搜索上界
        x_min=-5,           # 搜索下界
        max_vel=10,        # 最大速度
        tol=1e-6,           # 收敛条件
        fitness=fit_fun,    # 适应度函数
        C1=2, C2=2, W=0.8   # 参数
    )

    fit_val_list, best_pos = pso.update_ndim()
    print("最优位置:", best_pos)
    print("最优解:", fit_val_list[-1])

    plt.plot(range(len(fit_val_list)), fit_val_list, alpha=0.7)
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.title("PSO 收敛曲线")
    plt.show()

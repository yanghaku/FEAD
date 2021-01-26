import numpy as np
from CNN_mawilab import CNN_mawilab


class GA:  # 遗传算法选择特征
    def __init__(self, DNA_size, pop_size, cross_rate, mutation_rate, n_generations):
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.pop = np.random.randint(0, 2 ** DNA_size, pop_size).astype(np.int32)
        self.cnn_test = CNN_mawilab()

    def cross_over(self, parent1, pop):  # 交叉
        if np.random.rand() < self.cross_rate:
            id = np.random.randint(0, self.pop_size, size=1)
            parent2 = pop[id]
            child = 0
            for i in range(self.DNA_size):
                if np.random.rand() > 0.5:
                    child = (child << 1) | ((parent1 >> i) & 1)
                else:
                    child = (child << 1) | ((parent2 >> i) & 1)
            return child
        return parent1

    def mutate(self, child):  # 变异
        change = 0
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                change = (change << 1) | 1
            else:
                change = change << 1
        return child ^ change

    # 选择
    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=fitness / fitness.sum())
        return pop[idx]

    def run(self):
        for _ in range(self.n_generations):
            Values = self.cnn_test.run(self.pop)

            fitness = Values + 1e-3 - np.min(Values)

            print("generation=", _ + 1, "  f1=", np.max(Values), "  feature=",
                  self.cnn_test.bin2name(self.pop[np.argmax(Values)]),
                  "feature binary is: ", self.pop[np.argmax(Values)],
                  "fitness=", np.sum(fitness) / fitness.shape[0])

            # 选择
            self.pop = self.select(self.pop, fitness)
            pop_copy = self.pop.copy()
            for i in range(self.pop_size):
                child = self.cross_over(self.pop[i], pop_copy)  # 交叉
                child = self.mutate(child)  # 变异
                self.pop[i] = child  # 替换


DNA_size = 13
cross_rate = 0.7  # 交叉概率
mutation_rate = 0.01  # 变异概率
n_generations = 20  # 迭代次数, 后代数量
pop_size = 40  # 种群个数

g = GA(DNA_size, pop_size, cross_rate, mutation_rate, n_generations)
g.cnn_test._run(780)
print(g.cnn_test.F1s[780])
g.run()

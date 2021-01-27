import numpy as np
from CNN_mawilab import CNN_mawilab


class P:
    def __init__(self, speed, position, fit, lBestPosition, lBestFit):
        self.speed = speed
        self.position = position
        self.fit = fit
        self.lBestFit = lBestFit
        self.lBestPosition = lBestPosition


class PSO:
    def __init__(self, num, w, c1, c2):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.cnn_test = CNN_mawilab()
        self.best_fit = 0
        self.best_pos = 0

        position = np.random.randint(0, 2 ** 13, num).astype(np.int32)
        f = self.cnn_test.run(position)
        speed = np.zeros(13)
        self.pop = []
        for i in range(num):
            self.pop.append(P(speed.copy(), position[i], f[i], position[i], f[i]))
            if f[i] > self.best_fit:
                self.best_fit = f[i]
                self.best_pos = position[i]

    def solve(self, epoch):
        for e in range(epoch):
            # update
            pos = []
            for p in self.pop:
                for i in range(p.speed.shape[0]):
                    local_p = (p.lBestPosition >> i) & 1
                    global_p = (self.best_pos >> i) & 1
                    this_p = (p.position >> i) & 1
                    p.speed[i] = self.w * p.speed[i] + self.c1 * np.random.rand() * (
                            local_p - this_p) + self.c2 * np.random.rand() * (global_p - this_p)
            for p in self.pop:
                for i in range(p.speed.shape[0]):
                    s = 1.0 / (1.0 + np.exp(- p.speed[i]))
                    if np.random.rand() < s:
                        p.position ^= (1 << i)
                pos.append(p.position)

            f = self.cnn_test.run(np.array(pos))

            for i in range(len(self.pop)):
                self.pop[i].fit = f[i]
                if f[i] > self.pop[i].lBestFit:
                    self.pop[i].lBestFit = f[i]
                    self.pop[i].lBestPosition = self.pop[i].position

            for p in self.pop:
                print("pos=", p.position, "f1=", p.fit, self.cnn_test.bin2name(p.position))
                if p.fit > self.best_fit:
                    self.best_fit = p.fit
                    self.best_pos = p.position

            print("epoch: ", e + 1, "best f1=", self.best_fit, "best feature=", self.best_pos,
                  self.cnn_test.bin2name(self.best_pos))


w = 1
c1 = c2 = 2
num = 20
p = PSO(num, w, c1, c2)

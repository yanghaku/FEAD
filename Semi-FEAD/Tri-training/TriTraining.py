import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
import sys

sys.path.append('../..')

import FEAD


class TriTraining:
    def __init__(self, sz, lr=0.001):
        self.classifiers = [0, 0, 0]
        self.classifiers[0] = FEAD.FEAD(sz, lr)
        self.classifiers[1] = FEAD.FEAD(sz, lr)
        self.classifiers[2] = FEAD.FEAD(sz, lr)

    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)
            # print("sample size is", sample[0].shape)
            self.classifiers[i].fit(*sample)
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        Li_X, Li_y = [[]] * 3, [[]] * 3  # to save proxy labeled data
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations
            # print("the iter is", self.iter)
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                # print("the e[i] is", e[i])
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    # print("Uyj and Uyk is", sum(U_y_j == U_y_k))
                    Li_X[i] = U_X[U_y_j == U_y_k]  # when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True

            for i in range(3):
                if update[i]:
                    # print("the Li_X is", len(Li_X[i]))
                    self.classifiers[i].fit(np.append(L_X, Li_X[i], axis=0), np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement

    def predict(self, X):
        pred = np.asarray([self.classifiers[i].predict(X) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def measure_error(self, X, y, j, k):
        j_pred = self.classifiers[j].predict(X)
        k_pred = self.classifiers[k].predict(X)
        # print(j_pred)
        # print(k_pred)
        # print(sum(j_pred == k_pred))

        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)

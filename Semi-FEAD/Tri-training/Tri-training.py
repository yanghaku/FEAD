import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import FEAD

RANDOM = 2020


class TriTraining:
    def __init__(self):
        # if sklearn.base.is_classifier(classifier):
        #     self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        # else:
        #     self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]
        self.classifiers = [0, 0, 0]
        self.classifiers[0] = FEAD.FEAD()
        self.classifiers[1] = FEAD.FEAD()
        self.classifiers[2] = FEAD.FEAD()

    def fit(self, L_X, L_y, U_X):
        for i in range(3):
            sample = sklearn.utils.resample(L_X, L_y)
            print("sample size is", sample[0].shape)
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
            print("the iter is", self.iter)
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k)
                print("the e[i] is", e[i])
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X)
                    U_y_k = self.classifiers[k].predict(U_X)
                    print("Uyj and Uyk is", sum(U_y_j == U_y_k))
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
                    print("the Li_X is", len(Li_X[i]))
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
        print(sum(j_pred == k_pred))

        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)


data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
# data_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\data_30w_des.tsv.npy"
# labels_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\labels_30w_des.csv.npy"
data = np.load(data_path).astype(np.float32)
labels = np.load(labels_path)

# ff = open("./data/res_tri.md", "w")

for label_train_size in [90000]:  # [180, 450, 900, 1800, 4500, 9000, 18000]:
    test_size = 10000
    train_size = 90000
    # test_size = 29999
    # train_size = 270000
    # label_train_size = 5000
    unlabel_train_size = train_size - label_train_size

    train_data = data[0:train_size]
    # train_label = labels[0:train_size]

    ran_dice = np.random.permutation(train_size)
    train_data = train_data[ran_dice]
    train_label = labels[ran_dice]

    # labeled_train = data[0:label_train_size,:]
    # train_label = labels[0:label_train_size]
    labeled_train = train_data[0:label_train_size, :]
    train_label = train_label[0:label_train_size]

    # unlabeled_train = data[label_train_size:label_train_size+unlabel_train_size,:]
    unlabeled_train = train_data[label_train_size:train_size, :]

    test = data[train_size:train_size + test_size, :]
    test_label = labels[train_size:train_size + test_size]

    TT = TriTraining()
    TT.fit(labeled_train, train_label, unlabeled_train)
    res = TT.predict(test)

    accuracy = accuracy_score(test_label, res)
    precision = precision_score(res, test_label)
    recall = recall_score(res, test_label)
    f1 = f1_score(res, test_label)
    test_error = 1 - accuracy
    PRECISION = []
    RECALL = []
    F1 = []
    PRECISION.append(precision)
    RECALL.append(recall)
    F1.append(f1)
    print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|",
          accuracy, "|", 1 - accuracy, "|", )#file=ff)

# ff.close()

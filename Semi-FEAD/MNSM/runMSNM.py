from MSNM import MSNM
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score
from matplotlib import pyplot as plt
import sys

sys.path.append("..")
from SemiDataLoader import getData

IDS = False

if IDS:
    ff = open("msnm_ids.md", "w")
    data_path = "../../MAWILab-GAfeature/ids_30w.npy"
    labels_path = "../../MAWILab-GAfeature/ids_label_30w.npy"
    test_size = 29999
    train_size = 270000
    lst = [540, 1350, 2700, 5400, 13500, 27000]
    threshold_precent = 1

else:
    ff = open("msnm_mawilab.md", "w")
    data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
    labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
    test_size = 10000
    train_size = 90000
    lst = [180, 450, 900, 1800, 4500, 9000, 18000]
    threshold_precent = 0.8

for train_labeled_size in lst:
    number = 10  # 取平均
    F1s = []
    ACCs = []
    for r in range(number):

        train_unlabeled_size = train_size - train_labeled_size

        train_labeled_data, train_label, _, test_data, test_label = getData(data_path, labels_path, train_labeled_size,
                                                                            train_unlabeled_size,
                                                                            test_size)
        indices = []
        for i in range(train_labeled_size):
            if train_label[i] == 0:
                indices.append(i)
        train_unlabeled_data = train_labeled_data[indices, :]

        model = MSNM(train_labeled_data.shape[1])
        print("training.....")

        K = 10
        rc = 0.001
        epoch = 5
        n_component = 2
        model.train(train_labeled_data, train_label, train_unlabeled_data, n_component, K, rc, epoch)

        print("test....")
        scores = model.test(test_data)
        auc = roc_auc_score(test_label, scores)
        # c = []
        # for i in test_label:
        #     if i == 1:
        #         c.append('r')
        #     else:
        #         c.append('g')
        # plt.scatter(range(len(scores)), scores, c=c)
        # plt.show()
        print("auc = ", auc)

        threshold = np.quantile(model.pca.fit_anomaly_score, threshold_precent)
        print("threshold = ", threshold)

        predict = np.empty(test_size)
        for i in range(test_size):
            if scores[i] > threshold:
                predict[i] = 1
            else:
                predict[i] = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predict)):
            if predict[i] == 0:
                if test_label[i] == 0:
                    TN = TN + 1
                else:
                    FN = FN + 1
            else:
                if test_label[i] == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1

        print("TP = ", TP, " FP = ", FP, " TN = ", TN, " FN = ", FN)
        f1 = f1_score(test_label, predict)
        # precision = precision_score(test_label, predict)
        # recall = recall_score(test_label, predict)
        acc = accuracy_score(test_label, predict)
        F1s.append(f1)
        ACCs.append(acc)
        print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1,
              "|", acc, "|", 1 - acc, "|")

    f1 = sum(F1s) / float(number)
    acc = sum(ACCs) / float(number)
    print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1,
          "|", acc, "|", 1.0 - acc, "|", file=ff)

ff.close()

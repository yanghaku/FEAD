import numpy as np
import newCNN
import time
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sklearn as sk
import FEAD

data_path = "./MAWILab-GAfeature/mawilab_ga.npy"
labels_path = "./MAWILab-GAfeature/mawilab_label_10w.npy"

train_size = 90000
test_size = 10000

if __name__ == "__main__":
    data = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path)
    model = FEAD.FEAD(data.shape[1])
    LEN = len(labels)
    all_train = data[0:train_size, :]
    all_train_label = labels[0:train_size]

    data_train = all_train[0:train_size, :]
    data_test = data[LEN - test_size:LEN, :]
    label_train = all_train_label[0:train_size]
    label_test = labels[LEN - test_size:LEN]

    model.fit(data_train, label_train)
    res = model.predict(data_test)

    TN = 0
    FN = 0
    TP = 0
    FP = 0
    for j in range(len(res)):
        if res[j] == 0:
            if label_test[j] == 0:
                TN = TN + 1
            else:
                FN += 1
        else:
            if label_test[j] == 0:
                FP = FP + 1
            else:
                TP += 1
    accuracy = accuracy_score(label_test, res)
    precision = precision_score(label_test, res)
    recall = recall_score(label_test, res)
    f1 = f1_score(label_test, res)
    if TN + FP > 0:
        fpr = FP / (TN + FP)
    else:
        fpr = 0
    print("Precision = ", precision, "Recall = ", recall, "F1-score = ", f1, "FPR=", fpr)

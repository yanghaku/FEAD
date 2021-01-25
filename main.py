import numpy as np
import data_loader
import newCNN
import time
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import sklearn as sk
import FEAD

data_path = ".\\20_30\\20180401_new_10w_20_30.tsv.npy"
labels_path = ".\\20_30\label_20180401_new_10w_20_30.csv.npy"


train_size = 80000
test_size = 19999

if __name__ == "__main__":
    data_in = data_loader.MnistLoader(data_path,labels_path)
    model = FEAD.FEAD()
    data = data_in.data
    labels = data_in.labels
    LEN = len(labels)
    all_train = data[0:80000,:]
    all_train_label = labels[0:80000]

    #dice = np.random.permutation(80000)
    #all_train = all_train[dice,:]
    #all_train_label = all_train_label[dice]

    data_train = all_train[0:train_size,:]
    data_test = data[LEN-test_size:LEN,:]
    label_train = all_train_label[0:train_size]
    label_test = labels[LEN-test_size:LEN]

    model.fit(data_train,label_train)
    res = model.predict(data_test)

    accuracy = accuracy_score(label_test, res)
    precision = sk.metrics.precision_score(res, label_test)
    recall = sk.metrics.recall_score(res, label_test)
    f1 = sk.metrics.f1_score(res, label_test)

    print("Precision  = ", precision, "TPR = ", recall, "F1-score", f1)
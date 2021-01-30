import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import FEAD

IDS = False

if IDS:
    data_path = "./MAWILab-GAfeature/ids_30w.npy"
    labels_path = "./MAWILab-GAfeature/ids_label_30w.npy"
    test_size = 29999
    train_size = 270000

else:
    data_path = "./MAWILab-GAfeature/mawilab_ga.npy"
    labels_path = "./MAWILab-GAfeature/mawilab_label_10w.npy"
    train_size = 90000
    test_size = 10000

if __name__ == "__main__":
    data = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path)
    model = FEAD.FEAD(data.shape[1])

    data_train = data[0:train_size, :]
    data_test = data[train_size:train_size + test_size, :]
    label_train = labels[0:train_size]
    label_test = labels[train_size:train_size + test_size]

    # for e in range(16):
    #     model.fit(data_train, label_train)
    #     res = model.predict(data_test)
    #     print("e=", e + 1, "f1=", f1_score(label_test, res))
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

    print("TN = ", TN, "TP= ", TP, "FN= ", FN, "FP=", FP)
    print("F1-score = ", f1, "Precision = ", precision, "Recall = ", recall, "FPR=", fpr)

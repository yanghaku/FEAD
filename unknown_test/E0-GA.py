import sys
sys.path.append("../")
import torch
torch.manual_seed(0)
import platform
import numpy as np
np.random.seed(10)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import FEAD

if "Windows" in platform.platform() :
    P_dir = 'D:\\Dataset\\KITSUNE\\GA90\\'
    dir = [
            P_dir + 'ARP_MitM\\',
            P_dir + 'Fuzzing\\',
            P_dir + 'Mirai\\',
            P_dir + 'SSDP_Flood\\',
            ]
    d_name = ['cut_10w.npy'] * 4
    l_name = ['cut_10w_label.npy'] * 4

else:
    dir = [
            '/home/zy/data2/ARP_MitM/',
            '/home/zy/data2/Fuzzing/',
            '/home/zy/data2/Mirai/',
            '/home/zy/data2/SSDP_Flood/',
            ]
    d_name = ['cut_10w.npy'] * 4
    l_name = ['cut_10w_label.npy'] * 4

if __name__ == "__main__":
    train_size = 90000
    test_size = 10000

    data_train = []
    data_test = []
    label_train = []
    label_test = []

    for i in range(4):
        data = np.load(dir[i] + d_name[i]).astype(np.float32)
        labels = np.load(dir[i] + l_name[i]).astype(np.int64)
        data_train.append(data[0:train_size, :])
        data_test.append(data[train_size:train_size + test_size, :])
        label_train.append(labels[0:train_size])
        label_test.append(labels[train_size:train_size + test_size])

    data_train = np.concatenate(data_train)
    data_test = np.concatenate(data_test)
    label_train = np.concatenate(label_train)
    label_test = np.concatenate(label_test)

    print(data_train.shape, data_test.shape, label_train.shape, label_test.shape)


    dice = np.random.permutation(train_size)
    data_train = data_train[dice,:]
    label_train = label_train[dice]

    model = FEAD.FEAD(data_train.shape[1])


    for ee in range(50):
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

        num = [0,0]
        for i in label_test:
            num[i]+=1
        print("test: ",num)
        print("TN = ", TN, "TP= ", TP, "FN= ", FN, "FP=", FP)
        print("F1-score = ", f1, "Precision = ", precision, "Recall = ", recall, "FPR=", fpr)
        torch.save(model.model, "E0-GA.model-"+str(ee))

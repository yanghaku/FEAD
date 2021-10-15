import sys
sys.path.append("../")
import platform
import torch
torch.manual_seed(1)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import FEAD

if "Windows" in platform.platform() :
    P_dir = 'D:\Dataset\KITSUNE\\kitsune-GA\\'
    dir = [
        P_dir + 'Mirai\\',
        P_dir + 'Fuzzing\\',
        P_dir + 'SSDP_Flood\\',
        P_dir + 'ARP_MitM\\',
        P_dir + 'Active_Wiretap\\',
        P_dir + 'SSL_Renegotiation\\',
        P_dir + 'SYN_DoS\\',
        P_dir + 'OS_Scan\\',
        P_dir + 'Video_Injection\\'
        ]
else:
    dir = [ '/home/zy/data2/Mirai/', '/home/zy/data2/Fuzzing/', '/home/zy/data2/SSDP_Flood/', '/home/zy/data2/ARP_MitM/',
        '/home/zy/data2/Active_Wiretap/', '/home/zy/data2/SSL_Renegotiation/', '/home/zy/data2/SYN_DoS/',
        '/home/zy/data2/OS_Scan/', '/home/zy/data2/Video_Injection/']

d_name = 'cut_10w.npy'
l_name = 'cut_10w_label.npy'


model = FEAD.FEAD(90)
model.model = torch.load("2-E0-GA.model").to(FEAD.device)

def test(msg, lll):
    train_size = 90000
    test_size = 10000


    data_test = []
    label_test = []

    for i in range(len(lll)):
        data = np.load(dir[lll[i]] + d_name).astype(np.float32)
        labels = np.load(dir[lll[i]] + l_name)

        data_test.append(data[train_size:train_size + test_size, :])
        label_test.append(labels[train_size:train_size + test_size])

    data_test = np.concatenate(data_test)
    label_test = np.concatenate(label_test)
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
    m=[]
    for i in lll:
        if "Windows" in platform.platform() :
            m.append(dir[i].split('\\')[4])
        else:
            m.append(dir[i].split('/')[4])

    print(msg, m)
    print("TN = ", TN, "TP= ", TP, "FN= ", FN, "FP=", FP)
    print("F1-score = ", f1, "Precision = ", precision, "Recall = ", recall, "FPR=", fpr, "\n\n")
    return msg, f1

ff = []
ff.append(test("E0",[0,1,2,3]))

ff.append(test("gE1",[0,1,2,4]))
ff.append(test("gE2",[0,2,4,6]))
ff.append(test("gE3",[0,4,6,7]))
ff.append(test("gE4",[4,6,7,8]))

ff.append(test("bE1",[0,1,2,5]))
ff.append(test("bE2",[0,2,5,8]))
ff.append(test("bE3",[0,5,8,7]))
ff.append(test("bE4",[5,8,7,6]))

for i in ff:
    print(i[0],i[1])
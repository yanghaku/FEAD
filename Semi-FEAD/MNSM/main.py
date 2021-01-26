from MSNM import MSNM
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

IDS = False

if IDS:
    data_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\data_30w_des.tsv.npy"
    labels_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\labels_30w_des.csv.npy"
    data = np.load(data_path).astype(np.float64)
    labels = np.load(labels_path)
    ff = open("./msnm_ids.md", "w")
    lst = [270, 540, 1350, 2700, 5400, 13500, 27000]

else:
    data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
    labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
    data = np.load(data_path).astype(np.float64)
    labels = np.load(labels_path)
    ff = open("./msnm_mawilab.md", "w")
    lst = [180, 450, 900, 1800, 4500, 9000, 18000]

for label_train_size in [180]:  # lst:
    if IDS:
        test_size = 29999
        train_size = 270000
    else:
        test_size = 10000
        train_size = 90000

    unlabeled_train_size = train_size - label_train_size

    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    ran_dice = np.random.permutation(train_size)
    all_train_data = data[ran_dice, :]
    all_train_label = labels[ran_dice]

    labeled_train = all_train_data[0:label_train_size, :]
    train_label = all_train_label[0:label_train_size]

    unlabeled_train = all_train_data[label_train_size: train_size, :]
    unlabeled_label = all_train_label[label_train_size: train_size]
    unlabeled_id = []
    for i in range(unlabeled_label.shape[0]):
        if unlabeled_label[i] == 0:
            unlabeled_id.append(i)
    unlabeled_train = unlabeled_train[unlabeled_id]

    all_test_data = data[train_size:train_size + test_size, :]
    all_test_label = labels[train_size:train_size + test_size]

    model = MSNM(data.shape[1])
    print("training.....")
    if IDS:
        n_component = 15
    else:  # mawilab
        n_component = 1
    K = 10
    rc = 0.01
    epoch = 5
    model.train(labeled_train, train_label, unlabeled_train, n_component, K, rc, epoch)

    print("test....")
    scores = model.test(all_test_data)
    auc = roc_auc_score(all_test_label, scores)
    c = []
    for i in all_test_label:
        if i == 1:
            c.append('r')
        else:
            c.append('g')
    plt.scatter(range(len(scores)), scores, c=c)
    plt.show()
    print("auc = ", auc)

    # predict = model.predict(all_test_data)
    x = model.pca.threshold
    predict = []
    for i in scores:
        if i > x:
            predict.append(1)
        else:
            predict.append(0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predict)):
        if predict[i] == 0:
            if all_test_label[i] == 0:
                TN = TN + 1
            else:
                FN = FN + 1
        else:
            if all_test_label[i] == 1:
                TP = TP + 1
            else:
                FP = FP + 1

    print("TP = ", TP, " FP = ", FP, " TN = ", TN, " FN = ", FN)
    f1 = f1_score(all_test_label, predict)
    precision = precision_score(all_test_label, predict)
    recall = recall_score(all_test_label, predict)
    acc = accuracy_score(all_test_label, predict)
    print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|",
          acc, "|", 1 - acc, "|")

    print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|",
          acc, "|", 1 - acc, "|", file=ff)

ff.close()

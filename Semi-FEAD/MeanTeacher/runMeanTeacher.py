import sys
import torch
import numpy as np
from meanTeacher import train, Test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)
sys.path.append("..")
from SemiDataLoader import getData

sys.path.append("../..")
import newCNN

IDS = False

if IDS:
    ff = open("./res_mean_ids.md", "w")
    data_path = "../../MAWILab-GAfeature/ids_30w.npy"
    labels_path = "../../MAWILab-GAfeature/ids_label_30w.npy"
    test_size = 29999
    train_size = 270000
    lst = [540, 1350, 2700, 5400, 13500, 27000]

else:
    ff = open("./res_mean-teacher.md", "w")
    data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
    labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
    test_size = 10000
    train_size = 90000
    lst = [180, 450, 900, 1800, 4500, 9000, 18000]

for train_labeled_size in lst:
    F1s = []
    Precisions = []
    Recalls = []
    ACCs = []
    number = 10  # 取number次均值

    train_unlabeled_size = train_size - train_labeled_size

    for __ in range(number):
        train_labeled_data, train_label, train_unlabeled_data, test_data, test_label = getData(data_path, labels_path,
                                                                                               train_labeled_size,
                                                                                               train_unlabeled_size,
                                                                                               test_size)
        unlabeled_label = np.array([-1] * train_unlabeled_size)
        indices = np.random.permutation(train_size)

        all_train_label = torch.from_numpy(
            (np.concatenate((train_label, unlabeled_label)))[indices].astype(np.longlong)).to(device)
        all_train_data = torch.from_numpy((np.concatenate((train_labeled_data, train_unlabeled_data)))[indices]).to(
            device)

        test_data = torch.from_numpy(test_data).to(device)
        test_label = torch.from_numpy(test_label)

        shape_1 = train_labeled_data.shape[1]

        step_counter = 0
        stu = newCNN.Model(shape_1).to(device)
        teacher = newCNN.Model(shape_1).to(device)
        for param in teacher.parameters():
            param.detach_()

        optimizer = torch.optim.Adam(stu.parameters())
        f1 = 0
        precision = 0
        recall = 0
        acc = 0
        for epoch in range(4):
            print("epoch: ", epoch)
            train(stu, teacher, all_train_data, all_train_label, optimizer, epoch, step_counter)

            print("test....")
            f1, precision, recall, acc = Test(stu, test_data, test_label, test_size)
            print("|", train_labeled_size, "|", f1, "|", precision, "|", recall, "|", acc, "|", 1 - acc, "|")

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        ACCs.append(acc)

    f1 = sum(F1s) / float(number)
    precision = sum(Precisions) / float(number)
    recall = sum(Recalls) / float(number)
    acc = sum(ACCs) / float(number)
    print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1, "|", acc, "|", 1.0 - acc, "|",
          file=ff)

ff.close()

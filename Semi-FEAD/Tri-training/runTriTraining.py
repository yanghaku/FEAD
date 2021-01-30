from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from TriTraining import TriTraining

sys.path.append('..')

from SemiDataLoader import getData

IDS = False

if IDS:
    ff = open("./res_triTraining-ids.md", "w")
    data_path = "../../MAWILab-GAfeature/ids_30w.npy"
    labels_path = "../../MAWILab-GAfeature/ids_label_30w.npy"
    test_size = 29999
    train_size = 270000
    lst = [540, 1350, 2700, 5400, 13500, 27000]
    lr = 0.0001

else:
    ff = open("./res_triTraining.md", "w")
    data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
    labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
    test_size = 10000
    train_size = 90000
    lst = [180, 450, 900, 1800, 4500, 9000, 18000]
    lr = 0.001

for train_labeled_size in lst:
    train_unlabeled_size = train_size - train_labeled_size

    number = 10  # 取10次平均
    F1s = []
    ACCs = []

    for r in range(number):
        train_labeled_data, train_label, train_unlabeled_data, test_data, test_label = getData(data_path, labels_path,
                                                                                               train_labeled_size,
                                                                                               train_unlabeled_size,
                                                                                               test_size)

        TT = TriTraining(train_labeled_data.shape[1], lr)

        TT.fit(train_labeled_data, train_label, train_unlabeled_data)
        res = TT.predict(test_data)

        accuracy = accuracy_score(test_label, res)
        precision = precision_score(test_label, res)
        recall = recall_score(test_label, res)
        f1 = f1_score(test_label, res)
        test_error = 1.0 - accuracy
        F1s.append(f1)
        ACCs.append(accuracy)

        print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1, "|", precision, "|", recall,
              "|", accuracy, "|", test_error, "|")

    f1 = sum(F1s) / float(number)
    acc = sum(ACCs) / float(number)
    print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1, "|", acc, "|", 1.0 - acc, "|",
          file=ff)

ff.close()

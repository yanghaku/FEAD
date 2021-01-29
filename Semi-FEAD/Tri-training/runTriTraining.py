from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
from TriTraining import TriTraining

sys.path.append('..')

from SemiDataLoader import getData

ff = open("./res_triTraining.md", "w")

for train_labeled_size in [180]:  # [180, 450, 900, 1800, 4500, 9000, 18000]:
    test_size = 10000
    train_size = 90000

    train_unlabeled_size = train_size - train_labeled_size

    train_labeled_data, train_label, train_unlabeled_data, test_data, test_label = getData(train_labeled_size,
                                                                                           train_unlabeled_size,
                                                                                           test_size)

    TT = TriTraining(train_labeled_data.shape[1])

    TT.fit(train_labeled_data, train_label, train_unlabeled_data)
    res = TT.predict(test_data)

    accuracy = accuracy_score(test_label, res)
    precision = precision_score(test_label, res)
    recall = recall_score(test_label, res)
    f1 = f1_score(test_label, res)
    test_error = 1.0 - accuracy

    print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1, "|", precision, "|", recall,
          "|", accuracy, "|", test_error, "|")
    print("|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", f1, "|", precision, "|", recall,
          "|", accuracy, "|", test_error, "|", file=ff)

ff.close()

import numpy as np


def getData(data_path, labels_path, train_labeled_size, train_unlabeled_size, test_size):
    # 加载数据
    data = np.load(data_path).astype(np.float32)
    labels = np.load(labels_path)

    train_size = train_labeled_size + train_unlabeled_size

    train_data = data[0:train_size]
    train_label = labels[0:train_size]

    choose_indice = np.zeros(train_size).astype(np.int)
    p = 2017  # 499 1009
    x = 0
    for _ in range(train_labeled_size):
        choose_indice[x] = 1
        x = (x + p) % train_size

    labeled_indice = np.zeros(train_labeled_size).astype(np.int)
    unlabeled_indice = np.zeros(train_unlabeled_size).astype(np.int)
    x = 0
    y = 0
    for i in range(train_size):
        if choose_indice[i] == 1:
            labeled_indice[x] = i
            x += 1
        else:
            unlabeled_indice[x] = i
            y += 1

    # train_labeled_data = train_data[0:train_labeled_size, :]
    # train_label = train_label[0:train_labeled_size]
    # train_unlabeled_data = train_data[train_labeled_size:train_size, :]

    train_labeled_data = train_data[labeled_indice, :]
    train_label = train_label[labeled_indice]
    train_unlabeled_data = train_data[unlabeled_indice, :]

    test_data = data[train_size:train_size + test_size, :]
    test_label = labels[train_size:train_size + test_size]

    return train_labeled_data, train_label, train_unlabeled_data, test_data, test_label

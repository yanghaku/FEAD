from get_mixmatch_loss import get_mixmatch_loss
from mixmatch_loader import MixMatchLoader
import torch

import sys

sys.path.append("../..")
import newCNN
import numpy as np
from DataLoad import DataLoad
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)

batch_size = 8
data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
data = np.load(data_path)
labels = np.load(labels_path)


def Test(model, test, test_label, test_size):
    model.eval()
    test_batch = test_size // batch_size
    prediction = np.zeros(test_size, dtype=np.uint8)
    for i in range(test_batch):
        inputs = Variable(test[i * batch_size:min((i + 1) * batch_size, test_size), :],
                          requires_grad=False).view(-1, 1, test.shape[1]).to(device)

        outputs = model(inputs)
        pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
        prediction[i * batch_size:min((i + 1) * batch_size, test_size)] = pred

    precision = precision_score(test_label, prediction)
    recall = recall_score(test_label, prediction)
    f1 = f1_score(test_label, prediction)
    acc = accuracy_score(test_label, prediction)

    return f1, precision, recall, acc



for label_train_size in [180]:  # [180, 450, 900, 1800, 4500, 9000, 18000]:  # [90000]:
    print("the label size is", label_train_size)

    test_size = 10000
    train_size = 90000
    unlabel_train_size = train_size - label_train_size

    all_train_data = data[0:train_size, :]
    all_train_label = labels[0:train_size]

    ran_dice = np.random.permutation(train_size)
    all_train_data = all_train_data[ran_dice, :]
    all_train_label = all_train_label[ran_dice]

    labeled_train = all_train_data[0:label_train_size, :]
    train_label = all_train_label[0:label_train_size]

    # unlabeled_train = data[label_train_size:label_train_size+unlabel_train_size,:]
    unlabeled_train = all_train_data[label_train_size: train_size, :]
    unlabeled_label = np.array([-1] * unlabel_train_size)

    test = torch.from_numpy(data[train_size:train_size + test_size, :])
    test_label = torch.from_numpy(labels[train_size:train_size + test_size])

    dataset_unlabeled = torch.utils.data.TensorDataset(torch.from_numpy(unlabeled_train))
    # loader_labeled = DataLoad(torch.from_numpy(labeled_train), torch.from_numpy(train_label), batch_size)# 1
    loader_labeled = DataLoad(labeled_train, train_label, batch_size)
    model = newCNN.Model(data.shape[1]).to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01

    # train model use labeled data
    print("pre-train")
    model.train()
    n_epochs = 30
    train_batch = label_train_size // batch_size

    for epoch in range(n_epochs):
        running_correct = 0
        loss_sum = 0
        for i in range(train_batch):
            inputs = Variable(
                torch.from_numpy(labeled_train[i * batch_size:min((i + 1) * batch_size, label_train_size)]),
                requires_grad=False).view(-1, 1, data.shape[1]).to(device)
            targets = Variable(
                torch.from_numpy(train_label[i * batch_size:min((i + 1) * batch_size, label_train_size)]),
                requires_grad=False).to(device)

            output = model(inputs)
            # print(output)
            _, pred = torch.max(output.data, 1)
            # print(pred)
            optimizer.zero_grad()
            loss = cost(output, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            running_correct += torch.sum(pred == targets).data

        print("the ", epoch, " epoch Loss is :{:.4f},Train acc is :{:.4f}%".format(loss_sum / label_train_size,
                                                                                   100 * running_correct // label_train_size
                                                                                   ))
    print("pre-train is over")

    loader_mixmatch = MixMatchLoader(
        loader_labeled,
        dataset_unlabeled,
        model,
        output_transform=torch.sigmoid,
        K=2,
        T=1.0,
        alpha=0.5
    )

    criterion = get_mixmatch_loss(
        criterion_labeled=torch.nn.BCEWithLogitsLoss(),
        output_transform=torch.sigmoid,
        K=2,
        weight_unlabeled=5.,  # 100.
        criterion_unlabeled=torch.nn.MSELoss()
    )

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_acc = 0.0
    f1, precision, recall, acc = Test(model, test, test_label, test_size)
    if f1 > best_f1:
        best_f1, best_precision, best_recall, best_acc = f1, precision, recall, acc
    print("|label size is", label_train_size, "|label percent is", label_train_size / train_size * 100, "|F1 is", f1,
          "|precision is", precision, "|recall is", recall,
          "|acc is",
          acc, "| 1-acc is", 1.0 - acc, "|")

    model.train()
    print("train: ")

    # for epoch in range(n_epochs):
    # loss_sum = 0
    # running_correct = 0
    # num = 0
    for epoch in range(10):
        print("the ", epoch, " th epoch")
        num = 0
        running_correct = 0
        loss_sum = 0
        for batch in loader_mixmatch:
            inputs = Variable(batch['features'].to(device), requires_grad=False).view(-1, 1, data.shape[1])
            label = Variable(batch['targets'].to(device), requires_grad=False)
            num = num + inputs.shape[0]
            logits = model(inputs)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(logits.data, 1)
            _, la = torch.max(label.data, 1)
            running_correct += torch.sum(pred == la)
            loss_sum += loss.data.cpu()
            # 混合训练
            index = random.randint(0, train_batch - 1)
            inputs = Variable(
                torch.from_numpy(labeled_train[index * batch_size:min((index + 1) * batch_size, label_train_size)]),
                requires_grad=False).view(-1, 1, data.shape[1]).to(device)
            targets = Variable(
                torch.from_numpy(train_label[index * batch_size:min((index + 1) * batch_size, label_train_size)]),
                requires_grad=False).to(device)
            output = model(inputs)
            optimizer.zero_grad()
            loss1 = cost(output, targets)
            loss1.backward()
            optimizer.step()
            # 混合训练结束

        print("the ", epoch, " epoch Loss is :{:.4f},Train acc is :{:.4f}%".format(loss_sum / num,
                                                                                   100 * running_correct // num
                                                                                   ))
        print("test....")
        f1, precision, recall, acc = Test(model, test, test_label, test_size)
        print("|label size is", label_train_size, "|label percent is", label_train_size / train_size * 100, "|F1 is",
              f1, "|precision is", precision, "|recall is", recall,
              "|acc is",
              acc, "| 1-acc is", 1 - acc, "|")
        if f1 > best_f1:
            best_f1, best_precision, best_recall, best_acc = f1, precision, recall, acc

    print("\n\n|", label_train_size, "|", label_train_size / train_size * 100, "|", best_f1, "|", best_precision, "|",
          best_recall, "|", best_acc, "|", 1.0 - best_acc, "|")

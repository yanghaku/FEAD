from get_mixmatch_loss import get_mixmatch_loss
from mixmatch_loader import MixMatchLoader
import torch

import sys

sys.path.append("../..")
import newCNN

sys.path.append("..")
from SemiDataLoader import getData

import numpy as np
from DataLoad import DataLoad
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)

batch_size = 8
data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
data = np.load(data_path)
labels = np.load(labels_path)


def Test(model, test_data, test_label, test_size):
    model.eval()
    test_batch = test_size // batch_size
    prediction = np.zeros(test_size, dtype=np.uint8)
    for i in range(test_batch):
        inputs = Variable(test_data[i * batch_size:min((i + 1) * batch_size, test_size), :],
                          requires_grad=False).view(-1, 1, test_data.shape[1])

        outputs = model(inputs)
        pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
        prediction[i * batch_size:min((i + 1) * batch_size, test_size)] = pred

    precision = precision_score(test_label, prediction)
    recall = recall_score(test_label, prediction)
    f1 = f1_score(test_label, prediction)
    acc = accuracy_score(test_label, prediction)

    return f1, precision, recall, acc


for train_labeled_size in [900]:  # [180, 450, 900, 1800, 4500, 9000, 18000]:  # [90000]:
    print("the label size is", train_labeled_size)
    test_size = 10000
    train_size = 90000

    train_unlabeled_size = train_size - train_labeled_size

    train_labeled_data, train_label, train_unlabeled_data, test_data, test_label = getData(train_labeled_size,
                                                                                           train_unlabeled_size,
                                                                                           test_size)
    test_data = torch.from_numpy(test_data).to(device)
    dataset_unlabeled = torch.utils.data.TensorDataset(torch.from_numpy(train_unlabeled_data))
    loader_labeled = DataLoad(train_labeled_data, train_label, batch_size)
    model = newCNN.Model(data.shape[1]).to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pre_train_epochs = 10
    train_epochs = 10

    # train model use labeled data
    print("pre-train")
    model.train()

    train_batch = train_labeled_size // batch_size

    for epoch in range(pre_train_epochs):
        running_correct = 0
        loss_sum = 0
        for i in range(train_batch):
            inputs = Variable(
                torch.from_numpy(train_labeled_data[i * batch_size:min((i + 1) * batch_size, train_labeled_size)]),
                requires_grad=False).view(-1, 1, data.shape[1]).to(device)
            targets = Variable(
                torch.from_numpy(train_label[i * batch_size:min((i + 1) * batch_size, train_labeled_size)]),
                requires_grad=False).to(device)

            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            optimizer.zero_grad()
            loss = cost(output, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            running_correct += sum(pred == targets).data

        print("the ", epoch, " epoch Loss is :{:.4f},Train acc is :{:.4f}%".
              format(loss_sum / train_labeled_size, 100 * running_correct / train_labeled_size))
    print("pre-train is over")

    loader_mixmatch = MixMatchLoader(
        loader_labeled,
        dataset_unlabeled,
        model,
        output_transform=torch.sigmoid,
        K=4,
        T=1.0,
        alpha=0.75
    )

    criterion = get_mixmatch_loss(
        criterion_labeled=torch.nn.BCEWithLogitsLoss(),
        output_transform=torch.sigmoid,
        K=4,
        weight_unlabeled=10.,  # 100.
        criterion_unlabeled=torch.nn.MSELoss()
    )

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_acc = 0.0
    f1, precision, recall, acc = Test(model, test_data, test_label, test_size)
    if f1 > best_f1:
        best_f1, best_precision, best_recall, best_acc = f1, precision, recall, acc
    print("|label size is", train_labeled_size, "|label percent is", train_labeled_size / train_size * 100, "|F1 is",
          f1, "|precision is", precision, "|recall is", recall,
          "|acc is", acc, "| 1-acc is", 1.0 - acc, "|")

    model.train()
    print("train: ")

    for epoch in range(train_epochs):
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
            running_correct += sum(pred == la)
            loss_sum += loss.data.cpu()
            # 混合训练
            # index = random.randint(0, train_batch - 1)
            # inputs = Variable(
            #     torch.from_numpy(labeled_train[index * batch_size:min((index + 1) * batch_size, label_train_size)]),
            #     requires_grad=False).view(-1, 1, data.shape[1]).to(device)
            # targets = Variable(
            #     torch.from_numpy(train_label[index * batch_size:min((index + 1) * batch_size, label_train_size)]),
            #     requires_grad=False).to(device)
            # output = model(inputs)
            # optimizer.zero_grad()
            # loss1 = cost(output, targets)
            # loss1.backward()
            # optimizer.step()
            # 混合训练结束

        print("the ", epoch, " epoch Loss is :{:.4f},Train acc is :{:.4f}%".format(loss_sum / num,
                                                                                   100 * running_correct // num
                                                                                   ))
        print("test....")
        f1, precision, recall, acc = Test(model, test_data, test_label, test_size)
        print("|label size is", train_labeled_size, "|label percent is", train_labeled_size / train_size * 100,
              "|F1 is", f1, "|precision is", precision, "|recall is", recall,
              "|acc is", acc, "| 1-acc is", 1 - acc, "|")
        if f1 > best_f1:
            best_f1, best_precision, best_recall, best_acc = f1, precision, recall, acc

    print("\n\n|", train_labeled_size, "|", train_labeled_size / train_size * 100, "|", best_f1, "|", best_precision,
          "|", best_recall, "|", best_acc, "|", 1.0 - best_acc, "|")

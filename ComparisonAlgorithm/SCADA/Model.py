import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class NN(nn.Module):
    def __init__(self, in_size):
        super(NN, self).__init__()

        self.in_size = in_size
        self.hidden_size = 100
        self.lstm = nn.LSTM(self.in_size, self.hidden_size)
        self.fnn1 = nn.Linear(self.in_size, self.hidden_size)

        self.fnn2 = nn.Linear(self.hidden_size*2, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        fnn_out = self.fnn1(x)

        h = torch.cat((torch.sigmoid(lstm_out),torch.sigmoid(fnn_out)),dim=-1)
        
        return self.fnn2(h).view((-1,2))

class Model:
    def __init__(self, in_size):
        self.sz = in_size
        self.model = NN(in_size)
        self.cost = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def fit(self, X, Y):
        data = X
        label = Y
        n_epochs = 4
        batch_size = 8
        self.model.train()
        train_size = len(X)
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        print("the size is", train_size)
        train_batch = train_size // batch_size

        for epoch in range(n_epochs):
            loss_sum = 0
            running_correct = 0
            print("training Epoch{}/{}".format(epoch, n_epochs))
            train_begin = time.time()
            for i in range(train_batch):
                inputs = Variable(data[i * batch_size:min((i + 1) * batch_size, train_size), :],
                                  requires_grad=False).view(-1, 1, self.sz)
                targets = Variable(label[i * batch_size:min((i + 1) * batch_size, train_size)],
                                   requires_grad=False)
                num = min((i + 1) * batch_size, train_size) - i * batch_size
                if num < batch_size:
                    break
                outputs = self.model(inputs)
                _, pred = torch.max(outputs.data, 1)
                self.optimizer.zero_grad()
                loss = self.cost(outputs, targets)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.data.cpu().numpy()
                running_correct += sum(pred == targets)

            stop = time.time()
            print(epoch, "epoch time is", stop - train_begin)
            print("Loss is : {:.4f},Train acc is :{:.4f}%".format(loss_sum / train_size,
                                                                  100 * running_correct / train_size))

    def predict(self, X):
        self.model.eval()
        batch_size = 8
        test_size = len(X)
        test_batch = test_size // batch_size
        data = torch.from_numpy(X)
        prediction = np.zeros(test_size, dtype=np.uint8)

        for i in range(test_batch):
            inputs = Variable(data[i * batch_size:min((i + 1) * batch_size, test_size), :],
                              requires_grad=False).view(-1, 1, self.sz)
            num = min((i + 1) * batch_size, test_size) - i * batch_size
            if num < batch_size:
                break
            outputs = self.model(inputs)
            pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
            prediction[i * batch_size:min((i + 1) * batch_size, test_size)] = pred

        return prediction



def measure(label_test, res):
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
    return f1
from __future__ import division, print_function
import torch
import newCNN
import numpy as np
import time
import sklearn as sk
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import pandas as pd
class FEAD:
    def __init__(self):
        self.model = newCNN.Model()
        self.cost = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01

    def fit(self,X,Y):
        data = X
        label = Y
        n_epochs = 4
        batch_size = 8
        self.model.train()
        train_size = len(X)
        print("the size is",train_size)
        train_batch = train_size//batch_size

        for epoch in range(n_epochs):
            loss_sum = 0
            running_correct = 0
            start = time.time()
            print("training Epoch{}/{}".format(epoch,n_epochs))
            train_begin = time.time()
            for i in range(train_batch):
                inputs = Variable(torch.from_numpy(data[i * batch_size:min((i + 1) * batch_size, train_size), :]),
                                  requires_grad=False).view(-1, 1, 90)
                targets = Variable(torch.from_numpy(label[i * batch_size:min((i + 1) * batch_size, train_size)]),
                                   requires_grad=False)

                outputs = self.model(inputs)
                _,pred = torch.max(outputs.data,1)
                self.optimizer.zero_grad()
                loss = self.cost(outputs,targets)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.data.cpu().numpy()
                running_correct += torch.sum(pred == targets)

            stop = time.time()
            print(epoch ,"epoch time is",stop - train_begin)
            print("Loss is : {:.4f},Train acc is :{:.4f}%".format(loss_sum / (train_size),
                                                                                      100 * running_correct / (train_size)))


    def predict(self,X):
        self.model.eval()
        batch_size = 8
        test_size = len(X)
        test_batch = test_size//batch_size
        data = X
        prediction = np.zeros(test_size,dtype=np.uint8)

        for i in range(test_batch):
            inputs = Variable(torch.from_numpy(data[i * batch_size:min((i + 1) * batch_size, test_size), :]),
                              requires_grad=False).view(-1, 1, 90)
            outputs = self.model(inputs)
            _,pred = torch.max(outputs.data,1)
            prediction[i*batch_size:min((i+1)*batch_size,test_size)] = np.argmax(outputs.data.cpu().numpy(),axis=1)
        return prediction


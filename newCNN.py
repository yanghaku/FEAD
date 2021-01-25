from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 1*90  1*100
            torch.nn.Conv1d(1, 4, kernel_size=5, stride=1),  # 86 1*96
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),  # 43 48

            torch.nn.Conv1d(4, 8, kernel_size=5, stride=1),  # 39 44
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),  # 19 22
            # torch.nn.Conv1d(16,120,kernel_size=5,stride=1),#15
            # torch.nn.ReLU()
        )
        self.dense = torch.nn.Sequential(
            #torch.nn.Linear(90,32),
            torch.nn.Linear(8*19,32),
            #torch.nn.Linear(22*8,32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(32,2),
        )

    def forward(self, x):
        x=self.conv1(x)
        #print(x.shape)
        x=x.view(-1, 8*19)
        #x = x.view(-1,22*8)
        x=self.dense(x)
        return x
        #return torch.nn.functional.log_softmax(x)
        #return torch.nn.functional.softmax(x)
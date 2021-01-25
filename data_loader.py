# data loader for mnist dataset
# the path is based on root dictory of this repo

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from FeatureExtractor import *
import pandas as pd
from sklearn.decomposition import PCA

class MnistLoader(object):                   # number of classes
    def __init__(self,  data_path='',labels_path=''):
        '''
        :param data_path: the path to mnist dataset
        '''

        type =data_path.split('.')[-1]
        if(type == "npy"):
            self.data = np.load(data_path)
            self.labels = np.load(labels_path)
        else:
            self.data = []
            self.labels = []
            self.FE = FE(data_path)

            self._load(data_path,labels_path)
            np.save(data_path+'.npy',self.data)
            np.save(labels_path+'.npy',self.labels)
            data_path += '.npy'
            labels_path += '.npy'
            self.data = np.load(data_path)
            self.labels = np.load(labels_path)


        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.int64)

        # from sklearn.preprocessing import StandardScaler
        # s1 = StandardScaler()
        # s1.fit(self.data)
        # self.data = s1.transform(self.data)

        self.data_size_0 = self.data.shape[0]
        self.data_size_1 = self.data.shape[1]
        print("the shape is ",self.data_size_0,self.data_size_1)
        print("the shape is",self.labels.shape)
    # load data according to different configurations
    def _load(self, data_path='data',labels_path = ""):
        index = 0
        start =time.time()

        while True:
            index += 1
            if index % 1000 == 0 :
                print(index)
            x = self.FE.get_next_vector()
            if len(x) == 0:
                break
            self.data.append(x)
        #in_labels = pd.read_csv(labels_path)
        #in_labels = pd.read_csv(labels_path) #names = ["index","x"]
        in_labels = pd.read_csv(labels_path)
        #in_labels = pd.read_csv(labels_path)
        siz = len(in_labels)
        for i in range(siz):
            self.labels.append(in_labels['0'][i])
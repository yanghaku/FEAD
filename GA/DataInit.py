import torch
from FeatureExtractor import FE
import numpy as np


# 0-1 正则化的类
class Normal01:
    def __init__(self, origin_dim):
        self.data_max = np.ones((origin_dim,)) * (-np.inf)
        self.data_min = np.ones((origin_dim,)) * np.inf

    # 0-1 正则化
    def norm(self, xx):
        for i in range(len(xx)):
            self.data_max[i] = max(self.data_max[i], xx[i])
            self.data_min[i] = min(self.data_min[i], xx[i])
        return ((xx - self.data_min) / (self.data_max - self.data_min + 1e-10)).astype(np.float32)


class DataManager:
    def __init__(self, data_file, label_file, save_data, save_label, L, R):
        print("data ", data_file, "init...")

        data_dim = 235
        self.norm01 = Normal01(data_dim)

        # # 读取label文件
        # f_label = open(label_file, "r", encoding='utf-8')
        # self.label = []
        # for row in f_label:
        #     x, y = row.strip().split(',')
        #     self.label.append(int(y))
        # f_label.close()

        self.fe = FE(data_file, np.inf)
        self.features = []

        x = 0
        while True:
            feature = self.fe.get_next_vector()

            if len(feature) == 0:
                break
            feature = self.norm01.norm(np.array(feature))
            self.features.append(feature)

            x += 1
            if x % 10000 == 0:
                print(x)

        # labels = np.array(self.label[L:R])
        # data = np.array(self.features[L:R])
        data = np.array(self.features)

        # # 打乱
        # indices = np.random.permutation(data.shape[0])
        # labels = labels[indices]
        # data = data[indices]
        # print("num: ", labels.shape[0], data.shape[0])
        # np.save(save_label, labels)
        np.save(save_data, data)

        print(data_file + " data init success")


# d1 = DataManager("D:\\Dataset\\KITSUNE\\Mirai\\test.pcap.tsv", "E:\\dataset\\kitsune\\Mirai_labels.csv",
#                  "./data/mirai_10w.npy", "./data/mirai_label_10w.npy", 50000, 150000)
# #
# d2 = DataManager("D:\\Dataset\\KITSUNE\\ARP_MitM\\test.pcap.tsv", "E:\\dataset\\kitsune\\ARP_MitM_labels.csv",
#                  "./data/arp_10w.npy", "./data/arp_label_10w.npy", 1240000, 1340000)
# d3 = DataManager("D:\\Dataset\\KITSUNE\\SSDP_Flood\\test.pcap.tsv", "E:\\dataset\\kitsune\\SSDP_Flood_labels.csv",
#                  "./data/ssdp_10w.npy", "./data/ssdp_label_10w.npy", 2560000, 2660000)
# #
# d4 = DataManager("D:\\Dataset\\KITSUNE\\Fuzzing\\test.pcap.tsv", "E:\\dataset\\kitsune\\Fuzzing_labels.csv",
#                  "./data/fuzzing_10w.npy", "./data/fuzzing_label_10w.npy", 1250000, 1350000)

# d = DataManager("D:\\Dataset\\mawilab20180401\\201806031400.pcap\\mawilab_20_30w.pcap.tsv",
#                 "", "./data/mawilab_10w.npy", "",0,100000)

d = DataManager("D:\\Dataset\\mawilab20180401\\201804011400.pcap\\20180401_20_30w.pcap.tsv",
                "", "./data/mawilab_0401_10w.npy", "", 0, 100000)

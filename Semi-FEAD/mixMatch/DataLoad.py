import torch
import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise


class DataLoad:

    def __init__(self, d, l, batch_size):
        self.data = d
        self.label = l
        self.now = 0
        self.len = d.shape[0]
        self.batch_size = batch_size
        self.num_workers = 0

        self.my_augmenter = (
            # TimeWarp()
            # + Crop(size=155)
            # + Quantize(n_levels=[10, 20, 30])
            # + Drift(max_drift=(0.1, 0.5))
            # + Reverse()
            AddNoise(scale=0.001)
        )

    def __iter__(self):
        self.now = 0
        return self

    def __next__(self):
        if self.now >= self.len:
            raise (StopIteration)

        fea = []
        label = []

        for i in range(self.now, min(self.now + self.batch_size, self.len)):
            # fea.append(self.my_augmenter.augment(self.data[i, :]))
            x = self.my_augmenter.augment(self.data[i, :])
            # print(self.data[i,:].shape)
            # fea.append(self.data[i,:])
            fea.append(x.astype(np.float32))
            ll = [0.0, 0.0]
            ll[self.label[i]] = 1.0
            label.append(ll)

        # print(fea)
        # print(label)
        # ll = [0.0, 0.0]
        # ll[self.label[self.now]] += 1.0
        # ans = dict({"features": self.data[self.now, :].unsqueeze(0), "targets": torch.Tensor(ll).unsqueeze(0)})
        # self.now += 1
        ans = dict({"features": torch.tensor(fea), "targets": torch.tensor(label)})
        self.now = self.now + self.batch_size
        return ans

    def __len__(self):
        return self.len

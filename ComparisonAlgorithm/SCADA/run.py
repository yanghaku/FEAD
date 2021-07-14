from Model import Model, measure
import numpy as np
import time

IDS = True

if IDS:
    train_size = 270000
    test_size = 30000
    path = 'D:\\Dataset\\IDS2017-Wednesday\\'
    labels = np.load(path + 'label.npy')[:train_size+test_size]
    name = 'ids_kitsune.npy'
else:
    train_size = 90000
    test_size = 10000
    path = 'D:\\Dataset\\mawilab20180401\\201806031400.pcap\\'
    name = 'mawilab_kitsune.npy'
    labels = np.load(path + 'mawilab_20_30w_label.npy')



if __name__ == "__main__":
    data = np.load(path + name).astype(np.float32)
    model = Model(data.shape[1])

    data_train = data[0:train_size, :]
    data_test = data[train_size:train_size + test_size, :]
    label_train = labels[0:train_size]
    label_test = labels[train_size:train_size + test_size]

    model.fit(data_train, label_train)

    st = time.time()
    res = model.predict(data_test)
    print("test time = ", time.time() - st)

    measure(label_test, res)
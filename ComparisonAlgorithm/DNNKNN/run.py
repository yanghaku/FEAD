from dnnknn import DNNkNN
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np



ACCEPTABLE_ERROR_RATE_FP = 0.05
ACCEPTABLE_ERROR_RATE_FN = 0.05
method = DNNkNN()
method.getDNN().setActivationFunctionHiddenLayer("tanh")
method.getDNN().setNumNeuronsHiddenLayer(200)
method.getDNN().setActivationFunctionOutputLayer("softmax")
method.getDNN().setNumNeuronsOutLayer(2)
method.getDNN().setNumEpochs(4)
method.getDNN().setOptimizer('adam')
method.getDNN().setLoss('binary_crossentropy')

method.getKNN().setKNeighbors(1)
method.getKNN().setAlgorithm('kd_tree')
method.getKNN().setWeights('uniform')

method.setAcceptableErrorRateFP(ACCEPTABLE_ERROR_RATE_FP)
method.setAcceptableErrorRateFN(ACCEPTABLE_ERROR_RATE_FN)


#Attributes to be removed by selection of attributes as Info Gain algorithm
# method.setDeleteAttributes(np.s_[0,1,6,8,9,10,12,13,14,15,16,17,18,19,20,21,23,26,27,35,39])

IDS = True

if IDS:
    train_size = 270000
    test_size = 30000
    path = 'D:\\Dataset\\IDS2017-Wednesday\\'
    labels = np.load(path + 'label.npy')[:train_size+test_size]
    name = 'ids_kitsune.npy'
    out = 'ids'
else:
    train_size = 90000
    test_size = 10000
    path = 'D:\\Dataset\\mawilab20180401\\201806031400.pcap\\'
    name = 'mawilab_kitsune.npy'
    out = 'mawilab'
    labels = np.load(path + 'mawilab_20_30w_label.npy')

data = np.load(path + name)
data_train = data[:train_size, :]
label_train = labels[:train_size]
data_test = data[train_size:train_size+test_size]
label_test = labels[train_size:train_size+test_size]
print(data.shape, data_train.shape, data_test.shape)

method.getDNN().setImputDimNeurons(100)
method.generateModel()

method.fit(data_train, label_train)

pre = method.predict(data_test)

print("test time = ", method.getTestTime())

TN = 0
FN = 0
TP = 0
FP = 0
for j in range(len(pre)):
    if pre[j] == 0:
        if label_test[j] == 0:
            TN = TN + 1
        else:
            FN += 1
    else:
        if label_test[j] == 0:
            FP = FP + 1
        else:
            TP += 1

# accuracy = accuracy_score(label_test, pre)
precision = precision_score(label_test, pre)
recall = recall_score(label_test, pre)
f1 = f1_score(label_test, pre)
if TN + FP > 0:
    fpr = FP / (TN + FP)
else:
    fpr = 0

print("TN = ", TN, "TP= ", TP, "FN= ", FN, "FP=", FP)
print("F1-score = ", f1, "Precision = ", precision, "Recall = ", recall, "FPR=", fpr)

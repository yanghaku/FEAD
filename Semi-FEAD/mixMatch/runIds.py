from mixmatch_pytorch import MixMatchLoader, get_mixmatch_loss
import torch
from torch import nn
import newCNN
import numpy as np
from DataLoad import DataLoad
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device is: ", device)

batch_size = 8
data_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\data_30w_des.tsv.npy"
labels_path = "D:\\Dataset\\IDS2017-Wednesday\\IDS2017-v4.1.0\\labels_30w_des.csv.npy"
data = np.load(data_path).astype(np.float32)
labels = np.load(labels_path)

ff = open("../data/mixmatch_ids.md", "w")


def Test(model, test, test_label, test_size):
    model.eval()
    test_batch = test_size // batch_size
    prediction = np.zeros(test_size, dtype=np.uint8)
    for i in range(test_batch):
        inputs = Variable(test[i * batch_size:min((i + 1) * batch_size, test_size), :],
                          requires_grad=False).view(-1, 1, test.shape[1])
        targets = Variable(test_label[i * batch_size:min((i + 1) * batch_size, test_size)],
                           requires_grad=False)

        num = min((i + 1) * batch_size, test_size) - i * batch_size
        if num < batch_size:
            break

        outputs = model(inputs)

        pred = np.argmax(outputs.data.cpu().numpy(), axis=1)
        prediction[i * batch_size:min((i + 1) * batch_size, test_size)] = pred

    precision = precision_score(test_label, prediction)
    recall = recall_score(test_label, prediction)
    f1 = f1_score(test_label, prediction)
    acc = accuracy_score(test_label, prediction)

    return f1, precision, recall, acc

for pp in range(50):
    for label_train_size in [2700*9]: #[270, 540, 1350, 2700, 5400, 13500, 27000]:
    # for label_train_size in [180, 450, 900, 1800, 4500, 9000, 18000, 90000]:  # [90000]:
        step_counter = 0
        stu = newCNN.Model(data.shape[1])
        teacher = newCNN.Model(data.shape[1])
        for param in teacher.parameters():
            param.detach_()

        test_size = 29999
        train_size = 270000
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
        loader_labeled = DataLoad(torch.from_numpy(labeled_train), torch.from_numpy(train_label), 2)

        model = newCNN.Model(90)

        loader_mixmatch = MixMatchLoader(
            loader_labeled,
            dataset_unlabeled,
            model,
            output_transform=torch.sigmoid,
            K=2,
            T=0.8,
            alpha=0.6
        )

        criterion = get_mixmatch_loss(
            criterion_labeled=nn.BCEWithLogitsLoss(),
            output_transform=torch.sigmoid,
            K=0,
            weight_unlabeled=5,
            criterion_unlabeled=nn.MSELoss()
        )

        optimizer = torch.optim.Adam(stu.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01

        print("train: ")
        for batch in loader_mixmatch:
            inputs = Variable(batch['features'].to(device), requires_grad=False).view(-1, 1, 90)
            logits = model(inputs)
            loss = criterion(logits, batch['targets'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("test....")
        f1, precision, recall, acc = Test(stu, test, test_label, test_size)
        print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|", acc, "|", 1 - acc, "|")

        print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|",
              acc, "|", 1 - acc, "|", file=ff)

ff.close()

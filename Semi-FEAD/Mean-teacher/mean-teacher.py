import torch
from torch import autograd
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F

import sys

sys.path.append("../..")
import newCNN

batch_size = 8


def update_teacher_variables(stu, teacher, alpha, global_step):
    alpha = min(1 - 10000 / (global_step + 10000), alpha)
    # print(1 - 10000 / (global_step + 10000), global_step, alpha)
    for ema_param, param in zip(teacher.parameters(), stu.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes


def noise(input):
    Size = len(input)
    a = np.random.normal(0.0, scale=0.5, size=(Size, 60))
    return (input + a).astype(np.float32)


def train(stu, teacher, train_data, train_labels, optimizer, epoch, step_counter):
    classify_loss_function = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    consistency_criterion = softmax_mse_loss
    residual_logit_criterion = symmetric_mse_loss
    alpha = 0.8
    consistency_weight = 1
    stu.train()
    teacher.train()
    epoch_loss = 0
    train_size = train_data.shape[0]
    train_batch = train_size // batch_size
    loss = 0
    class_loss = 0
    consistency_loss = 0
    for i in range(train_batch):
        input1 = autograd.Variable(
            torch.from_numpy((train_data[i * batch_size:min((i + 1) * batch_size, train_size), :])),
            requires_grad=False).view(-1, 1, train_data.shape[1])
        input2 = autograd.Variable(
            torch.from_numpy(noise(train_data[i * batch_size:min((i + 1) * batch_size, train_size), :])),
            requires_grad=False).view(-1, 1, train_data.shape[1])
        targets = autograd.Variable(
            torch.from_numpy(train_labels[i * batch_size:min((i + 1) * batch_size, train_size)]),
            requires_grad=False)

        num = min((i + 1) * batch_size, train_size) - i * batch_size + 1
        if num < batch_size:
            continue

        labeled_minibatch_size = targets.data.ne(-1).sum()

        stu_out = stu(input1)
        teacher_out = teacher(input2)

        logit1 = stu_out
        ema_logit = teacher_out

        ema_logit = ema_logit.detach().data

        class_logit, cons_logit = logit1, logit1

        class_loss = classify_loss_function(class_logit, targets) / num
        # ema_class_loss = classify_loss_function(ema_logit, targets) / num
        consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / num

        loss = class_loss  # + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_counter += 1
        epoch_loss += loss
        update_teacher_variables(stu, teacher, alpha, step_counter)

    print("epoch: {} , epoch_loss: {:.3f}".format(epoch, epoch_loss))
    print("loss, class_loss, consistency_loss: ", loss.data.numpy(), class_loss.data.numpy(),
          consistency_loss.data.numpy())

    return step_counter


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


data_path = "../../MAWILab-GAfeature/mawilab_ga.npy"
labels_path = "../../MAWILab-GAfeature/mawilab_label_10w.npy"
data = np.load(data_path)
labels = np.load(labels_path)

ff = open("./res_mean-teacher.md", "w")

for label_train_size in [180, 450, 900, 1800, 4500, 9000, 18000]:  # [90000]:
    step_counter = 0
    stu = newCNN.Model(data.shape[1])
    teacher = newCNN.Model(data.shape[1])
    for param in teacher.parameters():
        param.detach_()

    # test_size = 29999
    # train_size = 270000
    test_size = 10000
    train_size = 90000
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

    # train_data = data[0:label_train_size+unlabel_train_size,:]
    train_data = all_train_data
    all_label_train = np.concatenate((train_label, unlabeled_label))
    indices = np.random.permutation(label_train_size + unlabel_train_size)

    new_data = train_data[indices, :]
    new_label = all_label_train[indices].astype(np.longlong)

    optimizer = torch.optim.Adam(stu.parameters(), lr=0.00001, weight_decay=0.01)  # 0.00001,0.01
    # optimizer = torch.optim.SGD(stu.parameters(), lr=0.01, momentum=0.05)
    f1 = 0
    precision = 0
    recall = 0
    acc = 0
    for epoch in range(1):
        print("epoch: ", epoch)
        train(stu, teacher, new_data, new_label, optimizer, epoch, step_counter)

        print("test....")
        f1, precision, recall, acc = Test(stu, test, test_label, test_size)
        print("|", label_train_size, "|", f1, "|", precision, "|", recall, "|", acc, "|", 1 - acc, "|")

    print("|", label_train_size, "|", label_train_size / train_size * 100, "|", f1, "|", precision, "|", recall, "|",
          acc, "|", 1 - acc, "|", file=ff)

ff.close()

from torch import autograd
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.nn.functional import mse_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8


def update_teacher_variables(stu, teacher, alpha, global_step):
    alpha = min(1 - 10000 / (global_step + 10000), alpha)
    # print(1 - 10000 / (global_step + 10000), global_step, alpha)
    for ema_param, param in zip(teacher.parameters(), stu.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.softmax(input_logits, dim=1)
    target_softmax = torch.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes


def noise(inputs, shape1):
    shape0 = len(inputs)
    a = torch.from_numpy(np.random.normal(0.0, scale=0.01, size=(shape0, shape1)).astype(np.float32)).to(device)
    return inputs + a


def train(stu, teacher, train_data, train_labels, optimizer, epoch, step_counter):
    classify_loss_function = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
    consistency_criterion = softmax_mse_loss
    # residual_logit_criterion = symmetric_mse_loss
    alpha = 0.9
    consistency_weight = 1
    stu.train()
    teacher.train()
    epoch_loss = 0
    train_size = train_data.shape[0]
    train_batch = train_size // batch_size
    loss = 0
    class_loss = 0
    consistency_loss = 0
    shape_1 = train_data.shape[1]
    for i in range(train_batch):
        input1 = autograd.Variable(
            (train_data[i * batch_size:min((i + 1) * batch_size, train_size), :]),
            requires_grad=False).view(-1, 1, shape_1)
        input2 = autograd.Variable(
            noise(train_data[i * batch_size:min((i + 1) * batch_size, train_size), :], shape_1),
            requires_grad=False).view(-1, 1, shape_1)
        targets = autograd.Variable(
            train_labels[i * batch_size:min((i + 1) * batch_size, train_size)],
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
    print("loss, class_loss, consistency_loss: ", loss.data.cpu().numpy(), class_loss.data.cpu().numpy(),
          consistency_loss.data.cpu().numpy())

    return step_counter


def Test(model, test, test_label, test_size):
    model.eval()
    test_batch = test_size // batch_size
    prediction = np.zeros(test_size, dtype=np.uint8)
    for i in range(test_batch):
        inputs = Variable(test[i * batch_size:min((i + 1) * batch_size, test_size), :],
                          requires_grad=False).view(-1, 1, test.shape[1])

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

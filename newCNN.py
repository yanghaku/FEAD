import torch


class Model(torch.nn.Module):
    def __init__(self, sz):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Sequential(  # 1 * size
            torch.nn.Conv1d(1, 4, kernel_size=5, stride=1),  # 4 * (size-4)
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),  # 4 * ((size-4)/2)

            torch.nn.Conv1d(4, 8, kernel_size=5, stride=1),  # 8 * ((size-4)/2)-4
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),  # 8  * (((size-4)/2)-4)/2
            # torch.nn.Conv1d(16,120,kernel_size=5,stride=1),#15
            # torch.nn.ReLU()
        )
        self.sz = 8 * ((((sz - 4) // 2) - 4) // 2)
        self.dense = torch.nn.Sequential(
            # torch.nn.Linear(90,32),
            torch.nn.Linear(self.sz, 32),
            # torch.nn.Linear(22*8,32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = x.view(-1, self.sz)
        # x = x.view(-1,22*8)
        x = self.dense(x)
        return torch.softmax(x, 1)
        # return torch.nn.functional.log_softmax(x)
        # return torch.nn.functional.softmax(x)

import torch.nn as nn


class EndiannessModel(nn.Module):
    def __init__(self, with_sigmoid=False):
        super().__init__()

        self.l1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(8, 1)
        self.with_sigmoid = with_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        if self.with_sigmoid:
            x = self.sigmoid(x)
        return x


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, with_sigmoid=False):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)
        self.with_sigmoid = with_sigmoid
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        if self.with_sigmoid:
            x = self.sig(x)
        return x


class MikkelEndiannessModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.sigmoid(x)
        return x

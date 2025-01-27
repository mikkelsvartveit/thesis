import torch.nn as nn
import torch


class MINOS(nn.Module):
    def __init__(self, num_classes: int):
        super(MINOS, self).__init__()
        # 100x100x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),  # 98x98x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 49x49x16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),  # 47x47x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 23x23x32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),  # 21x21x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10x64
        )
        self.flat = nn.Flatten()  # 6400
        self.fc1 = nn.Sequential(
            nn.Linear(10 * 10 * 64, num_classes),
            nn.Softmax(dim=1) if num_classes > 1 else nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        return x


if __name__ == "__main__":
    model = MINOS(num_classes=5)
    x = torch.randn(1, 1, 100, 100)
    print(model(x))

from torch import nn
import torch.nn.functional as F


class EmbeddingAndCNNModel(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.3):
        super(EmbeddingAndCNNModel, self).__init__()

        self.embedding = nn.Embedding(256, 128)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()

        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)

        return x


class EmbeddingAnd2dCNNModel(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.0):
        super(EmbeddingAnd2dCNNModel, self).__init__()

        assert input_length == 32 * 16, "For a 32x16 grid, input_length must be 512."

        self.embedding = nn.Embedding(256, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.spatial_dim_h = 32
        self.spatial_dim_w = 16

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2
            ),
        )

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.view(x.size(0), self.spatial_dim_h, self.spatial_dim_w, 128)
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)

        # x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

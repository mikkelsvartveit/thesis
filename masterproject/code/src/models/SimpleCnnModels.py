from torch import nn


class Simple2dCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.0):
        super(Simple2dCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Block 1: Note the input channel is now 1 instead of 128.
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2
            ),
        )

        # Global pooling and dense layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is expected to have shape (batch, 1, 32, 32)
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.block3(x)
        x = self.dropout(x)

        x = self.global_pool(x)  # -> (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # flatten -> (batch, 128)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        # Optionally, if you need probabilities, uncomment the next line.
        # x = self.softmax(x)

        return x

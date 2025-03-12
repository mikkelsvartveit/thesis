from torch import nn


class Simple1d(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.3):
        super(Simple1d, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.block0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1, stride=1),
        )

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

        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()

        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)

        x = self.block0(x)
        x = self.dropout(x)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        # Reshape the tensor to match the expected input shape for dense1
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)

        return x


class Simple2d(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.0):
        super(Simple2d, self).__init__()

        assert input_length == 32 * 16, "For a 32x16 grid, input_length must be 512."

        self.dropout = nn.Dropout(p=dropout_rate)

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1, stride=1),
        )

        # Block 1: Note the input channel is now 1 instead of 128.
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1
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

        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block0(x)
        x = self.dropout(x)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # flatten -> (batch, 128)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

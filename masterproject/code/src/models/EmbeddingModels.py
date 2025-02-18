from torch import nn
import torch.nn.functional as F


class EmbeddingAndCNNModel(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.0):
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
    def __init__(self, input_length=1024, num_classes=2, dropout_rate=0.0):
        super(EmbeddingAnd2dCNNModel, self).__init__()

        # Ensure the input length matches 32x32 = 1024 tokens.
        assert input_length == 32 * 32, "For a 32x32 grid, input_length must be 1024."

        self.embedding = nn.Embedding(256, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.spatial_dim = 32  # since 32 x 32 = 1024

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

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(128, 8)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(8, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is expected to have shape (batch, 1024)
        x = self.embedding(x)  # -> (batch, 1024, 128)
        x = self.dropout(x)
        # Reshape to a 32x32 grid.
        x = x.view(x.size(0), self.spatial_dim, self.spatial_dim, 128)
        # Permute to channel-first format: (batch, 128, 32, 32)
        x = x.permute(0, 3, 1, 2)

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
        # If using CrossEntropyLoss later, you can return logits directly.
        # Otherwise, uncomment the next line to output probabilities.
        # x = self.softmax(x)

        return x


class Simple1DCNN(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.5):
        super(Simple1DCNN, self).__init__()
        # Since our input is a single vector of length 512,
        # we treat it as having 1 channel (unsqueezed to shape: (batch, 1, 512)).
        # Define a couple of convolutional layers followed by max pooling.
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool1d(
            kernel_size=2, stride=2
        )  # halves the length: 512 -> 256

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool1d(
            kernel_size=2, stride=2
        )  # halves the length: 256 -> 128

        # After the two pooling layers the sequence length becomes:
        #   512 / 2 / 2 = 128.
        # The number of output channels is 64, so the flattened feature size is 64 * 128.
        self.fc1 = nn.Linear(64 * (input_length // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.float()

        # Expect input x of shape: (batch_size, 512)
        # Add a channel dimension -> (batch_size, 1, 512)
        x = x.unsqueeze(1)

        # Convolutional block 1
        x = F.relu(self.conv1(x))  # -> (batch_size, 32, 512)
        x = self.pool1(x)  # -> (batch_size, 32, 256)

        # Convolutional block 2
        x = F.relu(self.conv2(x))  # -> (batch_size, 64, 256)
        x = self.pool2(x)  # -> (batch_size, 64, 128)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # -> (batch_size, 64*128)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # -> (batch_size, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # -> (batch_size, num_classes)
        return x


class Cnn1dModel(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.0):
        super(Cnn1dModel, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Conv1d(
                in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.dense1 = nn.Linear(128 * (input_length // 64), 1024)
        self.relu = nn.ReLU()

        self.dense2 = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)

        return x

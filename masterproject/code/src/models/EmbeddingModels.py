from torch import nn


class EmbeddingAndCNNModel(nn.Module):
    def __init__(self, input_length=512, num_classes=2, dropout_rate=0.5):
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
        x = self.embedding(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x = self.block1(x)
        x = self.dropout(x)

        x = self.block2(x)
        x = self.dropout(x)

        x = self.block3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)

        return x

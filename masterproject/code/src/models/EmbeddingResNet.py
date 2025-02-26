import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


class EmbeddingResNet50(nn.Module):
    def __init__(
        self,
        input_length=512,
        num_classes=2,
        dropout_rate=0.0,
        embedding_dim=128,
        vocab_size=256,
    ):
        super(EmbeddingResNet50, self).__init__()

        # Make sure input length can be arranged as a grid
        # Assuming a square grid for simplicity, but it could be rectangular
        self.grid_size = int(input_length**0.5)
        assert (
            self.grid_size**2 == input_length
        ), "Input length must be a perfect square for reshaping"

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Load ResNet50 model without pre-training
        self.resnet = resnet50(pretrained=False)

        # Modify the first layer to accept our embedding dimension instead of 3 (RGB)
        self.resnet.conv1 = nn.Conv2d(
            embedding_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify the final fully connected layer for our number of classes
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Apply embedding
        x = self.embedding(
            x
        )  # (batch, sequence_length) -> (batch, sequence_length, embedding_dim)
        x = self.dropout(x)

        # Reshape to 2D for convolution
        batch_size = x.size(0)
        x = x.view(
            batch_size, self.grid_size, self.grid_size, -1
        )  # Reshape to square grid
        x = x.permute(0, 3, 1, 2)  # (batch, embedding_dim, grid_size, grid_size)

        # Pass through ResNet
        x = self.resnet(x)

        return x

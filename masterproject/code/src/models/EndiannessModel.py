import torch
import torch.nn as nn

from src.dataset_loaders.ISAdetect_mips_mipsel import (
    create_train_test_dataloaders as mips_mipsel_dataloader,
)


class EndiannessModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(4, 2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    # Replace with your actual paths
    mips_dir = "dataset/ISAdetect/ISA_detect_full_dataset/mips"
    mipsel_dir = "dataset/ISAdetect/ISA_detect_full_dataset/mipsel"

    # Create train and test loaders with 80-20 split
    train_loader, test_loader = mips_mipsel_dataloader(
        mips_dir=mips_dir,
        mipsel_dir=mipsel_dir,
        test_split=0.2,
    )

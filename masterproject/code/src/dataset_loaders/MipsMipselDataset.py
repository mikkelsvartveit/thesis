from os import PathLike
import torch
from torch.utils.data import Dataset, DataLoader
from src.dataset_loaders.utils import random_train_test_split
from src.transforms.EndiannessCount import EndiannessCount
from pathlib import Path
import numpy as np


class MipsMipselDataset(Dataset):
    def __init__(self, dataset_path: PathLike, transform=None):
        self.transform = transform
        self.files = []
        self.labels = []

        mips_path = Path(dataset_path) / Path("mips")
        mipsel_path = Path(dataset_path) / Path("mipsel")

        # Collect MIPS files (label 0)
        mips_files = mips_path.glob("*.code")
        for file_path in mips_files:
            self.files.append(file_path)
            self.labels.append(0)  # 0 for MIPS

        # Collect MIPSEL files (label 1)
        mipsel_files = mipsel_path.glob("*.code")
        for file_path in mipsel_files:
            self.files.append(file_path)
            self.labels.append(1)  # 1 for MIPSEL

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # Read binary file
        with open(file_path, "rb") as f:
            binary_data = f.read()

        # Convert to numpy array for easier processing
        data = np.frombuffer(binary_data, dtype=np.uint8)
        data = torch.frombuffer(data.copy(), dtype=torch.uint8)

        # Apply transforms if any
        if self.transform:
            features = self.transform(data)
        else:
            features = torch.from_numpy(data)

        label = torch.tensor(label, dtype=torch.float32)

        return features, label


if __name__ == "__main__":
    # Replace with your actual paths
    mips_dir = "dataset/ISAdetect/ISAdetect_full_dataset/mips"
    mipsel_dir = "dataset/ISAdetect/ISAdetect_full_dataset/mipsel"

    train, test = random_train_test_split(
        MipsMipselDataset(mips_dir, mipsel_dir, transform=EndiannessCount()),
        test_split=0.2,
    )

    train_loader = DataLoader(train, batch_size=4, shuffle=True)
    test_loader = DataLoader(test, batch_size=4, shuffle=False)

    # Print dataset sizes
    print(
        f"Training size: {len(train_loader)*train_loader.batch_size}, Training batches: {len(train_loader)}"
    )
    print(
        f"test size: {len(train_loader)*train_loader.batch_size}, Test batches: {len(test_loader)}"
    )

    # Test the loaders
    print("\nSample from training loader:")
    for batch_features, batch_labels in train_loader:
        print("Feature shape:", batch_features.shape)
        print("Labels shape:", batch_labels.shape)
        print("Sample features:", batch_features[0])
        print("Sample label:", batch_labels[0])
        break

    print("\nSample from test loader:")
    for batch_features, batch_labels in test_loader:
        print("Feature shape:", batch_features.shape)
        print("Labels shape:", batch_labels.shape)
        print("Sample features:", batch_features[0])
        print("Sample label:", batch_labels[0])
        break

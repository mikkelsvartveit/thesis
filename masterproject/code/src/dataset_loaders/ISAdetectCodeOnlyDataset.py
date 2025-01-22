from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.dataset_loaders.utils import architecture_metadata_info


class ISAdetectCodeOnlyDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        per_architecture_limit=None,
        file_byte_read_limit: int | None = 2**10,  # 1 KB
    ):
        self.transform = transform
        self.files = []
        self.labels = []
        self.file_byte_read_limit = file_byte_read_limit

        # Collect files
        for architecture in Path(dataset_path).iterdir():
            if architecture.is_dir():
                file_count = 0
                metadata = architecture_metadata_info(architecture, architecture.name)
                for file_path in architecture.glob("*.code"):
                    self.files.append(file_path)
                    self.labels.append(metadata)
                    file_count += 1
                    if per_architecture_limit and file_count >= per_architecture_limit:
                        print(architecture.name, "limit reached")
                        break

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        file_path = self.files[idx]
        label = self.labels[idx]

        # Read binary file
        with open(file_path, "rb") as f:
            binary_data = f.read(self.file_byte_read_limit)

        # Convert to numpy array for easier processing
        data = torch.from_numpy(np.frombuffer(binary_data, dtype=np.uint8).copy())

        # Apply transforms if any
        if self.transform:
            features = self.transform(data)
        else:
            features = data

        return features, label


if __name__ == "__main__":
    # Replace with your actual path
    dataset_path = "dataset/ISAdetect/ISAdetect_full_dataset"

    dataset = ISAdetectCodeOnlyDataset(
        dataset_path, file_byte_read_limit=2**10, per_architecture_limit=1
    )
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(dataset[0][1])
    print(dataset[-1])
    print(dataset[-1][0].shape)
    print(dataset[-1][1])
    print(dataset[-1][1])

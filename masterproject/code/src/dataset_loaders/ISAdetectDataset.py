from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.dataset_loaders.utils import architecture_metadata_info


class ISAdetectDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        per_architecture_limit=None,
        file_byte_read_limit: int | None = 2**10,  # 1 KB
        use_code_only: bool = True,
    ):
        self.transform = transform
        self.files = []
        self.metadata = []
        self.file_byte_read_limit = file_byte_read_limit

        self.use_code_only = use_code_only

        # Collect files
        for isa in Path(dataset_path).iterdir():
            if isa.is_dir():
                file_count = 0
                metadata = architecture_metadata_info(isa, isa.name)
                for file_path in isa.glob("*.code"):
                    self.files.append(file_path)
                    self.metadata.append(metadata)
                    file_count += 1
                    if per_architecture_limit and file_count >= per_architecture_limit:
                        print(isa.name, "limit reached")
                        break

    def set_code_only(self, use_code_only: bool):
        self.use_code_only = use_code_only

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        file_path = self.files[idx]
        if not self.use_code_only:
            file_path = file_path.with_suffix("")  # remove .code extension

        labels = self.metadata[idx]

        # Read binary file
        with open(file_path, "rb") as f:
            binary_data = f.read(self.file_byte_read_limit)

        # Convert to numpy array for easier processing
        data = torch.from_numpy(np.frombuffer(binary_data, dtype=np.uint8).copy())

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return data, labels


if __name__ == "__main__":
    # Replace with your actual path
    dataset_path = "dataset/ISAdetect/ISAdetect_full_dataset"

    dataset = ISAdetectDataset(
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

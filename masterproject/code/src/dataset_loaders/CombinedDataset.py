from os import PathLike
from pathlib import Path
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from src.dataset_loaders.ISAdetectDataset import ISAdetectDataset
from src.dataset_loaders.BuildCrossDataset import BuildCrossDataset


class CombinedDataset(Dataset):
    def __init__(
        self,
        isadetect_dataset_path: PathLike | None,
        buildcross_dataset_path: PathLike | None,
        isadetect_feature_csv_path: str | None,
        buildcross_feature_csv_path: str | None,
        target_feature: str,
        transform=None,
        per_architecture_limit=None,
        file_byte_read_limit: int | None = 2**10,  # 1 KB
        use_code_only: bool = True,
        isadetect_max_file_splits: int | None = None,
        buildcross_max_file_splits: int | None = None,
    ):
        """
        A dataset class that combines ISAdetect and BuildCross datasets.

        Parameters:
            isadetect_dataset_path: Path to ISAdetect dataset (can be None if not using)
            buildcross_dataset_path: Path to BuildCross dataset (can be None if not using)
            isadetect_feature_csv_path: Path to ISAdetect features CSV
            buildcross_feature_csv_path: Path to BuildCross features CSV
            target_feature: Feature to target (e.g. 'endianness', 'instructionwidth')
            transform: Optional transforms to apply to the data
            per_architecture_limit: Maximum number of samples per architecture
            file_byte_read_limit: Maximum number of bytes to read from each file
            use_code_only: For ISAdetect, whether to use .code files or full binaries
            isadetect_max_file_splits: Maximum file splits for ISAdetect files
            buildcross_max_file_splits: Maximum file splits for BuildCross files
        """
        self.datasets = []

        # Initialize the ISAdetect dataset if path is provided
        if isadetect_dataset_path and isadetect_feature_csv_path:
            isadetect_dataset = ISAdetectDataset(
                dataset_path=isadetect_dataset_path,
                feature_csv_path=isadetect_feature_csv_path,
                target_feature=target_feature,
                transform=transform,
                per_architecture_limit=per_architecture_limit,
                file_byte_read_limit=file_byte_read_limit,
                use_code_only=use_code_only,
                max_file_splits=isadetect_max_file_splits,
            )
            self.datasets.append(isadetect_dataset)

        # Initialize the BuildCross dataset if path is provided
        if buildcross_dataset_path and buildcross_feature_csv_path:
            buildcross_dataset = BuildCrossDataset(
                dataset_path=buildcross_dataset_path,
                feature_csv_path=buildcross_feature_csv_path,
                target_feature=target_feature,
                transform=transform,
                per_architecture_limit=per_architecture_limit,
                file_byte_read_limit=file_byte_read_limit,
                max_file_splits=buildcross_max_file_splits,
            )
            self.datasets.append(buildcross_dataset)

        # Combine the datasets
        self.combined_dataset = ConcatDataset(self.datasets)

        # Keep track of dataset sizes for indexing
        self.dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.dataset_cumulative_sizes = [0]
        cumulative = 0
        for size in self.dataset_sizes:
            cumulative += size
            self.dataset_cumulative_sizes.append(cumulative)

        # Collect metadata from each dataset
        self.metadata = []
        for dataset in self.datasets:
            if hasattr(dataset, "metadata"):
                self.metadata.extend(dataset.metadata)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]

    def get_dataset_source(self, idx):
        """Returns which source dataset the sample at index idx comes from (0 for ISAdetect, 1 for BuildCross)"""
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        dataset_idx = 0
        for i, cumulative_size in enumerate(self.dataset_cumulative_sizes[1:]):
            if idx < cumulative_size:
                break
            dataset_idx = i + 1

        return dataset_idx


if __name__ == "__main__":
    # Example usage
    isadetect_path = "dataset/ISAdetect/ISAdetect_full_dataset"
    buildcross_path = "dataset/buildcross/text_bin"
    isadetect_csv = "dataset/ISAdetect-features.csv"
    buildcross_csv = "dataset/buildcross/labels.csv"

    # Test with both datasets
    dataset = CombinedDataset(
        isadetect_dataset_path=isadetect_path,
        buildcross_dataset_path=buildcross_path,
        isadetect_feature_csv_path=isadetect_csv,
        buildcross_feature_csv_path=buildcross_csv,
        target_feature="endianness",
        file_byte_read_limit=512,
        use_code_only=True,
        isadetect_max_file_splits=2,
        buildcross_max_file_splits=1,
    )

    # Print dataset information
    print(f"Combined dataset size: {len(dataset)}")

    # Test individual samples
    print("\nFirst sample:")
    data, labels = dataset[0]
    print(f"Data shape: {data.shape}")
    print(f"Labels: {labels}")
    print(
        f"Source dataset: {'ISAdetect' if dataset.get_dataset_source(0) == 0 else 'BuildCross'}"
    )

    print("\nLast sample:")
    data, labels = dataset[-1]
    print(f"Data shape: {data.shape}")
    print(f"Labels: {labels}")
    print(
        f"Source dataset: {'ISAdetect' if dataset.get_dataset_source(-1) == 0 else 'BuildCross'}"
    )

    # Test with ISAdetect only
    isadetect_only = CombinedDataset(
        isadetect_dataset_path=isadetect_path,
        buildcross_dataset_path=None,
        isadetect_feature_csv_path=isadetect_csv,
        buildcross_feature_csv_path=None,
        target_feature="endianness",
        file_byte_read_limit=512,
        use_code_only=True,
        isadetect_max_file_splits=2,
    )

    print(f"\nISAdetect-only dataset size: {len(isadetect_only)}")

    # Test with BuildCross only
    buildcross_only = CombinedDataset(
        isadetect_dataset_path=None,
        buildcross_dataset_path=buildcross_path,
        isadetect_feature_csv_path=None,
        buildcross_feature_csv_path=buildcross_csv,
        target_feature="endianness",
        file_byte_read_limit=512,
        buildcross_max_file_splits=1,
    )

    print(f"\nBuildCross-only dataset size: {len(buildcross_only)}")

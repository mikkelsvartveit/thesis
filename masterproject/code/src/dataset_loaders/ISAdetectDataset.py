from os import PathLike
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.dataset_loaders.utils import get_architecture_features, get_elf_header_end


class ISAdetectDataset(Dataset):
    def __init__(
        self,
        dataset_path: PathLike,
        feature_csv_path,
        target_feature: str,
        transform=None,
        per_architecture_limit=None,
        file_byte_read_limit: int | None = 2**10,  # 1 KB
        use_code_only: bool = True,
        max_file_splits: int | None = None,
    ):
        self.transform = transform
        self.file_paths = []
        self.file_byte_offset = []
        self.metadata = []
        self.file_byte_read_limit = file_byte_read_limit
        self.use_code_only = use_code_only

        # Collect files
        metadata_errors = []
        for isa in Path(dataset_path).iterdir():
            if isa.is_dir():
                file_count = 0
                metadata = get_architecture_features(feature_csv_path, isa.name)

                if not metadata:
                    metadata_errors.append(isa.name)
                    continue

                if (
                    target_feature == "instructionwidth"
                    and metadata["instructionwidth_type"] == "variable"
                ):
                    continue

                for file_path in isa.glob("*.code"):
                    # if using full binary, remove .code and add offset skipping elf header
                    elf_end_offset = 0
                    if not use_code_only:
                        file_path = file_path.with_suffix("")
                        elf_end_offset = get_elf_header_end(file_path)

                    # Split file into file_byte_read_limit chunks
                    file_splits = 1
                    if max_file_splits and file_byte_read_limit:
                        file_size = file_path.stat().st_size
                        file_splits = file_size // file_byte_read_limit
                        file_splits = min(file_splits, max_file_splits)

                    for i in range(file_splits):
                        self.file_paths.append(file_path)
                        self.file_byte_offset.append(
                            elf_end_offset + i * file_byte_read_limit
                        )
                        self.metadata.append(metadata)
                        file_count += 1
                        if (
                            per_architecture_limit
                            and file_count >= per_architecture_limit
                        ):
                            break
                    if per_architecture_limit and file_count >= per_architecture_limit:
                        print(isa.name, "limit reached")
                        break

        if metadata_errors:
            print("Metadata errors:", metadata_errors)
            raise ValueError("Metadata errors for dataset")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict, str]:
        file_path = str(self.file_paths[idx])
        labels = self.metadata[idx].copy()
        labels["file_path"] = file_path

        # Read binary file
        with open(file_path, "rb") as f:
            f.seek(self.file_byte_offset[idx])
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
    feature_csv_path = "dataset/ISAdetect-features.csv"

    dataset = ISAdetectDataset(
        dataset_path,
        feature_csv_path=feature_csv_path,
        file_byte_read_limit=2**10,
        use_code_only=True,
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

    # elf_count = 0
    # elf_full_code_count = 0
    # total_files = 0
    # elf_length_set = {}

    # print("Counting ELF files")
    # for isa in Path(dataset_path).iterdir():
    #     num = 0
    #     print(isa.name)
    #     elf_length_set[isa.name] = set()
    #     if isa.is_dir():
    #         for file_path in isa.glob("*"):
    #             num += 1
    #             if num > 1000:
    #                 break

    #             total_files += 1
    #             elf_end = get_elf_header_end(file_path)

    #             if elf_end == None:
    #                 continue

    #             elf_count += 1
    #             elf_length_set[isa.name].add(elf_end)

    #             if file_path.suffix != ".code":
    #                 elf_full_code_count += 1

    # print(f"Total files: {total_files}")
    # print(f"Total ELF files: {elf_count}")
    # print(f"Total ELF full code files: {elf_full_code_count}")
    # print(f"Total ELF lengths: {elf_length_set}")

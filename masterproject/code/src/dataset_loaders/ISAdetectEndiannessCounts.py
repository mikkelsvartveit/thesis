import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from src.dataset_loaders.utils import architecture_metadata_info
from src.transforms.EndiannessCount import EndiannessCount


class ISAdetectEndiannessCountsDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None,
        use_code_only: bool = True,
    ):
        self.transform = transform
        self.counts_code_only = []
        self.counts_full_program = []
        self.labels = []
        self.use_code_only = use_code_only

        for architecture in Path(dataset_path).iterdir():
            # check if endianness.json exists
            metadata = architecture_metadata_info(architecture, architecture.name)
            endianness_json_path = architecture / "_endianness_test.json"
            if not endianness_json_path.exists():
                print(f"generating json for {architecture.name}")
                endiannes_json = []
                # generate endianness.json
                for file_path in architecture.glob("*.code"):
                    # read .code file
                    with open(file_path, "rb") as f:
                        binary_data = f.read()

                    # count .code file
                    data_code_only = torch.frombuffer(
                        binary_data, dtype=torch.uint8, requires_grad=False
                    )
                    count_code_only_tensor = EndiannessCount()(data_code_only)
                    count_code_only_arr = count_code_only_tensor.tolist()

                    # read full program file
                    file_path_full_program = file_path.with_suffix("")
                    with open(file_path_full_program, "rb") as f:
                        binary_data = f.read()

                    # count full program file
                    data_full_program = torch.frombuffer(
                        binary_data, dtype=torch.uint8, requires_grad=False
                    )
                    count_full_program_tensor = EndiannessCount()(data_full_program)
                    count_full_program_arr = count_full_program_tensor.tolist()

                    endiannes_json.append(
                        {
                            "file_name": file_path_full_program.name,
                            "counts_code": count_code_only_arr,
                            "counts_full": count_full_program_arr,
                        }
                    )

                    self.counts_code_only.append(count_code_only_arr)
                    self.counts_full_program.append(count_full_program_arr)
                    self.labels.append(metadata)

                with open(endianness_json_path, "w") as f:
                    json.dump(endiannes_json, f)
                print("Generated endianness.json for", architecture.name)
            else:
                # load endianness.json
                endiannes_json = json.load(endianness_json_path.open())
                for endiannes in endiannes_json:
                    self.counts_code_only.append(endiannes["counts_code"])
                    self.counts_full_program.append(endiannes["counts_full"])
                    self.labels.append(metadata)

        self.counts_code_only = torch.tensor(self.counts_code_only)
        self.counts_full_program = torch.tensor(self.counts_full_program)

    def set_code_only(self, use_code_only: bool):
        self.use_code_only = use_code_only

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        counts = (
            self.counts_code_only[idx]
            if self.use_code_only
            else self.counts_full_program[idx]
        )
        label = self.labels[idx]

        if self.transform:
            counts = self.transform(counts)

        return counts, label


if __name__ == "__main__":
    # Replace with your actual path
    dataset_path = "dataset/ISAdetect/ISAdetect_full_dataset"

    dataset = ISAdetectEndiannessCountsDataset(dataset_path=dataset_path)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(dataset[0][1])
    print(dataset[-1])
    print(dataset[-1][0].shape)
    print(dataset[-1][1])
    print(dataset[-1][1])

    dataset.use_code_only = False
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(dataset[0][1])
    print(dataset[-1])
    print(dataset[-1][0].shape)
    print(dataset[-1][1])
    print(dataset[-1][1])

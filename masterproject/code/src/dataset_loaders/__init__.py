from .MipsMipselDataset import *
from .utils import *
from .ISAdetectDataset import *
from .ISAdetectEndiannessCounts import *


def get_dataset(transform, **kwargs):
    dataset_name = kwargs["name"]

    if dataset_name == "MipsMipselDataset":
        return MipsMipselDataset(
            data_dir=kwargs["data_dir"],
            transform=transform,
        )
    elif dataset_name == "ISAdetectDataset":
        return ISAdetectDataset(
            dataset_path=kwargs["dataset_path"],
            feature_csv_path=kwargs["feature_csv_path"],
            per_architecture_limit=kwargs["per_architecture_limit"],
            file_byte_read_limit=kwargs["file_byte_read_limit"],
            use_code_only=kwargs["use_code_only"],
            transform=transform,
        )
    elif dataset_name == "ISAdetectEndiannessCounts":
        return ISAdetectEndiannessCounts(
            data_dir=kwargs["data_dir"],
            transform=transform,
        )

from .MipsMipselDataset import *
from .utils import *
from .ISAdetectDataset import *
from .ISAdetectEndiannessCounts import *


def get_dataset(transform, **kwargs):
    dataset_name = kwargs["name"]

    if dataset_name == "MipsMipselDataset":
        return MipsMipselDataset(
            transform=transform,
            **kwargs["params"]
        )
    elif dataset_name == "ISAdetectDataset":
        return ISAdetectDataset(
            transform=transform,
            **kwargs["params"],
        )
    elif dataset_name == "ISAdetectEndiannessCounts":
        return ISAdetectEndiannessCounts(
            transform=transform,
            **kwargs["params"]
        )

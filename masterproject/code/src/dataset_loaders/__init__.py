from .MipsMipselDataset import *
from .utils import *
from .ISAdetectDataset import *
from .ISAdetectEndiannessCounts import *
from .CpuRecDataset import *
from .BuildCrossDataset import *
from .CombinedDataset import *


def get_dataset(
    transform, dataset_base_path: PathLike, target_feature: str | None = None, **kwargs
):
    dataset_name = kwargs["name"]

    params = kwargs["params"].copy()

    # Prepend dataset_base_path to all relevant paths
    for key in list(params.keys()):
        if key.endswith("dataset_path") or key.endswith("csv_path"):
            params[key] = Path(dataset_base_path) / Path(params[key])

    if dataset_name == "MipsMipselDataset":
        return MipsMipselDataset(transform=transform, **params)
    elif dataset_name == "ISAdetectDataset":
        return ISAdetectDataset(
            transform=transform,
            target_feature=target_feature,
            **params,
        )
    elif dataset_name == "CpuRecDataset":
        return CpuRecDataset(
            transform=transform,
            target_feature=target_feature,
            **params,
        )
    elif dataset_name == "BuildCrossDataset":
        return BuildCrossDataset(
            transform=transform,
            target_feature=target_feature,
            **params,
        )
    elif dataset_name == "CombinedDataset":
        return CombinedDataset(
            transform=transform,
            target_feature=target_feature,
            **params,
        )
    elif dataset_name == "ISAdetectEndiannessCountsDataset":
        return ISAdetectEndiannessCountsDataset(
            transform=transform,
            **params,
        )

from .MipsMipselDataset import *
from .utils import *
from .ISAdetectDataset import *
from .ISAdetectEndiannessCounts import *
from .CpuRecDataset import *
from .BuildCrossDataset import *


def get_dataset(
    transform, dataset_base_path: PathLike, target_feature: str | None = None, **kwargs
):
    dataset_name = kwargs["name"]

    params = kwargs["params"].copy()
    params["dataset_path"] = Path(dataset_base_path) / Path(params["dataset_path"])
    if "feature_csv_path" in params:
        params["feature_csv_path"] = Path(dataset_base_path) / Path(
            params.get("feature_csv_path")
        )

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
    elif dataset_name == "ISAdetectEndiannessCountsDataset":
        return ISAdetectEndiannessCountsDataset(
            transform=transform,
            **params,
        )

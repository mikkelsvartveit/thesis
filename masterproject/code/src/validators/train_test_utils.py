import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union
from os import PathLike
import wandb


def set_seed(seed: int) -> None:
    """Set random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # For reproducible behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_as_onnx(
    model: nn.Module,
    model_name: str,
    sample_dataset: Dataset,
    save_dir: PathLike = None,
):
    save_path = (
        f"wandb/{model_name}.onnx"
        if save_dir is None
        else f"{save_dir}/{model_name}.onnx"
    )
    original_device = next(model.parameters()).device

    try:
        model = model.to("cpu")

        sample_input = DataLoader(sample_dataset, batch_size=1, shuffle=False)
        sample_input = next(iter(sample_input))[0]

        torch.onnx.export(
            model,
            sample_input,
            save_path,
        )

        wandb.save(save_path, policy="now")

    except Exception as e:
        print(f"Failed to save model as onnx: {e}")
    finally:
        import os

        model = model.to(original_device)
        os.remove(save_path)

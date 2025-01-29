from src.dataset_loaders import ISAdetectDataset
from typing_extensions import Buffer
import torch
import numpy as np
from torch import Tensor


class GrayScaleImage:
    """
    Transforms binary data into a grayscale image tensor
    Output tensor shape: torch.tensor([1, dimx, dimy])
    """

    def __init__(self, dimx, dimy):
        self.dimx = dimx
        self.dimy = dimy

    def __call__(self, binary_data: torch.Tensor) -> torch.Tensor:
        # ensure tensor
        if not isinstance(binary_data, torch.Tensor):
            raise ValueError(
                "data is not a torch.Tensor. Check that dataset returns tensor"
            )

        # pad with zeros if necessary
        if len(binary_data) < self.dimx * self.dimy:
            padd = torch.zeros(
                self.dimx * self.dimy - len(binary_data), dtype=torch.uint8
            )
            binary_data = torch.cat((binary_data, padd))

        truncated_data = binary_data[: (self.dimx * self.dimy)]

        gray_scale = truncated_data.view(self.dimx, self.dimy)
        gray_scale = gray_scale.unsqueeze(0)
        return gray_scale


if __name__ == "__main__":
    # Replace with your actual path
    dataset_path = "dataset/ISAdetect/ISAdetect_full_dataset"
    transform = GrayScaleImage(100, 100)
    dataset = ISAdetectDataset(
        dataset_path, transform=transform, file_byte_read_limit=128 * 128
    )
    print(dataset[0])
    print(dataset[1])
    print(dataset[0][0].shape)

from src.dataset_loaders import ISAdetectDataset
from typing_extensions import Buffer
import torch
import numpy as np
from torch import Tensor


class GrayScaleImage:
    """
    Transforms binary data into a grayscale image tensor
    Output tensor shape: torch.tensor([1, dimx, dimy])
    if normalize is True, the tensor values are scaled to [0, 1]
    if duplicate_channels is set, the tensor is duplicated to n channels tensor([n, dimx, dimy])
    """

    def __init__(self, dimx, dimy, normalize=True, duplicate_to_n_channels: int = None):
        self.dimx = dimx
        self.dimy = dimy
        self.normalize = normalize
        self.duplicate_to_n_channels = duplicate_to_n_channels

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

        if self.normalize:
            gray_scale = gray_scale.float() / 255.0

        if self.duplicate_to_n_channels is not None:
            gray_scale = gray_scale.repeat(self.duplicate_to_n_channels, 1, 1)

        return gray_scale


class RGBImage:
    """
    Transforms binary data into a RGB image tensor
    Output tensor shape: torch.tensor([num_channels, dimx, dimy])
    If normalize is True, the tensor values are scaled to [0, 1]
    Consecutive bytes are interpreted as one channel
    """

    def __init__(self, dimx, dimy, num_channels=3, normalize=True):
        self.dimx = dimx
        self.dimy = dimy
        self.normalize = normalize
        self.num_channels = num_channels

    def __call__(self, binary_data: torch.Tensor) -> torch.Tensor:
        # ensure tensor
        if not isinstance(binary_data, torch.Tensor):
            raise ValueError(
                "data is not a torch.Tensor. Check that dataset returns tensor"
            )

        required_size = self.dimx * self.dimy * self.num_channels
        # pad with zeros if necessary
        if len(binary_data) < required_size:
            padd = torch.zeros(required_size - len(binary_data), dtype=torch.uint8)
            binary_data = torch.cat((binary_data, padd))

        truncated_data = binary_data[:required_size]

        # Reshape to [height, width, 3], each pixel three consecutive bytes
        #   (pixel0) = (a, b, c)
        #   (pixel1) = (d, e, f), etc.
        image_hw3 = truncated_data.view(self.dimx, self.dimy, self.num_channels)

        # Rearrange to PyTorch's [channels, height, width]
        rgb = image_hw3.permute(2, 0, 1)  # [3, dimx, dimy]

        if self.normalize:
            rgb = rgb.float() / 255.0

        return rgb


if __name__ == "__main__":
    # Replace with your actual path
    dataset_path = "dataset/ISAdetect/ISAdetect_full_dataset"
    transform = GrayScaleImage(100, 100, duplicate_to_n_channels=5)
    dataset = ISAdetectDataset(
        dataset_path,
        transform=transform,
        file_byte_read_limit=128 * 128,
        per_architecture_limit=1,
    )
    print(dataset[0])
    print(dataset[1])
    print(dataset[0][0].shape)

    transform = RGBImage(100, 100, num_channels=4)
    dataset = ISAdetectDataset(
        dataset_path,
        transform=transform,
        file_byte_read_limit=128 * 128 * 3,
        per_architecture_limit=1,
    )
    print(dataset[0])
    print(dataset[1])
    print(dataset[0][0].shape)

import torch


class Vector1D:
    def __init__(self, length):
        self.length = length

    def __call__(self, binary_data: torch.Tensor) -> torch.Tensor:
        # ensure tensor
        if not isinstance(binary_data, torch.Tensor):
            raise ValueError(
                "data is not a torch.Tensor. Check that dataset returns tensor"
            )

        # pad with zeros if necessary
        if len(binary_data) < self.length:
            padd = torch.zeros(self.length - len(binary_data), dtype=torch.uint8)
            binary_data = torch.cat((binary_data, padd))

        truncated_data = binary_data[: self.length]

        # Convert to int
        truncated_data = truncated_data.int()

        return truncated_data

from typing_extensions import Buffer
import torch
import numpy as np
from torch import Tensor


class EndiannessCount:
    def __init__(self):
        self.patterns = [
            b"\x00\x01",  # 0x0001
            b"\x01\x00",  # 0x0100
            b"\xfe\xff",  # 0x0100
            b"\xff\xfe",  # 0x1011
        ]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # ensure tensor
        if not isinstance(data, torch.Tensor):
            raise ValueError(
                "data is not a torch.Tensor. Check that dataset returns tensor"
            )

        data: bytearray = data.numpy().tobytes()

        # Count occurrences of each pattern
        counts = []
        for pattern in self.patterns:
            count = data.count(pattern)
            counts.append(count)

        return torch.tensor(counts, dtype=torch.float32)

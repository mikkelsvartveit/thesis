from typing_extensions import Buffer
import torch
import numpy as np


class EndiannessCount:
    def __init__(self):
        self.patterns = [
            b"\x00\x01",  # 0x0001
            b"\x01\x00",  # 0x0100
            b"\xfe\xff",  # 0x0100
            b"\xff\xfe",  # 0x1011
        ]

    def __call__(self, data: np.ndarray | Buffer) -> torch.Tensor:
        # ensure np array
        if not isinstance(data, np.ndarray):
            data = np.frombuffer(data, dtype=np.uint8)

        # Convert data to bytes if it's not already
        if isinstance(data, np.ndarray):
            data = data.tobytes()

        # Count occurrences of each pattern
        counts = []
        for pattern in self.patterns:
            count = data.count(pattern)
            counts.append(count)

        return torch.tensor(counts, dtype=torch.float32)

from typing import List
import numpy as np
from classes.consts import FilterSize as fs


class Filter:
    # for sake of simplicity we
    def __init__(self) -> None:
        self.channels: int = fs.C
        self.r: int = fs.R
        self.s: int = fs.S
        self.bias: float = np.random.rand()
        self.kernel = np.random.rand(self.channels, self.r, self.s)

    def get_bias(self) -> float:
        return self.bias

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for r in range(self.r):
                row = " ".join(f"{self.kernel[c][r][s]:.2f}" for s in range(self.s))
                output.append(row)
        return "\n".join(output)

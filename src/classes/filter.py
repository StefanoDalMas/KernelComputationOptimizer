from typing import List
import numpy as np
from classes.consts import FilterSize as fs


class Filter:
    # for sake of simplicity we
    def __init__(self) -> None:
        self.channels = fs.C
        self.r = fs.R
        self.s = fs.S
        self.kernel = np.random.rand(self.channels, self.r, self.s)

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for r in range(self.r):
                row = " ".join(f"{self.kernel[c][r][s]:.2f}" for s in range(self.s))
                output.append(row)
        return "\n".join(output)

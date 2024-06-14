from typing import List
import numpy as np
from classes.consts import InputFmapSize as fs


class FeatureMap:
    # for sake of simplicity we
    def __init__(self) -> None:
        self.channels = fs.C
        self.h = fs.H
        self.w = fs.W
        self.kernel = np.random.rand(self.channels, self.h, self.w)

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for h in range(self.h):
                row = " ".join(f"{self.kernel[c][h][w]:.2f}" for w in range(self.w))
                output.append(row)
        return "\n".join(output)

from typing import List
import numpy as np
from classes.consts import InputFmapSize as fs


class InputFeatureMap:
    # for sake of simplicity we
    def __init__(self) -> None:
        self.channels: int = fs.C
        self.h: int = fs.H
        self.w: int = fs.W
        self.fmap = np.random.rand(self.channels, self.h, self.w)

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for h in range(self.h):
                row = " ".join(f"{self.fmap[c][h][w]:.2f}" for w in range(self.w))
                output.append(row)
        return "\n".join(output)

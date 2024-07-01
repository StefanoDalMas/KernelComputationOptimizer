from typing import List
import json
from classes.consts import FilterSize as fs
import numpy as np

id = 0
class Filter:
    # for sake of simplicity we
    def __init__(self, *args) -> None:
        self.channels: int = fs.C
        self.r: int = fs.R
        self.s: int = fs.S
        if len(args) == 2:
            self.bias: float = args[0]
            self.kernel = args[1]
        else:
            self.bias: float = np.random.rand()
            self.kernel = np.random.rand(self.channels, self.r, self.s)
        global id
        self.id = id 
        id += 1

    def get_bias(self) -> float:
        return self.bias
    
    def get_id(self) -> int:
        return f"Filter " + str(self.id)

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for r in range(self.r):
                row = " ".join(f"{self.kernel[c][r][s]:.2f}" for s in range(self.s))
                output.append(row)
        return "\n".join(output)
    
    def to_dict(self):
        kernel = {}
        for c in range(self.channels):
            kernel[f"channel_{c}"] = {}
            for r in range(self.r):
                kernel[f"channel_{c}"][f"row_{r}"] = self.kernel[c][r].tolist()
        data = {
            "bias": self.bias,
            "kernel": kernel
        }
        return data


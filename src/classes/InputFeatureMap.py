from typing import List
import json
import numpy as np
from classes.consts import InputFmapSize as fs

f_id = 0
class InputFeatureMap:
    # for sake of simplicity we
    def __init__(self, *args) -> None:
        self.channels: int = fs.C
        self.h: int = fs.H
        self.w: int = fs.W
        if len(args) == 1:
            self.fmap = args[0]
        else:
            self.fmap = np.random.rand(self.channels, self.h, self.w)
        global f_id
        self.id = f_id
        f_id += 1

    def __str__(self) -> str:
        output = []
        for c in range(self.channels):
            output.append(f"Channel {c}:")
            for h in range(self.h):
                row = " ".join(f"{self.fmap[c][h][w]:.2f}" for w in range(self.w))
                output.append(row)
        return "\n".join(output)
    
    def to_dict(self) -> dict:
        # we have to do the same as in Filter
        fmap = {}
        for c in range(self.channels):
            fmap[f"channel_{c}"] = {}
            for h in range(self.h):
                fmap[f"channel_{c}"][f"row_{h}"] = self.fmap[c][h].tolist()
        data = {
            "fmap": fmap
        }
        return data
    

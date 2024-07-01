from typing import List, Any, Dict
import numpy as np
import json
from classes.InputFeatureMap import InputFeatureMap
from classes.memoryModel import Memory
from classes.filter import Filter
from classes.consts import (
    FilterSize as fs,
    InputFmapSize as ifs,
    OutputFmapSize as ofs,
    U as stride,
)
from collections import defaultdict


def data_loader() -> tuple[Dict[int, Filter], Dict[int, float], List[InputFeatureMap]]:
    # this function loads data from json files and returns it in a cool tuple
    filters: Dict[int, Filter] = defaultdict(Filter)
    with open("data/filters.json", "r") as f:
        filters_json = json.load(f)
        for m in range(fs.M):
            bias = filters_json[m]["bias"]
            kernel = []
            for c in range(fs.C):
                kernel.append([])
                for r in range(fs.R):
                    kernel[c].append(filters_json[m]["kernel"][f"channel_{c}"][f"row_{r}"])
            filters[m] = Filter(bias, np.array(kernel))
    biases: Dict[int, float] = defaultdict(float)
    for m in range(fs.M):
        biases[m] = filters[m].get_bias()
    inputFmaps: List[InputFeatureMap] = []
    with open("data/input_fmaps.json", "r") as f:
        input_fmaps_json = json.load(f)
        for n in range(ifs.N):
            fmap = []
            for c in range(ifs.C):
                fmap.append([])
                for h in range(ifs.H):
                    fmap[c].append(input_fmaps_json[n]["fmap"][f"channel_{c}"][f"row_{h}"])
            inputFmaps.append(InputFeatureMap(np.array(fmap)))
    return (filters, biases, inputFmaps)

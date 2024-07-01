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

def generate_data() -> None:
    # here we would like to create an example of kernel computation

    # FILTERS:
    # we are going to have M filters, each one with C channels and R x S kernel size

    # INPUT FMAP:
    # we are going to have C channels and H x W size

    # OUTPUT FMAP:
    # we are going to have N outputs, each one with M channels and P x Q size


    # initialize the M filters

    filters: Dict[int, Filter] = defaultdict(Filter)
    for m in range(fs.M):
        filters[m] = Filter()


    # each filter has a bias, so for now we simply map it with a dict to quickly access it
    biases: Dict[int, float] = defaultdict(float)
    for m in range(fs.M):
        biases[m] = filters[m].get_bias()

    # we initialize the N fmaps with random values
    inputFmaps: List[InputFeatureMap] = []
    for n in range(ifs.N):
        inputFmaps.append(InputFeatureMap())

    filters_json = [filter.to_dict() for filter in filters.values()]
    with open("data/filters.json", "w") as f:
        json.dump(filters_json, f, indent=4)
    input_fmaps_json = [inputFmap.to_dict() for inputFmap in inputFmaps]
    with open("data/input_fmaps.json", "w") as f:
        json.dump(input_fmaps_json, f, indent=4)

if __name__ == "_main_":
    generate_data()
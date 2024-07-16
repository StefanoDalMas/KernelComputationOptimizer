from typing import List, Any, Dict
import numpy as np
import pandas as pd
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
from tools.convolution import convolution, flattened_convolution
from tools.generate_data import generate_data
from tools.data_loader import data_loader
from tools.benchmarkParser import main as benchmark_main


# uncomment this one to generate new json files
generate_data()

inputFmaps: List[InputFeatureMap] = []
biases: Dict[int, float] = defaultdict(float)
filters: Dict[int, Filter] = defaultdict(Filter)

filters, biases, inputFmaps = data_loader()
for m in range(fs.M):
    print(f"Filter {m + 1}:")
    print(filters[m])
for n in range(ifs.N):
    print(f"Input Feature Map {n + 1}:")
    print(inputFmaps[n])
for m in range(fs.M):
    print(f"Filter {m + 1} bias: {biases[m]:.2f}")


# now we need to perform the convolution, we need the sizes P and Q
P = (ifs.H - fs.R) // stride + 1
Q = (ifs.W - fs.S) // stride + 1
outputFmaps: List[Any] = []
# initialize the output fmaps with zeroes using np.zeros
for n in range(ifs.N):
    outputFmaps.append(np.zeros((fs.M, P, Q)))

outputFmaps = convolution(outputFmaps, inputFmaps, filters, biases, P, Q)

# Print output feature maps
for n in range(ofs.N):
    for m in range(fs.M):
        print(f"Output Feature Map {n + 1}, Channel {m + 1}:")
        for i in range(P):
            row = " ".join(f"{outputFmaps[n][m][i][j]:.2f}" for j in range(Q))
            print(row)


# now we try with the flattened filters and input fmaps
flattened_filters = np.array([filter.kernel.flatten() for filter in filters.values()])
flattened_input_fmaps = np.array([inputFmap.fmap.flatten() for inputFmap in inputFmaps])

outputFmaps = np.zeros((ifs.N, fs.M, P, Q))

outputFmapsWithFlattened = flattened_convolution(
    outputFmaps, flattened_input_fmaps, flattened_filters, biases, P, Q
)

# Print output feature maps
for n in range(ofs.N):
    for m in range(fs.M):
        print(f"Output Feature Map {n + 1}, Channel {m + 1}:")
        for i in range(P):
            row = " ".join(
                f"{outputFmapsWithFlattened[n][m][i][j]:.2f}" for j in range(Q)
            )
            print(row)


# testing loading into Volatile Memory
memory = Memory()
outputFmaps = np.zeros((ifs.N, fs.M, P, Q))

# this one is used to check if power failure works correctly and some other stuff
# memory.alloc(filters, volatile=True)
# memory.alloc(biases, volatile=True)
# memory.alloc(inputFmaps, volatile=True)
# memory.alloc(outputFmaps, volatile = True)


# memory.monitor_convolution(outputFmaps, inputFmaps, filters, biases, P, Q)

memory.perform_all_convolutions(
    outputFmaps, inputFmaps, filters, biases, P, Q, tiling=False, all_nonvolatile=True
)


# memory = Memory()
# print("Convolution Volatile NO TILING")
# outputFmaps = np.zeros((ifs.N, fs.M, P, Q))
# memory.perform_all_convolutions(
#     outputFmaps, inputFmaps, filters, biases, P, Q, tiling=False, all_nonvolatile=False
# )


memory = Memory()
print("Convolution Volatile TILING")
outputFmaps = np.zeros((ifs.N, fs.M, P, Q))
memory.perform_all_convolutions(
    outputFmaps, inputFmaps, filters, biases, P, Q, tiling=True, all_nonvolatile=False
)


# benchmark_main("data/benchmarks.txt", "data/benchmarks_diff.csv")
benchmark_main("data/benchmarks_tiling.txt", "data/benchmarks_diff_tiling.csv")
benchmark_main("data/benchmarks_all_nonvolatile.txt", "data/benchmarks_diff_all_nonvolatile.csv")

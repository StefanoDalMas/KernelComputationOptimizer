from typing import List, Any, Dict
import numpy as np
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
    print(f"Filter {m} bias: {biases[m]}")

for k, filter in filters.items():
    print(f"Filter {k}\n", filter)

# we initialize the N fmaps with random values
inputFmaps: List[InputFeatureMap] = []
for n in range(ifs.N):
    inputFmaps.append(InputFeatureMap())

for n in range(ifs.N):
    print("Input Feature Map\n", inputFmaps[n])

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
# outputFmaps = np.zeros((ifs.N, fs.M, P, Q))
# memory.alloc(filters, volatile=True)
# memory.alloc(biases, volatile=True)
# memory.alloc(inputFmaps, volatile=True)
# memory.alloc(outputFmaps, volatile = True) # not necessary to state true, it is hardcoded in the monitorConvolution function!!!!

# # # Testing Nonvolatile Memory costs
# memory.monitor_convolution(outputFmaps, inputFmaps, filters, biases, P, Q)

# # Print output feature maps
# for n in range(ofs.N):
#     for m in range(fs.M):
#         print(f"Output Feature Map {n + 1}, Channel {m + 1}:")
#         for i in range(P):
#             row = " ".join(
#                 f"{outputFmapsWithFlattened[n][m][i][j]:.2f}" for j in range(Q)
#             )
#             print(row)


outputFmaps = np.zeros((ifs.N, fs.M, P, Q))
memory.perform_all_convolutions(outputFmaps, inputFmaps, filters, biases, P, Q)

# Print output feature maps
for n in range(ofs.N):
    for m in range(fs.M):
        print(f"Output Feature Map {n + 1}, Channel {m + 1}:")
        for i in range(P):
            row = " ".join(
                f"{outputFmapsWithFlattened[n][m][i][j]:.2f}" for j in range(Q)
            )
            print(row)

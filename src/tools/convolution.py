from classes.consts import (
    FilterSize as fs,
    InputFmapSize as ifs,
    OutputFmapSize as ofs,
    U as stride,
)
from typing import List, Dict, Any
from classes.InputFeatureMap import InputFeatureMap
from classes.filter import Filter


def convolution(
    outputFmaps: List[Any],
    inputFmaps: List[InputFeatureMap],
    filters: Dict[int, Filter],
    biases: Dict[int, float],
    P: int,
    Q: int,
):

    # now we can start the convolution
    for n in range(ifs.N):
        for m in range(fs.M):
            for x in range(P):
                for y in range(Q):
                    outputFmaps[n][m][x][y] = biases[m]
                    for i in range(fs.R):
                        for j in range(fs.S):
                            for k in range(fs.C):
                                outputFmaps[n][m][x][y] += (
                                    inputFmaps[n].fmap[k][x * stride + i][
                                        y * stride + j
                                    ]
                                    * filters[m].kernel[k][i][j]
                                )
                    # Apply activation function (ReLU)
                    outputFmaps[n][m][x][y] = max(0, outputFmaps[n][m][x][y])
    return outputFmaps


def flattened_convolution(outputFmaps: List[Any],
    flattened_input_fmaps: List[InputFeatureMap],
    flattened_filters: Dict[int, Filter],
    biases: Dict[int, float],
    P: int,
    Q: int):
    for n in range(ifs.N):
        for m in range(fs.M):
            for x in range(P):
                for y in range(Q):
                    # Initialize the output value with bias
                    output_value = biases[m]
                    for i in range(fs.R):
                        for j in range(fs.S):
                            for k in range(fs.C):
                                # Calculate the indices for the input feature map
                                input_x = x * stride + i
                                input_y = y * stride + j
                                # Flatten the 3D index to 1D index
                                input_idx = k * (ifs.H * ifs.W) + input_x * ifs.W + input_y
                                filter_idx = k * (fs.R * fs.S) + i * fs.S + j
                                output_value += flattened_input_fmaps[n, input_idx] * flattened_filters[m, filter_idx]
                    outputFmaps[n, m, x, y] = output_value
    return outputFmaps
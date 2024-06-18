from classes.consts import (
    FilterSize as fs,
    InputFmapSize as ifs,
    OutputFmapSize as ofs,
    U as stride,
)
from typing import List, Dict, Any
from classes.InputFeatureMap import InputFeatureMap
from classes.filter import Filter


# def convolution(
#     outputFmaps: List[Any],
#     inputFmaps: List[InputFeatureMap],
#     filters: Dict[int, Filter],
#     biases: Dict[int, float],
#     P: int,
#     Q: int,
# ):

#     # now we can start the convolution
#     for n in range(ifs.N):
#         for m in range(fs.M):
#             for x in range(P):
#                 for y in range(Q):
#                     outputFmaps[n][m][x][y] = biases[m]
#                     for i in range(fs.R):
#                         for j in range(fs.S):
#                             for k in range(fs.C):
#                                 outputFmaps[n][m][x][y] += (
#                                     inputFmaps[n].fmap[k][x * stride + i][
#                                         y * stride + j
#                                     ]
#                                     * filters[m].kernel[k][i][j]
#                                 )
#                     # Apply activation function (ReLU)
#                     outputFmaps[n][m][x][y] = max(0, outputFmaps[n][m][x][y])
#     return outputFmaps

def convolution(
    outputFmaps: List[Any],
    inputFmaps: List[InputFeatureMap],
    filters: Dict[int, Filter],
    biases: Dict[int, float],
    P: int,
    Q: int,
):
    # Iterate over each input feature map
    for n in range(ifs.N):
        # Iterate over each filter
        for m in range(fs.M):
            # Iterate over each output element in the P x Q grid
            for x in range(P):
                for y in range(Q):
                    # Load the bias value for the current filter
                    bias_value = biases[m]  # 1 load
                    output_value = bias_value  # No additional load/store
                    # Perform the convolution operation
                    for i in range(fs.R):
                        for j in range(fs.S):
                            for k in range(fs.C):
                                # Load the input feature map value
                                input_value = inputFmaps[n].fmap[k][x * stride + i][y * stride + j]  # 1 load
                                # Load the filter kernel value
                                filter_value = filters[m].kernel[k][i][j]  # 1 load
                                # Multiply input and filter values
                                product = input_value * filter_value  # No additional load/store
                                # Accumulate the result
                                output_value += product  # 1 load (for output_value) + 1 store (for output_value)

                    # Apply activation function (ReLU)
                    if output_value < 0:
                        output_value = 0  # No additional load/store
                    # Store the final output value
                    outputFmaps[n][m][x][y] = output_value  # 1 store

    return outputFmaps
# memory accesses cost is (N*M*P*Q) * (1(initialization)+(2+2)×(fs.R×fs.S×fs.C)+1(final store)
# in this case it's (2*2*3*3) * (1+(4*(3*3*3)+1) => 36 * (1+108+1) => 36 * 110 => 3960 memory accesses!


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
from classes.consts import MemoryModel as mm, UnitModel as um
from classes.filter import Filter
from classes.InputFeatureMap import InputFeatureMap
from typing import List, Dict, Any, Union
import numpy as np
from classes.consts import (
    FilterSize as fs,
    InputFmapSize as ifs,
    OutputFmapSize as ofs,
    U as stride,
)


# this class is going to model the energy costs of read
# and write operations in volatile and non-volatile memories
class Memory:
    def __init__(self, memory_type: str) -> None:
        self.memory_type = memory_type.lower()
        if self.memory_type == "volatile":
            self.read_cost = mm.VOLATILE_READ
            self.write_cost = mm.NONVOLATILE_READ
            self.memory_size = mm.VOLATILE_MEMORY_SIZE
        elif self.memory_type == "nonvolatile":
            self.read_cost = mm.NONVOLATILE_READ
            self.write_cost = mm.NONVOLATILE_WRITE
            self.memory_size = mm.NONVOLATILE_MEMORY_SIZE
        else:
            raise ValueError("Invalid memory type. Choose 'volatile' or 'nonvolatile'.")
        self.reads = 0
        self.writes = 0
        self.memory_usage = 0

    def read(self) -> None:
        self.reads += 1

    def write(self) -> None:
        self.writes += 1

    def reset(self) -> None:
        self.reads = 0
        self.writes = 0
        self.memory_usage = 0

    def load(
        self, data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap]
    ) -> None:
        if isinstance(data, np.ndarray):
            size_to_add = data.size * data.itemsize
            if self.memory_usage + size_to_add > self.memory_size:
                raise ValueError("Memory overflow.")
            self.memory_usage += size_to_add
        elif isinstance(data, list):
            for item in data:
                self.load(item)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, int):
                    size_to_add = um.SIZE_OF_INT
                else:
                    size_to_add = len(str(key))

                if self.memory_usage + size_to_add > self.memory_size:
                    raise ValueError("Memory overflow.")
                self.memory_usage += size_to_add

                if isinstance(value, int):
                    size_to_add = um.SIZE_OF_INT
                elif isinstance(value, float):
                    size_to_add = um.SIZE_OF_FLOAT
                elif isinstance(value, Filter):
                    size_to_add = value.kernel.size * um.SIZE_OF_FLOAT
                elif isinstance(value, InputFeatureMap):
                    size_to_add = value.fmap.size * um.SIZE_OF_FLOAT
                else:
                    self.load(value)
                    continue

                if self.memory_usage + size_to_add > self.memory_size:
                    raise ValueError("Memory overflow.")
                self.memory_usage += size_to_add
        elif isinstance(data, InputFeatureMap):
            size_to_add = data.fmap.size * um.SIZE_OF_FLOAT
            if self.memory_usage + size_to_add > self.memory_size:
                raise ValueError("Memory overflow.")
            self.memory_usage += size_to_add
        else:
            raise ValueError("Unsupported data type for memory load.")
        print(f"Current memory usage: {self.memory_usage} bytes")

    def get_total_memory_accesses(self) -> int:
        return self.reads + self.writes

    def get_total_energy_cost(self) -> float:
        return (self.reads * self.read_cost) + (self.writes * self.write_cost)

    def monitor_convolution(
        self,
        outputFmaps: List[Any],
        inputFmaps: List[InputFeatureMap],
        filters: Dict[int, Filter],
        biases: Dict[int, float],
        P: int,
        Q: int,
    ) -> None:
        self.reset()  # Reset the operation counters before starting the convolution

        # Define the monitored convolution function
        def monitored_convolution(
            outputFmaps: List[Any],
            inputFmaps: List[InputFeatureMap],
            filters: Dict[int, Filter],
            biases: Dict[int, float],
            P: int,
            Q: int,
        ) -> None:

            for n in range(ifs.N):
                for m in range(fs.M):
                    for x in range(P):
                        for y in range(Q):
                            # Load the bias value for the current filter
                            output_value = biases[m]  # No additional load/store
                            self.read()  # Track the memory read operation
                            self.write()
                            # Perform the convolution operation
                            for i in range(fs.R):
                                for j in range(fs.S):
                                    for k in range(fs.C):
                                        # Load the input feature map value
                                        input_value = inputFmaps[n].fmap[k][x * stride + i][y * stride + j]
                                        self.read()  
                                        # Load the filter kernel value
                                        filter_value = filters[m].kernel[k][i][j]  # 1 load
                                        self.read()  # Track the memory read operation
                                        # Multiply input and filter values
                                        product = (input_value * filter_value)  # No additional load/store
                                        # Accumulate the result
                                        output_value += product  # 1 load (for output_value) + 1 store (for output_value)
                                        self.read()  # Track the memory read operation
                                        self.write()  # Track the memory write operation

                            # Apply activation function (ReLU)
                            self.read() # we have to read the output value
                            if output_value < 0:
                                output_value = 0
                            outputFmaps[n][m][x][y] = output_value 
                            self.write()  # Track the memory write operation as store

            return outputFmaps
        # we have 
        #   for the bias load 1 R and 1 W => NMPQ * (1R+1W)
        #   for the inner loop 3 R and 1 W => NMPQ * R * S * C * (3R+1W) 
        #   for the activation function 1 R and 1 W => NMPQ * (1R+1W)
        # the total cost would be 2 * NMPQ * Read + 2 * NMPQ * Write + 3 * NMPQ * R * S * C  * Read + NMPQ * R * S * C * Write
        
        
        result = monitored_convolution(outputFmaps, inputFmaps, filters, biases, P, Q)
        total_energy_cost = self.get_total_energy_cost()
        total_memory_accesses = self.get_total_memory_accesses()
        print(
            f"Total memory accesses during the convolution operation: {total_memory_accesses}"
        )
        print(f"Total energy cost of the convolution operation: {total_energy_cost}")
        return result
    


if __name__ == "main":
    # Example usage:
    volatile_memory = Memory("volatile")
    nonvolatile_memory = Memory("nonvolatile")

    print(f"Volatile Memory Read Cost: {volatile_memory.read()}")  # Output: 100
    print(f"Volatile Memory Write Cost: {volatile_memory.write()}")  # Output: 100
    print(f"Non-Volatile Memory Read Cost: {nonvolatile_memory.read()}")  # Output: 250
    print(
        f"Non-Volatile Memory Write Cost: {nonvolatile_memory.write()}"
    )  # Output: 250

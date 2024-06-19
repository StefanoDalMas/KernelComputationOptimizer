from classes.consts import MemoryModel as mm, UnitModel as um
from classes.filter import Filter
from classes.InputFeatureMap import InputFeatureMap
from typing import List, Dict, Any, Union
import numpy as np
from classes.consts import FilterSize as fs, InputFmapSize as ifs, OutputFmapSize as ofs, U as stride


class Memory:
    def __init__(self) -> None:
        # Initialize the costs and sizes for both types of memory
        self.volatile_read_cost = mm.VOLATILE_READ
        self.volatile_write_cost = mm.VOLATILE_WRITE
        self.volatile_memory_size = mm.VOLATILE_MEMORY_SIZE
        self.nonvolatile_read_cost = mm.NONVOLATILE_READ
        self.nonvolatile_write_cost = mm.NONVOLATILE_WRITE
        self.nonvolatile_memory_size = mm.NONVOLATILE_MEMORY_SIZE
        
        # Initialize counters for reads and writes
        self.volatile_reads = 0
        self.volatile_writes = 0
        self.nonvolatile_reads = 0
        self.nonvolatile_writes = 0
        
        # Initialize memory usage and memory allocator
        self.volatile_memory_usage = 0
        self.nonvolatile_memory_usage = 0
        self.volatile_allocator : List[Any]= []
        self.nonvolatile_allocator: List[Any] = []
        

    def read(self, volatile: bool = True) -> None:
        if volatile:
            self.volatile_reads += 1
        else:
            self.nonvolatile_reads += 1

    def write(self, volatile: bool = True) -> None:
        if volatile:
            self.volatile_writes += 1
        else:
            self.nonvolatile_writes += 1

    def reset(self) -> None:
        self.volatile_reads = 0
        self.volatile_writes = 0
        self.nonvolatile_reads = 0
        self.nonvolatile_writes = 0
        self.volatile_memory_usage = 0
        self.nonvolatile_memory_usage = 0
    
    # this method lets us both know what we are allocating and also how much memory we are using
    def update_memory_usage(self, data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap], adding: bool = True, volatile: bool = False) -> None:
        if isinstance(data, np.ndarray):
            size = data.size * data.itemsize
            if adding:
                if volatile:
                    if self.volatile_memory_usage + size > self.volatile_memory_size:
                        raise ValueError("Volatile memory overflow.")
                    self.volatile_memory_usage += size
                else:
                    if self.nonvolatile_memory_usage + size > self.nonvolatile_memory_size:
                        raise ValueError("Non-volatile memory overflow.")
                    self.nonvolatile_memory_usage += size
            else:
                if volatile:
                    if size > self.volatile_memory_usage:
                        raise ValueError("Invalid operation: size to remove exceeds volatile memory usage.")
                    self.volatile_memory_usage -= size
                else:
                    if size > self.nonvolatile_memory_usage:
                        raise ValueError("Invalid operation: size to remove exceeds non-volatile memory usage.")
                    self.nonvolatile_memory_usage -= size
        elif isinstance(data, list):
            for item in data:
                self.update_memory_usage(item, adding, volatile)
        elif isinstance(data, dict):
            for key, value in data.items():
                size = um.SIZE_OF_INT if isinstance(key, int) else len(str(key))
                if adding:
                    if volatile:
                        if self.volatile_memory_usage + size > self.volatile_memory_size:
                            raise ValueError("Volatile memory overflow.")
                        self.volatile_memory_usage += size
                    else:
                        if self.nonvolatile_memory_usage + size > self.nonvolatile_memory_size:
                            raise ValueError("Non-volatile memory overflow.")
                        self.nonvolatile_memory_usage += size
                else:
                    if volatile:
                        if size > self.volatile_memory_usage:
                            raise ValueError("Invalid operation: size to remove exceeds volatile memory usage.")
                        self.volatile_memory_usage -= size
                    else:
                        if size > self.nonvolatile_memory_usage:
                            raise ValueError("Invalid operation: size to remove exceeds non-volatile memory usage.")
                        self.nonvolatile_memory_usage -= size

                if isinstance(value, int):
                    size = um.SIZE_OF_INT
                elif isinstance(value, float):
                    size = um.SIZE_OF_FLOAT
                elif isinstance(value, Filter):
                    size = value.kernel.size * um.SIZE_OF_FLOAT
                elif isinstance(value, InputFeatureMap):
                    size = value.fmap.size * um.SIZE_OF_FLOAT
                else:
                    self.update_memory_usage(value, adding, volatile)
                    continue

                if adding:
                    if volatile:
                        if self.volatile_memory_usage + size > self.volatile_memory_size:
                            raise ValueError("Volatile memory overflow.")
                        self.volatile_memory_usage += size
                    else:
                        if self.nonvolatile_memory_usage + size > self.nonvolatile_memory_size:
                            raise ValueError("Non-volatile memory overflow.")
                        self.nonvolatile_memory_usage += size
                else:
                    if volatile:
                        if size > self.volatile_memory_usage:
                            raise ValueError("Invalid operation: size to remove exceeds volatile memory usage.")
                        self.volatile_memory_usage -= size
                    else:
                        if size > self.nonvolatile_memory_usage:
                            raise ValueError("Invalid operation: size to remove exceeds non-volatile memory usage.")
                        self.nonvolatile_memory_usage -= size
        elif isinstance(data, InputFeatureMap):
            size = data.fmap.size * um.SIZE_OF_FLOAT
            if adding:
                if volatile:
                    if self.volatile_memory_usage + size > self.volatile_memory_size:
                        raise ValueError("Volatile memory overflow.")
                    self.volatile_memory_usage += size
                else:
                    if self.nonvolatile_memory_usage + size > self.nonvolatile_memory_size:
                        raise ValueError("Non-volatile memory overflow.")
                    self.nonvolatile_memory_usage += size
            else:
                if volatile:
                    if size > self.volatile_memory_usage:
                        raise ValueError("Invalid operation: size to remove exceeds volatile memory usage.")
                    self.volatile_memory_usage -= size
                else:
                    if size > self.nonvolatile_memory_usage:
                        raise ValueError("Invalid operation: size to remove exceeds non-volatile memory usage.")
                    self.nonvolatile_memory_usage -= size
        else:
            raise ValueError("Unsupported data type for memory update.")
        print(f"Current volatile memory usage: {self.volatile_memory_usage} bytes")
        print(f"Current non-volatile memory usage: {self.nonvolatile_memory_usage} bytes")

    def alloc(self, data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap], volatile: bool = False) -> None:
        if volatile:
            self.volatile_allocator.append(data)
        else:
            self.nonvolatile_allocator.append(data)
        self.update_memory_usage(data, adding=True, volatile=volatile)

    def free(self, data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap], volatile: bool = False) -> None:
        if volatile:
            self.volatile_allocator.remove(data)
        else:
            self.nonvolatile_allocator.remove(data)
        self.update_memory_usage(data, adding=False, volatile=volatile)

    def get_volatile_memory_accesses(self) -> int:
        return self.volatile_reads + self.volatile_writes
    
    def get_nonvolatile_memory_accesses(self) -> int:
        return self.nonvolatile_reads + self.nonvolatile_writes

    def get_total_memory_accesses(self) -> int:
        return self.get_volatile_memory_accesses() + self.get_nonvolatile_memory_accesses()
    
    def get_volatile_energy_cost(self) -> int:
        return (self.volatile_reads * self.volatile_read_cost) + (self.volatile_writes * self.volatile_write_cost)
    
    def get_nonvolatile_energy_cost(self) -> int:
        return (self.nonvolatile_reads * self.nonvolatile_read_cost) + (self.nonvolatile_writes * self.nonvolatile_write_cost)

    def get_total_energy_cost(self) -> float:
        return self.get_volatile_energy_cost() + self.get_nonvolatile_energy_cost()

    def monitor_convolution(self, outputFmaps: List[Any], inputFmaps: List[InputFeatureMap], filters: Dict[int, Filter], biases: Dict[int, float], P: int, Q: int) -> None:
        self.reset()  # Reset the operation counters before starting the convolution

        # Define the monitored convolution function
        def monitored_convolution(outputFmaps: List[Any], inputFmaps: List[InputFeatureMap], filters: Dict[int, Filter], biases: Dict[int, float], P: int, Q: int) -> None:
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
        print(f"Total memory accesses during the convolution operation: {total_memory_accesses}")
        print(f"Total energy cost of the convolution operation: {total_energy_cost}")
        return result


if __name__ == "main":
    # Example usage:
    memory = Memory()

    print(f"Volatile Memory Read Cost: {memory.read()}")  # Output: 100
    print(f"Volatile Memory Write Cost: {memory.write()}")  # Output: 100
    print(f"Non-Volatile Memory Read Cost: {memory.read(volatile=False)}")  # Output: 250
    print(f"Non-Volatile Memory Write Cost: {memory.write(volatile=False)}")  # Output: 250

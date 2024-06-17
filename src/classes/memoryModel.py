from classes.consts import MemoryModel as mm, UnitModel as um
from classes.filter import Filter
from classes.InputFeatureMap import InputFeatureMap
from typing import List, Dict, Any, Callable, Union
import numpy as np
from functools import partial


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

    def get_total_energy_cost(self) -> float:
        return (self.reads * self.read_cost) + (self.writes * self.write_cost)

    def monitor_convolution(self, func: Callable, *args) -> Any:
        self.reset()  # Reset the operation counters before starting the convolution
        result = func(*args)
        total_energy_cost = self.get_total_energy_cost()
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

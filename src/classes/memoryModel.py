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
    EnergyModel as em,
)


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
        self.volatile_allocator: List[Any] = []
        self.nonvolatile_allocator: List[Any] = []

    def read(self, volatile: bool = False) -> None:
        if volatile:
            self.volatile_reads += 1
        else:
            self.nonvolatile_reads += 1

    def write(self, volatile: bool = False) -> None:
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
    def update_memory_usage(
        self,
        data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap],
        adding: bool = True,
        volatile: bool = False,
    ) -> None:
        if isinstance(data, np.ndarray):
            size = data.size * data.itemsize
            if adding:
                if volatile:
                    if self.volatile_memory_usage + size > self.volatile_memory_size:
                        raise ValueError("Volatile memory overflow.")
                    self.volatile_memory_usage += size
                else:
                    if (
                        self.nonvolatile_memory_usage + size
                        > self.nonvolatile_memory_size
                    ):
                        raise ValueError("Non-volatile memory overflow.")
                    self.nonvolatile_memory_usage += size
            else:
                if volatile:
                    if size > self.volatile_memory_usage:
                        raise ValueError(
                            "Invalid operation: size to remove exceeds volatile memory usage."
                        )
                    self.volatile_memory_usage -= size
                else:
                    if size > self.nonvolatile_memory_usage:
                        raise ValueError(
                            "Invalid operation: size to remove exceeds non-volatile memory usage."
                        )
                    self.nonvolatile_memory_usage -= size
        elif isinstance(data, list):
            for item in data:
                self.update_memory_usage(item, adding, volatile)
        elif isinstance(data, dict):
            for key, value in data.items():
                size = um.SIZE_OF_INT if isinstance(key, int) else len(str(key))
                if adding:
                    if volatile:
                        if (
                            self.volatile_memory_usage + size
                            > self.volatile_memory_size
                        ):
                            raise ValueError("Volatile memory overflow.")
                        self.volatile_memory_usage += size
                    else:
                        if (
                            self.nonvolatile_memory_usage + size
                            > self.nonvolatile_memory_size
                        ):
                            raise ValueError("Non-volatile memory overflow.")
                        self.nonvolatile_memory_usage += size
                else:
                    if volatile:
                        if size > self.volatile_memory_usage:
                            raise ValueError(
                                "Invalid operation: size to remove exceeds volatile memory usage."
                            )
                        self.volatile_memory_usage -= size
                    else:
                        if size > self.nonvolatile_memory_usage:
                            raise ValueError(
                                "Invalid operation: size to remove exceeds non-volatile memory usage."
                            )
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
                        if (
                            self.volatile_memory_usage + size
                            > self.volatile_memory_size
                        ):
                            raise ValueError("Volatile memory overflow.")
                        self.volatile_memory_usage += size
                    else:
                        if (
                            self.nonvolatile_memory_usage + size
                            > self.nonvolatile_memory_size
                        ):
                            raise ValueError("Non-volatile memory overflow.")
                        self.nonvolatile_memory_usage += size
                else:
                    if volatile:
                        if size > self.volatile_memory_usage:
                            raise ValueError(
                                "Invalid operation: size to remove exceeds volatile memory usage."
                            )
                        self.volatile_memory_usage -= size
                    else:
                        if size > self.nonvolatile_memory_usage:
                            raise ValueError(
                                "Invalid operation: size to remove exceeds non-volatile memory usage."
                            )
                        self.nonvolatile_memory_usage -= size
        elif isinstance(data, InputFeatureMap):
            size = data.fmap.size * um.SIZE_OF_FLOAT
            if adding:
                if volatile:
                    if self.volatile_memory_usage + size > self.volatile_memory_size:
                        raise ValueError("Volatile memory overflow.")
                    self.volatile_memory_usage += size
                else:
                    if (
                        self.nonvolatile_memory_usage + size
                        > self.nonvolatile_memory_size
                    ):
                        raise ValueError("Non-volatile memory overflow.")
                    self.nonvolatile_memory_usage += size
            else:
                if volatile:
                    if size > self.volatile_memory_usage:
                        raise ValueError(
                            "Invalid operation: size to remove exceeds volatile memory usage."
                        )
                    self.volatile_memory_usage -= size
                else:
                    if size > self.nonvolatile_memory_usage:
                        raise ValueError(
                            "Invalid operation: size to remove exceeds non-volatile memory usage."
                        )
                    self.nonvolatile_memory_usage -= size
        else:
            raise ValueError("Unsupported data type for memory update.")
        print(f"Current volatile memory usage: {self.volatile_memory_usage} bytes")
        print(
            f"Current non-volatile memory usage: {self.nonvolatile_memory_usage} bytes"
        )

    def alloc(
        self,
        data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap],
        volatile: bool = False,
    ) -> None:
        if volatile:
            self.volatile_allocator.append(data)
        else:
            self.nonvolatile_allocator.append(data)
        self.update_memory_usage(data, adding=True, volatile=volatile)

    def free(
        self,
        data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap],
        volatile: bool = False,
    ) -> None:
        if volatile:
            if isinstance(data,np.ndarray):
                for el in self.volatile_allocator:
                    if np.array_equal(el,data):
                        self.volatile_allocator.remove(el)
            else:
                self.volatile_allocator.remove(data)
        else:
            self.nonvolatile_allocator.remove(data)
        self.update_memory_usage(data, adding=False, volatile=volatile)

    def check_if_volatile(
        self, data: Union[np.ndarray, List, Dict, Filter, InputFeatureMap]
    ) -> bool:
        def compare_data(item, data):
            if isinstance(item, np.ndarray) and isinstance(data, np.ndarray):
                return np.array_equal(item, data)
            elif isinstance(item, list) and isinstance(data, list):
                return all(compare_data(i, d) for i, d in zip(item, data)) and len(
                    item
                ) == len(data)
            elif isinstance(item, dict) and isinstance(data, dict):
                return all(
                    key in item and compare_data(item[key], data[key]) for key in data
                ) and len(item) == len(data)
            elif isinstance(item, Filter) and isinstance(data, Filter):
                return compare_data(item.kernel, data.kernel)
            elif isinstance(item, InputFeatureMap) and isinstance(
                data, InputFeatureMap
            ):
                return compare_data(item.fmap, data.fmap)
            else:
                return item == data

        for item in self.volatile_allocator:
            if type(data) == type(item):
                if compare_data(item, data):
                    return True
        return False

    def get_volatile_memory_accesses(self) -> int:
        return self.volatile_reads + self.volatile_writes

    def get_nonvolatile_memory_accesses(self) -> int:
        return self.nonvolatile_reads + self.nonvolatile_writes

    def get_total_memory_accesses(self) -> int:
        return (
            self.get_volatile_memory_accesses() + self.get_nonvolatile_memory_accesses()
        )

    def get_volatile_energy_cost(self) -> int:
        return (self.volatile_reads * self.volatile_read_cost) + (
            self.volatile_writes * self.volatile_write_cost
        )

    def get_nonvolatile_energy_cost(self) -> int:
        return (self.nonvolatile_reads * self.nonvolatile_read_cost) + (
            self.nonvolatile_writes * self.nonvolatile_write_cost
        )

    def get_total_energy_cost(self) -> float:
        return self.get_volatile_energy_cost() + self.get_nonvolatile_energy_cost()

    def power_failure(self) -> None:
        # Power failure policy : If it happens, save the volatile memory to non-volatile memory and restore it back
        # generate a random number, if it is less than 0.3, we perform a power failure
        if np.random.rand() < em.POWER_FAILURE_PROBABILITY:
            # Create a backup of the volatile allocator
            self.checkpoint()

    def checkpoint(self) -> None:
        backup_volatile_allocator = list(self.volatile_allocator)

        for data in backup_volatile_allocator:
            self.free(data, volatile=True)

        # Write the backup data to non-volatile memory and then read it back
        for data in backup_volatile_allocator:
            self.write(volatile=False)  # Writing to non-volatile memory
            self.read(volatile=False)  # Reading from non-volatile memory

            # Re-allocate the data back to volatile memory

            self.alloc(data, volatile=True)
        print(
            "Power failure occurred. Data has been restored from non-volatile memory."
        )

    def monitor_convolution(
        self,
        outputFmaps: List[Any],
        inputFmaps: List[InputFeatureMap],
        filters: Dict[int, Filter],
        biases: Dict[int, float],
        P: int,
        Q: int,
    ) -> None:
        # self.reset()  # Reset the operation counters before starting the convolution

        # Define the monitored convolution function
        def monitored_convolution(
            outputFmaps: List[Any],
            inputFmaps: List[InputFeatureMap],
            filters: Dict[int, Filter],
            biases: Dict[int, float],
            P: int,
            Q: int,
            volatile_filters: bool,
            volatile_input_fmap: bool,
            volatile_biases: bool,
            volatile_output_fmap: bool,
        ) -> None:
            for n in range(ifs.N):
                for m in range(fs.M):
                    for x in range(P):
                        for y in range(Q):
                            # Load the bias value for the current filter
                            output_value = biases[m]  # No additional load/store
                            self.read(
                                volatile=volatile_biases
                            )  # Track the memory read operation
                            self.write(volatile=volatile_biases)
                            # Perform the convolution operation
                            self.power_failure()  # Check for power failure
                            for i in range(fs.R):
                                for j in range(fs.S):
                                    for k in range(fs.C):
                                        # Load the input feature map value
                                        input_value = inputFmaps[n].fmap[k][
                                            x * stride + i
                                        ][y * stride + j]
                                        self.read(volatile=volatile_input_fmap)
                                        # Load the filter kernel value
                                        filter_value = filters[m].kernel[k][i][
                                            j
                                        ]  # 1 load
                                        self.read(
                                            volatile=volatile_filters
                                        )  # Track the memory read operation
                                        # Multiply input and filter values
                                        product = (
                                            input_value * filter_value
                                        )  # No additional load/store
                                        # Accumulate the result
                                        output_value += product  # 1 load (for output_value) + 1 store (for output_value)
                                        self.read(
                                            volatile=volatile_output_fmap
                                        )  # read output value from memory
                                        self.write(
                                            volatile=volatile_output_fmap
                                        )  # update output value to memory

                            # Apply activation function (ReLU)
                            self.read(
                                volatile=volatile_output_fmap
                            )  # we have to read the output value
                            if output_value < 0:
                                output_value = 0
                            outputFmaps[n][m][x][y] = output_value
                            self.write(
                                volatile=volatile_output_fmap
                            )  # Track the memory write operation as store

            return outputFmaps

        # we have
        #   for the bias load 1 R and 1 W => NMPQ * (1R+1W)
        #   for the inner loop 3 R and 1 W => NMPQ * R * S * C * (3R+1W)
        #   for the activation function 1 R and 1 W => NMPQ * (1R+1W)
        # the total cost would be 2 * NMPQ * Read + 2 * NMPQ * Write + 3 * NMPQ * R * S * C  * Read + NMPQ * R * S * C * Write
        volatile_biases = self.check_if_volatile(biases)
        volatile_filters = self.check_if_volatile(filters)
        volatile_input_fmap = self.check_if_volatile(inputFmaps)
        volatile_output_fmap = (
            True  # for now we assume to always save everything into Volatile!
        )
        result = monitored_convolution(
            outputFmaps,
            inputFmaps,
            filters,
            biases,
            P,
            Q,
            volatile_filters,
            volatile_input_fmap,
            volatile_biases,
            volatile_output_fmap,
        )
        volatile_energy_cost = self.get_volatile_energy_cost()
        nonvolatile_energy_cost = self.get_nonvolatile_energy_cost()
        volatile_memory_accesses = self.get_volatile_memory_accesses()
        nonvolatile_memory_accesses = self.get_nonvolatile_memory_accesses()
        total_energy_cost = self.get_total_energy_cost()
        total_memory_accesses = self.get_total_memory_accesses()
        print(
            "Total energy cost: ",
            total_energy_cost,
            " In volatile : ",
            volatile_energy_cost,
            " In non-volatile : ",
            nonvolatile_energy_cost,
        )
        print(
            "Total memory accesses: ",
            total_memory_accesses,
            " In volatile : ",
            volatile_memory_accesses,
            " In non-volatile : ",
            nonvolatile_memory_accesses,
        )
        return result

    # in this version of the monitor we will perform the convolution one by one
    def perform_all_convolutions(
        self,
        outputFmaps: List[Any],
        inputFmaps: List[InputFeatureMap],
        filters: Dict[int, Filter],
        biases: Dict[int, float],
        P: int,
        Q: int,
    ) -> None:
        for n in range(ifs.N):
            for m in range(fs.M):
                # Perform the monitored convolution for each channel
                for k in range(fs.C):
                    for channel in range(fs.C):
                        self.monitor_convolution_one_by_one(
                            outputFmaps,
                            inputFmaps,
                            filters,
                            biases,
                            P,
                            Q,
                            n,
                            m,
                            k,
                            channel,
                        )

    def monitor_convolution_one_by_one(
        self,
        outputFmaps: List[Any],
        inputFmaps: List[InputFeatureMap],
        filters: Dict[int, Filter],
        biases: Dict[int, float],
        P: int,
        Q: int,
        n: int,
        m: int,
        k: int,
        channel: int,
    ) -> None:
        def monitored_convolution(
            outputFmaps: List[Any],
            inputFmaps: List[InputFeatureMap],
            filters: Dict[int, Filter],
            biases: Dict[int, float],
            P: int,
            Q: int,
            n: int,
            m: int,
            k: int,
            channel: int,
        ) -> None:
            # Bring the filter and corresponding input fmap into volatile memory
            self.alloc(filters[m].kernel[k], volatile=True) # number of : is fs.M - 1
            self.alloc(inputFmaps[n].fmap[k], volatile=True)

            for x in range(P):
                for y in range(Q):
                    output_value = biases[m]
                    self.read(volatile=True)
                    self.write(volatile=True)

                    self.power_failure()

                    for i in range(fs.R):
                        for j in range(fs.S):
                            # Load the input feature map value
                            input_value = inputFmaps[n].fmap[k][x * stride + i][
                                y * stride + j
                            ]
                            self.read(volatile=True)

                            # Load the filter kernel value
                            filter_value = filters[m].kernel[k][i][j]
                            self.read(volatile=True)

                            # Multiply input and filter values
                            product = input_value * filter_value

                            # Accumulate the result
                            output_value += product
                            self.read(volatile=True)
                            self.write(volatile=True)

                    # Apply activation function (ReLU)
                    self.read(volatile=True)
                    if output_value < 0:
                        output_value = 0
                    outputFmaps[n][m][x][y] = output_value
                    self.write(volatile=True)

            # Save the output fmap in non-volatile memory
            self.write(volatile=False)

            # Free the filter and input fmap from volatile memory
            self.free(filters[m].kernel[k], volatile=True)
            self.free(inputFmaps[n].fmap[k], volatile=True)

            return outputFmaps

        result = monitored_convolution(
            outputFmaps, inputFmaps, filters, biases, P, Q, n, m, k, channel
        )

        volatile_energy_cost = self.get_volatile_energy_cost()
        nonvolatile_energy_cost = self.get_nonvolatile_energy_cost()
        volatile_memory_accesses = self.get_volatile_memory_accesses()
        nonvolatile_memory_accesses = self.get_nonvolatile_memory_accesses()
        total_energy_cost = self.get_total_energy_cost()
        total_memory_accesses = self.get_total_memory_accesses()

        print(
            "Total energy cost: ",
            total_energy_cost,
            " In volatile : ",
            volatile_energy_cost,
            " In non-volatile : ",
            nonvolatile_energy_cost,
        )
        print(
            "Total memory accesses: ",
            total_memory_accesses,
            " In volatile : ",
            volatile_memory_accesses,
            " In non-volatile : ",
            nonvolatile_memory_accesses,
        )
        return result


if __name__ == "main":
    # Example usage:
    memory = Memory()

    print(f"Volatile Memory Read Cost: {memory.read()}")  # Output: 100
    print(f"Volatile Memory Write Cost: {memory.write()}")  # Output: 100
    print(
        f"Non-Volatile Memory Read Cost: {memory.read(volatile=False)}"
    )  # Output: 250
    print(
        f"Non-Volatile Memory Write Cost: {memory.write(volatile=False)}"
    )  # Output: 250

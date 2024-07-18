class FilterSize:
    M: int = 2
    C: int = 3
    R: int = 2
    S: int = 2


class InputFmapSize:
    N: int = 2
    C: int = 3
    H: int = 32
    W: int = 32


class OutputFmapSize:
    N: int = 2
    M: int = 3
    P: int = 14
    Q: int = 14


# U is the stride
U: int = 2

# obtained by estimating 3.0V, 25C, 0.5mA for SRAM and 1.5mA Read / 2.0mA Write for FRAM


class MemoryModel:
    VOLATILE_MEMORY_SIZE: int = 8 * 1024  # 8KB
    VOLATILE_READ: float = 93.75
    VOLATILE_WRITE: float = 93.75  # pJ/bit
    NONVOLATILE_MEMORY_SIZE: int = 256 * 1024  # 256KB
    NONVOLATILE_READ: float = 281.25
    NONVOLATILE_WRITE: float = 375  # pJ/bit


# the overall cost should be

# volatile_read_cost = (2 + 3 * fs.R * fs.S) * P * Q * 90
# volatile_write_cost = (2 + 1 * fs.R * fs.S) * P * Q * 93.75
# nonvolatile_write_cost = P * Q * 375


class UnitModel:
    SIZE_OF_INT: int = 4
    SIZE_OF_FLOAT: int = 8


class EnergyModel:
    POWER_FAILURE_PROBABILITY: float = 0.5

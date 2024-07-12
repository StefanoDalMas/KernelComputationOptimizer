class FilterSize:
    M: int = 2
    C: int = 3
    R: int = 3
    S: int = 3


class InputFmapSize:
    N: int = 2
    C: int = 3
    H: int = 6
    W: int = 6


class OutputFmapSize:
    N: int = 2
    M: int = 3
    P: int = 3
    Q: int = 3


# U is the stride
U: int = 3


class MemoryModel:
    VOLATILE_MEMORY_SIZE: int = 8 * 1024 # 8KB
    VOLATILE_READ: float = 100
    VOLATILE_WRITE: float = 100
    NONVOLATILE_MEMORY_SIZE: int = 256 * 1024 # 256KB
    NONVOLATILE_READ: float = 250
    NONVOLATILE_WRITE: float = 250

class UnitModel:
    SIZE_OF_INT: int = 4
    SIZE_OF_FLOAT: int = 8

class EnergyModel:
    POWER_FAILURE_PROBABILITY : int = 0.3
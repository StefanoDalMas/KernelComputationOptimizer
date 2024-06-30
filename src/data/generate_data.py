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


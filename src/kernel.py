from typing import List
from classes.FeatureMap import FeatureMap
from classes.filter import Filter
from classes.consts import FilterSize as fs, InputFmapSize as ifs

# here we would like to create an example of kernel computation

# FILTERS:
# we are going to have M filters, each one with C channels and R x S kernel size

# INPUT FMAP:
# we are going to have C channels and H x W size

# OUTPUT FMAP:
# we are going to have N outputs, each one with M channels and P x Q size


# initialize the filters

filters: List[Filter] = []
for m in range(fs.M):
    filters.append(Filter())

for m in range(fs.M):
    print(f"Filter {m}\n", filters[m])


inputFmaps: List[FeatureMap] = []
for c in range(ifs.C):
    inputFmaps.append(FeatureMap())

for c in range(ifs.C):
    print("Input Feature Map\n", inputFmaps[c])

import warnings

import numpy as np
from tqdm.contrib.concurrent import process_map

from ImageSplitting import splitter

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

if __name__ == "__main__":
    total = 21806
    res = process_map(splitter._foo, range(total), max_workers=16, chunksize=1)

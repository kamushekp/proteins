import splitter
import numpy as np
from tqdm.contrib.concurrent import process_map
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

if __name__ == "__main__":
    total = 21806
    res = process_map(splitter._foo, range(total), max_workers = 32, chunksize=1)

       
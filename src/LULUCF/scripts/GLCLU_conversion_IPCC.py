import os

# dask/parallelization libraries
import coiled
import dask
from dask.distributed import Client, LocalCluster
from dask.distributed import print as dask_print
import dask.config
import distributed

# scipy basics
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows

from numba import jit
import concurrent.futures

import boto3
import time
import math
import ctypes
import pandas as pd

def aggregate():

    print("Hello world")

    # Local cluster with multiple workers
    local_cluster = LocalCluster()
    local_client = Client(local_cluster)
    print(local_client)



if __name__ == '__main__':

    aggregate()




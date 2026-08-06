"""
With assistance from https://chatgpt.com/share/e/e91c2e03-8c71-44f9-9872-61d25c51bc87
"""

import argparse
import coiled
import dask
from dask.distributed import Client
import dask.array as da
from create_cluster_test import create_cluster_test
from run_operation_test import run_operation_test
from terminate_cluster_test import terminate_cluster_test

def main(n_workers, worker_memory):
    # Step 1: Create the cluster
    cluster = create_cluster_test(n_workers, worker_memory)

    # Step 2: Run a test operation
    result = run_operation_test(cluster.name)

    # Step 3: Delete the cluster
    terminate_cluster_test(cluster.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='8', help='Memory per worker (e.g., 8GiB)')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.n_workers, args.worker_memory)

"""
With assistance from https://chatgpt.com/share/e/e91c2e03-8c71-44f9-9872-61d25c51bc87
"""

import argparse
import coiled
import dask
from dask.distributed import Client
import dask.array as da
from create_cluster import create_cluster
from run_test_operation import run_test_operation
from terminate_cluster import terminate_cluster

def main(n_workers, worker_memory):
    # Step 1: Create the cluster
    cluster = create_cluster(n_workers, worker_memory)

    # Step 2: Run a test operation
    result = run_test_operation(cluster.name)

    # Step 3: Delete the cluster
    terminate_cluster(cluster.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='8', help='Memory per worker (e.g., 8GiB)')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.n_workers, args.worker_memory)

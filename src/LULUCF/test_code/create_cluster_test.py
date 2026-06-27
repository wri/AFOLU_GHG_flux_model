"""
With assistance from https://chatgpt.com/share/e/e91c2e03-8c71-44f9-9872-61d25c51bc87
"""

import coiled
import argparse

def create_cluster_test(n_workers, worker_memory):

    # Convert worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"

    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="DAG_test_py_connection",
        account='wri-forest-research',
        worker_memory=worker_memory_str
    )
    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}, Worker memory: {worker_memory_str}")
    return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='8', help='Memory per worker (e.g., 8GiB)')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    create_cluster_test(args.n_workers, args.worker_memory)

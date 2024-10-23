"""
Run from src/LULUCF/
python -m scripts.utilities.create_cluster -n 1 -m 8
"""

import coiled
import argparse

def create_cluster(n_workers, worker_memory, worker_cpu):

    # Convert worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"

    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="AFOLU_flux_model_scripts",
        workspace='wri-forest-research',
        # mount_bucket="s3://gfw2-data",
        worker_memory = worker_memory_str,
        worker_cpu = worker_cpu
    )
    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}; Worker memory: {worker_memory_str}; cpus per worker: {worker_cpu}")
    return cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=str, default='8', help='Memory per worker (e.g., 8GiB)')
    parser.add_argument('-c', '--worker_cpu', type=str, default='4', help='Number of CPUs per worker')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    create_cluster(args.n_workers, args.worker_memory, args.worker_cpu)

"""
python -m scripts.utilities.terminate_cluster --cluster_name AFOLU_flux_model_scripts
"""

import coiled
import argparse

def terminate_cluster(cluster_name):
    # Connect to the existing cluster using the cluster's name
    cluster = coiled.Cluster(name=cluster_name)

    # Terminate the cluster
    cluster.shutdown()

    print(f"Cluster '{cluster_name}' has been terminated.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')

    args = parser.parse_args()

    cluster_name = args.cluster_name


    if cluster_name:
        terminate_cluster(cluster_name)
    else:
        print("Please provide the cluster name as an argument.")

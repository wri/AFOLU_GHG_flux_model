"""
With assistance from https://chatgpt.com/share/e/e91c2e03-8c71-44f9-9872-61d25c51bc87
"""

import coiled

def terminate_cluster_test(cluster_name):
    # Connect to the existing cluster using the cluster's name
    cluster = coiled.Cluster(name=cluster_name)

    # Terminate the cluster
    cluster.shutdown()

    print(f"Cluster '{cluster_name}' has been terminated.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cluster_name = sys.argv[1]
        terminate_cluster_test(cluster_name)
    else:
        print("Please provide the cluster name as an argument.")

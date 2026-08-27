import argparse
import coiled

from src.scripts.utilities import constants_and_names as cn


def resize_coiled_cluster(cluster_name, n_workers):
    matches = [
        cluster
        for cluster in coiled.list_clusters(workspace=cn.Coiled_workspace)
        if cluster.get("name") == cluster_name
    ]
    if not matches:
        raise RuntimeError(
            f"Cluster {cluster_name!r} does not exist in workspace "
            f"{cn.Coiled_workspace!r}."
        )

    cluster = coiled.Cluster(
        name=cluster_name,
        workspace=cn.Coiled_workspace,
        shutdown_on_close=False,
        package_sync=False,
    )

    # Resize the cluster to the specified number of workers
    cluster.scale(n_workers)

    print(f"Cluster '{cluster_name}' resized to {n_workers} workers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an existing Coiled cluster to a specified number of workers.")
    parser.add_argument('cluster_name', type=str, help='Name of the Coiled cluster to resize')
    parser.add_argument('-n', '--n_workers', type=int, required=True, help='Number of workers to resize the cluster to')

    args = parser.parse_args()

    # Resize the cluster with the provided arguments
    resize_coiled_cluster(args.cluster_name, args.n_workers)

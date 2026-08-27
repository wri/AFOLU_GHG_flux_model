"""Terminate an existing Coiled cluster without creating one."""

from __future__ import annotations

import argparse
from typing import Sequence

import coiled

from src.scripts.utilities import constants_and_names as cn


def _existing_cluster(cluster_name: str) -> dict:
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
    return matches[0]


def terminate_cluster(cluster_name: str) -> None:
    """Terminate the exact existing cluster named by ``cluster_name``."""

    cluster = _existing_cluster(cluster_name)
    cluster_id = cluster.get("id") or cluster.get("cluster_id")
    if cluster_id is None:
        raise RuntimeError(f"Cluster {cluster_name!r} has no resolvable identifier.")
    coiled.delete_cluster(
        cluster_id=int(cluster_id),
        workspace=cn.Coiled_workspace,
    )
    print(f"Cluster {cluster_name!r} has been terminated.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminate an existing Coiled cluster.")
    parser.add_argument("cluster_name", help="Exact name of the cluster to terminate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    terminate_cluster(args.cluster_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

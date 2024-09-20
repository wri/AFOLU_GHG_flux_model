# utils/cluster_utils.py

import coiled
from dask.distributed import Client, LocalCluster

def setup_coiled_cluster(n_workers: int = 60,
                         use_best_zone: bool = True,
                         compute_purchase_option: str = "spot_with_fallback",
                         idle_timeout: str = "10 minutes",
                         region: str = "us-east-1",
                         name: str = "AFOLU_flux_model",
                         account: str = 'wri-forest-research',
                         worker_cpu: int = 4,
                         worker_memory: str = "32GiB") -> coiled.Cluster:
    """
    Sets up a Coiled cluster with specified configurations.

    Args:
        n_workers (int): Number of worker nodes.
        use_best_zone (bool): Whether to use the best available zone.
        compute_purchase_option (str): Compute purchase option.
        idle_timeout (str): Time before idle workers are shut down.
        region (str): AWS region.
        name (str): Cluster name.
        account (str): Coiled account name.
        worker_cpu (int): Number of CPUs per worker.
        worker_memory (str): Memory per worker.

    Returns:
        coiled.Cluster: Initialized Coiled cluster.
    """
    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=use_best_zone,
        compute_purchase_option=compute_purchase_option,
        idle_timeout=idle_timeout,
        region=region,
        name=name,
        account=account,
        worker_cpu=worker_cpu,
        worker_memory=worker_memory
    )
    return cluster

def setup_test_coiled_cluster() -> coiled.Cluster:
    """
    Sets up a test Coiled cluster with minimal resources.

    Returns:
        coiled.Cluster: Initialized test Coiled cluster.
    """
    cluster = coiled.Cluster(
        n_workers=1,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="20 minutes",
        region="us-east-1",
        name="AFOLU_flux_model",
        account='wri-forest-research',
        worker_cpu=4,
        worker_memory="32GiB"
    )
    return cluster

def setup_local_single_process_cluster() -> Client:
    """
    Sets up a local single-process Dask cluster.

    Returns:
        Client: Dask client connected to the local cluster.
    """
    client = Client(processes=False)
    return client

def setup_local_multi_process_cluster() -> Client:
    """
    Sets up a local multi-process Dask cluster.

    Returns:
        Client: Dask client connected to the local multi-process cluster.
    """
    local_cluster = LocalCluster()
    client = Client(local_cluster)
    return client

def shutdown_cluster(client: Client, cluster: coiled.Cluster = None):
    """
    Shuts down the Dask client and optionally the Coiled cluster.

    Args:
        client (Client): Dask client to shut down.
        cluster (coiled.Cluster, optional): Coiled cluster to shut down. Defaults to None.
    """
    client.shutdown()
    if cluster:
        cluster.shutdown()

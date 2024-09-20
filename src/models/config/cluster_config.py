# config/cluster_config.py

import logging
import coiled
from dask.distributed import Client, LocalCluster
from config.constants import (
    s3_region_name,
    s3_bucket_name
)

def setup_coiled_cluster():
    """
    Sets up a Coiled Dask cluster for scalable cloud-based processing.

    Returns:
        tuple: A tuple containing the Dask Client and Coiled Cluster objects.
    """
    try:
        # Initialize the Coiled cluster with desired configurations
        cluster = coiled.Cluster(
            n_workers=60,
            use_best_zone=True,
            compute_purchase_option="spot_with_fallback",
            idle_timeout="10 minutes",
            region=s3_region_name,
            name="afolu_flux_model_cluster",
            account='wri-forest-research',  # Replace with your Coiled account if different
            worker_cpu=4,
            worker_memory="32GiB",
            software="default"  # You can customize the software environment if needed
        )
        client = cluster.get_client()
        logging.info("Coiled cluster successfully initialized.")
        return client, cluster
    except Exception as e:
        logging.error(f"Failed to set up Coiled cluster: {e}")
        raise

def setup_local_cluster():
    """
    Sets up a local Dask cluster for development and testing.

    Returns:
        tuple: A tuple containing the Dask Client and Local Cluster objects.
    """
    try:
        # Initialize the Local Dask cluster with desired configurations
        cluster = LocalCluster(
            n_workers=4,            # Number of worker processes
            threads_per_worker=2,   # Number of threads per worker
            memory_limit='32GB',    # Memory limit per worker
            dashboard_address=':8787'  # Port for the Dask dashboard
        )
        client = Client(cluster)
        logging.info("Local Dask cluster successfully initialized.")
        return client, cluster
    except Exception as e:
        logging.error(f"Failed to set up local Dask cluster: {e}")
        raise

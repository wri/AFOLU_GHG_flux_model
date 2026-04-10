"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/
python -m src.utilities.create_cluster -n 1 -t 1 -m 16 -cn LULUCF_model
python -m src.utilities.create_cluster -n 5 -t 1 -m 32 -cn LULUCF_model
python -m src.utilities.create_cluster -n 20 -t 1 -m 64 -cn LULUCF_model

Table of instance types (and pricing): https://instances.vantage.sh/?id=9c1a108b13a45889fc00951e867ca5295e82dd2c
Table of spot pricing: https://aws.amazon.com/ec2/spot/pricing/
These are the cheapest worker types and they have fewer vCPUs than usual for the memory.
This makes them less costly on AWS and use fewer Coiled credits.

List available worker types for Coiled clusters with: coiled.list_instance_types() in the Python shell

Using more than 1 thread/worker slows down processing a lot when there are more tasks than workers for the core LULUCF model,
which is the situation for large analyses, obviously.
"""

import coiled
import argparse
import sys
import os
import base64
from dask.distributed import Client
from dask import config

# Function to write Google Cloud Project credentials to all workers
def write_gcp_creds():
    import os, base64
    destination = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    b64 = os.environ["GCP_CREDENTIALS_B64"]
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(base64.b64decode(b64))
    return destination, os.path.exists(destination)


def create_cluster(cluster_name, n_workers, worker_memory, threads_per_worker=None, on_demand=False, zonal_stats=False, gcp=None):

    # Converts worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"
    scheduler_memory_str = f"{worker_memory}GiB"

    if worker_memory == 128:
        idle_timeout = 10
        scheduler_vm_type = "x2iedn.xlarge"    # 4 vCPU/worker
        worker_vm_type = "x2iedn.xlarge"

    elif worker_memory == 64:
        idle_timeout = 15
        # scheduler_vm_type = "x8g.xlarge"    # 4 vCPU/worker
        # worker_vm_type = "x8g.xlarge"
        scheduler_vm_type = "r7g.xlarge"    # 8 vCPU/worker, what Solomon used for zonal stats
        worker_vm_type = "r7g.2xlarge"

    elif worker_memory == 32:
        idle_timeout = 20
        # scheduler_vm_type = "r7g.large"    # 4 vCPU/worker, same series as Solomon used for zonal stats
        # worker_vm_type = "r7g.xlarge"
        scheduler_vm_type = "x8g.large"   # 2 vCPU/worker. x2gd.large also has this ratio, and theoretically lower interruption rates but has worse hardware.
        worker_vm_type = "x8g.large"      # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3
        # scheduler_vm_type = "x2gd.large"   # 2 vCPU/worker. x8g.large also has this ratio. x2gd.large theoretically has a lower interruption rate but seems older and slower.
        # worker_vm_type = "x2gd.large"      # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3

    elif worker_memory == 16:
        idle_timeout = 25
        scheduler_vm_type = "x2gd.medium"   # 1 vCPU/worker
        worker_vm_type = "x2gd.medium"

    elif worker_memory == 8:
        idle_timeout = 25
        scheduler_vm_type = "r8g.medium"   # 1 vCPU/worker
        worker_vm_type = "r8g.medium"

    elif worker_memory == 4:
        idle_timeout = 25
        scheduler_vm_type = "m8g.medium"   # 1 vCPU/worker
        worker_vm_type = "m8g.medium"

    # # t2.small not available with Coiled. t3.small has 2 vCPUs, so it's not actually Coiled credit-effective.
    # elif worker_memory == 2:
    #     idle_timeout = 25
    #     scheduler_vm_type = "t3.small"
    #     worker_vm_type = "t3.small"

    # # Couldn't get a cluster started that used 1GB workers using t3a.micro, t3.micro, or t4g.micro. Don't know why.
    # elif worker_memory == 1:
    #     idle_timeout = 25
    #     scheduler_vm_type = "t3a.micro"
    #     worker_vm_type = "t3a.micro"

    else:
        sys.exit('Memory argument not 2, 4, 8, 16, 32, 64, or 128 GB')

    idle_timeout = f"{idle_timeout} minutes"

    worker_options = {}
    if threads_per_worker is not None:
        worker_options["nthreads"] = threads_per_worker

    if zonal_stats:
        purchase_option = "on-demand"
        use_best_zone = False
        allow_cross_zone = False
        software_environment = "afolu-env_202512222"  # pins zarr==3.1.3 for xr.open_zarr compatibility
    else:
        software_environment = None  # use default package sync (uploads local src wheel)
        # Uses on-demand workers for large jobs. Otherwise, prefers spot workers.
        if n_workers > 120:
            purchase_option = "on-demand"
            use_best_zone = False  # Should allow workers to be split across different zones, to help obtain large requested amount of workers

            # Allows workers in different availability zones, to help obtain large requested amount of workers.
            # Has costs for transferring data between workers in different zones, which happens for zonal stats but not model runs.
            # So, can't allow cross zone for zonal stats.
            allow_cross_zone = True
        elif on_demand:
            purchase_option = "on-demand"
            use_best_zone = False
            allow_cross_zone = False
        else:
            purchase_option = "spot_with_fallback"
            use_best_zone = True
            allow_cross_zone = False

    # If gcp flag is initialized, pass in local GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS to all workers
    env = {}
    gcp_creds_b64 = None

    if gcp:
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        gcp_credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        gcp_credentials_dest = "/tmp/gcp.json"

        if not gcp_project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set in your environment.")
        if not gcp_credentials_file:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set in your environment.")
        if not os.path.exists(gcp_credentials_file):
            raise FileNotFoundError(f"Credentials file not found: {gcp_credentials_file}")

        env["GOOGLE_CLOUD_PROJECT"] = gcp_project
        env["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_dest

        with open(gcp_credentials_file, "rb") as f:
            gcp_creds_b64 = base64.b64encode(f.read()).decode("ascii")


    cluster = coiled.Cluster(
        n_workers=n_workers,
        use_best_zone=use_best_zone,
        compute_purchase_option=purchase_option,
        idle_timeout=idle_timeout,
        region="us-east-1",
        name=cluster_name,
        workspace='wri-forest-research',
        tags = {"project": "AFOLU_flux_model"},
        allow_cross_zone=allow_cross_zone,
        scheduler_vm_types = scheduler_vm_type,
        worker_vm_types = worker_vm_type,
        worker_options = worker_options,
        # software="afolu-env_coiled_20251119",  # Specifies all the Python package versions
        # package_sync=True,  # also upload local src package as wheel to workers
        environ=env  # pass env vars to scheduler/workers
        # send_dask_config = True
    )

    client = Client(cluster)

    # If gcp flag is initialized, write Google credentials file onto every worker
    if gcp:
        cluster.send_private_envs({"GCP_CREDENTIALS_B64": gcp_creds_b64})
        client.run(write_gcp_creds)

    print(f"Cluster created with name: {cluster.name}")
    print(f"Number of workers: {n_workers}; worker memory: {worker_memory_str}; scheduler memory: {scheduler_memory_str}; "
          f"'threads per worker: {threads_per_worker}; worker purchase option: {purchase_option}")
    return cluster


# # Gets worker memory configuration that is specified in ~/.config/dask/distributed.yaml
# # Check from https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
# def check_worker_memory_config():
#     return {
#         "target": config.get("distributed.worker.memory.target"),
#         "spill": config.get("distributed.worker.memory.spill"),
#         "pause": config.get("distributed.worker.memory.pause"),
#         "terminate": config.get("distributed.worker.memory.terminate"),
#     }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers for the cluster')
    parser.add_argument('-m', '--worker_memory', type=int, help='Memory per worker')
    parser.add_argument('-t', '--threads_per_worker', type=int, help='Number of threads/worker')
    parser.add_argument('-od', '--on_demand', action='store_true', help='Use on-demand workers (not spot workers)')
    parser.add_argument('-zs', '--zonal_stats', action='store_true', help='Use zonal stats worker configuration')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    n_workers = args.n_workers
    worker_memory = args.worker_memory
    threads_per_worker = args.threads_per_worker
    on_demand = args.on_demand
    zonal_stats = args.zonal_stats

    # Create the cluster with command line arguments
    cluster = create_cluster(cluster_name, n_workers, worker_memory, threads_per_worker, on_demand=on_demand, zonal_stats=zonal_stats)

    # client = Client(cluster)
    # print(client.run(check_worker_memory_config))



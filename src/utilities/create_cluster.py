"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/
python -m src.utilities.create_cluster -n 1 -t 1 -m 16 -cn LULUCF_model
python -m src.utilities.create_cluster -n 5 -t 1 -m 32 -cn LULUCF_model
python -m src.utilities.create_cluster -n 20 -t 1 -m 64 -cn LULUCF_model

To pass in Google Cloud local environment variables:
python -m src.utilities.create_cluster -n 1 -m 4 -cn GEE_assets --gcp

Table of instance types (and pricing): https://instances.vantage.sh/?id=9c1a108b13a45889fc00951e867ca5295e82dd2c
Table of spot pricing: https://aws.amazon.com/ec2/spot/pricing/
These are the cheapest worker types and they have fewer vCPUs than usual for the memory.
This makes them less costly on AWS and use fewer Coiled credits.

List available worker types for Coiled clusters with: coiled.list_instance_types() in the Python shell

Clusters for zonal stats and non-zonal stats runs have different configurations:
    --Zonal stats clusters need zarr v3.1.3. Those clusters use their own Coiled software environment.
        They do no use functions from the repo on workers and use .compute(), so they just need a stable conda environment
        but not the project src files to be distributed to workers.
    --Non-zonal stats clusters need zarr v3.1.6. Because some other packages that I hadn't pinned keep changing,
      I made a different software environment for these clusters with a more fixed package list.
      But because these clusters use functions from the repo (src) on workers, they also need src available to them,
      so that's where the src zipping and uploading below comes in.
      This is per Claude session 'Coiled cluster creation error'.
    --Claude summary: " .compute() sends data operations;
      client.submit() with a src.* function sends a module reference that the worker has to be able to import."


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
import subprocess
import glob
import tempfile
import os
import zipfile

# Project imports
from src.utilities import constants_and_names as cn


# Function to write Google Cloud Project credentials to all workers
def write_gcp_creds():
    import os, base64
    destination = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    b64 = os.environ["GCP_CREDENTIALS_B64"]
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(base64.b64decode(b64))
    return destination, os.path.exists(destination)


def create_cluster(cluster_name, n_workers, worker_memory, threads_per_worker=None, disk_space = None, on_demand=False, zonal_stats=False, cog=False, gcp=None):

    if zonal_stats or ("zonal" in cluster_name) or ("stats" in cluster_name):
        print("Using zonal stats worker configuration")
        zonal_stats = True
    else:
        zonal_stats = False

    if cog or ("cog" in cluster_name.lower()):
        print("Using cog worker configuration")
        cog = True
    else:
        cog = False

    # Converts worker_memory from an integer to the required format (e.g., 8 to "8GiB")
    worker_memory_str = f"{worker_memory}GiB"
    scheduler_memory_str = f"{worker_memory}GiB"

    if worker_memory == 128:
        idle_timeout = 10
        scheduler_vm_type = "x2iedn.xlarge"    # 4 vCPU/worker
        worker_vm_type = "x2iedn.xlarge"

    elif worker_memory == 64:
        idle_timeout = 15
        if zonal_stats == True:
            scheduler_vm_type = "r8g.xlarge"  # 8 vCPU/worker, what Solomon used for zonal stats
            worker_vm_type = "r8g.2xlarge"
        elif cog == True:
            scheduler_vm_type = "r8g.xlarge"  # 8 vCPU/worker, used for COG creation
            worker_vm_type = "r8g.2xlarge"
        else:
            scheduler_vm_type = "x8g.xlarge"    # 4 vCPU/worker
            worker_vm_type = "x8g.xlarge"

    elif worker_memory == 32:
        idle_timeout = 20
        if zonal_stats == True:
            scheduler_vm_type = "r8g.large"    # 4 vCPU/worker, same series as Solomon used for zonal stats
            worker_vm_type = "r8g.xlarge"
        elif cog == True:
            scheduler_vm_type = "m4.xlarge"    # 8 vCPU/worker, used for COG creation
            worker_vm_type = "m4.2xlarge"
        else:
            scheduler_vm_type = "x8g.large"   # 2 vCPU/worker. x2gd.large also has this ratio, and theoretically lower interruption rates but has worse hardware.
            worker_vm_type = "x8g.large"      # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3
        # scheduler_vm_type = "x2gd.large"   # 2 vCPU/worker. x8g.large also has this ratio. x2gd.large theoretically has a lower interruption rate but seems older and slower.
        # worker_vm_type = "x2gd.large"      # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3

    elif worker_memory == 16:
        idle_timeout = 25
        if zonal_stats == True:
            scheduler_vm_type = "r8g.medium"    # 2 vCPU/worker, same series as Solomon used for zonal stats
            worker_vm_type = "r8g.large"
        elif cog == True:
            scheduler_vm_type = "m4.xlarge"     # 4 vCPU/worker, used for COG creation
            worker_vm_type = "m4.xlarge"
        else:
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

    # Special settings for zonal stats clusters: can't have workers across zones (to prevent inter-zone data transfer), and need to set zarr version
    if zonal_stats == True:
        purchase_option = "on-demand"
        use_best_zone = False
        allow_cross_zone = False
        software = "afolu_zonal_stats_20251222"  # pins zarr==3.1.3 for xr.open_zarr compatibility
    else:
        print("Not using zonal stats worker configuration")
        software = "afolu_not_zonal_stats_20251119"  # pins zarr==3.1.6
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
            allow_cross_zone = True
        else:
            purchase_option = "spot_with_fallback"
            use_best_zone = True
            allow_cross_zone = True

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

    if disk_space is not None:
        cluster = coiled.Cluster(
            n_workers=n_workers,
            use_best_zone=use_best_zone,
            compute_purchase_option=purchase_option,
            idle_timeout=idle_timeout,
            region="us-east-1",
            name=cluster_name,
            workspace=cn.Coiled_workspace,
            tags = {"wri:project": "AFOLU_flux_model", "wri:program": "FLW"},
            scheduler_vm_types=scheduler_vm_type,
            worker_vm_types=worker_vm_type,
            worker_options=worker_options,
            worker_disk_size=f"{disk_space} GiB",
            environ=env,  # pass env vars to scheduler/workers
            **({'software': software} if software else {}),
        )
    else:
        cluster = coiled.Cluster(
            n_workers=n_workers,
            use_best_zone=use_best_zone,
            compute_purchase_option=purchase_option,
            idle_timeout=idle_timeout,
            region="us-east-1",
            name=cluster_name,
            workspace=cn.Coiled_workspace,
            tags = {"wri:project": "AFOLU_flux_model", "wri:program": "FLW"},
            allow_cross_zone=allow_cross_zone,
            scheduler_vm_types = scheduler_vm_type,
            worker_vm_types = worker_vm_type,
            worker_options = worker_options,
            environ=env,  # pass env vars to scheduler/workers
            **({'software': software} if software else {}),
        )

    client = Client(cluster)

    # Adds the src files to the workers so that they have access to those.
    # Otherwise, workers don't have access to src files, which only matters for non-zonal stats runs
    # (not for zonal stats because src functions aren't assigned to workers).
    # This is not actually necessary for zonal stats clusters right now because they don't use src on workers,
    # but there's no harm in making this happen for all clusters.
    # Per Claude session 'Coiled cluster creation error'
    print("Uploading src to workers")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "src.zip")
        project_root = "/mnt/c/GIS/git/AFOLU_GHG_flux_model"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(os.path.join(project_root, "src")):
                for file in files:
                    if file.endswith(".py"):
                        filepath = os.path.join(root, file)
                        zf.write(filepath, os.path.relpath(filepath, project_root))
        client.upload_file(zip_path)
    print("Uploaded src to workers")

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
    parser.add_argument('-d', '--disk_space', type=int, help='Disk space')
    parser.add_argument('-od', '--on_demand', action='store_true', help='Use on-demand workers (not spot workers)')
    parser.add_argument('-zs', '--zonal_stats', action='store_true', help='Use zonal stats worker configuration')
    parser.add_argument('-c', '--cog', action='store_true', help='Use cog worker configuration')

    # Options to copy certain local environments into Coiled workers
    parser.add_argument("--gcp", action="store_true", help="If set, copy local GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS into the Coiled cluster.")

    args = parser.parse_args()

    cluster_name = args.cluster_name
    n_workers = args.n_workers
    worker_memory = args.worker_memory
    threads_per_worker = args.threads_per_worker
    disk_space = args.disk_space
    on_demand = args.on_demand
    zonal_stats = args.zonal_stats
    cog = args.cog
    gcp = args.gcp

    # Create the cluster with command line arguments
    cluster = create_cluster(cluster_name, n_workers, worker_memory, threads_per_worker, disk_space=disk_space, on_demand=on_demand, zonal_stats=zonal_stats, cog=cog, gcp=gcp)

    # client = Client(cluster)
    # print(client.run(check_worker_memory_config))



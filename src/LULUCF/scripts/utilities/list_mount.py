import os

# def list_files_in_directory(path):
#     if os.path.exists(path):
#         print(f"Files in {path}:")
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 print(os.path.join(root, file))
#     else:
#         print(f"Directory {path} does not exist.")
#
# list_files_in_directory("/mnt/")


# def list_mounted_directories():
#     root_path = "/mnt/"
#     if os.path.exists(root_path):
#         print(f"Directories in {root_path}:")
#         for directory in os.listdir(root_path):
#             print(directory)
#     else:
#         print(f"Mount point {root_path} does not exist.")
#
# list_mounted_directories()

import coiled
import os
from dask.distributed import Client
from dask.distributed import print

def list_mounted_directories():
    root_path = "/mounty_mountebank/"  # Mount point /mounty_mountebank/ does not exist.
    if os.path.exists(root_path):
        print(f"Directories in {root_path}:")
        for directory in os.listdir(root_path):
            print(directory)
    else:
        print(f"Mount point {root_path} does not exist.")

    root_path = "/mount/"    # Directories in /mount/:
    if os.path.exists(root_path):
        print(f"Directories in {root_path}:")
        for directory in os.listdir(root_path):
            print(directory)
    else:
        print(f"Mount point {root_path} does not exist.")

    root_path = "/mnt/"    # Directories in /mnt/:
    if os.path.exists(root_path):
        print(f"Directories in {root_path}:")
        for directory in os.listdir(root_path):
            print(directory)
    else:
        print(f"Mount point {root_path} does not exist.")

def main():
    # Connect to or create the Coiled cluster
    cluster = coiled.Cluster(
        n_workers=1,
        name="AFOLU_flux_model_mount_test",
        mount_bucket="s3://gfw2-data",
        allow_ssh = True
    )
    client = Client(cluster)

    print("Made cluster")

    # Run the directory listing on a Dask worker
    future = client.submit(list_mounted_directories)
    future.result()

    print("Sent results from cluster")

    # # Close the client and cluster after the operation
    # client.close()
    # cluster.close()

if __name__ == "__main__":
    main()


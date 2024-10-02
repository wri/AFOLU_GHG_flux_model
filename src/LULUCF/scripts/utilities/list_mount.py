import coiled
import rasterio
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
        for directory in os.listdir(f"{root_path}gfw2-data/climate/"):
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



    # Define the S3 URI and convert it to the local mounted path
    s3_uri = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/40000_pixels/20240821/00N_000E__AGC_density_MgC_ha_2000.tif"
    local_uri = s3_uri.replace("s3://gfw2-data/", "/mount/gfw2-data/")

    # Access the file using rasterio
    try:
        with rasterio.open(local_uri) as dataset:
            # Read metadata and a small portion of the data to test access
            print("Dataset metadata:")
            print(dataset.meta)

            # Read a small portion of data (e.g., the first window)
            window = rasterio.windows.Window(0, 0, 100, 100)
            data = dataset.read(1, window=window)
            print("Data sample:")
            print(data)
            print(data.max())

    except Exception as e:
        print(f"Failed to access the dataset: {e}")

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
import coiled
import rioxarray as rxr
import xarray as xr
import numpy as np
import boto3
import logging
import rasterio
from rasterio.transform import from_origin
from dask.distributed import Client

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def setup_coiled_cluster():
    coiled_cluster = coiled.Cluster(
        n_workers=1,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="test_coiled_connection",
        account='wri-forest-research',
        worker_memory="32GiB"
    )
    coiled_client = coiled_cluster.get_client()
    return coiled_client, coiled_cluster


def create_test_raster(output_raster_path):
    # Create a test raster (100x100 grid with values from 1 to 10000)
    data = np.arange(1, 10001, dtype=np.float32).reshape((100, 100))
    transform = from_origin(0, 100, 1, 1)  # Affine transformation

    # Create xarray DataArray with geospatial attributes
    raster = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.linspace(100 - 0.5, 0.5, 100), "x": np.linspace(0.5, 100 - 0.5, 100)},
    )
    raster = raster.rio.write_crs("EPSG:4326", inplace=True)
    raster = raster.rio.write_transform(transform, inplace=True)
    raster.rio.to_raster(output_raster_path, compress='lzw')

    logging.info(f"Test raster created and saved to {output_raster_path}")


def upload_to_s3(local_path, s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    try:
        logging.info(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        logging.info(f"Successfully uploaded {local_path} to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to s3://{s3_bucket}/{s3_key}: {e}")


def main():
    client, cluster = setup_coiled_cluster()
    logging.info(f"Coiled cluster initialized: {cluster.name}")

    try:
        output_raster_path = "/tmp/test_raster.tif"
        create_test_raster(output_raster_path)

        # Define your S3 bucket and key
        s3_bucket = 'gfw2-data'
        s3_key = 'climate/AFOLU_flux_model/organic_soils/test/test_raster.tif'

        upload_to_s3(output_raster_path, s3_bucket, s3_key)

    finally:
        client.close()
        cluster.close()
        logging.info("Coiled cluster closed")


if __name__ == "__main__":
    main()

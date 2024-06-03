"""
This script processes Engert et al. (2024) data for road density by clipping and reprojecting
the raw density data to match the bounds and resolution of the given tiles. The processed data
is then saved locally and uploaded to an S3 bucket.

Steps:
1. Download the specified tiles from the S3 bucket if they do not already exist locally.
2. Clip and reproject the Engert road density raster to match each tile.
3. Save the processed raster locally.
4. Upload the processed rasters to the S3 bucket.
5. Delete local files after processing to manage disk space.

TODO:
- Harmonize with utilities and variables.
- Harmonize with other scripts.

Usage:
The script currently processes a hardcoded list of tiles. To process additional tiles,
add the tile IDs to the `tiles_list` variable.

Functions:
- delete_local_file: Attempts to delete a local file with retries on failure.
- handle_file_operations: Manages the download, processing, and cleanup for a single tile.
"""

import os
import rioxarray
import boto3
import logging
import time
import subprocess
from pp_get_tile_list import get_tile_ids_from_raster

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density'

# Local paths
tiles_dir = r"C:\GIS\Temp\GFW_Global_Peatlands"
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\engert"
engert_density_raw = r"C:\GIS\Data\Global\Wetlands\Raw\Tropics\Engert_roads\engert_asiapac_ghrdens_1k_project.tif"
tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
tiles_list = get_tile_ids_from_raster(engert_density_raw, tile_index_path)


def delete_local_file(file_path):
    """
    Attempts to delete a local file with retries on failure.

    Args:
        file_path (str): Path to the file to delete.
    """
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            logging.info(f"Deleted local file {file_path}")
            break
        except PermissionError as e:
            if attempt < max_attempts - 1:
                logging.warning(f"Failed to delete {file_path}, retrying... ({e})")
                time.sleep(1)  # Wait a bit before retrying
            else:
                logging.error(f"Could not delete {file_path} after several attempts. ({e})")


def handle_file_operations(tile_name):
    """
    Manages the download, processing, and cleanup for a single tile.

    Args:
        tile_name (str): The tile ID to process.
    """
    local_tile_path = os.path.join(tiles_dir, f"{tile_name}.tif")
    local_output_path = os.path.join(output_dir, f"engert_{tile_name}.tif")
    s3_tile_path = f"{s3_tiles_prefix}{tile_name}_peat_mask_processed.tif"
    s3_output_path = f"{s3_output_prefix}/engert_{tile_name}.tif"

    # Check if the output file already exists on S3
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=s3_output_path)
        logging.info(f"Output file {s3_output_path} already exists on S3, skipping processing.")
        return
    except s3_client.exceptions.ClientError:
        logging.info(f"Output file {s3_output_path} does not exist on S3, proceeding with processing.")

    if os.path.exists(local_output_path):
        logging.info(f"Output file {local_output_path} already exists locally, skipping processing.")
        return

    if not os.path.exists(local_tile_path):
        try:
            s3_client.download_file(s3_bucket_name, s3_tile_path, local_tile_path)
            logging.info(f"Downloaded {s3_tile_path} to {local_tile_path}")
        except Exception as e:
            logging.error(f"Failed to download {s3_tile_path}. Error: {e}")
            return
    else:
        logging.info(f"Tile {local_tile_path} already exists locally, skipping download.")

    try:
        tile_ds = rioxarray.open_rasterio(local_tile_path, masked=True)
        engert_ds = rioxarray.open_rasterio(engert_density_raw, masked=True)
        clipped_engert = engert_ds.rio.clip_box(*tile_ds.rio.bounds())
        clipped_engert = clipped_engert.rio.reproject_match(tile_ds)
        clipped_engert.rio.to_raster(local_output_path)
        logging.info(f"Processed and saved locally {local_output_path}")

    #     # Upload to S3
    #     try:
    #         s3_client.upload_file(local_output_path, s3_bucket_name, s3_output_path)
    #         logging.info(f"Uploaded {local_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
    #     except Exception as e:
    #         logging.error(f"Failed to upload {local_output_path} to S3. Error: {e}")
    finally:
        print("Complete")
    #     if os.path.exists(local_output_path):
    #         delete_local_file(local_output_path)


# Ensure directories exist
os.makedirs(tiles_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Process each tile in the list
for tile in tiles_list:
    handle_file_operations(tile)

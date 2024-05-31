"""
This script processes Dadap et al. (2021) data for canal density by clipping and reprojecting
the raw density data to match the bounds and resolution of the given tiles. The processed data
is then saved locally and uploaded to an S3 bucket.

Steps:
1. Download the specified tiles from the S3 bucket if they do not already exist locally.
2. Clip and reproject the Dadap canal density raster to match each tile.
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
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/dev'

# Local paths
tiles_dir = r"C:\GIS\Temp\GFW_Global_Peatlands"
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\dadap\dev"
dadap_density_raw = r"C:\GIS\Data\Global\Wetlands\Raw\Tropics\canal_length_data\canal_length_data\canal_length_1km.tif"
tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
# tiles_list = ["00N_090E", "00N_100E", "00N_110E", "10N_090E", "10N_100E", "10N_110E"] #hardcoded for comparison
tiles_list = get_tile_ids_from_raster(dadap_density_raw,tile_index_path)


def handle_file_operations(tile_name):
    """
    Manages the download, processing, and cleanup for a single tile.

    Args:
        tile_name (str): The tile ID to process.
    """
    local_tile_path = os.path.join(tiles_dir, f"{tile_name}.tif")
    local_output_path = os.path.join(output_dir, f"dadap_{tile_name}.tif")
    s3_tile_path = f"{s3_tiles_prefix}{tile_name}_peat_mask_processed.tif"
    s3_output_path = f"{s3_output_prefix}dadap_{tile_name}.tif"

    if os.path.exists(local_output_path):
        logging.info(f"Output file {local_output_path} already exists, skipping processing.")
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
        dadap_ds = rioxarray.open_rasterio(dadap_density_raw, masked=True)
        clipped_dadap = dadap_ds.rio.clip_box(*tile_ds.rio.bounds())
        clipped_dadap = clipped_dadap.rio.reproject_match(tile_ds)
        clipped_dadap.rio.to_raster(local_output_path)
        logging.info(f"Processed and saved locally {local_output_path}")

        # Upload to S3
        try:
            s3_client.upload_file(local_output_path, s3_bucket_name, s3_output_path)
            logging.info(f"Uploaded {local_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
        except Exception as e:
            logging.error(f"Failed to upload {local_output_path} to S3. Error: {e}")
    finally:
        if os.path.exists(local_output_path):
            delete_local_file(local_output_path)

os.makedirs(tiles_dir, exist_ok=True)

# Ensure directories exist
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
os.makedirs(output_dir, exist_ok=True)

# Process each tile in the list
for tile in tiles_list:
    handle_file_operations(tile)
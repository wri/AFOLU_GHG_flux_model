import os
import rioxarray
import boto3
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_roads/'

# Local paths
tiles_dir = r"C:\GIS\Temp\GFW_Global_Peatlands"
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\engert"
engert_roads_raw = r"C:\GIS\Data\Global\Wetlands\Raw\Tropics\Engert_roads\engert_asiapac_ghrdens_1k.tif"

# List of tile IDs from the image
tiles_list = [
    "00N_090E", "00N_100E", "00N_110E", "00N_120E", "00N_130E", "00N_140E", "00N_150E",
    "10N_090E", "10N_100E", "10N_110E", "10N_120E", "10N_130E", "10N_140E", "10S_150E"
]


def handle_file_operations(tile_name):
    """
    Manages the download, processing, and cleanup for a single tile.

    Args:
        tile_name (str): The tile ID to process.
    """
    local_tile_path = os.path.join(tiles_dir, f"{tile_name}.tif")
    local_output_path = os.path.join(output_dir, f"engert_{tile_name}.tif")
    s3_tile_path = f"{s3_tiles_prefix}{tile_name}_peat_mask_processed.tif"
    s3_output_path = f"{s3_output_prefix}engert_{tile_name}.tif"

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
        engert_ds = rioxarray.open_rasterio(engert_roads_raw, masked=True)

        # Ensure valid bounds for clipping
        bounds = tile_ds.rio.bounds()
        if bounds[0] == bounds[2] or bounds[1] == bounds[3]:
            logging.error(f"Invalid bounds for tile {tile_name}. Skipping.")
            return

        clipped_engert = engert_ds.rio.clip_box(*bounds)
        clipped_engert = clipped_engert.rio.reproject_match(tile_ds)
        clipped_engert.rio.to_raster(local_output_path)
        logging.info(f"Processed and saved locally {local_output_path}")
    except Exception as e:
        logging.error(f"Failed to process tile {tile_name}. Error: {e}")


# Ensure directories exist
os.makedirs(tiles_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Process each tile in the list
for tile in tiles_list:
    handle_file_operations(tile)

# After processing all tiles, upload the entire directory to S3
local_directory = output_dir
s3_output_location = f"s3://{s3_bucket_name}/{s3_output_prefix}"
upload_command = ["aws", "s3", "cp", local_directory, s3_output_location, "--recursive"]

try:
    subprocess.run(upload_command, check=True)
    logging.info(f"Uploaded all processed tiles from {local_directory} to {s3_output_location}")
except FileNotFoundError:
    logging.error("AWS CLI not found. Ensure AWS CLI is installed and in the system PATH.")

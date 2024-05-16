import os
import rioxarray
import boto3
import logging
import time
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/'

# Local paths
tiles_dir = r"C:\GIS\Temp\GFW_Global_Peatlands"
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\dadap"
dadap_density_raw = r"C:\GIS\Data\Global\Wetlands\Raw\Tropics\canal_length_data\canal_length_data\canal_length_1km.tif"
tiles_list = ["00N_090E", "00N_100E", "00N_110E", "10N_090E", "10N_100E", "10N_110E"]

def delete_local_file(file_path):
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
    finally:
        if os.path.exists(local_tile_path):
            try:
                os.remove(local_tile_path)
                logging.info(f"Deleted local file {local_tile_path}")
            except PermissionError as e:
                logging.error(f"Could not delete {local_tile_path}. Error: {e}")

os.makedirs(tiles_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for tile in tiles_list:
    handle_file_operations(tile)

# After processing all tiles, upload the entire directory to S3
local_directory = output_dir
s3_output_location = f"s3://{s3_bucket_name}/{s3_output_prefix}"
upload_command = ["aws", "s3", "cp", local_directory, s3_output_location, "--recursive"]
subprocess.run(upload_command, check=True)
logging.info(f"Uploaded all processed tiles from {local_directory} to {s3_output_location}")

import os
import logging
import gc
import boto3
import subprocess
from pp_utilities import compress_file, get_existing_s3_files, hansenize

"""
This script processes raster tiles by resampling them to a specified resolution,
clipping them to tile bounds, and uploading the processed tiles to S3.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'

# Local paths
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
raw_rasters = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/engert_roads/engert_asiapac_ghrdens_1km_resample_30m.tif',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/canals/Dadap_SEA_Drainage/canal_length_data/canal_length_1km_resample_30m.tif'
}
output_prefixes = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m'
}

def process_tile(tile_key, dataset, run_mode='default'):
    """
    Processes a single tile: resamples, reprojects, clips, and uploads to S3.

    Parameters:
    tile_key (str): S3 key of the tile to process.
    dataset (str): The dataset type (e.g., 'engert' or 'dadap').
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    output_dir = local_temp_dir if run_mode == 'test' else '/tmp/output'
    os.makedirs(output_dir, exist_ok=True)

    s3_output_dir = output_prefixes[dataset]
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}.tif")
    compressed_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}_compressed.tif")
    s3_output_path = f"{s3_output_dir}/{dataset}_{tile_id}_compressed.tif".replace("\\", "/")

    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except:
            logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")

    logging.info(f"Starting processing of the tile {tile_id}")
    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'
    raw_raster_path = f'/vsis3/{s3_bucket_name}/{raw_rasters[dataset]}'
    logging.info(f"Reference raster path: {s3_input_path}")

    try:
        hansenize(raw_raster_path, output_dir, s3_input_path, s3_bucket_name, s3_output_dir, dataset, run_mode)
        logging.info(f"Running gdalwarp command")

        if run_mode == 'default':
            os.remove(local_output_path)
            os.remove(compressed_output_path)

        logging.info("Processing completed")
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(dataset, run_mode='default'):
    """
    Processes all tiles for the given dataset.

    Parameters:
    dataset (str): The dataset type (e.g., 'engert' or 'dadap').
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_tiles_prefix)
    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith('_peat_mask_processed.tif'):
                    tile_keys.append(tile_key)

    for tile_key in tile_keys:
        process_tile(tile_key, dataset, run_mode)

def main(tile_id=None, dataset='engert', run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
    tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
    dataset (str, optional): The dataset type (e.g., 'engert' or 'dadap'). Defaults to 'engert'.
    run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
    None
    """
    try:
        if tile_id:
            tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
            process_tile(tile_key, dataset, run_mode)
        else:
            process_all_tiles(dataset, run_mode)

    finally:
        logging.info("Processing completed")

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    main(tile_id='00N_110E', dataset='engert', run_mode='default')
    main(tile_id='00N_110E', dataset='dadap', run_mode='default')

    # Process datasets separately
    # main(dataset='engert', run_mode='default')
    # main(dataset='dadap', run_mode='default')

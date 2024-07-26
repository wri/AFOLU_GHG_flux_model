import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import boto3
import logging
import os
import fiona
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import gc
import subprocess
import rioxarray
import warnings
from utilities import (
    s3_file_exists, list_s3_files, compress_file, get_existing_s3_files,
    compress_and_upload_directory_to_s3, get_raster_bounds, resample_raster,
    create_fishnet_from_raster, reproject_gdf, read_tiled_features,
    assign_segments_to_cells, convert_length_to_density, fishnet_to_raster,
    resample_to_30m
)

"""
This script processes OSM and GRIP data specifically for roads and canals using tiled shapefiles.
It performs the following steps:
1. Reads raster tiles from S3.
2. Resamples the raster to a target resolution.
3. Creates a fishnet grid.
4. Reads corresponding roads or canals shapefiles for each tile.
5. Assigns road/canal lengths to the fishnet cells.
6. Converts the lengths to density.
7. Saves the results as raster files locally and uploads them to S3.

The script uses Dask to parallelize the processing of multiple tiles.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from', UserWarning)

# AWS S3 setup
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'

# Input directories
feature_directories = {
    'osm_roads': 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/osm_roads/roads_by_tile/',
    'osm_canals': 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/osm_roads/canals_by_tile/',
    'grip_roads': 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/roads_by_tile/'
}

# Output directories
output_directories = {
    'osm_roads': {
        'local': 'C:/GIS/Data/Global/OSM/osm_roads_density/',
        's3': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/'
    },
    'osm_canals': {
        'local': 'C:/GIS/Data/Global/OSM/osm_canals_density/',
        's3': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/'
    },
    'grip_roads': {
        'local': r'C:\GIS\Data\Global\Wetlands\Processed\grip_density',
        's3': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/'
    }
}

local_temp_dir = r"C:\GIS\Data\Global\Wetlands\Processed\30_m_temp"

# Ensure local output directories exist
for key, paths in output_directories.items():
    os.makedirs(paths['local'], exist_ok=True)
os.makedirs(local_temp_dir, exist_ok=True)

logging.info("Directories and paths set up")

def process_tile(tile_key, feature_type, run_mode='default'):
    """
    Processes a single tile.

    Parameters:
    tile_key (str): Key of the tile in the S3 bucket.
    feature_type (str): Type of feature ('osm_roads', 'osm_canals', or 'grip_roads').
    run_mode (str): Mode to run the script ('default' or 'test').

    Returns:
    None
    """
    output_dir = output_directories[feature_type]['local']
    s3_output_dir = output_directories[feature_type]['s3']
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{feature_type}_density_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}{feature_type}_density_{tile_id}.tif"

    # Check if the file already exists on S3
    s3_client = boto3.client('s3')
    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except:
            logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'

    try:
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_path) as src:
                target_resolution = 1000

                # Step 1: Resample raster to target resolution
                resampled_data, resampled_profile = resample_raster(src, target_resolution)

                # Step 2: Mask the raster data
                masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)

                # Step 3: Create fishnet from raster data
                fishnet_gdf = create_fishnet_from_raster(masked_data, resampled_profile['transform'])
                fishnet_gdf = reproject_gdf(fishnet_gdf, 3395)  # Reproject to EPSG:3395

                # Step 4: Read and reproject tiled features
                features_gdf = read_tiled_features(tile_id, feature_type, feature_directories)

                # Step 5: Assign segments to fishnet cells and calculate lengths
                fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, features_gdf)

                # Step 6: Convert lengths to density
                fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

                # Step 7: Convert fishnet to raster and save
                fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

                # Save intermediate products if in test mode
                if run_mode == 'test':
                    intermediate_dir = os.path.join(output_dir, 'intermediate')
                    os.makedirs(intermediate_dir, exist_ok=True)

                    resampled_path = os.path.join(intermediate_dir, f'resampled_{tile_id}.tif')
                    masked_path = os.path.join(intermediate_dir, f'masked_{tile_id}.tif')
                    fishnet_path = os.path.join(intermediate_dir, f'fishnet_{tile_id}.shp')
                    lengths_path = os.path.join(intermediate_dir, f'lengths_{tile_id}.shp')
                    density_path = os.path.join(intermediate_dir, f'density_{tile_id}.shp')

                    # Save resampled raster
                    with rasterio.open(resampled_path, 'w', **resampled_profile) as dst:
                        dst.write(resampled_data[0], 1)

                    # Save masked raster
                    with rasterio.open(masked_path, 'w', **masked_profile) as dst:
                        dst.write(masked_data, 1)

                    # Save fishnet GeoDataFrame
                    fishnet_gdf.to_file(fishnet_path)

                    # Save fishnet with lengths
                    fishnet_with_lengths.to_file(lengths_path)

                    # Save fishnet with density
                    fishnet_with_density.to_file(density_path)

                logging.info(f"Saved {local_output_path}")

                # Step 8: Resample to 30 meters
                reference_path = f'/vsis3/{s3_bucket_name}/{tile_key}'
                local_30m_output_path = os.path.join(local_temp_dir, os.path.basename(local_output_path))
                resample_to_30m(local_output_path, local_30m_output_path, reference_path)

                if run_mode == 'test':
                    logging.info(f"Test mode: Outputs saved locally at {local_output_path} and {local_30m_output_path}")
                else:
                    logging.info(f"Uploading {local_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
                    s3_client.upload_file(local_output_path, s3_bucket_name, s3_output_path)

                    logging.info(f"Uploading {local_30m_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
                    s3_client.upload_file(local_30m_output_path, s3_bucket_name, s3_output_path)

                    logging.info(
                        f"Uploaded {local_output_path} and {local_30m_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
                    os.remove(local_output_path)
                    os.remove(local_30m_output_path)

                del resampled_data, masked_data, fishnet_gdf, features_gdf, fishnet_with_lengths, fishnet_with_density
                gc.collect()
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(feature_type, run_mode='default'):
    """
    Processes all tiles using Dask for parallelization.

    Parameters:
    feature_type (str): Type of feature ('osm_roads', 'osm_canals', or 'grip_roads').
    run_mode (str): Mode to run the script ('default' or 'test').

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

    dask_tiles = [dask.delayed(process_tile)(tile_key, feature_type, run_mode) for tile_key in tile_keys]
    with ProgressBar():
        dask.compute(*dask_tiles)

def main(tile_id=None, feature_type='osm_roads', run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
    tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
    feature_type (str, optional): Type of feature ('osm_roads', 'osm_canals', or 'grip_roads'). Defaults to 'osm_roads'.
    run_mode (str, optional): Mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
    None
    """
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        if tile_id:
            tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
            process_tile(tile_key, feature_type, run_mode)
        else:
            process_all_tiles(feature_type, run_mode)

        if run_mode != 'test':
            compress_and_upload_directory_to_s3(output_directories[feature_type]['local'], s3_bucket_name,
                                                output_directories[feature_type]['s3'])
    finally:
        client.close()

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    main(tile_id='00N_110E', feature_type='osm_roads', run_mode='test')
    main(tile_id='00N_110E', feature_type='grip_roads', run_mode='test')

    # Process roads and canals separately
    # main(feature_type='osm_roads', run_mode='default')
    # main(feature_type='osm_canals', run_mode='default')
    # main(feature_type='grip_roads', run_mode='default')

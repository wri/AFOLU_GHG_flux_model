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
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import rioxarray
import warnings

import pp_utilities as uu
import constants_and_names as cn

"""
This script processes OSM and GRIP data specifically for roads and canals using tiled shapefiles.
It performs the following steps:
1. Reads raster tiles from S3.
2. Resamples the raster to a target resolution.
3. Masks the raster based on specific values.
4. Creates a fishnet grid based on the raster data.
5. Reads corresponding roads or canals shapefiles for each tile.
6. Assigns road/canal lengths to the fishnet cells.
7. Converts the lengths to density (km/km^2).
8. Saves the results as raster files locally and uploads them to S3.

The script can be run either using a local Dask client or a Coiled cluster.

Functions:
- get_raster_bounds: Reads and returns the bounds of a raster file.
- resample_raster: Resamples a raster to a target resolution.
- mask_raster: Masks raster data to highlight specific values.
- create_fishnet_from_raster: Creates a fishnet grid from raster data.
- reproject_gdf: Reprojects a GeoDataFrame to a specified EPSG code.
- read_tiled_features: Reads and reprojects shapefiles (roads or canals) for a given tile.
- assign_segments_to_cells: Assigns features to fishnet cells and calculates lengths.
- convert_length_to_density: Converts lengths of features to density (km/km^2).
- fishnet_to_raster: Converts a fishnet GeoDataFrame to a raster and saves it.
- process_tile: Processes a single tile.
- process_all_tiles: Processes all tiles using Dask for parallelization.
- main: Main function to execute the processing based on provided arguments.

Command-Line Usage Examples:
1. To process a specific tile for OSM roads:
    python pp_roads_canals.py --tile_id 00N_110E --feature_type osm_roads --run_mode default --client local

2. To process all tiles for GRIP roads using Coiled:
    python pp_roads_canals.py --feature_type grip_roads --run_mode default --client coiled

3. To run the script in test mode for OSM canals:
    python pp_roads_canals.py --tile_id 00N_110E --feature_type osm_canals --run_mode test --client local
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from', UserWarning)

# Ensure local output directories exist
for dataset_key, dataset_info in cn.datasets.items():
    if dataset_key in ['osm', 'grip']:
        for sub_key, sub_dataset in dataset_info.items():
            os.makedirs(sub_dataset['local_processed'], exist_ok=True)
os.makedirs(cn.local_temp_dir, exist_ok=True)
logging.info("Directories and paths set up")

def get_raster_bounds(raster_path):
    """
    Reads the bounds of a raster file.

    Parameters:
    raster_path (str): Path to the raster file.

    Returns:
    bounds (tuple): The bounds of the raster (left, bottom, right, top).
    """
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds

def resample_raster(src, target_resolution_m):
    """
    Resamples a raster to a target resolution in meters.

    Parameters:
    src (rasterio.io.DatasetReader): The source raster dataset.
    target_resolution_m (float): Target resolution in meters.

    Returns:
    data (np.ndarray): Resampled raster data.
    profile (dict): Updated raster profile.
    """
    logging.info(f"Resampling raster to {target_resolution_m} meter resolution (1 km by 1 km)")
    target_resolution_deg = target_resolution_m / 111320

    width = int((src.bounds.right - src.bounds.left) / target_resolution_deg)
    height = int((src.bounds.top - src.bounds.bottom) / target_resolution_deg)

    new_transform = rasterio.transform.from_bounds(
        src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, width, height)

    profile = src.profile
    profile.update(transform=new_transform, width=width, height=height)

    data = src.read(
        out_shape=(src.count, height, width),
        resampling=Resampling.nearest
    )

    # Skip if the resampled data is all nodata
    if np.all(data == profile['nodata']):
        logging.info("Resampled data contains only nodata values. Skipping further processing.")
        return None, None

    return data, profile

def mask_raster(data, profile):
    """
    Masks raster data to retain only values equal to 1.

    Parameters:
    data (np.ndarray): The raster data to be masked.
    profile (dict): The raster profile.

    Returns:
    masked_data (np.ndarray): Masked raster data.
    updated_profile (dict): Updated raster profile.
    """
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    if not mask.any():
        logging.info("Masked data contains no valid data. Skipping further processing.")
        return None, None

    profile.update(dtype=rasterio.uint8)
    return mask.astype(rasterio.uint8), profile

def create_fishnet_from_raster(data, transform):
    """
    Creates a fishnet grid based on the raster data.

    Parameters:
    data (np.ndarray): The raster data.
    transform (Affine): The affine transformation of the raster.

    Returns:
    fishnet_gdf (GeoDataFrame): A GeoDataFrame representing the fishnet grid.
    """
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
                x, y = transform * (col, row)
                polygons.append(box(x, y, x + transform[0], y + transform[4]))

    if not polygons:
        logging.info("No polygons created for fishnet. Skipping further processing.")
        return gpd.GeoDataFrame(columns=['geometry'])

    fishnet_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def reproject_gdf(gdf, epsg):
    """
    Reprojects a GeoDataFrame to a specified EPSG code.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to reproject.
    epsg (int): The target EPSG code.

    Returns:
    reprojected_gdf (GeoDataFrame): The reprojected GeoDataFrame.
    """
    if gdf.empty:
        logging.info("GeoDataFrame is empty. Skipping reprojection.")
        return gdf

    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)

def read_tiled_features(tile_id, feature_type):
    """
    Reads and reprojects shapefiles (roads or canals) for a given tile.

    Parameters:
    tile_id (str): Tile ID.
    feature_type (str): Type of feature ('osm_roads', 'osm_canals', or 'grip_roads').

    Returns:
    features_gdf (GeoDataFrame): Reprojected features GeoDataFrame.
    """
    try:
        feature_key = feature_type.split('_')
        feature_tile_dir = cn.datasets[feature_key[0]][feature_key[1]]['s3_raw']

        tile_id = '_'.join(tile_id.split('_')[:2])
        s3_file_path = os.path.join(feature_tile_dir, f"{feature_key[1]}_{tile_id}.shp")
        full_s3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_file_path}"

        logging.info(f"Constructed S3 file path: {full_s3_path}")

        features_gdf = gpd.read_file(full_s3_path)
        if features_gdf.empty:
            logging.info(f"No data found in shapefile for tile {tile_id} at {full_s3_path}. Skipping.")
            return gpd.GeoDataFrame(columns=['geometry'])

        logging.info(f"Read {len(features_gdf)} {feature_type} features for tile {tile_id}")
        return reproject_gdf(features_gdf, 3395)

    except fiona.errors.DriverError as e:
        logging.error(f"Error reading {feature_type} shapefile: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])

    except Exception as e:
        logging.error(f"Unexpected error occurred while reading {feature_type} for tile {tile_id}: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])

def assign_segments_to_cells(fishnet_gdf, features_gdf):
    """
    Assigns features to fishnet cells and calculates lengths.

    Parameters:
    fishnet_gdf (GeoDataFrame): The fishnet GeoDataFrame.
    features_gdf (GeoDataFrame): The features GeoDataFrame.

    Returns:
    fishnet_gdf (GeoDataFrame): The updated fishnet GeoDataFrame with feature lengths.
    """
    if fishnet_gdf.empty or features_gdf.empty:
        logging.info("Fishnet or features GeoDataFrame is empty. Skipping assignment of segments.")
        return fishnet_gdf

    logging.info("Assigning features segments to fishnet cells and calculating lengths")
    feature_lengths = []

    for idx, cell in fishnet_gdf.iterrows():
        features_in_cell = gpd.clip(features_gdf, cell.geometry)
        total_length = features_in_cell.geometry.length.sum()
        feature_lengths.append(total_length)

    fishnet_gdf['length'] = feature_lengths
    logging.info(f"Fishnet with feature lengths: {fishnet_gdf.head()}")
    return fishnet_gdf

def resample_to_30m(input_path, output_path, reference_path):
    input_raster = rioxarray.open_rasterio(input_path, masked=True)
    reference_raster = rioxarray.open_rasterio(reference_path, masked=True)

    logging.info(f"Resampling {input_path} to match {reference_path}")
    clipped_resampled_raster = input_raster.rio.clip_box(*reference_raster.rio.bounds())
    clipped_resampled_raster = clipped_resampled_raster.rio.reproject_match(reference_raster)

    clipped_resampled_raster.rio.to_raster(output_path)

    if os.path.exists(output_path):
        logging.info(f"Successfully saved resampled raster to {output_path}")
    else:
        logging.error(f"Failed to save resampled raster to {output_path}")


def convert_length_to_density(fishnet_gdf, crs):
    """
    Converts lengths of features to density (km/km2).

    Parameters:
    fishnet_gdf (GeoDataFrame): The fishnet GeoDataFrame.
    crs (CRS): The coordinate reference system of the fishnet.

    Returns:
    fishnet_gdf (GeoDataFrame): The updated fishnet GeoDataFrame with density values.
    """
    if fishnet_gdf.empty:
        logging.info("Fishnet GeoDataFrame is empty. Skipping conversion of length to density.")
        return fishnet_gdf

    logging.info("Converting length to density (km/km2)")
    if crs.axis_info[0].unit_name == 'metre':
        fishnet_gdf['length_km'] = fishnet_gdf['length'] / 1000
        fishnet_gdf['density'] = fishnet_gdf['length_km']
    else:
        raise ValueError("Unsupported CRS units")

    logging.info(f"Density values: {fishnet_gdf[['length', 'density']]}")
    return fishnet_gdf

def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    """
    Converts a fishnet GeoDataFrame to a raster and saves it.

    Parameters:
    fishnet_gdf (GeoDataFrame): The fishnet GeoDataFrame with density values.
    profile (dict): The raster profile.
    output_raster_path (str): The path where the output raster will be saved.
    """
    if fishnet_gdf.empty:
        logging.info("Fishnet GeoDataFrame is empty. Skipping rasterization.")
        return

    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    transform = profile['transform']
    out_shape = (profile['height'], profile['width'])
    fishnet_gdf = fishnet_gdf.to_crs(profile['crs'])

    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry, fishnet_gdf['density'])],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.float32
    )

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(rasterized, 1)

    logging.info("Fishnet converted to raster and saved")

def process_tile(tile_key, feature_type, run_mode='default'):
    """
    Processes a single tile by resampling, masking, creating a fishnet, and calculating density.

    Parameters:
    tile_key (str): The key for the tile in the S3 bucket.
    feature_type (str): The type of feature to process (e.g., 'osm_roads').
    run_mode (str): The mode to run the processing in ('default' or 'test').
    """
    output_dir = cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['local_processed']
    s3_output_dir = cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['s3_processed']
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{feature_type}_density_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}{feature_type}_density_{tile_id}.tif"

    s3_client = boto3.client('s3')
    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=cn.s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except:
            logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{cn.s3_bucket_name}/{tile_key}'
    logging.info(f"Constructed S3 input path: {s3_input_path}")

    try:
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_path) as src:
                target_resolution = 1000

                resampled_data, resampled_profile = resample_raster(src, target_resolution)
                if resampled_data is None:
                    return

                masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)
                if masked_data is None:
                    return

                fishnet_gdf = create_fishnet_from_raster(masked_data, resampled_profile['transform'])
                fishnet_gdf = reproject_gdf(fishnet_gdf, 3395)

                features_gdf = read_tiled_features(tile_id, feature_type)

                fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, features_gdf)

                fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

                fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

                if run_mode == 'test':
                    intermediate_dir = os.path.join(output_dir, 'intermediate')
                    os.makedirs(intermediate_dir, exist_ok=True)

                    resampled_path = os.path.join(intermediate_dir, f'resampled_{tile_id}.tif')
                    masked_path = os.path.join(intermediate_dir, f'masked_{tile_id}.tif')
                    fishnet_path = os.path.join(intermediate_dir, f'fishnet_{tile_id}.shp')
                    lengths_path = os.path.join(intermediate_dir, f'lengths_{tile_id}.shp')
                    density_path = os.path.join(intermediate_dir, f'density_{tile_id}.shp')

                    with rasterio.open(resampled_path, 'w', **resampled_profile) as dst:
                        dst.write(resampled_data[0], 1)

                    with rasterio.open(masked_path, 'w', **masked_profile) as dst:
                        dst.write(masked_data, 1)

                    fishnet_gdf.to_file(fishnet_path)

                    fishnet_with_lengths.to_file(lengths_path)

                    fishnet_with_density.to_file(density_path)

                logging.info(f"Saved {local_output_path}")

                reference_path = f'/vsis3/{cn.s3_bucket_name}/{tile_key}'
                local_30m_output_path = os.path.join(cn.local_temp_dir, os.path.basename(local_output_path))
                resample_to_30m(local_output_path, local_30m_output_path, reference_path)

                if run_mode == 'test':
                    logging.info(f"Test mode: Outputs saved locally at {local_output_path} and {local_30m_output_path}")
                else:
                    upload_final_output_to_s3(local_output_path, s3_output_path)
                    upload_final_output_to_s3(local_30m_output_path, s3_output_path.replace('.tif', '_30m.tif'))

                del resampled_data, masked_data, fishnet_gdf, features_gdf, fishnet_with_lengths, fishnet_with_density
                gc.collect()
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(feature_type, run_mode='default'):
    """
    Processes all tiles in the S3 bucket.

    Parameters:
    feature_type (str): The type of feature to process (e.g., 'osm_roads').
    run_mode (str): The mode to run the processing in ('default' or 'test').
    """
    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=cn.peat_tiles_prefix)
    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith(cn.peat_pattern):
                    tile_keys.append(tile_key)

    dask_tiles = [dask.delayed(process_tile)(tile_key, feature_type, run_mode) for tile_key in tile_keys]
    with ProgressBar():
        dask.compute(*dask_tiles)

def main(tile_id=None, feature_type='osm_roads', run_mode='default', client_type='local'):
    """
    Main function to execute the processing based on provided arguments.

    Parameters:
    tile_id (str): The tile ID to process.
    feature_type (str): The type of feature to process (e.g., 'osm_roads').
    run_mode (str): The mode to run the processing in ('default' or 'test').
    client_type (str): The Dask client type to use ('local' or 'coiled').
    """
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
    else:
        cluster = LocalCluster()
        client = Client(cluster)

    try:
        if tile_id:
            tile_key = f"{cn.peat_tiles_prefix}{tile_id}{cn.peat_pattern}"
            process_tile(tile_key, feature_type, run_mode)
        else:
            process_all_tiles(feature_type, run_mode)
    finally:
        client.close()
        if client_type == 'coiled':
            cluster.close()

if __name__ == "__main__":
    import argparse
    import sys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process OSM and GRIP data for roads and canals.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--feature_type', type=str, choices=['osm_roads', 'osm_canals', 'grip_roads'], default='osm_roads', help='Type of feature to process')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default', help='Run mode (default or test)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local', help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    if not any(sys.argv[1:]):  # Check if there are no command-line arguments
        # Direct execution examples for PyCharm
        tile_id = '00N_110E'  # Replace with your desired tile ID
        feature_type = 'osm_canals'  # Replace with your desired feature type
        run_mode = 'default'  # Replace with your desired run mode
        client_type = 'local'  # Replace with 'coiled' if you want to use the Coiled cluster

        main(tile_id=tile_id, feature_type=feature_type, run_mode=run_mode, client_type=client_type)
    else:
        main(tile_id=args.tile_id, feature_type=args.feature_type, run_mode=args.run_mode, client_type=args.client)

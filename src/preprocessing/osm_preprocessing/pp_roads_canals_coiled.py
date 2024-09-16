import os
import geopandas as gpd
import logging
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd
import argparse
import sys
import boto3
import gc
import subprocess
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from shapely.geometry import box
import rioxarray
import warnings
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

import pp_utilities as uu
import constants_and_names as cn

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
- resample_to_30m: Resamples a raster to 30 meters resolution.
- compress_file: Compresses a file using GDAL.
- get_existing_s3_files: Gets a list of existing files in an S3 bucket.
- upload_final_output_to_s3: Uploads the final output file to S3 and deletes the local file.
- process_tile: Processes a single tile.
- process_all_tiles: Processes all tiles using Dask for parallelization.
- main: Main function to execute the processing based on provided arguments.

Usage examples:
- Process a specific tile (00N_110E):
  python script.py --tile_id 00N_110E --feature_type osm_roads --client local --run_mode default

- Process all tiles:
  python script.py --feature_type osm_roads --client local --run_mode default

Dependencies:
- geopandas
- pandas
- shapely
- numpy
- rasterio
- rioxarray
- boto3
- fiona
- dask
- dask.distributed
- dask.diagnostics
- subprocess
- botocore
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
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds

def resample_raster(src, target_resolution_m):
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

    return data, profile

def mask_raster(data, profile):
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    profile.update(dtype=rasterio.uint8)
    return mask.astype(rasterio.uint8), profile

def create_fishnet_from_raster(data, transform):
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
                x, y = transform * (col, row)
                polygons.append(box(x, y, x + transform[0], y + transform[4]))

    fishnet_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def reproject_gdf(gdf, epsg):
    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)

def read_tiled_features(tile_id, feature_type):
    """
    Reads and reprojects shapefiles (roads or canals) for a given tile.

    Parameters:
    tile_id (str): Tile ID.
    feature_type (str): Type of feature ('osm_roads', 'osm_canals', or 'grip_roads').

    Returns:
    GeoDataFrame: Reprojected features GeoDataFrame.
    """
    try:
        # Extract relevant directories
        feature_key = feature_type.split('_')
        feature_tile_dir = cn.datasets[feature_key[0]][feature_key[1]]['s3_raw']

        # Construct the S3 path for the shapefile
        tile_id = '_'.join(tile_id.split('_')[:2])
        s3_file_path = os.path.join(feature_tile_dir, f"{feature_key[1]}_{tile_id}.shp")
        full_s3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_file_path}"

        logging.info(f"Constructed S3 file path: {full_s3_path}")

        # Attempt to read the shapefile using GDAL virtual file system
        features_gdf = gpd.read_file(full_s3_path)
        if not features_gdf.empty:
            logging.info(f"Read {len(features_gdf)} {feature_type} features for tile {tile_id}")
            features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
            return features_gdf
        else:
            logging.warning(f"No data found in shapefile for tile {tile_id} at {full_s3_path}")
            return gpd.GeoDataFrame(columns=['geometry'])

    except fiona.errors.DriverError as e:
        logging.error(f"Error reading {feature_type} shapefile: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])

    except Exception as e:
        logging.error(f"Unexpected error occurred while reading {feature_type} for tile {tile_id}: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])

def assign_segments_to_cells(fishnet_gdf, features_gdf):
    logging.info("Assigning features segments to fishnet cells and calculating lengths")
    feature_lengths = []

    for idx, cell in fishnet_gdf.iterrows():
        features_in_cell = gpd.clip(features_gdf, cell.geometry)
        total_length = features_in_cell.geometry.length.sum()
        feature_lengths.append(total_length)

    fishnet_gdf['length'] = feature_lengths
    logging.info(f"Fishnet with feature lengths: {fishnet_gdf.head()}")
    return fishnet_gdf

def convert_length_to_density(fishnet_gdf, crs):
    logging.info("Converting length to density (km/km2)")
    if crs.axis_info[0].unit_name == 'metre':
        fishnet_gdf['length_km'] = fishnet_gdf['length'] / 1000
        fishnet_gdf['density'] = fishnet_gdf['length_km']
    else:
        raise ValueError("Unsupported CRS units")
    logging.info(f"Density values: {fishnet_gdf[['length', 'density']]}")
    return fishnet_gdf

def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    transform = profile['transform']
    out_shape = (profile['height'], profile['width'])
    fishnet_gdf = fishnet_gdf.to_crs(profile['crs'])

    if fishnet_gdf.empty:
        logging.info(f"No valid geometries found for {output_raster_path}. Skipping rasterization.")
        return

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

def compress_file(input_file, output_file):
    try:
        subprocess.run(
            ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', input_file, output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {input_file}: {e}")

def get_existing_s3_files(s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
    existing_files = set()

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                existing_files.add(obj['Key'])

    return existing_files

def upload_final_output_to_s3(local_output_path, s3_output_path):
    """
    Uploads the final output file to S3 and deletes the local file.

    Parameters:
    local_output_path (str): Path to the local output file.
    s3_output_path (str): Path to the S3 destination.

    Returns:
    None
    """
    s3_client = boto3.client('s3')
    try:
        logging.info(f"Uploading {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        s3_client.upload_file(local_output_path, cn.s3_bucket_name, s3_output_path)

        logging.info(f"Successfully uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        os.remove(local_output_path)
        logging.info(f"Deleted local file: {local_output_path}")

    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Credentials error: {e}")
    except Exception as e:
        logging.error(f"Failed to upload {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}: {e}")

def process_tile(tile_key, feature_type, run_mode='default'):
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

                masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)

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
    # Initialize Dask client based on the argument
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
    else:
        cluster = LocalCluster()
        client = Client(cluster)

    logging.info(f"Dask client initialized with {client_type} cluster")

    try:
        if tile_id:
            tile_key = f"{cn.peat_tiles_prefix}{tile_id}{cn.peat_pattern}"
            process_tile(tile_key, feature_type, run_mode)
        else:
            process_all_tiles(feature_type, run_mode)

    finally:
        client.close()
        logging.info("Dask client closed")
        if client_type == 'coiled':
            cluster.close()
            logging.info("Coiled cluster closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process OSM and GRIP data by tiles.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--feature_type', type=str, choices=['osm_roads', 'osm_canals', 'grip_roads'], default='osm_roads', help='Type of feature to process')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local', help='Dask client type to use (local or coiled)')
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default', help='Run mode for processing (default or test)')
    args = parser.parse_args()

    if not any(sys.argv[1:]):  # Check if there are no command-line arguments
        # Direct execution examples for PyCharm
        # Example usage for processing a specific tile with the local Dask client
        main(tile_id='00N_110E', feature_type='osm_roads', run_mode='default', client_type='local')

        # Example usage for processing all tiles with the local Dask client
        # main(feature_type='osm_roads', run_mode='default', client_type='local')

    else:
        main(tile_id=args.tile_id, feature_type=args.feature_type, run_mode=args.run_mode, client_type=args.client)
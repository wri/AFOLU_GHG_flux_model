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
- compress_and_upload_directory_to_s3: Compresses and uploads a directory to S3.
- process_tile: Processes a single tile.
- process_all_tiles: Processes all tiles using Dask for parallelization.
- main: Main function to execute the processing based on provided arguments.


TODO: get this working with coiled
Update compression 
Stop uploading both functions to s3
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from', UserWarning)

# Ensure local output directories exist
for key, paths in cn.output_directories.items():
    os.makedirs(paths['local'], exist_ok=True)
os.makedirs(cn.local_temp_dir, exist_ok=True)

logging.info("Directories and paths set up")


def get_raster_bounds(raster_path):
    """
    Reads and returns the bounds of a raster file.

    Parameters:
    raster_path (str): Path to the raster file.

    Returns:
    bounds (rasterio.coords.BoundingBox): Bounds of the raster.
    """
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds


def resample_raster(src, target_resolution_m):
    """
    Resamples a raster to a target resolution.

    Parameters:
    src (rasterio.io.DatasetReader): Source raster object.
    target_resolution_m (float): Target resolution in meters.

    Returns:
    data (numpy.ndarray): Resampled raster data.
    profile (dict): Updated raster profile.
    """
    logging.info(f"Resampling raster to {target_resolution_m} meter resolution (1 km by 1 km)")
    target_resolution_deg = target_resolution_m / 111320  # Approximate conversion factor

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
    """
    Masks raster data to highlight specific values.

    Parameters:
    data (numpy.ndarray): Raster data.
    profile (dict): Raster profile.

    Returns:
    masked_data (numpy.ndarray): Masked raster data.
    masked_profile (dict): Updated raster profile.
    """
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    profile.update(dtype=rasterio.uint8)
    return mask.astype(rasterio.uint8), profile


def create_fishnet_from_raster(data, transform):
    """
    Creates a fishnet grid from raster data.

    Parameters:
    data (numpy.ndarray): Raster data.
    transform (Affine): Affine transformation for the raster.

    Returns:
    fishnet_gdf (GeoDataFrame): Fishnet grid as a GeoDataFrame.
    """
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
    """
    Reprojects a GeoDataFrame to a specified EPSG code.

    Parameters:
    gdf (GeoDataFrame): GeoDataFrame to reproject.
    epsg (int): Target EPSG code.

    Returns:
    GeoDataFrame: Reprojected GeoDataFrame.
    """
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
    feature_tile_dir = cn.feature_directories[feature_type]
    logging.info(f"Reading tiled {feature_type} shapefile for tile ID: {tile_id}")
    try:
        tile_id = '_'.join(tile_id.split('_')[:2])
        file_path = os.path.join(feature_tile_dir, f"{feature_type.split('_')[1]}_{tile_id}.shp")
        if os.path.exists(file_path) or file_path.startswith('s3://'):
            features_gdf = gpd.read_file(file_path)
            logging.info(f"Read {len(features_gdf)} {feature_type} features for tile {tile_id}")
            features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
            return features_gdf
        else:
            logging.warning(f"No shapefile found for tile {tile_id}")
            return gpd.GeoDataFrame(columns=['geometry'])
    except fiona.errors.DriverError as e:
        logging.error(f"Error reading {feature_type} shapefile: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])


def assign_segments_to_cells(fishnet_gdf, features_gdf):
    """
    Assigns features to fishnet cells and calculates lengths.

    Parameters:
    fishnet_gdf (GeoDataFrame): Fishnet grid GeoDataFrame.
    features_gdf (GeoDataFrame): Features GeoDataFrame.

    Returns:
    GeoDataFrame: Fishnet grid with calculated feature lengths.
    """
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
    """
    Converts lengths of features to density (km/km2).

    Parameters:
    fishnet_gdf (GeoDataFrame): Fishnet grid with feature lengths.
    crs (pyproj.CRS): Coordinate reference system of the GeoDataFrame.

    Returns:
    GeoDataFrame: Fishnet grid with feature densities.
    """
    logging.info("Converting length to density (km/km2)")
    if crs.axis_info[0].unit_name == 'metre':
        fishnet_gdf['length_km'] = fishnet_gdf['length'] / 1000  # Convert lengths from meters to kilometers
        fishnet_gdf['density'] = fishnet_gdf['length_km']  # Cell area is 1 km² since each cell is 1 km x 1 km
    else:
        raise ValueError("Unsupported CRS units")
    logging.info(f"Density values: {fishnet_gdf[['length', 'density']]}")
    return fishnet_gdf


def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    """
    Converts a fishnet GeoDataFrame to a raster and saves it.

    Parameters:
    fishnet_gdf (GeoDataFrame): Fishnet grid with feature densities.
    profile (dict): Raster profile.
    output_raster_path (str): Path to save the output raster file.

    Returns:
    None
    """
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
    """
    Resamples a raster to match a reference raster's resolution and extent.

    Parameters:
    input_path (str): Path to the input raster file.
    output_path (str): Path to save the resampled raster file.
    reference_path (str): Path to the reference raster file.

    Returns:
    None
    """
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
    """
    Compresses a file using GDAL.

    Parameters:
    input_file (str): Path to the input file.
    output_file (str): Path to save the compressed file.

    Returns:
    None
    """
    try:
        subprocess.run(
            ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', input_file, output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {input_file}: {e}")


def get_existing_s3_files(s3_bucket, s3_prefix):
    """
    Gets a list of existing files in an S3 bucket.

    Parameters:
    s3_bucket (str): Name of the S3 bucket.
    s3_prefix (str): Prefix path in the S3 bucket.

    Returns:
    set: Set of existing files in the S3 bucket.
    """
    s3_client = boto3.client('s3')
    existing_files = set()

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                existing_files.add(obj['Key'])

    return existing_files


def compress_and_upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    """
    Compresses and uploads a directory to S3.

    Parameters:
    local_directory (str): Path to the local directory.
    s3_bucket (str): Name of the S3 bucket.
    s3_prefix (str): Prefix path in the S3 bucket.

    Returns:
    None
    """
    s3_client = boto3.client('s3')
    existing_files = get_existing_s3_files(s3_bucket, s3_prefix)

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            compressed_file_path = os.path.join(root, f"compressed_{file}")
            s3_file_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, s3_file_path).replace("\\", "/")

            if s3_key in existing_files:
                logging.info(f"File {s3_key} already exists in S3. Skipping upload.")
            else:
                try:
                    logging.info(f"Compressing {local_file_path}")
                    compress_file(local_file_path, compressed_file_path)

                    logging.info(f"Uploading {compressed_file_path} to s3://{s3_bucket}/{s3_key}")
                    s3_client.upload_file(compressed_file_path, s3_bucket, s3_key)

                    logging.info(f"Successfully uploaded {compressed_file_path} to s3://{s3_bucket}/{s3_key}")

                    os.remove(compressed_file_path)
                except (NoCredentialsError, PartialCredentialsError) as e:
                    logging.error(f"Credentials error: {e}")
                    return
                except Exception as e:
                    logging.error(f"Failed to upload {local_file_path} to s3://{s3_bucket}/{s3_key}: {e}")


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
    output_dir = cn.output_directories[feature_type]['local']
    s3_output_dir = cn.output_directories[feature_type]['s3']
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{feature_type}_density_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}{feature_type}_density_{tile_id}.tif"

    # Check if the file already exists on S3
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
                features_gdf = read_tiled_features(tile_id, feature_type)

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
                reference_path = f'/vsis3/{cn.s3_bucket_name}/{tile_key}'
                local_30m_output_path = os.path.join(cn.local_temp_dir, os.path.basename(local_output_path))
                resample_to_30m(local_output_path, local_30m_output_path, reference_path)

                if run_mode == 'test':
                    logging.info(f"Test mode: Outputs saved locally at {local_output_path} and {local_30m_output_path}")
                else:
                    logging.info(f"Uploading {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
                    s3_client.upload_file(local_output_path, cn.s3_bucket_name, s3_output_path)

                    logging.info(f"Uploading {local_30m_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
                    s3_client.upload_file(local_30m_output_path, cn.s3_bucket_name, s3_output_path)

                    logging.info(
                        f"Uploaded {local_output_path} and {local_30m_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
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
    page_iterator = paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=cn.s3_tiles_prefix)
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
            tile_key = f"{cn.s3_tiles_prefix}{tile_id}{cn.peat_pattern}"
            process_tile(tile_key, feature_type, run_mode)
        else:
            process_all_tiles(feature_type, run_mode)

        if run_mode != 'test':
            compress_and_upload_directory_to_s3(cn.output_directories[feature_type]['local'], cn.s3_bucket_name,
                                                cn.output_directories[feature_type]['s3'])
    finally:
        client.close()


# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    # main(tile_id='00N_110E', feature_type='osm_roads', run_mode='test')
    main(tile_id='00N_110E', feature_type='grip_roads', run_mode='test')

    # Process roads and canals separately
    # main(feature_type='osm_roads', run_mode='default')
    # main(feature_type='osm_canals', run_mode='default')
    # main(feature_type='grip_roads', run_mode='default')

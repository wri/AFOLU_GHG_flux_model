"""
This script processes OSM (OpenStreetMap) roads and canals by tiles using a pre-existing raster template from S3.
It reads the raster tiles from S3, resamples them to a target resolution, creates a fishnet grid, reads corresponding
roads or canals shapefiles for each tile, assigns road/canal lengths to the fishnet cells, converts the lengths to density,
and saves the results as raster files. This will have to be re-run with the updated organic soils extent.

The script uses Dask to parallelize the processing of multiple tiles.

Functions:
- get_raster_bounds: Reads and returns the bounds of a raster file.
- resample_raster: Resamples a raster to a target resolution.
- mask_raster: Masks raster data to highlight specific values.
- create_fishnet_from_raster: Creates a fishnet grid from raster data.
- reproject_gdf: Reprojects a GeoDataFrame to a specified EPSG code.
- read_tiled_features: Reads and reprojects shapefile for a given tile.
- assign_segments_to_cells: Assigns segments to fishnet cells and calculates lengths.
- convert_length_to_density: Converts lengths to density.
- fishnet_to_raster: Converts a fishnet GeoDataFrame to a raster and saves it.
- process_tile: Processes a single tile (Dask delayed function).
- process_all_tiles: Processes all tiles using Dask for parallelization.
- main: Main function to orchestrate the processing based on provided arguments.

Usage examples:
- Process a specific tile (00N_110E):
  python script.py --tile_id 00N_110E --feature_type roads

- Process all tiles:
  python script.py --feature_type roads

Dependencies:
- geopandas
- pandas
- shapely
- numpy
- rasterio
- boto3
- logging
- os
- fiona
- dask
- dask.distributed
- dask.diagnostics

Note: The script relies on the presence of AWS credentials for accessing S3 buckets.
"""

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

# Increase Dask communication timeouts
os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "60s"
os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "60s"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'

# OSM directories
roads_tiles_directory = 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/osm_roads/roads_by_tile/'
canals_tiles_directory = 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/osm_roads/canals_by_tile/'

# Output directories
output_dir_roads = 'C:/GIS/Data/Global/OSM/osm_roads_density/'  # Local directories for intermediate storage
output_dir_canals = 'C:/GIS/Data/Global/OSM/osm_canals_density/'  # Local directories for intermediate storage

s3_output_dir_roads = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/'
s3_output_dir_canals = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/'

# Ensure local output directories exist
os.makedirs(output_dir_roads, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)

logging.info("Directories and paths set up")


def get_raster_bounds(raster_path):
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds


def resample_raster(src, target_resolution_m):
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


def read_tiled_features(tile_id, feature_tile_dir, epsg, feature_type):
    logging.info(f"Reading tiled {feature_type} shapefile for tile ID: {tile_id}")
    try:
        tile_id = '_'.join(tile_id.split('_')[:2])  # e.g., '00N_110E'
        file_path = os.path.join(feature_tile_dir, f"{feature_type}_{tile_id}.shp")
        if os.path.exists(file_path) or file_path.startswith('s3://'):
            features_gdf = gpd.read_file(file_path)
            logging.info(f"Read {len(features_gdf)} {feature_type} features for tile {tile_id}")
            features_gdf = reproject_gdf(features_gdf, epsg)
            return features_gdf
        else:
            logging.warning(f"No shapefile found for tile {tile_id}")
            return gpd.GeoDataFrame(columns=['geometry'])
    except fiona.errors.DriverError as e:
        logging.error(f"Error reading {feature_type} shapefile: {e}")
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
        fishnet_gdf['density'] = fishnet_gdf['length'] / (1 * 1)  # lengths are in meters, cell area in km2
    elif crs.axis_info[0].unit_name == 'kilometre':
        fishnet_gdf['density'] = fishnet_gdf['length']  # lengths are already in km, cell area in km2
    else:
        raise ValueError("Unsupported CRS units")
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


def process_tile(tile_key, feature_type, feature_tile_dir, output_dir, s3_output_dir):
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{feature_type}_density_{tile_id}.tif")

    if os.path.exists(local_output_path):
        logging.info(f"{local_output_path} already exists. Skipping processing.")
        return

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'

    try:
        # Create a session for each tile to avoid serialization issues
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_path) as src:
                target_resolution = 1000  # 1 km resolution in meters

                resampled_data, resampled_profile = resample_raster(src, target_resolution)

                masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)

                fishnet_gdf = create_fishnet_from_raster(masked_data, resampled_profile['transform'])

                fishnet_gdf = reproject_gdf(fishnet_gdf, 5070)

                features_gdf = read_tiled_features(tile_id, feature_tile_dir, 5070, feature_type)

                fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, features_gdf)

                fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

                fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

                logging.info(f"Saved {local_output_path}")

                # Upload the file to S3
                s3_client = boto3.client('s3')
                s3_output_path = f"{s3_output_dir}{feature_type}_density_{tile_id}.tif"
                s3_client.upload_file(local_output_path, s3_bucket_name, s3_output_path)
                logging.info(f"Uploaded {local_output_path} to s3://{s3_bucket_name}/{s3_output_path}")

                # Optionally, remove the local file after upload
                os.remove(local_output_path)

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")


def process_all_tiles(feature_type, feature_tile_dir, output_dir, s3_output_dir):
    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_tiles_prefix)

    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith('_peat_mask_processed.tif'):
                    tile_keys.append(tile_key)

    dask_tiles = [dask.delayed(process_tile)(tile_key, feature_type, feature_tile_dir, output_dir, s3_output_dir) for tile_key in tile_keys]
    with ProgressBar():
        dask.compute(*dask_tiles)


def main(tile_id=None, feature_type='roads'):
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        if feature_type == 'roads':
            feature_tile_dir = roads_tiles_directory
            output_dir = output_dir_roads
            s3_output_dir = s3_output_dir_roads
        elif feature_type == 'canals':
            feature_tile_dir = canals_tiles_directory
            output_dir = output_dir_canals
            s3_output_dir = s3_output_dir_canals
        else:
            raise ValueError("Unsupported feature type")

        if tile_id:
            tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
            process_tile(tile_key, feature_type, feature_tile_dir, output_dir, s3_output_dir)
        else:
            process_all_tiles(feature_type, feature_tile_dir, output_dir, s3_output_dir)
    finally:
        client.close()

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    # main(tile_id='00N_110E', feature_type='canals')

    # Process roads and canals separately
    # main(feature_type='roads')
    main(feature_type='canals')

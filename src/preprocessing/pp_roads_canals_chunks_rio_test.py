import dask_geopandas as dgpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
import numpy as np
from shapely.geometry import box  # Import box to create rectangular polygons
import boto3
import geopandas as gpd
import logging
import os
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import gc
import subprocess
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import warnings
import time
import math
from rasterio.enums import Resampling  # Import Resampling for use with rioxarray

import pp_utilities as uu
import constants_and_names as cn

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


# Utility functions

def timestr():
    return time.strftime("%Y%m%d_%H_%M_%S")


def boundstr(bounds):
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


def calc_chunk_length_pixels(bounds):
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


def get_10x10_tile_bounds(tile_id):
    if "S" in tile_id:
        max_y = -1 * (int(tile_id[:2]))
        min_y = -1 * (int(tile_id[:2]) + 10)
    else:
        max_y = (int(tile_id[:2]))
        min_y = (int(tile_id[:2]) - 10)

    if "W" in tile_id:
        max_x = -1 * (int(tile_id[4:7]) - 10)
        min_x = -1 * (int(tile_id[4:7]))
    else:
        max_x = (int(tile_id[4:7]) + 10)
        min_x = (int(tile_id[4:7]))

    return min_x, min_y, max_x, max_y  # W, S, E, N


def get_chunk_bounds(chunk_params):
    min_x = chunk_params[0]
    min_y = chunk_params[1]
    max_x = chunk_params[2]
    max_y = chunk_params[3]
    chunk_size = chunk_params[4]

    x, y = (min_x, min_y)
    chunks = []

    while y < max_y:
        while x < max_x:
            bounds = [
                x,
                y,
                x + chunk_size,
                y + chunk_size,
            ]
            chunks.append(bounds)
            x += chunk_size
        x = min_x
        y += chunk_size

    return chunks


def xy_to_tile_id(top_left_x, top_left_y):
    lat_ceil = math.ceil(top_left_y / 10.0) * 10
    lng_floor = math.floor(top_left_x / 10.0) * 10

    lng = f"{str(lng_floor).zfill(3)}E" if (lng_floor >= 0) else f"{str(-lng_floor).zfill(3)}W"
    lat = f"{str(lat_ceil).zfill(2)}N" if (lat_ceil >= 0) else f"{str(-lat_ceil).zfill(2)}S"

    return f"{lat}_{lng}"


def get_raster_bounds(raster_path):
    logging.info(f"Reading raster bounds from {raster_path}")
    with rxr.open_rasterio(raster_path) as src:
        bounds = src.rio.bounds()
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds


def mask_raster(data, profile):
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    profile.update(dtype=np.uint8)
    return mask.astype(np.uint8), profile


def create_fishnet_from_raster(data, transform):
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
                x, y = transform * (col, row)
                x1, y1 = transform * (col + 1, row + 1)
                polygons.append(box(x, y, x1, y1))

    # Create the GeoDataFrame
    fishnet_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

    # Convert the GeoDataFrame to a Dask GeoDataFrame with multiple partitions
    fishnet_dgdf = dgpd.from_geopandas(fishnet_gdf, npartitions=10)

    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_dgdf


def reproject_gdf(gdf, epsg):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have a CRS. Please set a CRS before reprojecting.")

    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)


def read_tiled_features(tile_id, feature_type):
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
        if not features_gdf.empty and 'geometry' in features_gdf:
            features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
            features_gdf = dgpd.from_geopandas(features_gdf, npartitions=10)
            return features_gdf
        else:
            logging.warning(f"No data found in shapefile for tile {tile_id} at {full_s3_path}")
            return dgpd.GeoDataFrame(columns=['geometry'])

    except Exception as e:
        logging.error(f"Error reading {feature_type} shapefile for tile {tile_id}: {e}")
        return dgpd.GeoDataFrame(columns=['geometry'])


def assign_segments_to_cells(fishnet_gdf, features_gdf):
    logging.info("Assigning feature segments to fishnet cells and calculating lengths")

    # Compute Dask GeoDataFrames to in-memory GeoDataFrames
    fishnet_gdf = fishnet_gdf.compute()
    features_gdf = features_gdf.compute()

    if fishnet_gdf.empty or 'geometry' not in fishnet_gdf:
        logging.warning("Fishnet GeoDataFrame is empty or lacks a geometry column. Skipping segment assignment.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=10)

    if features_gdf.empty or 'geometry' not in features_gdf:
        logging.warning("Features GeoDataFrame is empty or lacks a geometry column. Skipping segment assignment.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=10)

    # Perform clipping
    clipped = gpd.clip(features_gdf, fishnet_gdf)
    if clipped.empty or 'geometry' not in clipped:
        logging.warning(
            "Clipping resulted in an empty GeoDataFrame or lacks a geometry column. Skipping further processing.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=10)

    clipped['length'] = clipped.geometry.length

    fishnet_with_lengths = clipped.dissolve(by='geometry', aggfunc='sum')
    logging.info(f"Fishnet with feature lengths: {fishnet_with_lengths.head()}")
    return dgpd.from_geopandas(fishnet_with_lengths, npartitions=10)


def convert_length_to_density(fishnet_gdf, crs):
    if fishnet_gdf is None or fishnet_gdf.empty:
        logging.warning("Fishnet GeoDataFrame is None or empty. Skipping density conversion.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=10)

    # Reproject fishnet to a projected CRS if necessary for accurate area calculation
    if fishnet_gdf.crs.is_geographic:
        fishnet_gdf = fishnet_gdf.to_crs(crs)

    # Calculate density as length per unit area
    fishnet_gdf['density'] = fishnet_gdf['length'] / 1000

    return fishnet_gdf


def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    logging.info('Updating profile...')

    profile.update(dtype=np.float32, count=1, compress='lzw')  # Store density as float
    logging.info('Profile updated.')

    transform = profile['transform']
    out_shape = (profile['height'], profile['width'])

    fishnet_gdf = fishnet_gdf.to_crs(profile['crs'])

    logging.info('Rasterizing fishnet...')

    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry.compute(), fishnet_gdf['density'].compute())],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32  # Ensure the raster is created with float values
    )

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    with rxr.open_rasterio(output_raster_path, mode='w', **profile) as dst:
        dst.write(rasterized, 1)

    logging.info("Fishnet converted to raster and saved")


@dask.delayed
def process_chunk(bounds, feature_type, tile_id):
    output_dir = cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['local_processed']
    bounds_str = boundstr(bounds)
    local_output_path = os.path.join(output_dir, f"{tile_id}_{bounds_str}_{feature_type}_density.tif")
    s3_output_path = f"{cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['s3_processed']}/{tile_id}_{bounds_str}_{feature_type}_density.tif"

    logging.info(f"Starting processing of the chunk {bounds_str} for tile {tile_id}")

    try:
        input_s3_path = f'/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix}{tile_id}{cn.peat_pattern}'
        with rxr.open_rasterio(input_s3_path) as src:
            chunk_raster = src.rio.clip_box(*bounds)

            if chunk_raster.isnull().all():
                logging.info(f"No data found in the raster for chunk {bounds_str}. Skipping processing.")
                return

            resampled_raster = chunk_raster.rio.reproject(
                dst_crs='EPSG:4326',
                resampling=Resampling.nearest
            )

            profile = {
                "driver": "GTiff",
                "dtype": str(resampled_raster.dtype),
                "nodata": resampled_raster.rio.nodata,
                "width": resampled_raster.rio.width,
                "height": resampled_raster.rio.height,
                "count": resampled_raster.rio.count,
                "crs": resampled_raster.rio.crs,
                "transform": resampled_raster.rio.transform(),
                "compress": "lzw"
            }

            masked_data, masked_profile = mask_raster(resampled_raster.values[0], profile)

            if np.all(masked_data == 0):
                logging.info(f"No data found in the masked raster for chunk {bounds_str}. Skipping processing.")
                return

            # Load and reproject the features
            features_gdf = read_tiled_features(tile_id, feature_type)
            features_gdf = reproject_gdf(features_gdf, 3395)

            # Convert features_gdf to a GeoPandas GeoDataFrame to use sindex
            features_gdf = features_gdf.compute()

            # Ensure the chunk geometry is in the same CRS as the features
            chunk_geom = box(*bounds)
            chunk_geom = gpd.GeoSeries([chunk_geom], crs=features_gdf.crs)

            # Use spatial index for efficient intersection
            if not features_gdf.sindex:
                logging.warning(f"Spatial index is not available for features_gdf in chunk {bounds_str}.")

            possible_matches_index = list(features_gdf.sindex.intersection(chunk_geom.geometry[0].bounds))
            possible_matches = features_gdf.iloc[possible_matches_index]

            precise_matches = possible_matches[possible_matches.intersects(chunk_geom.geometry[0])]

            # Check if there are any features in the chunk bounds
            if precise_matches.empty:
                logging.info(f"No features found within chunk bounds {bounds_str}. Skipping processing.")
                return

            # Proceed with fishnet creation and further processing only if features exist
            fishnet_gdf = create_fishnet_from_raster(masked_data, masked_profile['transform'])

            fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, precise_matches)

            fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

            fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

            reference_path = input_s3_path
            local_30m_output_path = os.path.join(cn.local_temp_dir, os.path.basename(local_output_path))
            resample_to_30m(local_output_path, local_30m_output_path, reference_path)

            upload_final_output_to_s3(local_output_path, s3_output_path)
            upload_final_output_to_s3(local_30m_output_path, s3_output_path.replace('.tif', '_30m.tif'))

            del chunk_raster, resampled_raster, masked_data, fishnet_gdf, features_gdf, fishnet_with_lengths, fishnet_with_density
            gc.collect()

    except Exception as e:
        logging.error(f"Error processing chunk {bounds_str} for tile {tile_id}: {e}", exc_info=True)


def process_tile(tile_key, feature_type, chunk_bounds=None, run_mode='default'):
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    tile_bounds = get_10x10_tile_bounds(tile_id)
    chunk_size = 0.5  # 1x1 degree chunks

    chunks = get_chunk_bounds([*tile_bounds, chunk_size])

    if chunk_bounds:
        chunks = [chunk_bounds]

    chunk_tasks = [process_chunk(bounds, feature_type, tile_id) for bounds in chunks]

    return chunk_tasks


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

    all_tasks = []
    for tile_key in tile_keys:
        all_tasks.extend(process_tile(tile_key, feature_type, run_mode=run_mode))

    with ProgressBar():
        dask.compute(*all_tasks)


def main(tile_id=None, feature_type='osm_roads', chunk_bounds=None, run_mode='default', client_type='local'):
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
    else:
        cluster = LocalCluster()
        client = Client(cluster)

    logging.info(f"Dask client initialized with {client_type} cluster")

    try:
        if tile_id:
            tile_key = f"{cn.peat_tiles_prefix}{tile_id}{cn.peat_pattern}"
            tasks = process_tile(tile_key, feature_type, chunk_bounds, run_mode)
            dask.compute(*tasks)
        else:
            process_all_tiles(feature_type, run_mode)
    finally:
        client.close()
        logging.info("Dask client closed")
        if client_type == 'coiled':
            cluster.close()
            logging.info("Coiled cluster closed")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Process OSM and GRIP data for roads and canals using tiled shapefiles.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--feature_type', type=str, choices=['osm_roads', 'osm_canals', 'grip_roads'],
                        default='osm_roads', help='Type of feature to process')
    parser.add_argument('--chunk_bounds', type=str,
                        help='Specific chunk bounds to process in the format "min_x,min_y,max_x,max_y"', default=None)
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode (default or test)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    chunk_bounds = None
    if args.chunk_bounds:
        chunk_bounds = tuple(map(float, args.chunk_bounds.split(',')))

    if not any(sys.argv[1:]):
        # Default values for running directly from PyCharm or an IDE without command-line arguments
        tile_id = '00N_110E'
        feature_type = 'osm_canals'
        chunk_bounds = (113, -3.5, 113.5, -3)  # Example specific chunk, adjust as needed
        run_mode = 'default'
        client_type = 'local'

        main(tile_id=tile_id, feature_type=feature_type, chunk_bounds=chunk_bounds, run_mode=run_mode,
             client_type=client_type)
    else:
        main(tile_id=args.tile_id, feature_type=args.feature_type, chunk_bounds=chunk_bounds, run_mode=args.run_mode,
             client_type=args.client)

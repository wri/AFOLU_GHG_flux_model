import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
import numpy as np
from rasterio.features import rasterize  # Importing rasterize
from shapely.geometry import box
import boto3
import logging
import os
import dask
from dask.distributed import Client, LocalCluster
import gc
import subprocess
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import warnings
import time
import math

import constants_and_names as cn
import pp_utilities as uu

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

def get_10x10_tile_bounds(tile_id):
    """Calculate the bounds for a 10x10 degree tile."""
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

def ensure_crs(gdf, target_crs):
    if gdf.crs is None:
        logging.warning(f"GeoDataFrame CRS is None, setting it to {target_crs}")
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs != target_crs:
        logging.info(f"Reprojecting GeoDataFrame from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf

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

def mask_raster(data, profile):
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    return mask.astype(np.uint8)

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

    # Convert to Dask GeoDataFrame
    fishnet_gdf = dgpd.from_geopandas(gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326"), npartitions=10)
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def read_tiled_features(tile_id, feature_type):
    try:
        # Extract relevant directories
        feature_key = feature_type.split('_')
        feature_tile_dir = cn.datasets[feature_key[0]][feature_key[1]]['s3_raw']

        # Construct the S3 path for the shapefile
        tile_id = '_'.join(tile_id.split('_')[:2])
        s3_file_path = os.path.join(feature_tile_dir, f"{feature_key[1]}_{tile_id}.shp")
        full_s3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_file_path}"

        logging.info(f"Reading tiled shapefile: {full_s3_path}")

        features_gdf = dgpd.read_file(full_s3_path, npartitions=1)  # Set npartitions
        # features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
        return features_gdf

    except Exception as e:
        logging.error(f"Error reading {feature_type} shapefile for tile {tile_id}: {e}")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=1)

    #     if not features_gdf.empty and 'geometry' in features_gdf:
    #         # Ensure CRS is set, and handle missing CRS
    #         if features_gdf.crs is None:
    #             logging.warning("Features GeoDataFrame CRS is None, setting it to EPSG:4326")
    #             features_gdf.set_crs("EPSG:4326", inplace=True)  # Set a default CRS
    #         features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
    #         return features_gdf
    #     else:
    #         logging.warning(f"No data found in shapefile for tile {tile_id} at {full_s3_path}")
    #         return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=1)
    #
    # except Exception as e:
    #     logging.error(f"Error reading {feature_type} shapefile for tile {tile_id}: {e}")
    #     return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=1)
def reproject_gdf(gdf, epsg):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have a CRS. Please set a CRS before reprojecting.")

    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")

    # Perform the CRS transformation without the meta argument
    return gdf.to_crs(epsg=epsg)


def assign_segments_to_cells(fishnet_gdf, features_gdf):
    logging.info("Assigning feature segments to fishnet cells and calculating lengths")

    # Log initial data types
    logging.info(f"Initial fishnet_gdf type: {type(fishnet_gdf)}")
    logging.info(f"Initial features_gdf type: {type(features_gdf)}")

    # Ensure both GeoDataFrames have the same CRS before clipping
    logging.info(f"Fishnet GeoDataFrame CRS: {fishnet_gdf.crs}")
    logging.info(f"Features GeoDataFrame CRS: {features_gdf.crs}")

    logging.info("Reprojecting features_gdf...")
    features_gdf = reproject_gdf(features_gdf, 3395)  # Reproject to EPSG:3395
    logging.info(f"Post-reprojection features_gdf type: {type(features_gdf)}")

    logging.info("Reprojecting fishnet_gdf...")
    fishnet_gdf = reproject_gdf(fishnet_gdf, 3395)  # Reproject to EPSG:3395
    logging.info(f"Post-reprojection fishnet_gdf type: {type(fishnet_gdf)}")

    # Log CRS after reprojection
    logging.info(f"Fishnet GeoDataFrame CRS: {fishnet_gdf.crs}")
    logging.info(f"Features GeoDataFrame CRS: {features_gdf.crs}")

    if fishnet_gdf.crs is None or features_gdf.crs is None:
        raise ValueError("One of the GeoDataFrames does not have a valid CRS. Cannot proceed with clipping.")

    if fishnet_gdf.crs != features_gdf.crs:
        fishnet_gdf = reproject_gdf(fishnet_gdf, features_gdf.crs.to_string())
        logging.info(f"Reprojected fishnet_gdf type: {type(fishnet_gdf)}")

    # Convert fishnet_gdf to a regular GeoDataFrame before clipping
    if isinstance(fishnet_gdf, dgpd.GeoDataFrame):
        logging.info("Converting fishnet_gdf from Dask GeoDataFrame to regular GeoDataFrame")
        fishnet_gdf = fishnet_gdf.compute()
        logging.info(f"Converted fishnet_gdf type: {type(fishnet_gdf)}")

    # Convert features_gdf to a regular GeoDataFrame if it's still a Dask GeoDataFrame
    if isinstance(features_gdf, dgpd.GeoDataFrame):
        logging.info("Converting features_gdf from Dask GeoDataFrame to regular GeoDataFrame")
        features_gdf = features_gdf.compute()
        logging.info(f"Converted features_gdf type: {type(features_gdf)}")

    # Perform clipping
    logging.info("Performing clipping...")
    clipped = gpd.clip(features_gdf, fishnet_gdf)
    logging.info(f"Clipped GeoDataFrame type: {type(clipped)}")

    # Check if the clipped GeoDataFrame is empty or lacks geometry
    if len(clipped.index) == 0 or 'geometry' not in clipped.columns:
        logging.warning(
            "Clipping resulted in an empty GeoDataFrame or lacks a geometry column. Skipping further processing.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=1)

    # Calculate lengths of features within each fishnet cell
    logging.info("Calculating lengths of clipped features")
    clipped['length'] = clipped.geometry.length

    # Dissolve by fishnet geometry to sum up lengths, providing metadata
    meta = gpd.GeoDataFrame(columns=clipped.columns, geometry=clipped.geometry.name, crs=clipped.crs)
    # fishnet_with_lengths = clipped.dissolve(by=clipped.index, aggfunc='sum', meta=meta)
    fishnet_with_lengths = clipped.dissolve(by=clipped.index, aggfunc='sum')

    logging.info(f"Final fishnet_with_lengths type: {type(fishnet_with_lengths)}")
    logging.info(f"Fishnet with feature lengths: {fishnet_with_lengths.head()}")

    return dgpd.from_geopandas(fishnet_with_lengths, npartitions=10)


# def convert_length_to_density(fishnet_gdf, crs):
#     logging.info("Converting lengths to density")
#
#     # Check if the GeoDataFrame is None or has no rows
#     if fishnet_gdf is None or len(fishnet_gdf.index) == 0:
#         logging.warning("Fishnet GeoDataFrame is None or empty. Skipping density conversion.")
#         return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry', 'density']), npartitions=1)
#
#     # Ensure the 'length' column exists before proceeding
#     if 'length' not in fishnet_gdf.columns:
#         logging.warning("Fishnet GeoDataFrame does not have a 'length' column. Skipping density conversion.")
#         return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry', 'density']), npartitions=1)
#
#     # Reproject fishnet to a projected CRS if necessary for accurate area calculation
#     # fishnet_gdf = ensure_crs(fishnet_gdf, crs)
#
#     # Calculate density as length per unit area
#     fishnet_gdf['density'] = fishnet_gdf['length'] / 1000
#
#     return fishnet_gdf

def convert_length_to_density(fishnet_gdf, crs):
    logging.info("Converting lengths to density")

    # Compute the fishnet GeoDataFrame before proceeding
    fishnet_gdf = fishnet_gdf.compute()

    # Check if the GeoDataFrame is None or has no rows
    if fishnet_gdf is None or len(fishnet_gdf.index) == 0:
        logging.warning("Fishnet GeoDataFrame is None or empty. Skipping density conversion.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry', 'density']), npartitions=1)

    # Ensure the 'length' column exists before proceeding
    if 'length' not in fishnet_gdf.columns:
        logging.warning("Fishnet GeoDataFrame does not have a 'length' column. Skipping density conversion.")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry', 'density']), npartitions=1)

    # Calculate density as length per unit area
    fishnet_gdf['density'] = fishnet_gdf['length'] / 1000

    return fishnet_gdf



def fishnet_to_raster(fishnet_gdf, chunk_raster, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    logging.info('Updating profile...')

    # Convert the GeoDataFrame to a format suitable for rasterization
    logging.info('Ensuring crs...')
    # fishnet_gdf = ensure_crs(fishnet_gdf, chunk_raster.rio.crs)

    logging.info('Rasterizing fishnet...')

    # Create a rasterization template using the 2D shape of the raster (ignoring the band dimension)
    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry.compute(), fishnet_gdf['density'].compute())],
        out_shape=chunk_raster.shape[1:],  # Use 2D shape
        transform=chunk_raster.rio.transform(),
        fill=0,
        all_touched=True,
        dtype=np.float32
    )

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    # Convert the rasterized array to an xarray DataArray and set attributes
    xr_rasterized = xr.DataArray(
        rasterized,
        dims=("y", "x"),
        coords={"y": chunk_raster.y, "x": chunk_raster.x},
    )
    xr_rasterized = xr_rasterized.rio.write_crs(chunk_raster.rio.crs, inplace=True)
    xr_rasterized = xr_rasterized.rio.write_transform(chunk_raster.rio.transform(), inplace=True)

    # Save the rasterized DataArray to a GeoTIFF file
    xr_rasterized.rio.to_raster(output_raster_path, compress='lzw')

    logging.info("Fishnet converted to raster and saved")

def upload_final_output_to_s3(local_output_path, s3_output_path):
    s3_client = boto3.client('s3')
    try:
        # Ensure no double slashes in the S3 path
        s3_output_path = s3_output_path.replace('//', '/')
        logging.info(f"Uploading {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        s3_client.upload_file(local_output_path, cn.s3_bucket_name, s3_output_path)
        logging.info(f"Successfully uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        os.remove(local_output_path)
        logging.info(f"Deleted local file: {local_output_path}")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Credentials error: {e}")
    except Exception as e:
        logging.error(f"Failed to upload {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}: {e}")

@dask.delayed
def process_chunk(bounds, feature_type, tile_id):
    output_dir = cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['local_processed']
    bounds_str = "_".join([str(round(x, 2)) for x in bounds])
    local_output_path = os.path.join(output_dir, f"{tile_id}_{bounds_str}_{feature_type}_density.tif")
    s3_output_path = f"{cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['s3_processed']}/{tile_id}_{bounds_str}_{feature_type}_density.tif"

    logging.info(f"Starting processing of the chunk {bounds_str} for tile {tile_id}")

    try:
        input_s3_path = f'/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix_1km}{tile_id}{cn.peat_pattern}'

        # Open the raster using rioxarray
        chunk_raster = rxr.open_rasterio(input_s3_path, masked=True)

        # Clip the raster to the specified bounds
        chunk_raster = chunk_raster.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3])

        if chunk_raster.isnull().all():
            logging.info(f"No data found in the raster for chunk {bounds_str}. Skipping processing.")
            return

        # Mask raster
        masked_data = mask_raster(chunk_raster[0].data, chunk_raster.rio)

        if np.all(masked_data == 0):
            logging.info(f"No data found in the masked raster for chunk {bounds_str}. Skipping processing.")
            return

        # Create fishnet and process features
        fishnet_gdf = create_fishnet_from_raster(masked_data, chunk_raster.rio.transform())

        if len(fishnet_gdf.index) == 0:  # Using len(df.index) to check for emptiness
            logging.info(f"Fishnet is empty for chunk {bounds_str}. Skipping processing.")
            return

        features_gdf = read_tiled_features(tile_id, feature_type)

        fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, features_gdf)

        fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

        if len(fishnet_with_density.columns) == 0 or 'density' not in fishnet_with_density.columns:
            logging.info(f"Skipping export of {local_output_path} as 'density' column is missing.")
            return

        # Save the fishnet to a raster using rioxarray
        fishnet_to_raster(fishnet_with_density, chunk_raster, local_output_path)

        upload_final_output_to_s3(local_output_path, s3_output_path)

        del chunk_raster, masked_data, fishnet_gdf, features_gdf, fishnet_with_lengths, fishnet_with_density
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing chunk {bounds_str} for tile {tile_id}: {e}", exc_info=True)


def process_tile(tile_key, feature_type, chunk_bounds=None, run_mode='default'):
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    tile_bounds = get_10x10_tile_bounds(tile_id)
    chunk_size = 2  # 1x1 degree chunks

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
        chunk_bounds = (112, -4, 114, -2)  # this chunk has data
        # chunk_bounds = None
        run_mode = 'test'
        client_type = 'local'

        main(tile_id=tile_id, feature_type=feature_type, chunk_bounds=chunk_bounds, run_mode=run_mode,
             client_type=client_type)
    else:
        main(tile_id=args.tile_id, feature_type=args.feature_type, chunk_bounds=chunk_bounds, run_mode=args.run_mode,
             client_type=args.client)

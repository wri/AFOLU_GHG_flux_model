import os
import logging
import boto3
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import pp_hansenize_gdal as hz  # Importing GDAL-based hansenize script
import constants_and_names as cn

"""
This script processes descals raster tiles by merging smaller input tiles,
clipping them to standard tile bounds, and uploading the processed tiles to S3.
This script is not currently using Dask but a version using Dask may be implemented.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')

def get_raster_files_from_s3(s3_directory):
    """
    Get a list of raster files from an S3 directory.

    Args:
        s3_directory (str): The S3 directory path.

    Returns:
        list: List of raster file paths in S3.
    """
    bucket, prefix = s3_directory.replace("s3://", "").split("/", 1)
    paginator = s3_client.get_paginator('list_objects_v2')
    raster_files = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.tif'):
                    raster_files.append(f"s3://{bucket}/{obj['Key']}")
    except Exception as e:
        logging.error(f"Error retrieving raster files from S3: {e}")
    return raster_files

def create_index_from_s3(s3_prefix, local_output_dir, dataset_name='descals'):
    """
    Create a shapefile index for all the tiles in a specified S3 directory.

    Parameters:
    s3_prefix (str): S3 prefix where the raster tiles are stored.
    local_output_dir (str): Local directory to save the index shapefile.
    dataset_name (str): The name of the dataset (default is 'descals').

    Returns:
    str: Path to the created shapefile index.
    """
    index_path = os.path.join(local_output_dir, f"{dataset_name}_tile_index.shp")

    if os.path.exists(index_path):
        logging.info(f"Descals tile index already exists at {index_path}. Loading...")
        return index_path

    logging.info(f"Creating index for {dataset_name} dataset from {s3_prefix}")

    # Retrieve raster file paths from S3
    raster_files = get_raster_files_from_s3(f"s3://{cn.s3_bucket_name}/{s3_prefix}")
    logging.info(f"Retrieved {len(raster_files)} raster files from {s3_prefix}")

    # Initialize an empty list to store the tile information
    tile_index = []

    # Loop through each raster file to collect bounds and create geometries
    for raster_file in raster_files:
        try:
            logging.info(f"Opening raster file {raster_file}")
            with rasterio.open(raster_file) as src:
                bounds = src.bounds
                tile_id = os.path.basename(raster_file).replace('.tif', '')
                geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                tile_index.append({'tile_id': tile_id, 'geometry': geometry, 'full_path': raster_file})
                logging.debug(f"Added tile {tile_id} with bounds {bounds} to index")
        except Exception as e:
            logging.error(f"Error processing raster file {raster_file}: {e}")
            continue

    # Convert the tile index list to a GeoDataFrame
    gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:4326")

    # Save the GeoDataFrame to a shapefile
    gdf.to_file(index_path, driver='ESRI Shapefile')
    logging.info(f"Index shapefile created at {index_path}")

    return index_path

def read_shapefile_from_s3(s3_prefix, local_dir):
    """
    Read a shapefile from S3 into a GeoDataFrame.

    Args:
        s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
        local_dir (str): The local directory where the files will be saved.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    try:
        extensions = ['.shp', '.shx', '.dbf', '.prj']
        for ext in extensions:
            s3_path = f"{s3_prefix}{ext}"
            local_path = os.path.join(local_dir, os.path.basename(s3_prefix) + ext)
            s3_client.download_file(cn.s3_bucket_name, s3_path, local_path)
        shapefile_path = os.path.join(local_dir, os.path.basename(s3_prefix) + '.shp')
        gdf = gpd.read_file(shapefile_path)
        logging.info(f"Shapefile {shapefile_path} successfully loaded")
    except Exception as e:
        logging.error(f"Error reading shapefile from S3: {e}")
        gdf = gpd.GeoDataFrame()  # Return an empty GeoDataFrame in case of error
    return gdf

def get_tile_bounds(index_shapefile, tile_id):
    """
    Retrieve the bounds of a specific tile from the global index shapefile.

    Args:
        index_shapefile (str): Path to the global index shapefile.
        tile_id (str): Tile ID to look for.

    Returns:
        tuple: Bounding box of the tile (minx, miny, maxx, maxy).
    """
    try:
        gdf = gpd.read_file(index_shapefile)
        tile = gdf[gdf['tile_id'] == tile_id]
        if tile.empty:
            logging.error(f"Tile {tile_id} not found in index shapefile.")
            return None
        bounds = tile.total_bounds
        logging.info(f"Tile bounds for {tile_id}: {bounds}")
    except Exception as e:
        logging.error(f"Error retrieving tile bounds for {tile_id}: {e}")
        bounds = None
    return bounds

import tempfile

def process_tile(tile_id, dataset, tile_bounds, descals_gdf, run_mode='default'):
    """
    Processes a single tile: merges descals tiles, clips to bounds, and uploads to S3.

    Parameters:
    tile_id (str): ID of the tile to process.
    dataset (str): The dataset type (e.g., 'descals').
    tile_bounds (tuple): Bounding box coordinates for the tile.
    descals_gdf (GeoDataFrame): GeoDataFrame of descals tiles.
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    output_dir = cn.datasets[dataset]['local_processed']
    os.makedirs(output_dir, exist_ok=True)

    s3_output_dir = cn.datasets[dataset]['s3_processed']
    local_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}/{dataset}_{tile_id}.tif".replace("\\", "/")

    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=cn.s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")
            else:
                logging.error(f"Error checking existence of {s3_output_path} on S3: {e}")
                return

    logging.info(f"Starting processing of the tile {tile_id}")

    try:
        if tile_bounds is None:
            logging.error(f"Tile bounds not found for {tile_id}")
            return

        # Find descals tiles that intersect with the peatland tile
        intersecting_descals = descals_gdf[descals_gdf.geometry.intersects(box(*tile_bounds))]
        descals_tile_paths = intersecting_descals['full_path'].tolist()

        if not descals_tile_paths:
            logging.info(f"No descals tiles found for tile {tile_id}")
            return

        logging.info(f"Found {len(descals_tile_paths)} descals tiles for tile {tile_id}: {descals_tile_paths}")

        # Download descals tiles locally
        local_tile_paths = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            for s3_path in descals_tile_paths:
                local_path = os.path.join(tmpdirname, os.path.basename(s3_path))
                bucket, key = s3_path.replace("s3://", "").split("/", 1)
                s3_client.download_file(bucket, key, local_path)
                local_tile_paths.append(local_path)

            # Use the GDAL version of hansenize to merge and clip descals tiles
            logging.info(f"Merging and clipping descals tiles for tile {tile_id} using GDAL")
            hz.hansenize_gdal(local_tile_paths, local_output_path, tile_bounds, nodata_value=0, dtype='Byte')

            # Replace the erroneous value of 3 with 0
            logging.info(f"Replacing erroneous value 3 with 0 in {local_output_path}")
            with rasterio.open(local_output_path, 'r+') as dst:
                data = dst.read(1)
                data[data == 3] = 0  # Replace 3 with 0
                dst.write(data, 1)

        logging.info(f"Tile {tile_id} processed successfully")
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")
    finally:
        if os.path.exists(local_output_path):
            logging.info("Success")
            # os.remove(local_output_path)  # Clean up the output raster after processing
            # logging.info(f"Intermediate output raster {local_output_path} removed")

def main(tile_id=None, dataset='descals', run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
    tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
    dataset (str, optional): The dataset type (default: 'descals').
    run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
    None
    """
    try:
        logging.info(f"Starting main processing routine for dataset {dataset}")

        # Ensure the global peatlands index is available
        peatlands_index_path = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        if not os.path.exists(peatlands_index_path):
            logging.info("Global peatlands index not found locally. Downloading...")
            read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir)

        # Ensure the descals tile index is available
        descals_index_path = create_index_from_s3(cn.datasets[dataset]['s3_raw'], cn.local_temp_dir)
        descals_gdf = gpd.read_file(descals_index_path)

        # Load peatlands index
        peatlands_gdf = gpd.read_file(peatlands_index_path)

        # If a specific tile ID is provided, filter peatlands_gdf to only include that tile
        if tile_id:
            peatlands_gdf = peatlands_gdf[peatlands_gdf['tile_id'] == tile_id]
            if peatlands_gdf.empty:
                logging.error(f"Tile {tile_id} not found in peatlands index.")
                return

        # Process selected tiles
        for _, peatland_tile in peatlands_gdf.iterrows():
            peatland_tile_id = peatland_tile['tile_id']
            tile_bounds = peatland_tile.geometry.bounds
            logging.info(f"Checking intersection for peatland tile {peatland_tile_id}")
            process_tile(peatland_tile_id, dataset, tile_bounds, descals_gdf, run_mode)

    except Exception as e:
        logging.error(f"Error in main processing routine: {e}")
    finally:
        logging.info("Processing completed")

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    main(tile_id='00N_110E', run_mode='test')

    # Process the entire descals dataset
    # main(run_mode='default')

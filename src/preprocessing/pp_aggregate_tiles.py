# src/preprocessing/pp_aggregate_tiles.py

import argparse
import logging
import dask
from dask.distributed import Client, LocalCluster
import os
import gc
import warnings
import sys
import re
import boto3
import subprocess

# Project imports
import constants_and_names as cn
import pp_utilities as uu  # Import utility functions from pp_utilities.py

"""
Script for Post-Processing LULUCF Fluxes

This script processes LULUCF flux data by merging small tiles into larger rasters.
It can be run with command-line arguments or directly within a Python script/IDE.
It also allows testing for a specific tile ID and processing specific datasets.
"""

# ----------------------------- Logging Setup -----------------------------

# Set up general logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from', UserWarning)

# Ensure local output directories exist
uu.create_directory_if_not_exists(cn.local_temp_dir)
logging.info("Directories and paths set up")

# ----------------------------- Utility Functions -----------------------------

def adjust_output_path(input_path):
    """
    Adjusts the output path by replacing '4000_pixels' with the value of 'cn.full_raster_dims'.

    Args:
        input_path (str): The original input path.

    Returns:
        str: The adjusted output path.
    """
    return input_path.replace('4000_pixels', f'{str(cn.full_raster_dims)}_pixels')

def extract_tile_id_from_filename(filename):
    """
    Extracts the tile ID from the given filename using regex.

    Args:
        filename (str): The name of the file.

    Returns:
        str or None: The extracted tile ID or None if not found.
    """
    match = re.match(r'(\d{2}[NS]_\d{3}[EW])', filename)
    if match:
        return match.group(1)
    else:
        return None

def list_tile_ids(bucket, prefix, pattern):
    """
    List all tile IDs in a specified S3 directory matching a given pattern.

    Args:
        bucket (str): The S3 bucket name.
        prefix (str): The prefix path in the S3 bucket.
        pattern (str): Regex pattern to match tile IDs.

    Returns:
        list: List of tile IDs.
    """
    s3_client = boto3.client('s3')  # Instantiate inside the function
    keys = []
    tile_ids = set()
    try:
        logging.info(f"Listing files in s3://{bucket}/{prefix} with pattern '{pattern}'")
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])

        logging.info(f"Retrieved {len(keys)} keys from S3")

        # Extract tile IDs from filenames
        for key in keys:
            filename = os.path.basename(key)
            logging.debug(f"Processing filename: {filename}")
            match = re.match(pattern, filename)
            if match:
                tile_ids.add(match.group(1))
            else:
                logging.debug(f"No match for filename: {filename}")
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")
    return list(tile_ids)

# ----------------------------- GDAL Hansenize Functions -----------------------------

def clip_raster_to_tile(input_path, output_path, bounds, nodata_value=None, dtype=None):
    """
    Clips a raster to specified bounds using GDAL's gdalwarp command.

    Args:
        input_path (str): Path to the input raster.
        output_path (str): Path to save the clipped raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
        nodata_value (float): NoData value to set for the output raster.
        dtype (str): Data type for the output raster.
    """
    try:
        minx, miny, maxx, maxy = bounds
        gdalwarp_cmd = [
            'gdalwarp',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite'
        ]

        if dtype:
            gdalwarp_cmd.extend(['-ot', dtype])  # Correctly add the data type option

        # Include any other options like -tr and -tap if needed
        gdalwarp_cmd.extend([
            '-tr', '0.00025', '0.00025',  # Set the output resolution explicitly
            '-tap'                         # Align pixels
        ])

        # Add input and output paths
        gdalwarp_cmd.extend([input_path, output_path])

        logging.info(f"Clipping raster with command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)
        logging.info(f"Raster clipped successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error during clipping: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during raster clipping: {e}")

def merge_and_clip_rasters_gdal(raster_paths, output_path, bounds, nodata_value=None, dtype=None):
    """
    Merges multiple rasters and clips to specified bounds using GDAL.
    """
    try:
        minx, miny, maxx, maxy = bounds
        gdalwarp_cmd = [
            'gdalwarp',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-tr', '0.00025', '0.00025',  # Set the output resolution explicitly
            '-r', 'near',                  # Resampling method
            '-t_srs', 'EPSG:4326',         # Set the output CRS
            '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite'
        ]

        if dtype:
            gdalwarp_cmd.extend(['-ot', dtype])  # Correctly add the data type option

        # Add raster paths and output path
        gdalwarp_cmd.extend(raster_paths + [output_path])

        logging.info(f"Merging and clipping rasters with command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)
        logging.info(f"Rasters merged and clipped successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error during merge and clip: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during raster merging and clipping: {e}")
        raise


def hansenize_gdal(input_paths, output_path, bounds, nodata_value=None, dtype=None):
    """
    Main function for processing using GDAL.

    Args:
        input_paths (str or list): Input raster path or list of paths to process.
        output_path (str): Path to save the processed raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for processing.
        nodata_value (float): NoData value to set for the output raster.
        dtype (str): Data type for the output raster.
    """
    if isinstance(input_paths, list):
        merge_and_clip_rasters_gdal(input_paths, output_path, bounds, nodata_value, dtype)
    else:
        clip_raster_to_tile(input_paths, output_path, bounds, nodata_value, dtype)

    gc.collect()

# ----------------------------- Main Processing Functions -----------------------------

@dask.delayed
def merge_tiles(tile_id, s3_in_folder, s3_out_folder, dataset_name, no_upload=False):
    """
    Merge small tiles into a larger raster for a given tile ID.

    Args:
        tile_id (str): The tile ID to process.
        s3_in_folder (str): The S3 input folder containing small tiles.
        s3_out_folder (str): The S3 output folder to save the merged raster.
        dataset_name (str): The name of the dataset being processed.
        no_upload (bool): If True, do not upload the output to S3.

    Returns:
        None
    """
    try:
        logging.info(f"Processing tile ID: {tile_id}")

        # Define local and S3 paths
        local_temp_dir = cn.local_temp_dir
        uu.create_directory_if_not_exists(local_temp_dir)

        # List of small tiles to merge
        s3_bucket_name = cn.s3_bucket_name
        prefix = s3_in_folder

        # List all files in the prefix
        all_files = uu.list_s3_files(s3_bucket_name, prefix)

        # Filter files for the specific tile_id
        small_tile_files = [key for key in all_files if extract_tile_id_from_filename(os.path.basename(key)) == tile_id]

        if not small_tile_files:
            logging.warning(f"No small tiles found for tile ID: {tile_id}")
            return

        # Download small tiles to local directory
        local_tile_files = []
        for s3_file in small_tile_files:
            local_file = os.path.join(local_temp_dir, os.path.basename(s3_file))
            uu.download_file_from_s3(s3_file, local_file, s3_bucket_name)
            local_tile_files.append(local_file)

        if not local_tile_files:
            logging.warning(f"No local tiles downloaded for tile ID: {tile_id}")
            return

        # Corrected code to parse tile_id
        # Extract latitude degrees and hemisphere
        lat_deg = int(tile_id[0:2])
        lat_hemisphere = tile_id[2]

        # Extract longitude degrees and hemisphere
        lon_deg = int(tile_id[4:7])
        lon_hemisphere = tile_id[7]

        # Adjust for hemisphere
        if lat_hemisphere == 'S':
            lat_deg = -lat_deg

        if lon_hemisphere == 'W':
            lon_deg = -lon_deg

        # Determine bounds (assuming 10x10 degree tiles)
        # For latitude, subtract 10 degrees to get the minimum latitude
        bounds = (lon_deg, lat_deg - 10, lon_deg + 10, lat_deg)

        # Merge and clip rasters using GDAL hansenize functions
        output_filename = f"{tile_id}_{dataset_name}.tif"
        merged_raster_path = os.path.join(local_temp_dir, output_filename)
        hansenize_gdal(local_tile_files, merged_raster_path, bounds)

        # Upload the merged raster if required
        if not no_upload:
            s3_output_path = os.path.join(s3_out_folder, output_filename)
            s3_output_path = s3_output_path.replace("\\", "/")  # Ensure S3 path uses forward slashes
            uu.upload_file_to_s3(merged_raster_path, s3_bucket_name, s3_output_path)

        # Clean up local files
        for local_file in local_tile_files + [merged_raster_path]:
            uu.delete_file_if_exists(local_file)

        gc.collect()

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}", exc_info=True)

def process_all_tiles(s3_in_folder, s3_out_folder, tile_id=None, no_upload=False):
    """
    Process all tiles or a specific tile by merging small tiles into larger rasters.

    Args:
        s3_in_folder (str): The S3 input folder containing small tiles.
        s3_out_folder (str): The S3 output folder to save the merged rasters.
        tile_id (str, optional): Specific tile ID to process. If None, process all tiles.
        no_upload (bool): If True, do not upload the outputs to S3.

    Returns:
        None
    """
    logging.info(f"Processing tiles in {s3_in_folder}")

    s3_bucket_name = cn.s3_bucket_name
    prefix = s3_in_folder

    # Correctly extract the dataset name from the folder path
    s3_path_parts = s3_in_folder.strip('/').split('/')
    dataset_name = ''
    for part in s3_path_parts:
        if part in ['osm_roads_density', 'osm_canals_density', 'grip_density']:
            dataset_name = part
            break

    if not dataset_name:
        logging.error(f"Could not determine dataset name from s3_in_folder: {s3_in_folder}")
        return

    logging.info(f"Dataset name: {dataset_name}")

    # Adjust the pattern to match your filenames
    pattern = rf"(\d{{2}}[NS]_\d{{3}}[EW])_.*_{dataset_name}\.tif"

    # List all tile IDs
    tile_ids = list_tile_ids(s3_bucket_name, prefix, pattern=pattern)

    if not tile_ids:
        logging.error("No tile IDs found to process.")
        return

    if tile_id:
        if tile_id not in tile_ids:
            logging.error(f"Tile ID {tile_id} not found in the list of available tiles.")
            return
        tile_ids = [tile_id]  # Process only the specified tile

    # Create delayed tasks for each tile
    tasks = []
    for tid in tile_ids:
        task = merge_tiles(tid, s3_in_folder, s3_out_folder, dataset_name, no_upload)
        tasks.append(task)

    # Compute the tasks
    logging.info(f"Computing {len(tasks)} tasks")
    dask.compute(*tasks)

# ----------------------------- Main Function -----------------------------

def main(date=cn.today_date, run_local=False, no_upload=False, tile_id=None, dataset=None):
    """
    Main function to handle post-processing of LULUCF fluxes.

    Args:
        date (str): Date in YYYYMMDD format to process.
        run_local (bool): Flag to run the script locally without Dask/Coiled.
        no_upload (bool): Flag to disable uploading outputs to S3.
        tile_id (str, optional): Specific tile ID to process.
        dataset (str, optional): Specific dataset to process.

    Returns:
        None
    """
    logging.info("Initializing main processing function")

    if run_local:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info("Running locally with Dask LocalCluster")
    else:
        client, cluster = uu.setup_coiled_cluster()
        logging.info(f"Coiled cluster initialized: {cluster.name}")

    try:
        # Define input and output S3 folders
        datasets_to_process = ['osm_roads_density', 'osm_canals_density', 'grip_density']
        if dataset:
            if dataset not in datasets_to_process:
                logging.error(f"Invalid dataset specified: {dataset}")
                return
            datasets_to_process = [dataset]

        for ds in datasets_to_process:
            s3_in_folder = os.path.join(cn.project_dir, cn.processed_dir, ds, '4000_pixels', date, '')
            s3_in_folder = s3_in_folder.replace("\\", "/")  # Ensure S3 path uses forward slashes
            s3_out_folder = adjust_output_path(s3_in_folder)
            logging.info(f"Processing S3 folder: {s3_in_folder}")
            logging.info(f"Adjusted output path: {s3_out_folder}")

            process_all_tiles(s3_in_folder, s3_out_folder, tile_id=tile_id, no_upload=no_upload)

    finally:
        client.close()
        logging.info("Dask client closed")
        if not run_local:
            cluster.close()
            logging.info("Coiled cluster closed")

# ----------------------------- Command-Line Interface -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing of LULUCF fluxes.")
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process', default=cn.today_date)
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to S3')
    parser.add_argument('--tile_id', type=str, help='Specific tile ID to process')
    parser.add_argument('--dataset', type=str, choices=['osm_roads_density', 'osm_canals_density', 'grip_density'], help='Specific dataset to process')
    args = parser.parse_args()

    # Ensure 'full_raster_dims' is a string
    if not hasattr(cn, 'full_raster_dims'):
        cn.full_raster_dims = '40000'  # Adjust as needed
        logging.info(f"Set 'full_raster_dims' to {cn.full_raster_dims}")
    else:
        cn.full_raster_dims = str(cn.full_raster_dims)

    # Check if script is run with command-line arguments
    if not any(sys.argv[1:]):
        # Default values for running directly from an IDE without command-line arguments
        date = '20240925'
        run_local = True
        no_upload = False
        tile_id = '00N_110E'  # Set a default tile ID for testing
        dataset = 'osm_roads_density'  # Specify a dataset to process

        main(date=date, run_local=run_local, no_upload=no_upload, tile_id=tile_id, dataset=dataset)
    else:
        main(date=args.date, run_local=args.run_local, no_upload=args.no_upload, tile_id=args.tile_id, dataset=args.dataset)


"""
Example Command-Line Usages:

1. Run for a specific date, tile, and dataset without uploading outputs:

   python pp_aggregate_tiles.py --date 20240101 --tile_id 00N_110E --dataset osm_canals_density --no_upload

2. Run for a specific date, tile, and dataset with uploading outputs:

   python pp_aggregate_tiles.py --date 20240822 --tile_id 00N_110E --dataset osm_canals_density

3. Run locally for all datasets and tiles for today's date:

   python pp_aggregate_tiles.py --run_local

4. Run using Coiled cluster for the 'grip_density' dataset:

   python pp_aggregate_tiles.py --dataset grip_density

5. Run for a specific tile across all datasets:

   python pp_aggregate_tiles.py --tile_id 00N_110E

"""

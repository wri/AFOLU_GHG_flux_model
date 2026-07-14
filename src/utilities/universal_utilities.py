import os
import fsspec
import coiled
import boto3
import time
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import psutil
import pytz
import sys
import rasterio
import rasterio.transform
import rasterio.windows
import subprocess
import re
import requests
import concurrent.futures
from botocore.config import Config
from dask.distributed import print
from dask.distributed import Client, LocalCluster
from dask import delayed
import dask.array as da
from datetime import datetime
from io import BytesIO
import xarray as xr
from numba import jit
from osgeo import gdal
from rasterio.transform import from_origin
from rasterio.session import AWSSession
import random
import zarr
import tempfile
import rasterio.errors
from urllib.parse import urlparse
from botocore.exceptions import ClientError
import botocore

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu

#TODO: Create separate utilities for land use specific functions. Keep all other general utilities and microservices here.


# Turns off a FutureWarning about gdal.UseExceptions() vs. gdal.DontUseExceptions()
gdal.UseExceptions()

session = boto3.Session()
aws_session = AWSSession(session)

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"

###################################################################################################
# S3 Utilities
###################################################################################################
# Splits a full s3 path "s3://bucket-name/rest_of_path" into "bucket-name" and "rest_of_path"
def split_s3_path(s3_path):
    s3_path = s3_path.replace("s3://", "")   # Remove the "s3://" prefix
    bucket, key = s3_path.split("/", 1)    # Split the remaining string by the first "/"
    return bucket, key

# List files in an S3 bucket with a certain pattern
def list_s3_files_with_pattern(s3_path, pattern, use_regex=False):
    s3 = boto3.client("s3")
    bucket_name, prefix = split_s3_path(s3_path)

    matching_files = []
    continuation_token = None  # For pagination

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

        # Check if there are any contents
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if use_regex:
                    compiled_pattern = re.compile(pattern)
                    filename = key.split("/")[-1]  # get just the filename from the S3 key
                    if compiled_pattern.fullmatch(filename):
                        matching_files.append(f"s3://{bucket_name}/{key}")
                else: 
                    if pattern in key:
                        matching_files.append(f"s3://{bucket_name}/{key}")

        # Check if there's more data to retrieve
        if response.get("IsTruncated"):  # If True, there are more pages to fetch
            continuation_token = response["NextContinuationToken"]
        else:
            break  # No more pages left

    return matching_files

def download_s3_file(s3_path, local_path):
    s3 = boto3.client('s3')
    bucket, key = split_s3_path(s3_path)
    s3.download_file(bucket, key, local_path)

def upload_s3_file(s3_path, local_path):
    s3 = boto3.client('s3')
    bucket, key = split_s3_path(s3_path)
    s3.upload_file(local_path, Bucket=bucket, Key=key)

def check_s3_file_created(s3_path):

    logger_worker = lu.setup_logging_worker()

    s3 = boto3.client('s3')
    bucket, key = split_s3_path(s3_path)

    try:
        s3.head_object(Bucket=bucket, Key=key)
        lu.print_and_log(f"File successfully created at: {s3_path}", False, logger_worker)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            raise RuntimeError(f"Failed to create file at: {s3_path}")
        else:
            raise RuntimeError(f"Error accessing S3: {e}")

def check_and_make_s3_dir(s3_directory, main_logger):
    if s3_directory.startswith("s3://"):
        # Normalize path to ensure it ends with a slash
        if not s3_directory.endswith("/"):
            s3_directory += "/"

        # Check if the S3 path exists
        try:
            check_cmd = ["aws", "s3", "ls", s3_directory]
            result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if result.stdout:
                main_logger.info(f"S3 processed directory exists: {s3_directory}")
            else:
                raise Exception("Empty result")

        except Exception:
            # Create a placeholder .keep file to establish the S3 "folder"
            placeholder_path = s3_directory + ".keep"
            create_cmd = ["aws", "s3", "cp", "-", placeholder_path]
            subprocess.run(create_cmd, input=b'', check=True)
            main_logger.info(f"Created S3 processed directory by uploading placeholder: {placeholder_path}")

    else:
        main_logger.warning(f"Processed directory is not an S3 path. Skipping creation: {s3_directory}")

# Uploads local rasters to s3 and deletes the local versions after
def upload_raster_to_s3(file_path, bucket, s3_key):

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(file_path, bucket, s3_key)
        os.remove(file_path)  # Remove local temp file after upload
    except Exception as e:
        print(f"Upload failed for {s3_key}: {e}")


# Saves arrays as rasters locally, then makes a list of tasks of rasters to upload. Does not actually upload.
# NoData value for outputs is optional.
# Parallelization based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67cf3a32-1bdc-800a-89c9-3ac153d999d4
def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id,
                                     bounds_str, output_dict, is_final, logger_worker,
                                     no_data_val=None):

    upload_tasks = []

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)

    file_info = f'{tile_id}__{bounds_str}'

    lu.print_and_log(f"Saving outputs in cluster for {bounds_str} in {tile_id}: {timestr()}", is_final, logger_worker)

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():

        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]
        full_s3_path = value[4]

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        # # Only prints if not a final run
        # # Disabled this because it prints sooooo many lines that it's annoying to scroll through
        # if not is_final:
        #     lu.print_and_log(f"Saving {key} for {bounds_str} in {tile_id} for {year_out}: {timestr()}", is_final, logger_worker)

        # Includes NoData value in output raster
        if no_data_val is not None:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw',
                               tiled=True, blockxsize=4000, blockysize=4000, nodata=no_data_val) as dst:
                dst.write(data_array, 1)

        # No NoData value in output raster
        else:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw',
                               tiled=True, blockxsize=4000, blockysize=4000) as dst:
                dst.write(data_array, 1)

        upload_tasks.append((f"/tmp/{file_name}", "gfw2-data", f"{full_s3_path}{file_name}"))

    return upload_tasks


# Returns list of rasters in an s3 folder and returns their names as a list (but not full paths)
def list_raster_names_in_s3_folder(full_in_folder):

    cmd = ['aws', 's3', 'ls', full_in_folder]
    s3_contents_bytes = subprocess.check_output(cmd)

    # Converts subprocess results to useful string
    s3_contents_str = s3_contents_bytes.decode('utf-8')
    s3_contents_list = s3_contents_str.splitlines()
    rasters = [line.split()[-1] for line in s3_contents_list]
    rasters = [i for i in rasters if "tif" in i]

    return rasters


# Returns list of rasters (full paths and names) in an s3 folder, and also returns the count of them
#Per https://chatgpt.com/share/e/67413a39-1b3c-800a-b582-72d1a8a17de1
def list_raster_full_paths_in_s3_folder_and_count(s3_path):
    """
    List all GeoTIFF files from a list of full S3 paths using boto3 and return the count.

    Args:
        s3_paths (list): List of S3 paths (e.g., "s3://bucket-name/prefix/").

    Returns:
        tuple: A tuple containing:
            - A flat list of GeoTIFF file paths.
            - The total count of GeoTIFF files.
    """

    # Initialize the S3 client
    s3_client = boto3.client('s3')
    geotiff_files = []

    try:
        # Parses bucket and prefix from the S3 path
        if s3_path.startswith("s3://"):
            path_parts = s3_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 path: {s3_path}")

        # Uses pagination to handle more than 1,000 objects (otherwise, limited to list of 1000 elements)
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                files = [obj['Key'] for obj in page['Contents']]
                geotiffs = [f"s3://{bucket_name}/{file}" for file in files if file.endswith(('.tif', '.tiff'))]
                geotiff_files.extend(geotiffs)

    except Exception as e:
        print(f"Error accessing {s3_path}: {e}")

    return geotiff_files, len(geotiff_files)


# Uploads a shapefile to s3
def upload_shp(in_folder, shp):

    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Uploading to {in_folder}{shp}: {timestr('time')}", True, logger_worker)

    shp_pattern = shp[:-4]

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(f"/tmp/{shp}", "gfw2-data", Key=f"{in_folder[cn.full_bucket_prefix_length:]}{shp}")
    s3_client.upload_file(f"/tmp/{shp_pattern}.dbf", "gfw2-data", Key=f"{in_folder[cn.full_bucket_prefix_length:]}{shp_pattern}.dbf")
    s3_client.upload_file(f"/tmp/{shp_pattern}.prj", "gfw2-data", Key=f"{in_folder[cn.full_bucket_prefix_length:]}{shp_pattern}.prj")
    s3_client.upload_file(f"/tmp/{shp_pattern}.shx", "gfw2-data", Key=f"{in_folder[cn.full_bucket_prefix_length:]}{shp_pattern}.shx")

    os.remove(f"/tmp/{shp}")
    os.remove(f"/tmp/{shp_pattern}.dbf")
    os.remove(f"/tmp/{shp_pattern}.prj")
    os.remove(f"/tmp/{shp_pattern}.shx")

    lu.print_and_log(f"Uploaded to {in_folder}{shp}: {timestr('time')}", True, logger_worker)

# Saves a data array locally as a raster and then uploads it to s3
def save_and_upload_raster_10x10(bounds, tile_length_pixels, tile_id,
                                 bounds_str, output_dict, is_final, logger_worker,
                                 no_data_val=None):

    upload_tasks = []

    transform = rasterio.transform.from_bounds(*bounds, width=tile_length_pixels, height=tile_length_pixels)

    lu.print_and_log(f"Saving output arrays to geotifs in cluster for {tile_id}: {timestr()}", is_final, logger_worker)

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():
        try:
            data_array = value[0]
            data_type = value[1]
            full_s3_path = value[4]

            #Does not upload the the raster if the array dimensions are not 40000x40000
            if data_array.shape != (cn.full_raster_dims, cn.full_raster_dims):
                lu.print_and_log( f"ERROR: Array shape {data_array.shape} for {key} does not match expected 40000x40000. Skipping.",is_final, logger_worker)
                continue

            if is_final:
                file_name = f"{tile_id}_{key}.tif"
            else:
                file_name = f"{tile_id}_{key}__{timestr()}.tif"

            # # Only prints if not a final run
            # # Disabled this because it prints sooooo many lines that it's annoying to scroll through
            # if not is_final:
            #     lu.print_and_log(f"Saving {key} for {bounds_str} in {tile_id} for {year_out}: {timestr()}", is_final, logger_worker)

            # Includes NoData value in output raster
            if no_data_val is not None:
                with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=tile_length_pixels,
                                   height=tile_length_pixels, count=1,
                                   dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw',
                                   tiled=True, blockxsize=4000, blockysize=4000, nodata=no_data_val) as dst:
                    dst.write(data_array, 1)

            # No NoData value in output raster
            else:
                with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=tile_length_pixels,
                                   height=tile_length_pixels, count=1,
                                   dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw',
                                   tiled=True, blockxsize=4000, blockysize=4000) as dst:
                    dst.write(data_array, 1)

            upload_tasks.append((f"/tmp/{file_name}", "gfw2-data", f"{full_s3_path}{file_name}"))

        except Exception as e:
            lu.print_and_log(f"ERROR saving {key}: {str(e)}", is_final, logger_worker)
            continue

    return upload_tasks


# Gets the name of the first file in a dictionary of dataset names and folders in s3.
# Returns dictionary of dataset names with the full path of the first file in the s3 folder.
# From https://chatgpt.com/share/e/9a7bf947-1c32-4898-ba6b-3b932a5220c1
def first_file_name_in_s3_folder(download_dict):

    s3_client = boto3.client("s3")

    # Initializes the dictionary to hold the first file paths
    first_tiles = {}

    # Iterates over the download_dict items
    for key, folder_path in download_dict.items():

        # Splits the path to get the directory part
        dir_path = os.path.dirname(folder_path)

        # Drops the s3://gfw2-data/ prefix and adds "/" to the end
        dir_path = dir_path[len(cn.full_bucket_prefix)+1:] + "/"

        # Lists metadata for everything in the bucket
        response = s3_client.list_objects_v2(Bucket=cn.short_bucket_prefix, Prefix=dir_path, Delimiter='/')

        # Checks if the folder contains any files
        if 'Contents' in response:
            # Filters the files to include only .tif files
            tif_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.tif')]

            # Check if any .tif files exist in the folder
            if tif_files:
                # Uses the first .tif file found
                first_tiles[key] = cn.full_bucket_prefix + "/" + tif_files[0]
            else:
                first_tiles[key] = None  # No .tif files found
                sys.exit(f"No tif files found in {key}, {folder_path}")
        else:
            first_tiles[key] = None  # No files found in the folder
            sys.exit(f"No files found in {key}, {folder_path}")

    return first_tiles


###################################################################################################
# LULUCF model utilities
###################################################################################################
# Time in Eastern US timezone as a string
def timestr(format="full"):

    # Define the Eastern Time timezone
    eastern = pytz.timezone('US/Eastern')

    # Get the current time in UTC and convert to Eastern Time
    eastern_time = datetime.now(eastern)

    # Format the time as a string
    if format == "time":
        return eastern_time.strftime("%H:%M:%S")
    else:
        return eastern_time.strftime("%Y%m%d_%H_%M_%S")

# Connects to a Coiled cluster of a specified name if the local flag isn't on.
# Does not create a Coiled cluster if the specified cluster name doesn't exist (contrary to default Coiled behavior).
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67fff45a-ec78-800a-83e1-8b3618a7e09a
def connect_to_Coiled_cluster(cluster_name, run_local, fallback_to_local_on_failure=True):

    # If local run flag is on, doesn't return a cluster or client
    if run_local:
        print("Running locally without Dask/Coiled.")
        return None, None, run_local

    # If no local run flag, it tries to attach to the named cluster
    try:
        # Gets info on all Coiled clusters (including terminated ones)
        all_clusters = coiled.list_clusters(workspace=cn.Coiled_workspace)

        # Iterates through clusters and identifies the running one of the correct name to connect to
        for cluster in all_clusters:
            if (cluster.get("name") == cluster_name) and (cluster.get("current_state", {}).get("state") in ['scaling', 'ready']):
                print(f"Connecting to running cluster '{cluster_name}'.")
                cluster = coiled.Cluster(name=cluster_name, workspace=cn.Coiled_workspace)
                client = Client(cluster)
                return cluster, client, run_local

        if fallback_to_local_on_failure:
            print(f"Cluster named {cluster_name} not found. Running locally.")
            return None, None, True
        else:
            raise RuntimeError(f"No running cluster named '{cluster_name}' found.")

    except Exception as e:
        if fallback_to_local_on_failure:
            print(f"Error while connecting to Coiled cluster: {e}\nRunning locally instead.")
            return None, None, True
        else:
            raise


# Chunk bounds as a string
def boundstr(bounds):
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


# Chunk length in pixels
def calc_chunk_length_pixels(bounds):
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


# Creates list of bounding boxes for chunks from a dataframe column structured as W_S_E_N.
# Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
def process_chunk_id(chunk_id):
    # Split by underscore
    bounding_box = list(map(float, chunk_id.split('_')))
    return bounding_box


# Maps GDAL data type to the appropriate string value
gdal_to_string_dtype_mapping = {
    gdal.GDT_Byte: 'Byte',
    gdal.GDT_UInt16: 'UInt16',
    gdal.GDT_Int16: 'Int16',
    gdal.GDT_UInt32: 'UInt32',
    gdal.GDT_Int32: 'Int32',
    gdal.GDT_Float32: 'Float32',
    gdal.GDT_Float64: 'Float64',
    'Int8': 'Int8',  # GDAL doesn't have int8, apparently. Outside Coiled, this converts it correctly.
    14: 'Int8'   # GDAL doesn't have int8, apparently. In Coiled, this converts it correctly.
}

# Maps GDAL data type to the appropriate string value
string_to_gdal_dtype_mapping = {
     'Byte': gdal.GDT_Byte,
     'UInt16': gdal.GDT_UInt16,
     'Int16': gdal.GDT_Int16,
     'UInt32': gdal.GDT_UInt32,
     'Int32': gdal.GDT_Int32,
     'Float32': gdal.GDT_Float32,
     'Float64': gdal.GDT_Float64
}

# Maps GDAL datatypes to numpy datatypes
def map_to_numpy_dtype(data_type):
    dtype_map = {
        'Float32': 'float32',
        'Float64': 'float64',
        'Byte': 'uint8',
        'Int32': 'int32',
        'UInt32': 'uint32',
        'Int16': 'int16',
        'UInt16': 'uint16',
        # Add more mappings as needed
    }
    return dtype_map.get(data_type, 'float32')  # Defaults to 'float32' if argument not found

# Gets the W, S, E, N bounds of a 10x10 degree tile
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


# Returns list of all chunk boundaries within a bounding box for chunks of a given size
def get_chunk_bounds_from_bounding_box(bounding_box, chunk_size):
    min_x = bounding_box[0]
    min_y = bounding_box[1]
    max_x = bounding_box[2]
    max_y = bounding_box[3]

    x, y = (min_x, min_y)
    chunks = []

    # Polygon Size
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


# Returns the encompassing tile_id string in the form YYN/S_XXXE/W based on a coordinate
def xy_to_tile_id(top_left_x, top_left_y):
    lat_ceil = math.ceil(top_left_y / 10.0) * 10
    lng_floor = math.floor(top_left_x / 10.0) * 10

    lng: str = f"{str(lng_floor).zfill(3)}E" if (lng_floor >= 0) else f"{str(-lng_floor).zfill(3)}W"
    lat: str = f"{str(lat_ceil).zfill(2)}N" if (lat_ceil >= 0) else f"{str(-lat_ceil).zfill(2)}S"

    return f"{lat}_{lng}"

# Interval info for model run.
# interval_year_diff is the difference between the start and end years of the interval, not the number of years in the interval.
# The difference between interval_length and interval_year_diff arises for 5-year intervals (e.g., 2016-2020), where there are 5 years in the interval
# but the difference between the start and end years is 4.
def get_interval_info(start_year, end_year, main_logger):

    if start_year == 2000 and end_year == 2020:
        interval_type = cn.intervals_five_years
        interval_length = [cn.five_year_interval_duration] * len(cn.interval_end_years_5_years)
        # interval_year_diff = [5, 5, 5, 5]  # Expected for 2000-2020
        interval_year_diff = [cn.five_year_interval_duration - 1] * len(cn.interval_end_years_5_years)  # -1 because the interval really starts one year after the end of the previous interval
        # interval_year_diff = [4, 4, 4, 4]  # Expected for 2000-2020
        output_years = cn.interval_end_years_5_years
    elif start_year == 2015 and end_year == cn.last_model_year_annual:
        interval_type = cn.intervals_annual
        interval_length = [1] * cn.end_year_count
        # interval_length = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # Expected for 2015-2024
        interval_year_diff = [1] * cn.end_year_count
        # interval_year_diff = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # Expected for 2015-2024
        output_years = cn.interval_end_years_annual
    elif start_year == 2000 and end_year == cn.last_model_year_annual:  # Hybrid model (2000-2024)
        interval_type = cn.intervals_hybrid
        interval_length = [cn.five_year_interval_duration] * len(cn.interval_end_years_5_years[:-1]) + [1] * cn.end_year_count
        # interval_length = [5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Expected for 2000-2024
        interval_year_diff = [cn.five_year_interval_duration - 1] * len(cn.interval_end_years_5_years[:-1]) + [1] * cn.end_year_count
        # interval_year_diff = [4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Expected for 2000-2024
        output_years = cn.interval_end_years_5_years[:-1] + cn.interval_end_years_annual
    else:
        main_logger.error("interval_type not valid")
        sys.exit(1)

    main_logger.info(f"Interval type: {interval_type}")
    main_logger.info(f"Interval end years/Output years: {output_years}")
    main_logger.info(f"Interval duration: {interval_length} years")
    main_logger.info(f"Interval year difference: {interval_year_diff} years")

    return interval_type, interval_year_diff, interval_length, output_years


# Creates the list of chunks to process given an approach: a bounding box or a shapefile attribute table
def create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger):

    # Makes list of chunks to analyze from the bounding box and chunk size (deg)
    # Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    if bounding_box and chunk_size_deg:

        chunk_size_pixels = int(cn.full_raster_dims * chunk_size_deg / 10)

        main_logger.info("Using bounding box and chunk size to determine chunks")
        main_logger.info(f"Chunk source: Bounding box {bounding_box} (W, S, E, N)")
        main_logger.info(f"Chunk size: {chunk_size_deg} degree, {chunk_size_pixels} pixels")
        chunk_list = get_chunk_bounds_from_bounding_box(bounding_box, chunk_size_deg)


    # Makes list of chunks to analyze from an attribute table of a shapefile of 1x1 degree chunks.
    # Attribute table column must be formatted as W_S_E_N.
    # Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    elif chunk_shapefile_uri:

        chunk_size_pixels = int(cn.full_raster_dims * 1/10)

        main_logger.info("Using chunk list shapefile (and optional number of test chunks) to determine 1x1 deg chunks")
        main_logger.info(f"Chunk source: 1x1 degree tile index shapefile {chunk_shapefile_uri}")
        main_logger.info(f"Chunk size: 1 degree, {chunk_size_pixels} pixels")

        # gdf = gpd.read_file(cn.fishnet_s3_uri)  # Reads shapefile attribute table
        fishnet_1x1_chunk_id_df = fishnet_iso_df[['chunk_id']]  # Creates dataframe

        # If argument for number of chunks in shapefile is supplied, limit to that
        if first_chunks:
            fishnet_1x1_chunk_id_df = fishnet_1x1_chunk_id_df[:first_chunks]

        # Converts dataframe column of chunk bounds to nested list
        # Per https://chatgpt.com/share/e/674747ee-d588-800a-995c-1f897a8ace31
        chunk_list = fishnet_1x1_chunk_id_df['chunk_id'].apply(process_chunk_id).tolist()

    else:
        main_logger.info("Chunk list cannot be determined")
        sys.exit()

    return chunk_list, chunk_size_pixels


# Calculates the elapsed time for a stage
def stage_duration(start_time_str, end_time_str, stage, logger, format="full"):

    if format == "time":
        logger.info(f"Stage {stage} ended at: {timestr('time')}")
    else:
        logger.info(f"Stage {stage} ended at: {end_time_str}")

    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
    end_time = datetime.strptime(end_time_str, "%Y%m%d_%H_%M_%S")

    logger.info(f"Elapsed time for {stage}: {end_time - start_time}" + "\n")


# Lazily opens tile within provided bounds (i.e. one chunk) and returns as a numpy array.
# If it can't open the uri for the chunk (tile does not exist), it creates a numpy array of all 0s
# of the correct datatype for that input.
# The returned chunk needs to have the correct datatype because it'll eventually be used in a
# numba function, which is very particular about datatypes.
# For example, a dataset that's float32 can't have NoData chunks that are uint8 because
# the Numba functions won't be able to handle that (since they're so particular about datatypes).
# So, that is addressed here through setting the array of 0s to the datatype of the dataset.
# Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages
def get_tile_dataset_rio(uri, bounds, chunk_length_pixels, logger_worker, data_type):

    bounds_str = boundstr(bounds)
    numpy_dtype = map_to_numpy_dtype(data_type)
    expected_shape = (chunk_length_pixels, chunk_length_pixels)

    # Number of retries for submitting requests to s3
    MAX_RETRIES = 11

    # Determines if the uri being accessed is from OpenGeoHub (because it needs special request staggering)
    is_OGH_data = urlparse(uri).netloc.endswith("s3.opengeohub.org")

    # If the uri exists, the relevant window is opened and returned and returned as an array.
    # Note that this chunk could still just have NoData values, which would be downloaded.
    # If the uri exists but the raster just doesn't extend there (e.g., far north), the array has to be padded to
    # reach the expected size.
    # Retries accessing the raster specified number of times in case too many requests to s3 are being made.
    # If too many requests to s3 are being made, the script terminates for safety.
    # https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68c3235e-a590-832d-bfdc-c1531416c311
    for attempt in range(MAX_RETRIES):
        try:

            # Accessing OGH data in a burst also causes fatal errors.
            # This specifically staggers and slows down requests for OGH data,
            # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69446f3d-0fbc-832a-9629-ae5469adeae3
            if is_OGH_data and attempt == 0:
                time.sleep(random.uniform(0.0, 0.2))

            if is_OGH_data and attempt > 0:
                base = 1.0
                cap = 30.0
                sleep_time = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 1.0)
                lu.print_and_log(f"Accessing OGH data from {uri} on retry {attempt}. Sleeping for {sleep_time:.2f}s...: {timestr()}",False, logger_worker)
                time.sleep(sleep_time)

            # Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
            # The more files in an s3 folder, the longer it takes to access them without this environment variable.
            # It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
            # Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
            with rasterio.Env(aws_session, AWS_REQUEST_PAYER='requester', GDAL_DISABLE_READDIR_ON_OPEN='TRUE'):
                with rasterio.open(uri) as ds:
                    window = rasterio.windows.from_bounds(*bounds, ds.transform)
                    data = ds.read(1, window=window)

                    # Checks if array shape is not what we expect (full chunk size) and pads the array with np.NaN if the array is incomplete.
                    # Was using 0s before but 0 has meaning and np.NaN doesn't.
                    # Per https://chatgpt.com/c/67dcb99b-edb8-800a-abd8-f718de76043c
                    if data.shape != expected_shape:
                        original_shape = data.shape

                        # Necessary for OGH vegetation height data. Otherwise, it errors because of a type issue.
                        # This is outside the OGH veg height extent.
                        # Per Claude session 'Quick task completion analysis'
                        fill_value = np.nan if np.issubdtype(np.dtype(numpy_dtype), np.floating) else 0
                        padded_data = np.full(expected_shape, fill_value, dtype=numpy_dtype)

                        # Calculates offset in pixels relative to chunk
                        row_offset = max(0, int(window.row_off))
                        col_offset = max(0, int(window.col_off))
                        rows, cols = data.shape
                        end_row = min(row_offset + rows, chunk_length_pixels)
                        end_col = min(col_offset + cols, chunk_length_pixels)

                        # Fills the correct slice of the padded array
                        padded_data[row_offset:end_row, col_offset:end_col] = data[:end_row - row_offset, :end_col - col_offset]

                        data = padded_data
                        status = f"padded {bounds_str} for {uri} from {original_shape} to {expected_shape} with data_type {data_type}/{numpy_dtype}"

                    else:
                        status = f"success- {bounds_str} for {uri} complete, no padding needed"

            # If previous attempts to download failed, log here that this attempt succeeded
            if attempt > 1:
                lu.print_and_log(f"Succeeded downloading {uri} on attempt {attempt}: {timestr()}",False, logger_worker)

            return data, status

        except rasterio.errors.WindowError as e:
            # WindowError ("Bounds and transform are inconsistent") can occur transiently when a
            # GeoTIFF header is read under high S3 concurrency and GDAL gets a garbled transform back.
            # Always retry -- if the file has a genuine data problem this will exhaust retries and raise.
            # Per Claude session 'Failed Coiled tasks diagnosis'
            if attempt < MAX_RETRIES - 1:
                sleep_time = min(30.0, 1.0 * (2 ** attempt)) + random.uniform(0.0, 1.0)
                lu.print_and_log(f"WindowError for {uri} on attempt {attempt} ({e}). Retrying in {sleep_time:.2f}s...: {timestr()}", False, logger_worker)
                time.sleep(sleep_time)
                continue
            else:
                raise RuntimeError(f"WindowError persisted after {MAX_RETRIES} retries for {uri}: {timestr()}") from e

        # From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68c3235e-a590-832d-bfdc-c1531416c311
        except rasterio.errors.RasterioIOError as e:
            err_msg = str(e)

            # Retryable errors-- these mean that the input exists but it's not being successfully accessed,
            # perhaps because of too many simultaneous requests to s3.
            # List of keywords for attempting retries is just from encountering various issues over time and including them here
            retryable_keywords = [
                # existing S3-ish/transient
                "SlowDown", "Please reduce", "503", "Read failed", "previous exception", "internal error",
                "not recognized",
                # HTTP/CURL-ish transient
                # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69446f3d-0fbc-832a-9629-ae5469adeae3
                "CURL", "Recv failure", "Connection reset by peer", "Connection timed out", "Timeout",
                "Operation timed out",
                "Could not resolve host", "Failed to connect", "Empty reply from server", "TLS", "SSL"
            ]

            if any(keyword in err_msg for keyword in retryable_keywords):
                if attempt < MAX_RETRIES - 1:
                    # Additional jitter for accessing OpenGeoHub because I was hitting it too quickly,
                    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69446f3d-0fbc-832a-9629-ae5469adeae3
                    base = 1.0
                    cap = 30.0
                    sleep_time = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 1.0)
                    lu.print_and_log(f"Retryable S3 error '{err_msg}' for {uri} on attempt {attempt}. Retrying in {sleep_time:.2f}s...: {timestr()}", False, logger_worker)
                    time.sleep(sleep_time)
                    continue
                else:
                    # Too many retries → fail hard
                    raise RuntimeError(
                        f"Retryable S3 error ('{err_msg}') persisted after {MAX_RETRIES} retries for {uri}: {timestr()}"
                    )

            # Non-retryable: missing key or other rasterio I/O issue
            else:
                data = np.full(expected_shape, 0, dtype=numpy_dtype)
                status = f"Can't access dataset {uri} for {bounds_str}. Returning array of all 0s: {err_msg}"
                return data, status


# Prepares list of chunks to download.
# Chunks are defined by a bounding box.
# Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages
def prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_final, logger_worker, stagger_download):

    # Only staggers downloads for scripts that require it because they're hitting individual s3 folders a lot, e.g., summative outputs.
    # Not all scripts hit individual s3 folders beyond s3's request limit.
    if stagger_download == True:
        # Staggers worker startup so that not all workers are requesting data from s3 at the same time, to prevent hitting request limit
        startup_delay = random.uniform(0, 3)
        time.sleep(startup_delay)

    futures = {}

    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])

    # Submits requests to S3 for input chunks but doesn't actually download them yet.
    # This queueing of the requests before downloading then speeds up the downloading.
    # Approach is to download all the input chunks up front for every year to make downloading more efficient, even though it means storing more upfront.
    # Without setting max_workers, it uses 1 per vCPU + 1 more (according to ChatGPT)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        lu.print_and_log(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger_worker)

        for key, value in download_dict.items():
            # print(key, value)

            # When the values are a list with just the file to download, without the datatype
            if len(value)==1:
                future = executor.submit(get_tile_dataset_rio, value[0], bounds, chunk_length_pixels, logger_worker, 'float32')

            # When the values are a list with the file to download and the datatype
            elif len(value)==2:
                future = executor.submit(get_tile_dataset_rio, value[0], bounds, chunk_length_pixels, logger_worker, value[1])

            else:
                sys.exit("Unexpected number of parameters in download dictionary")

            futures[future] = key  # Stores Future objects (data and status) as keys, layer names as values

            if stagger_download:
                # Staggers submissions to avoid burst traffic to S3
                time.sleep(random.uniform(0.05, 0.5))

    return futures


# Checks if tiles exist at all
def check_for_tile(download_dict, is_final, logger):

    # Configures S3 client with increased retries; retries can max out for global analyses
    s3_config = Config(
        retries={
            'max_attempts': 10,  # Increases the number of retry attempts
            'mode': 'standard'
        }
    )
    s3_client = boto3.client("s3", config=s3_config)  # Uses the configured client with more retries

    i = 0

    while i < len(list(download_dict.values())):

        # Tile path and name in s3, without s3://gfw2-data/ (hence, [len(cn.full_bucket_prefix)+1:])
        # [0] is to select the s3 path element of the list in the dictionary value (as opposed to the datatype, which is [1]
        s3_key = list(download_dict.values())[i][0][len(cn.full_bucket_prefix)+1:]

        tile_id = re.findall(cn.tile_id_pattern, list(download_dict.values())[i][0])[0]  # Extracts the tile_id from the s3 path

        # Breaks the loop if the tile exists. No need to keep checking other tiles because one exists.
        try:
            s3_client.head_object(Bucket='gfw2-data', Key=s3_key)

            lu.print_and_log(f"Tile id {tile_id} exists for some inputs. Proceeding: {timestr()} ", is_final, logger)

            return True
        except:
            pass

        i += 1

    lu.print_and_log(f"Tile id {tile_id} does not exist. Skipped chunk: {timestr()}", is_final, logger)

    return False


# Turns a list of basic output directory names into a list of fully specified directories based on output chunk size, run date, model type, and output years
def create_output_dir_name_list(dir_list, interval_type, start_year, chunk_size_pixels,
                                model_type, model_version, model_path_description, output_years, interval_duration,
                                run_date, include_full_period_totals, pixel_meaning=None):

    # List of directories for outputs
    output_full_dirs = []

    # Replaces placeholders in paths with values specific to the run
    dir_list = [path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}") for path in dir_list]
    dir_list = [path.replace("MODEL_INTERVAL_TYPE", interval_type) for path in dir_list]
    dir_list = [path.replace("RUN_DATE", run_date) for path in dir_list]

    # Replaces the chunk_size part of the path with global if this is a global aggregation
    if chunk_size_pixels == "global":
        dir_list = [path.replace("CHUNK_SIZE_pixels", chunk_size_pixels) for path in dir_list]
    else:
        dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in dir_list]

    # # Any entry that covers the entire model period, from start year to end of last interval
    # dir_list = [path.replace("FULL_MODEL", f"{start_year}_{output_years[-1]}") for path in dir_list]

    # Replaces the pixel meaning placeholder with the per-ha or per-pixel meanings.
    # Pixel meanings are formulated slightly differently depending on whether the output is C density or flux.
    for i, basic_output in enumerate(dir_list):
        if "density" in basic_output:  # Changes C density outputs
            if pixel_meaning == "per_ha":
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning)
            elif pixel_meaning == cn.C_density_aggreg_pixel_meaning:
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.C_density_aggreg_pixel_meaning)
            else:
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.C_per_pixel_pixel_meaning)
        else:  # Changes flux outputs and removal factors
            if pixel_meaning == "per_ha":
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.flux_density_pixel_meaning)
            elif pixel_meaning == cn.flux_aggreg_pixel_meaning:
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.flux_aggreg_pixel_meaning)
            else:
                updated_path = basic_output.replace("PER_HA_OR_PIXEL", cn.flux_per_pixel_pixel_meaning)

        # Updates dir_list
        dir_list[i] = updated_path

    # Iterates through the list of core output directories and adds the correct output years (stocks) or year ranges (fluxes) to each.
    for basic_output in dir_list:

        # Sample output directory (given year) for each set of outputs, that will be used to create the path for the full model period output
        sample_output_dir = None

        for count, output_year in enumerate(output_years):

            # For outputs that are a specific year (stocks)
            if "YEAR" in basic_output:
                output_dir = basic_output.replace('YEAR', str(output_year))
            # For outputs that cover the start of the model to the end of the current interval
            elif "RUNSTART_END" in basic_output:
                output_dir = basic_output.replace('RUNSTART_END', f"{str(start_year)}_{str(output_year)}")
            # For outputs that cover an interval (fluxes)
            else:
                if interval_type == cn.intervals_five_years:
                    output_dir = basic_output.replace('START_END', str(output_year))
                elif interval_type == cn.intervals_annual:
                    output_dir = basic_output.replace('START_END', str(output_year))
                else:  # Hybrid model (2000-2024)
                    output_dir = basic_output.replace('START_END', str(output_year))

            sample_output_dir = basic_output
            output_full_dirs.append(output_dir)

        # Creates the full model period path (2015-ENDYEAR) and adds it to the list of paths.
        # Only used for select outputs.
        if include_full_period_totals:
            full_model_period_dir = sample_output_dir.replace('START_END', f"{cn.first_model_year_annual}_{cn.last_model_year_annual}")
            output_full_dirs.append(full_model_period_dir)

    return output_full_dirs


# Checks if a geotif has data in it.
# Goes window by window so that the entire raster isn't read into memory.
# https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67b8c1d9-c850-800a-8f50-5f0c8edeeb40
def check_geotiff_has_data(file_path, chunk_size=1000):

    with rasterio.open(file_path) as src:
        nodata_value = src.nodata

        # Reads the raster in chunks (windows)
        for j in range(0, src.height, chunk_size):
            for i in range(0, src.width, chunk_size):
                window = rasterio.windows.Window(i, j, min(chunk_size, src.width - i), min(chunk_size, src.height - j))
                band = src.read(1, window=window)  # Reads only the chunk

                # Checks for valid data
                if nodata_value is not None:
                    if np.any(band != nodata_value):
                        return True
                else:
                    if np.any(~np.isnan(band)):
                        return True

    print(f"{file_path} is empty or contains only NoData values.")
    return False


# Checks whether a chunk has data in it.
# There are two options for how to assess if a chunk has data (any_or_all argument): if any assessed input has data, or if all assessed inputs have data.
# Any: To have data, a chunk have have at least one of the assessed inputs (layers).
# All: To have data, a chunk must have all necessary inputs (layers).
# If one or more necessary input is missing, the loop is terminated and the chunk ultimately skipped.
def check_chunk_for_data(required_layers, bounds_str, tile_id, any_or_all, is_final, logger):
    # Checks if ANY of the assessed inputs are present
    if any_or_all == "any":

        i = 0

        while i < len(list(required_layers.values())):

            # Checks if all the pixels have the nodata value.
            # Assume no data in the chunk if the min and max values are the same for EVERY input raster.
            # Can't use np.all because it doesn't work in chunks that are mostly water; says nodata in chunk even if there is land
            # So, instead compare np.min and np.max.
            min = np.min(list(required_layers.values())[i])

            # Breaks the loop if there is data in the chunk.
            # Don't need to keep checking chunk for data because the condition has been met
            # (at least one chunk has data).
            # The one print statement regardless of whether the model is full-scale or not.
            if min != None:  # if min exists, there must be data in the chunk
                logger.info(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                print(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                return True

            i += 1

        # Printed regardless of whether or not the model is full-scale
        logger.info(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        print(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        return False

    # Checks if ALL of the assessed inputs are present
    elif any_or_all == "all":

        # Iterates through all the required input layers
        for i, (key, value) in enumerate(required_layers.items()):

            # Assume no data in the chunk if the min and max values are the same for EVERY input raster.
            # Can't use np.all because it doesn't work in chunks that are mostly water; says nodata in chunk even if there is land
            # So, instead compare np.min and np.max.
            min = np.min(value)
            max = np.max(value)

            # Breaks the loop if min and max are the same, i.e. chunk doesn't exist.
            # We assume that if min and max are the same, there are no valid pixels
            # Don't need to keep checking chunk for data because at least one input doesn't have data,
            # so not ALL of the inputs exist
            if min == max:
                # Printed regardless of whether or not the model is full-scale
                logger.info(f"flm: Chunk {bounds_str} does not exist for {key}. Skipped chunk: {timestr()}")  # The one print statement regardless of whether the model is full-scale or not
                print(f"flm: Chunk {bounds_str} does not exist for {key}. Skipped chunk: {timestr()}")
                return False

        # If all required inputs are checked (for loop is completed), ALL inputs exist.
        # Printed regardless of whether or not the model is full-scale.
        logger.info(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")  # The one print statement regardless of whether the model is full-scale or not
        print(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        return True

    else:

        raise Exception("any_or_all argument not valid")


# Makes a shapefile of the footprints of rasters in a folder, for checking geographical completeness of rasters
def make_tile_footprint_shp(input_dict, no_upload):

    logger_worker = lu.setup_logging_worker()

    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    # Task properties
    lu.print_and_log(f"flm: Making tile index shapefile for: {in_folder}: {timestr()}", True, logger_worker)

    # Folder including s3 key
    s3_in_folder = in_folder
    vsis3_in_folder = f'/vsis3/{in_folder[5:]}' #[5] drops the s3:// at the front

    # List of all the filenames in the folder
    filenames = list_raster_names_in_s3_folder(s3_in_folder)

    # List of the tile paths in the folder
    tile_paths = [vsis3_in_folder + filename for filename in filenames]

    file_paths_txt = f's3_paths_{pattern}.txt'

    with open(f"/tmp/{file_paths_txt}", 'w') as file:
        for item in tile_paths:
            file.write(item + '\n')

    # Output shapefile name
    shp = f"raster_footprints_{pattern}.shp"

    cmd = ["gdaltindex", "-t_srs", "EPSG:4326", f"/tmp/{shp}", "--optfile", f"/tmp/{file_paths_txt}"]
    subprocess.check_call(cmd)

    # Uploads shapefile to s3 if upload not disabled
    if not no_upload:
        upload_shp(s3_in_folder, shp)

    os.remove(f"/tmp/{file_paths_txt}")

    return(f"Index shapefile for {pattern} completed: {timestr()}")


# Creates a list of 10x10 deg tiles to create, where the list is a list of dictionaries of the form
# [{'s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr/2000_2005/4000_pixels/20241121/': ['00N_000E__gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr_2000_2005.tif']},
# {'s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr/2000_2005/4000_pixels/20241121/': ['00N_010E__gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr_2000_2005.tif']}, ... ]
# The keys are s3 destination foldes and the values are the output 10x10 deg file names
def create_list_for_aggregation(s3_in_folders, main_logger):

    list_of_s3_names_total = []  # Final list of dictionaries of input s3 paths and output aggregated 10x10 raster names

    # Iterates through all the input s3 folders
    for i, s3_in_folder in enumerate(s3_in_folders):

        try:
            main_logger.info(f"Listing files in folder {i} out of {len(s3_in_folders)}: {s3_in_folder}")

            simple_output_file_names = []  # List of output aggregated output 10x10 rasters

            # Raw filenames in an input folder, e.g., ['00N_000E__6_-2_8_0__IPCC_classes_2020.tif', '00N_000E__6_-4_8_-2__IPCC_classes_2020.tif',...]
            filenames = list_raster_names_in_s3_folder(s3_in_folder)

            # Iterates through all the files in a folder and converts them to the output names.
            # Essentially [tile_id]__[pattern].tif. Drops the chunk bounds from the middle.
            for filename in filenames:
                result = re.sub(cn.small_chunk_pattern, '__', filename)
                simple_output_file_names.append(result)  # New list of simplified file names used for 10x10 degree outputs

            # Removes duplicate simplified file names.
            # There are duplicates because each 10x10 output raster has many constituent chunks, each of which have the same aggregated, final name
            # e.g., ['00N_000E__IPCC_classes_2020.tif', '00N_010E__IPCC_classes_2020.tif', ...]
            simple_output_file_names = np.unique(simple_output_file_names).tolist()

            # Makes nested lists of the file names. Nested for next step.
            # e.g., [['00N_110E__AGC_density_MgC_ha_2000.tif']]
            simple_output_file_names = [[item] for item in simple_output_file_names]

            # Makes a list of dictionaries, where the key is the input s3 path and the value is the output aggregated name
            # e.g., [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__AGC_density_MgC_ha_2000.tif']}]
            list_of_s3_name_dicts = [{key: value} for value in simple_output_file_names for key in [s3_in_folder]]

            # Adds the dictionary of s3 paths and output names for this folder to the list for all folders
            list_of_s3_names_total.append(list_of_s3_name_dicts)

        except Exception as e:
            main_logger.error(f"Failed processing folder {s3_in_folder} due to error: {e}")
            continue

    # Combines all the lists from individual output folders into a single list
    # Now it's:
    # [{'s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr/2000_2005/4000_pixels/20241121/': ['00N_000E__gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr_2000_2005.tif']},
    # {'s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr/2000_2005/4000_pixels/20241121/': ['00N_010E__gross_emissions_all_C_pools_all_gases__MgCO2_ha_yr_2000_2005.tif']}, ... ]
    list_of_s3_names_total = flatten_list(list_of_s3_names_total)

    main_logger.info(f"There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders.")

    return list_of_s3_names_total


# Flattens a nested list
def flatten_list(nested_list):
    return [x for xs in nested_list for x in xs]


# Merges rasters that are <10x10 degrees into 10x10 degree rasters in the standard grid.
# Approach is to merge rasters with gdal.Warp and then upload them to s3.
# Commented out COG creation; it just outputs basic geotifs for now.
def merge_small_tiles_gdal(s3_name_dict, is_final, no_upload, output_dir=None, stat_type=None):

    process = psutil.Process(os.getpid())

    chunk_start_time = time.time()

    # Retry parameters in case of failure the first time
    max_retries = 3
    retry_delay = 5  # seconds between retries

    ### Part 1: Merges 1x1 deg rasters to 10x10 deg

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call for uploading to work

    logger_worker = lu.setup_logging_worker()

    in_folder = list(s3_name_dict.keys())[0]  # The input s3 folder for the small rasters
    out_file_name = list(s3_name_dict.values())[0][0]  # The output file name for the combined rasters

    s3_in_folder = in_folder  # The input s3 folder with s3:// prepended
    vsis3_in_folder = f'/vsis3/{in_folder[5:]}'  # The input s3 folder with /vsis3/ prepended

    # Lists all the rasters in the specified s3 folder
    filenames = list_raster_names_in_s3_folder(s3_in_folder)

    # Gets the tile_id from the output file name in the standard format
    tile_id = out_file_name[:8]

    # Limits the input rasters to the specified tile_id (the relevant 10x10 area)
    filenames_in_focus_area = [i for i in filenames if tile_id in i]

    # Lists the tile paths for the relevant rasters
    tile_paths = [vsis3_in_folder + filename for filename in filenames_in_focus_area]

    lu.print_and_log(f"flm: Merging small rasters in {tile_id} in {vsis3_in_folder}: {timestr()}", False, logger_worker)

    # Names the output folder. Same as the input folder but with the dimensions in pixels replaced
    out_folder = re.sub(r'\d+_pixels', f'{cn.full_raster_dims}_pixels', in_folder)

    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)

    # Dynamically sets the datatype for the merged raster based on the input rasters (courtesy of https://chatgpt.com/share/e/a91c4c98-b2b1-4680-a4a7-453f1a878052)
    # Determines the data type of the first raster.
    # Retries in case of failure (likely because of too many simultaneous requests to s3)
    first_raster_path = tile_paths[0]
    for attempt in range(1, max_retries + 1):
        try:
            startup_delay = random.uniform(0, 1.5)  # Slight delay before s3 is accessed so that request quota isn't exceeded
            time.sleep(startup_delay)
            ds = gdal.Open(first_raster_path)
            raster_datatype = ds.GetRasterBand(1).DataType
            raster_nodata_value = ds.GetRasterBand(1).GetNoDataValue()
            if raster_nodata_value == None:  # In case no NoData value is assigned
                raster_nodata_value = 0
            ds = None
        except RuntimeError as e:
            lu.print_and_log(f"Error accessing {first_raster_path} for data type extraction (attempt {attempt}/{max_retries}): {e}: {timestr()}", False, logger_worker)
            if attempt < max_retries:
                lu.print_and_log(f"Retrying accessing {first_raster_path} for data type extraction (attempt {attempt}/{max_retries}): {timestr()}", False, logger_worker)
                time.sleep(retry_delay)
            else:
                return f"Error: failure accessing {first_raster_path} after {max_retries} attempts: {timestr()}"

    # Defaults to Float32 if not found
    dtype_str = gdal_to_string_dtype_mapping.get(raster_datatype, 'Float32')

    # Merges the rasters (courtesy of ChatGPT: https://chatgpt.com/share/e/13158ebb-dd0a-41d8-8dfb-9ee12e4c804e)
    # This is the only system I found that maintains the extent of all the constituent rasters and doesn't change their resolution or pixel size or shift them.
    # I also tried various gdal_translate, build_vrt, and numpy padding approaches, none of which worked in all cases.
    merged_file = f"/tmp/merged_non_COG_{out_file_name}"

    merge_command = [
        'gdal_merge.py',
        '-o', merged_file,
        '-of', 'GTiff',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'TILED=YES', # If not included, the size of the merged small rasters can be many times their sum. Answer at https://gis.stackexchange.com/a/258215
        '-co', 'BLOCKXSIZE=400',  # Internal tiling
        '-co', 'BLOCKYSIZE=400',  # Internal tiling
        '-ul_lr', str(min_x), str(max_y), str(max_x), str(min_y),
        '-ot', dtype_str,
        '-a_nodata', str(raster_nodata_value)
    ]

    # Add the input tile paths
    merge_command.extend(tile_paths)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.check_call(merge_command)
            lu.print_and_log(f"Successfully merged rasters into {merged_file} on attempt {attempt}: {timestr()}", is_final, logger_worker)
            lu.print_and_log(f"After creating geotif for {merged_file}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
            break  # exit loop if successful
        except subprocess.CalledProcessError as e:
            lu.print_and_log(f"Error merging rasters (attempt {attempt}/{max_retries}): {e}: {timestr()}", False, logger_worker)
            if attempt < max_retries:
                lu.print_and_log(f"Retrying {merged_file} for attempt {attempt}: {timestr()}", False, logger_worker)
                time.sleep(retry_delay)
            else:
                return f"Error: failure merging {s3_name_dict} after {max_retries} attempts: {timestr()}"

    chunk_non_cog_end_time = time.time()
    lu.print_and_log(f"Merging {merged_file} took {round(chunk_non_cog_end_time - chunk_start_time)} seconds: {timestr()}", False, logger_worker)

    # ### Part 2: Converts geotifs to COGs
    #
    # # Convert to Cloud Optimized GeoTIFF
    # # https://gfw.atlassian.net/wiki/spaces/LCL/pages/1918238725/STAC-API+pre-flight+checklist
    # merged_cog_file = f"/tmp/merged_cog_{out_file_name}"
    # translate_command = [
    #     'gdal_translate',
    #     merged_file,
    #     merged_cog_file,
    #     '-of', 'COG',
    #     '-co', 'COMPRESS=DEFLATE',
    #     '-co', 'PREDICTOR=2',
    #     '-co', 'BIGTIFF=IF_SAFER',
    #     '-co', 'OVERVIEW_RESAMPLING=average'
    # ]
    #
    # try:
    #     subprocess.check_call(translate_command)
    #     lu.print_and_log(f"Successfully created COG: {merged_cog_file}: {timestr()}", is_final, logger_worker)
    #     lu.print_and_log(f"After creating COG for {merged_cog_file}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    # except subprocess.CalledProcessError as e:
    #     lu.print_and_log(f"Error converting to COG: {e}: {timestr()}", False, logger_worker)
    #     return f"failure converting to COG for {s3_name_dict}"
    #
    # chunk_cog_end_time = time.time()
    # lu.print_and_log(f"Through COG creation for {merged_cog_file} took {round(chunk_cog_end_time - chunk_start_time)} seconds: {timestr()}", False, logger_worker)

    ### Part 3: Counts non-No Data pixels in 10x10 raster (for comparison with summed 1x1 rasters)

    # Computes valid pixel count in the output 10x10 raster for comparison with the sum of the constituent 1x1s
    lu.print_and_log(f"Counting pixels in {tile_id} for {out_file_name}: {timestr()}", is_final, logger_worker)

    # Computes count of valid pixels by reading raster in chunks (can't read full 10x10 into memory all at once
    try:
        # ds = gdal.Open(merged_cog_file)
        ds = gdal.Open(merged_file)
        if ds is not None:
            band = ds.GetRasterBand(1)
            valid_pixel_count = 0

            # Gets raster dimensions
            x_size = band.XSize
            y_size = band.YSize

            # Reads in chunks to avoid high memory usage
            block_size_x, block_size_y = band.GetBlockSize()

            for y in range(0, y_size, block_size_y):
                rows_to_read = min(block_size_y, y_size - y)
                for x in range(0, x_size, block_size_x):
                    cols_to_read = min(block_size_x, x_size - x)

                    # Reads only a portion of the raster at a time
                    block = band.ReadAsArray(x, y, cols_to_read, rows_to_read)

                    if block is not None:
                        valid_pixel_count += np.count_nonzero(block != raster_nodata_value)

            ds = None  # Closes dataset
        else:
            valid_pixel_count = -1  # Failed to open file
    except Exception as e:
        # lu.print_and_log(f"Error counting pixels for {merged_cog_file}: {e}", is_final, logger_worker)
        lu.print_and_log(f"Error counting pixels for {merged_file}: {e}", is_final, logger_worker)
        # print(f"Error counting pixels for {merged_cog_file}: {e}")
        print(f"Error counting pixels for {merged_file}: {e}")
        return f"failure counting pixels for {s3_name_dict}"

    # Gets the output file pattern and year/year_range
    out_pattern, year_range = strip_and_extract_years(out_file_name)

    # Most stats for the 10x10 aren't calculated.
    # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
    # Dictionary is in a list because it's necessary for chunk stats processing later.
    chunk_stats = [{
        'chunk_id': 'N/A',
        'tile_id': tile_id,
        'layer_name': out_file_name,
        'tile_name': out_file_name,
        'in_out': 'output_layer',
        'pattern': out_pattern,
        'years': year_range,
        'min_value': 'no data',
        'mean_value': 'no data',
        'max_value': 'no data',
        'count_value': valid_pixel_count,
        'sum_value': 'no data',
        'data_type': 'no data'
    }]

    ### Part 4: Uploads 10x10 to s3 using multipart uploading
    ### https://chatgpt.com/share/e/67d848cf-8b08-800a-b0e8-79a72c9eb49a.

    if no_upload == False:

        # For testing!!! Redirects outputs to a different version folder.
        # out_folder = out_folder.replace(cn.model_version_underscore, f"{cn.model_version_underscore}_small_GDAL_geotif_test")

        # Because boto3 does multipart uploading for files >100MB, this only adds multipart uploading for files
        # between part_size and 100MB.
        try:
            lu.print_and_log(f"Saving {out_file_name} to s3: {out_folder}{out_file_name}: {timestr()}", is_final, logger_worker)
            part_size = 20 * 1024 * 1024  # 20MB chunks

            # Starts multipart upload
            response = s3_client.create_multipart_upload(Bucket=cn.short_bucket_prefix, Key=f"{out_folder[cn.full_bucket_prefix_length:]}{out_file_name}")
            upload_id = response['UploadId']

            parts = []
            # with open(merged_cog_file, 'rb') as f:
            with open(merged_file, 'rb') as f:
                part_number = 1
                while chunk := f.read(part_size):
                    response = s3_client.upload_part(
                        Bucket=cn.short_bucket_prefix,
                        Key=f"{out_folder[cn.full_bucket_prefix_length:]}{out_file_name}",
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk
                    )
                    parts.append({'PartNumber': part_number, 'ETag': response['ETag']})
                    part_number += 1

            # Completes the multipart upload
            s3_client.complete_multipart_upload(
                Bucket=cn.short_bucket_prefix,
                Key=f"{out_folder[cn.full_bucket_prefix_length:]}{out_file_name}",
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            lu.print_and_log(f"Uploaded {out_file_name} to s3: {timestr()}", is_final, logger_worker)


        except Exception as e:
            lu.print_and_log(f"Error uploading file to s3: {e}: {timestr()}", is_final, logger_worker)
            print(f"Error uploading file to s3: {e}: {timestr()}")
            return f"failure uploading {s3_name_dict}"

    # Deletes the local merged raster
    os.remove(merged_file)
    # os.remove(merged_cog_file)

    chunk_end_time = time.time()
    # lu.print_and_log(f"Full processing for {merged_cog_file} took {round(chunk_end_time - chunk_start_time)} seconds: {timestr()}", False, logger_worker)
    lu.print_and_log(f"Full processing for {merged_file} took {round(chunk_end_time - chunk_start_time)} seconds: {timestr()}", False, logger_worker)

    return f"Success merging {s3_name_dict}", chunk_stats


# Creates numpy array of rates or ratios from a tab in an Excel spreadsheet, e.g., removal factors or carbon pool ratios.
# Tries to read from s3 first. If that doesn't work (because I'm in the office), it reads from my computer.
# Courtesy of ChatGPT: https://chatgpt.com/share/e/aff31681-c9a7-40fe-85c1-73a1cab62066
def convert_lookup_table_to_array(spreadsheet, sheet_name, fields_to_keep):

    try:
        # Try fetching the file from the S3 URL
        # print(f"Attempting to download file from URL: {spreadsheet}")
        response = requests.get(spreadsheet, timeout=10)
        response.raise_for_status()
        excel_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Failed to download file from S3. Falling back to local file. Error: {e}")
        # Define the local fallback path
        fallback_dir = r"/mnt/c/Users/David.Gibbs/OneDrive - World Resources Institute/Documents/Projects/AFOLU_flux_model__all_land_all_carbon/rate_ratio_lookup_tables"
        fallback_filename = os.path.basename(spreadsheet)
        fallback_path = os.path.join(fallback_dir, fallback_filename)

        if not os.path.exists(fallback_path):
            raise FileNotFoundError(f"Fallback file not found at {fallback_path}")

        print(f"Reading file from local path: {fallback_path}")
        excel_df = pd.read_excel(fallback_path, sheet_name=sheet_name)

    # Retains only the relevant columns
    filtered_data = excel_df[fields_to_keep]

    # Converts from dataframe to Numpy array
    filtered_array = filtered_data.to_numpy().astype(float)  # Need to convert Pandas dataframe to numpy array because Numba jit-decorated function can't use dataframes.
    filtered_array = filtered_array.astype(float)  # Convert from object dtype to float dtype-- necessary for numba to use it

    return filtered_array

# Creates arrays of 0s for any missing inputs and puts them in the corresponding typed dictionary
def complete_inputs(existing_input_list, typed_dict, datatype, chunk_length_pixels, bounds_str, tile_id, is_final, logger):
    for dataset_name in existing_input_list:
        if dataset_name not in typed_dict.keys():
            typed_dict[dataset_name] = np.full((chunk_length_pixels, chunk_length_pixels), 0, dtype=datatype)
            lu.print_and_log(f"Created {dataset_name} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
    return typed_dict


# Counts the number of successful and skipped chunks after processing
# Based on https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
def count_successful_chunks(chunk_list, is_final, main_logger, results):

    # Arrays of chunk stats and return messages
    all_stats = []
    return_messages = []

    # Initializes counters for different types of return messages
    success_count = 0
    skipping_chunk_count = 0
    error_chunk_count = 0
    other_message_count = 0

    # Processes the chunk stats and returned messages
    # Results are the messages from the chunks and chunk stats
    for result in results:
        try:  # Counts errored tasks correctly, per Claude session 'Quick task completion analysis'
            if isinstance(result, dict) and result.get("status") == "failed":
                error_chunk_count += 1
                continue
            return_message, chunk_stats = result
        except Exception as e:
            main_logger.error(f"Malformed result: {result} | Error: {e}")
            continue

        if "Success" in return_message:
            success_count += 1
        elif "Skipped chunk" in return_message:
            skipping_chunk_count += 1
        elif ("Error" in return_message) or ("failure" in return_message):
            error_chunk_count += 1
        else:
            other_message_count += 1

        if return_message:
            return_messages.append(return_message)

        # Ensures chunk_stats is a list. Needs to be a list for further processing.
        chunk_stats = chunk_stats if isinstance(chunk_stats, list) else [chunk_stats]

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)


    # Prints the returned messages if not a large (is_final) run
    if not is_final:
        for message in return_messages:
            main_logger.info(message)

    # Print the counts of successful and skipped chunks
    main_logger.info(f"Number of 'Success' chunks: {success_count}")
    main_logger.info(f"Number of 'Skipped' chunks: {skipping_chunk_count}")
    main_logger.info(f"Number of 'Error' chunks: {error_chunk_count}")
    main_logger.info(f"Number of 'Other message' chunks: {other_message_count}")

    # Doesn't compare the difference between submitted and processed chunks if it is reporting on
    # merging 1x1 deg rasters because calculating the difference is too complicated.
    if return_messages and "Success merging" not in return_messages[0]:
        main_logger.info(f"Difference between submitted chunks and processed chunks: {len(chunk_list) - (success_count + skipping_chunk_count + error_chunk_count + other_message_count)}")
    main_logger.info("\n")

    return success_count, all_stats


# Calculates stats for a chunk (numpy array), mostly using per hectare values
# but optionally summing per pixel values to get a chunk total.
# Also joins ISO from GADM to each entry.
# Stats calculations adapted from https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
# Joining iso adapted from https://chatgpt.com/share/e/6744de08-6b64-800a-b8c4-6a20833f7e3a
# def calculate_stats(array_per_ha, name, bounds_str, tile_id, in_out, fishnet_iso_df, array_per_pixel=None):
def calculate_stats(array_per_ha, name, bounds_str, tile_id, in_out, array_per_pixel=None):

    # Sums the per pixel totals if relevant
    if in_out == 'output_layer' and array_per_pixel is not None:
        sum_value = np.nansum(array_per_pixel)  # Need nansum because SOC timeseries uses NaN
    else:
        sum_value = 'N/A- input layer or no per-pixel array supplied'

    # Gets the output file pattern and year/year_range
    out_pattern, year_range = strip_and_extract_years(name)

    if array_per_ha is None or not np.any(array_per_ha):  # Checks if the array is None or empty
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'pattern': out_pattern,
            'years': year_range,
            'chunk_name': f'{tile_id}__{bounds_str}__{out_pattern}_{year_range}.tif',
            'tile_name': f'{tile_id}__{out_pattern}_{year_range}.tif',
            'in_out': in_out,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'no data',
            'sum_value': sum_value,
            'data_type': 'no data'
        }
    else:    # Only calculates stats if there is data in the array
        # Ignores NaN in chunk stats calculation. Otherwise, min, mean and max might not be calculated, even if count is.
        # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/698215a2-bbdc-8332-991d-ab2ff90bb0ef
        if np.isnan(array_per_ha).all():
            min_val = mean_val = max_val = 'nan'
            count_val = 0
        else:
            min_val = float(np.nanmin(array_per_ha))
            mean_val = float(np.nanmean(array_per_ha))
            max_val = float(np.nanmax(array_per_ha))
            # count_val = np.count_nonzero(~np.isnan(array_per_ha) & (array_per_ha != 0))  # Counts non-0 and non-NaN only
            count_val = int(np.count_nonzero(~np.isnan(array_per_ha)))  # Counts non-NaN only

        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'pattern': out_pattern,
            'years': year_range,
            'chunk_name': f'{tile_id}__{bounds_str}__{out_pattern}_{year_range}.tif',
            'tile_name': f'{tile_id}__{out_pattern}_{year_range}.tif',
            'in_out': in_out,
            'min_value': min_val,
            'mean_value': mean_val,
            'max_value': max_val,
            'count_value': count_val,
            'sum_value': sum_value,
            'data_type': array_per_ha.dtype.name
        }


# Calculates summary statistics for an IPCC land-use output raster at the chunk level. 
# Reports metadata, minimum/maximum values, dominant class, unique classes, and either pixel counts or area (ha) by class, depending on whether a pixel area raster is provided
def calculate_ipcc_stats(array, name, bounds_str, tile_id, in_out, pixel_area=None):

    out_pattern, year_range = strip_and_extract_years(name)

    base = {
        "chunk_id": bounds_str,
        "tile_id": tile_id,
        "layer_name": name,
        "pattern": out_pattern,
        "years": year_range,
        "chunk_name": f"{tile_id}__{bounds_str}__{out_pattern}_{year_range}.tif",
        "tile_name": f"{tile_id}__{out_pattern}_{year_range}.tif",
        "in_out": in_out,
    }

    if array is None or not np.any(array):
        return {
            **base,
            "min_value": "no data",
            "max_value": "no data",
            "count_value": "no data",
            "mode_value": "no data",
            "unique_values": "no data",
            "pixel_counts": "no data",
            "data_type": "no data",
        }

    valid = array != 0
    values = np.unique(array[valid])

    if pixel_area is not None:

        pixel_area_ha = pixel_area * cn.m2_to_ha

        flat_classes = array.ravel().astype(np.int32)
        flat_area = pixel_area_ha.ravel()

        area_sums = np.bincount( flat_classes, weights=flat_area)

        values = np.nonzero(area_sums)[0]
        values = values[values != 0]

        count_cols = { f"count_{int(v)}": float(area_sums[v]) for v in values}
        total_area = float(area_sums[values].sum())
        mode_value = int(values[np.argmax(area_sums[values])])

    else:

        _, counts = np.unique(array[valid], return_counts=True)

        count_cols = {f"count_{int(value)}": int(count) for value, count in zip(values, counts)}
        mode_value = int(values[np.argmax(counts)])
        total_area = int(np.sum(counts))

    return {
        **base,
        "min_value": int(np.min(values)),
        "max_value": int(np.max(values)),
        "count_value": total_area,
        "mode_value": mode_value,
        "unique_values": [int(v) for v in values],
        **count_cols,
        "data_type": array.dtype.name,
    }

# Makes sure that all columns in output chunk stats Pandas dataframe are indeed numeric
# From https://chatgpt.com/c/68751cbe-6888-800a-bf9d-3657b048a810
def sanitize_numeric_columns(df, numeric_cols):
    df = df.copy()  # Prevents SettingWithCopyWarning
    for col in numeric_cols:
        if col not in df.columns:
            continue

        # Safely coerces non-numeric to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# Calculates chunk-level stats for all inputs and outputs and saves to Excel spreadsheet.
# Calculates the min and max value for each input and output across all chunks.
# Calculates difference between pixel counts in all 1x1s in a 10x10 vs. the corresponding 10x10
# to make sure that aggregation of 1x1s didn't lose any data (difference should be 0).
# From https://chatgpt.com/share/e/67d5d68d-7168-800a-ada1-e42f8c3e9253
# Returns the local path to the saved chunk stats, whether parquet or Excel.
def compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger):

    ### Part 1: Organizes chunk stats for 1x1 degree chunks (inputs and outputs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    main_logger.info(f"Starting to aggregate and export tile stats: {timestr()}")

    # Converts accumulated 1x1 chunk statistics to a DataFrame
    df_all_1x1_stats = pd.DataFrame(all_1x1_stats)

    # Converts problematic non-numeric values to NaN
    df_all_1x1_stats['min_value'] = pd.to_numeric(df_all_1x1_stats['min_value'], errors='coerce')
    df_all_1x1_stats['max_value'] = pd.to_numeric(df_all_1x1_stats['max_value'], errors='coerce')

    # Sorts the DataFrame by 'in_out' and 'layer_name'
    main_logger.info(f"Sorting 1x1 tile stats by properties: {timestr()}")
    sorted_1x1_stats = df_all_1x1_stats.sort_values(by=['in_out', 'layer_name']).reset_index(drop=True)

    # Calculates the min and max values for each layer_name across all chunks
    main_logger.info(f"Calculating min and max values across all 1x1 chunks: {timestr()}")
    min_max_1x1_stats = df_all_1x1_stats.groupby('layer_name').agg(
        min_value=('min_value', 'min'),
        max_value=('max_value', 'max'),
        count=('layer_name', 'count')
    ).reset_index()

    # Reads the shapefile from S3 to extract "chunk_id" and "iso" fields
    # Based on https://chatgpt.com/share/e/6744de08-6b64-800a-b8c4-6a20833f7e3a
    gdf = gpd.read_file(chunk_shapefile_uri)

    # Creates a DataFrame with "chunk_id" and "iso" fields
    fishnet_shapefile_df = gdf[['chunk_id', 'iso']]

    # Merges the shapefile data with the statistics DataFrame
    main_logger.info(f"Merging country code to 1x1 chunk stats table: {timestr()}")
    merged_1x1_stats = sorted_1x1_stats.merge(fishnet_shapefile_df, on='chunk_id', how='left')

    # When iso isn't assigned, empty cells are filled.
    # iso is only assigned when the chunks are 1x1 deg (since that's what the fishnet uses)
    merged_1x1_stats['iso'] = merged_1x1_stats['iso'].fillna('no iso assigned')

    # There are so many chunks with so many inputs and outputs in a full model run that Excel can't handle all the rows
    # and they need to be split across multiple workbook tabs.

    # Separates input rows (in_out == 'input') and output rows (in_out == 'output' or anything from a zarr)
    main_logger.info(f"Separating 1x1 outputs into different tables: {timestr()}")
    input_1x1_rows = merged_1x1_stats[merged_1x1_stats['in_out'] == 'input_layer']
    output_1x1_rows = merged_1x1_stats[merged_1x1_stats['in_out'].isin(['output_layer', 'zarr_stats'])]

    # Groups inputs that are a timeseries so they can go in their own tab so that no tab is too many rows
    timeseries_input_layers = (f'{cn.burned_area_final_pattern}|'
                               f'{cn.forest_disturbance_layer_name}|'
                               f'{cn.vegetation_height_pattern}|'
                               f'{cn.land_cover_pattern}|'
                               f'{cn.mangrove_extent_processed_pattern}|'
                               f'{cn.GPW_MVH_pattern}')

    # Splits input rows based on whether they are a timeseries input
    annual_1x1_inputs = input_1x1_rows[input_1x1_rows['layer_name'].str.contains(timeseries_input_layers, case=False, na=False)]

    # Puts output rows that aren't a timeseries input in their own tab
    other_1x1_inputs = input_1x1_rows[~input_1x1_rows['layer_name'].str.contains(timeseries_input_layers, case=False, na=False)]

    # Splits output rows based on 'layer_name' containing 'flux', 'gross', or 'net'
    gross_flux_1x1_outputs = output_1x1_rows[output_1x1_rows['layer_name'].str.contains('gross', case=False, na=False)]
    net_flux_1x1_outputs = output_1x1_rows[output_1x1_rows['layer_name'].str.contains('net|flux', case=False, na=False)]

    # Puts output rows that don't contain 'flux|gross|net' in a separate tab
    other_1x1_outputs = output_1x1_rows[~output_1x1_rows['layer_name'].str.contains('flux|gross|net', case=False, na=False)]


    ### Part 2: Sums pixel counts for 1x1 degree outputs to their 10x10s, to make sure pixels aren't being lost in aggregation

    # Counts the number of pixels in 1x1 chunks within each 10x10 chunk
    main_logger.info(f"Calculating number of pixels in 1x1 chunk within each 10x10 chunk: {timestr()}")

    # Forces pixel counts to numeric if they're not
    output_1x1_rows.loc[:, 'count_value'] = pd.to_numeric(output_1x1_rows['count_value'], errors='coerce').fillna(0)

    # Groups by tile_id and layer_name, summing count_value and sum_value
    sum_1x1_to_10x10 = output_1x1_rows.groupby(['tile_id', 'layer_name']).agg(total_count=('count_value', 'sum')).reset_index()

    # Creates the tile_name column
    sum_1x1_to_10x10['tile_name'] = sum_1x1_to_10x10['tile_id'] + '__' + sum_1x1_to_10x10['layer_name'] + '.tif'

    # Reorders columns to place tile_name between layer_name and total_count
    sum_1x1_to_10x10 = sum_1x1_to_10x10[['tile_id', 'layer_name', 'tile_name', 'total_count']]


    ### Part 3: Saves dataframes to Excel spreadsheet if three of the main output tabs are <900,000 rows; otherwise, saves to parquet format
    # other_1x1_outputs is the output table that has the most rows, so it's the best way to judge what's output is too large for Excel.
    # Excel's row limit is more like 1.5 million, but that'd be a really unwieldy spreadsheet.
    if (len(other_1x1_outputs) > 900000) or (len(net_flux_1x1_outputs) > 900000) or (len(gross_flux_1x1_outputs) > 900000):
    # if (len(other_1x1_outputs) > 2) or (len(net_flux_1x1_outputs) > 2) or (len(gross_flux_1x1_outputs) > 2):   # large-scale testing
        main_logger.info(f"Row count {len(other_1x1_outputs)} greater than 900,000. Writing all outputs to Parquet.")

        # Saves each output DataFrame as Parquet
        out_base = f"{stage}_{timestr()}"
        output_dir = cn.local_chunk_stats_path

        # Makes sure that numeric columns are indeed numeric before saving them to parquet.
        # Parquet is picky about this.
        # From https://chatgpt.com/c/68751cbe-6888-800a-bf9d-3657b048a810
        main_logger.info(f"Cleaning non-numeric outputs for saving to Parquet: {timestr()}")

        # Numeric columns that need cleaning
        numeric_columns = ['min_value', 'mean_value', 'max_value', 'count_value', 'sum_value']

        (
            annual_1x1_inputs, other_1x1_inputs, gross_flux_1x1_outputs, net_flux_1x1_outputs,
            other_1x1_outputs, min_max_1x1_stats, sum_1x1_to_10x10,
        ) = [
            sanitize_numeric_columns(df, numeric_columns)
            for df in [
                annual_1x1_inputs, other_1x1_inputs, gross_flux_1x1_outputs, net_flux_1x1_outputs,
                other_1x1_outputs, min_max_1x1_stats, sum_1x1_to_10x10,
            ]
        ]

        # Saves to Parquet
        # Groups output files from model run into a timestamped folder
        timestamp = timestr()
        parquet_folder = Path(f"{output_dir}parquet_{timestamp}/")
        parquet_folder.mkdir(parents=True, exist_ok=True)

        annual_1x1_inputs.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.annual_1x1_inputs}.parquet", index=False)
        other_1x1_inputs.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.other_1x1_inputs}.parquet", index=False)
        gross_flux_1x1_outputs.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.gross_outputs_1x1}.parquet", index=False)
        net_flux_1x1_outputs.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.net_outputs_1x1}.parquet", index=False)
        other_1x1_outputs.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.other_outputs_1x1}.parquet", index=False)
        min_max_1x1_stats.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.min_max_for_layers_1x1}.parquet", index=False)
        sum_1x1_to_10x10.to_parquet(f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}__{cn.counts_1x1_in_10x10}.parquet", index=False)

        # Uploads to S3 if needed
        parquet_files = {
                cn.annual_1x1_inputs: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.annual_1x1_inputs}.parquet",
                cn.other_1x1_inputs: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.other_1x1_inputs}.parquet",
                cn.gross_outputs_1x1: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.gross_outputs_1x1}.parquet",
                cn.net_outputs_1x1: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.net_outputs_1x1}.parquet",
                cn.other_outputs_1x1: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.other_outputs_1x1}.parquet",
                cn.min_max_for_layers_1x1: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.min_max_for_layers_1x1}.parquet",
                cn.counts_1x1_in_10x10: f"{out_base}__v{cn.veg_model_version_underscore}__{cn.counts_1x1_in_10x10}.parquet",
            }

        if not no_upload:
            for key, filename in parquet_files.items():
                full_path = f"{parquet_folder}/{filename}"
                s3_key = f"{cn.s3_chunk_stats_path}parquet_{timestamp}/{filename}"
                main_logger.info(f"Uploading {filename} to S3: {timestr()}")
                s3_client.upload_file(full_path, cn.short_bucket_prefix, Key=s3_key)

        # Returns the names of all the parquet files
        return f"{parquet_folder}/{out_base}__v{cn.veg_model_version_underscore}"

    # Saves chunk stats to Excel
    else:

        # Writes the data to a single Excel file with separate sheets.
        # Should continue with model post-processing even if chunk stats don't work for some reason
        # (e.g., more many rows output than rows in an Excel spreadsheet)
        out_spreadsheet = f'{stage}_1x1_chunk_statistics_{timestr()}.xlsx'
        local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"

        main_logger.info(f"Writing tile stats to spreadsheet: {timestr()}")
        try:
            with pd.ExcelWriter(local_spreadsheet) as writer:

                # Writes input rows to one sheet
                main_logger.info(f"Writing inputs to spreadsheet: {timestr()}")
                annual_1x1_inputs.to_excel(writer, sheet_name=cn.annual_1x1_inputs, index=False)
                other_1x1_inputs.to_excel(writer, sheet_name=cn.other_1x1_inputs, index=False)

                # Writes output rows based on layer_name conditions to separate sheets
                main_logger.info(f"Writing outputs to spreadsheet: {timestr()}")
                gross_flux_1x1_outputs.to_excel(writer, sheet_name=cn.gross_outputs_1x1, index=False)
                net_flux_1x1_outputs.to_excel(writer, sheet_name=cn.net_outputs_1x1, index=False)
                other_1x1_outputs.to_excel(writer, sheet_name=cn.other_outputs_1x1, index=False)

                # Writes the min and max statistics to the second sheet
                min_max_1x1_stats.to_excel(writer, sheet_name=cn.min_max_for_layers_1x1, index=False)

                # Writes the 1x1s summed to 10x10, if available
                sum_1x1_to_10x10.to_excel(writer, sheet_name=cn.counts_1x1_in_10x10, index=False)

            main_logger.info(merged_1x1_stats.head())  # Show first few rows of the stats DataFrame for inspection

            main_logger.info(f"Done aggregating and exporting tile stats: {timestr()}")

        except Exception as e:
            main_logger.info(f"Can't save chunk stats to Excel: {e}")

        if not no_upload:
            main_logger.info(f"Uploading chunk stats spreadsheet to s3: {timestr()}")
            try:
                s3_client.upload_file(local_spreadsheet, cn.short_bucket_prefix, Key=f"{cn.s3_chunk_stats_path}{out_spreadsheet}")
                main_logger.info(f"Chunk stats spreadsheet uploaded to {cn.full_bucket_prefix}/{cn.s3_chunk_stats_path}{out_spreadsheet}: {timestr()}")
            except Exception as e:
                main_logger.warning(f"Chunk stats upload to S3 failed: {e}. Continuing without halting.")

        # Returns path to spreadsheet
        return local_spreadsheet


def compile_ipcc_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger):
    """
    Compiles IPCC 1x1 chunk stats to Excel with separate tabs for:
      - raw_chunk_stats
      - class
      - node_code
      - change
      - summary

    count_* columns represent either:
      - total area in hectares, if calculate_ipcc_stats was called with pixel_area
      - pixel counts, if calculate_ipcc_stats was called without pixel_area
    """

    s3_client = boto3.client("s3")

    main_logger.info(f"Starting to aggregate and export IPCC chunk stats: {timestr()}")

    df = pd.DataFrame(all_1x1_stats)

    if df.empty:
        main_logger.warning("No IPCC chunk stats to compile.")
        return None

    # Add ISO/country code from fishnet
    gdf = gpd.read_file(chunk_shapefile_uri)
    fishnet_shapefile_df = gdf[["chunk_id", "iso"]]

    df = df.merge(fishnet_shapefile_df, on="chunk_id", how="left")
    df["iso"] = df["iso"].fillna("unknown")

    # Identify count columns. These are count_* but may contain area in ha.
    count_cols = [col for col in df.columns if col.startswith("count_")]

    valid_ipcc_class_count_cols = [f"count_{code}" for code in cn.ipcc_class_codes]
    valid_ipcc_node_count_cols = [f"count_{code}" for code in cn.ipcc_node_codes]
    valid_ipcc_change_summary_count_cols = [f"count_{code}" for code in cn.ipcc_change_codes]

    for col in count_cols + ["count_value", "min_value", "max_value", "mode_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Layer-type grouping
    def ipcc_layer_group(pattern):
        if pattern == cn.IPCC_class_pattern:
            return "class"
        if pattern == cn.IPCC_node_pattern:
            return "node_code"
        if pattern == cn.IPCC_change_pattern:
            return "change"
        if pattern == cn.IPCC_summary_pattern:
            return "summary"
        return "other"

    df["ipcc_layer_group"] = df["pattern"].apply(ipcc_layer_group)

    # Summarize each IPCC layer type by ISO, years, and class/code columns
    def summarize_group(group_name, valid_count_cols):
        group_df = df[df["ipcc_layer_group"] == group_name].copy()

        if group_df.empty:
            return pd.DataFrame()

        group_cols = ["iso", "pattern", "years"]

        # Keep only count columns valid for this IPCC output type.
        # Add missing valid columns as 0 so every tab has consistent columns.
        for col in valid_count_cols:
            if col not in group_df.columns:
                group_df[col] = 0

        summary = (group_df.groupby(group_cols, dropna=False)[valid_count_cols].sum(numeric_only=True).reset_index())
        summary["total_count_or_area"] = summary[valid_count_cols].sum(axis=1)

        return summary

    class_stats = summarize_group("class", valid_ipcc_class_count_cols)
    node_code_stats = summarize_group("node_code", valid_ipcc_node_count_cols)
    change_stats = summarize_group("change", valid_ipcc_change_summary_count_cols)
    summary_stats = summarize_group("summary", valid_ipcc_change_summary_count_cols)

    # Save to Excel
    out_spreadsheet = f"{stage}__IPCC_1x1_chunk_stats__{timestr()}.xlsx"
    local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"

    with pd.ExcelWriter(local_spreadsheet) as writer:
        df.to_excel(writer, sheet_name="raw_chunk_stats", index=False)
        class_stats.to_excel(writer, sheet_name="class", index=False)
        node_code_stats.to_excel(writer, sheet_name="node_code", index=False)
        change_stats.to_excel(writer, sheet_name="change", index=False)
        summary_stats.to_excel(writer, sheet_name="summary", index=False)

    main_logger.info(f"Saved IPCC chunk stats locally: {local_spreadsheet}")

    if not no_upload:
        s3_path = f"{cn.s3_chunk_stats_path}{out_spreadsheet}"
        bucket, key = split_s3_path(f"s3://{cn.short_bucket_prefix}/{s3_path}")
        s3_client.upload_file(local_spreadsheet, Bucket=bucket, Key=key)
        main_logger.info(f"Uploaded IPCC chunk stats to s3://{bucket}/{key}")

    return local_spreadsheet

def aggregate_10x10_chunk_stats(counts_10x10_df, stage, no_upload, main_logger):

    ### Part 1: Organizes chunk stats for 1x1 degree chunks (inputs and outputs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    main_logger.info(f"Starting to aggregate and export tile stats: {timestr()}")

    # Writes the data to a single Excel file with separate sheets.
    # Should continue with model post-processing even if chunk stats don't work for some reason
    # (e.g., more many rows output than rows in an Excel spreadsheet)
    out_spreadsheet = f'{stage}_10x10_chunk_statistics_{timestr()}.xlsx'
    local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"

    main_logger.info(f"Writing tile stats to spreadsheet: {timestr()}")
    try:
        with pd.ExcelWriter(local_spreadsheet) as writer:

            counts_10x10_df.to_excel(writer, sheet_name='pix_counts_compa_10x10_1x1', index=False)

        main_logger.info(counts_10x10_df.head())  # Show first few rows of the stats DataFrame for inspection

        main_logger.info(f"Done aggregating and exporting tile stats: {timestr()}")

    except Exception as e:
        main_logger.info(f"Can't print chunk stats: {e}")

    if not no_upload:
        main_logger.info(f"Uploading chunk stats spreadsheet to s3: {timestr()}")
        s3_client.upload_file(local_spreadsheet, cn.short_bucket_prefix, Key=f"{cn.s3_chunk_stats_path}{out_spreadsheet}")
        main_logger.info(f"Chunk stats spreadsheet uploaded to {cn.full_bucket_prefix}/{cn.s3_chunk_stats_path}{out_spreadsheet}: {timestr()}")


# Gets the datatype of a raster in s3.
# This seems much faster than the rasterio version that ChatGPT suggested later in the chat.
# From https://chatgpt.com/share/e/a48c768d-0331-43da-9fc6-ef8a84af586c
def get_dtype_from_s3(key, s3_path):

    # Constructs the /vsis3/ path
    try:
        vsis3_path = f'/vsis3/{s3_path[len("s3://"):]}'
        data_type = get_dtype_from_raster(vsis3_path)
        return data_type

    except Exception as e:
        print(f"Error: s3 path not available for getting datatype of first tile in {key}: {e}")
        sys.exit(1)

# Gets the datatype of a raster in a Coiled cluster
def get_dtype_from_coiled(s3_path, local_path):
    file = download_s3_file(s3_path, local_path)
    data_type = get_dtype_from_raster(file)
    return data_type

# Gets the datatype of a raster
def get_dtype_from_raster(file_path):

    try:
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise ValueError(f"Could not open file {file_path}")

        band = dataset.GetRasterBand(1)
        if band is None:
            raise ValueError(f"No raster bands found in file {file_path}")

        data_type = gdal.GetDataTypeName(band.DataType)
        return data_type

    except Exception as e:
        print(f"Error: {e}")
        return None  # Return None or an appropriate fallback value



# Creates a dictionary of inputs where the keys are the dataset names and the values are a list with the first
# tile of the dataset in s3 and the datatype,
# e.g., {'land_cover_2000': ['s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/landcover/composite/2000/raw/00N_010E.tif', 'Byte'],
# 'agc_2000': ['s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/40000_pixels/20240821/00N_010E__AGC_density_MgC_ha_2000.tif', 'Float32'],
# 'drivers': ['s3://gfw2-data/climate/carbon_model/other_emissions_inputs/tree_cover_loss_drivers/processed/drivers_2022/20230407/00N_010E_tree_cover_loss_driver_processed.tif', 'Byte']}
def add_file_type_to_dict(first_tiles):

    # Dictionary where the keys are the dataset names and the values are a list with the first
    # tile of the dataset in s3 and the datatype
    download_dict_with_data_types = {}

    # Iterates through the first tile of each tile set in s3 in the input dictionary
    for key, file_path in first_tiles.items():

        # Gets the datatype from the first tile of the dataset in s3
        dtype = get_dtype_from_s3(key, file_path)
        # Adds file path and dtype as a list as the value in the dictionary
        download_dict_with_data_types[key] = [file_path, dtype]

        # print(f"Key: {key}, File Path: {file_path}, Data Type: {dtype}")

    return download_dict_with_data_types


# Replaces a tile_id in s3 paths in a dictionary with another tile_id
def replace_tile_id_in_dict(data_dict, new_tile_id):

    # Loop through the dictionary and modify the values
    for key, value in data_dict.items():
        # Assuming value is a list where the first item is the file path
        file_path = value[0]
        # Replace the pattern in the file path with the new tile_id
        updated_file_path = re.sub(cn.tile_id_pattern, new_tile_id, file_path)

        # Update the dictionary with the new file path
        data_dict[key][0] = updated_file_path

    return data_dict


# Identifies all chunks (1x1 deg tiles) that are adjacent to a focal chunk, plus the focal chunk.
# Does not limit to chunks that actually exist.
def get_adjacent_1x1_chunks(bbox):
    """
    Given a bounding box (W, S, E, N), return a list of 1x1 degree tile bounding boxes
    that intersect it. Each tile is assumed to be aligned on integer degrees.

    Parameters:
    - bbox: tuple (W, S, E, N)

    Returns:
    - List of (xmin, ymin, xmax, ymax) tuples for intersecting tiles
    """
    west, south, east, north = bbox

    # Floor west/south, ceil east/north to get inclusive range of tiles
    xmin_tiles = int(np.floor(west))
    xmax_tiles = int(np.ceil(east))
    ymin_tiles = int(np.floor(south))
    ymax_tiles = int(np.ceil(north))

    tile_bounds = []
    for x in range(xmin_tiles, xmax_tiles):
        for y in range(ymin_tiles, ymax_tiles):
            tile_bounds.append((x, y, x + 1, y + 1))  # 1x1 degree tile

    return tile_bounds



# Fills any missing chunks (layers) with NoData (0s) of the correct datatype.
# The 0s must be the correct datatype so that the numba function receives consistent datatypes for each input dataset.
# Needs to be expanded if additional datatypes are being used.
def fill_missing_input_layers_with_no_data(layers, uint8_list, int16_list, int32_list, float32_list,
                                           bounds_str, tile_id, is_final, logger):

   # Fills missing layers with arrays of the appropriate data type and size
    for key, array in layers.items():
        if array is None:

            # Determines the appropriate dtype based on the categorized lists
            if key in uint8_list:
                dtype = np.uint8
            elif key in int16_list:
                dtype = np.int16
            elif key in int32_list:
                dtype = np.int32
            elif key in float32_list:
                dtype = np.float32
            else:
                raise ValueError(f"Key {key} for chunk {bounds_str} in {tile_id} not found in any data type lists: {timestr()}")

            # Finds an existing array to use as a template for size
            existing_array = next((arr for arr in layers.values() if arr is not None), None)
            if existing_array is not None:
                # Creates an array of zeros with the same shape and the determined dtype
                layers[key] = np.zeros(existing_array.shape, dtype=dtype)
                # print(f"Filled missing layer '{key}' with an array of zeros (dtype={dtype}).")
                lu.print_and_log(f"Created {key} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
            else:
                # Handles the case where no data exists at all
                raise ValueError(f"No data available to determine the size for the missing layer {key} for chunk {bounds_str} in {tile_id}: {timestr()}")

    return layers


# Extracts the file name pattern and year (or year range) from a string
# Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/670e9a35-71b8-800a-aae1-9bbff7485a30 (including the pattern)
def strip_and_extract_years(key):

    pattern = re.sub(cn.date_date_range_pattern, '', key)

    try:
        year_range = re.search(cn.date_date_range_pattern, key).group()[1:]
        year_range = year_range.lstrip('_')  # Removes any leading _
    except:
        year_range = 'no year range'

    return pattern, year_range


# Removes the pixel meaning (per-ha or per-pixel) from a string
def strip_pixel_meaning(key):

    # Try each suffix in order
    for meaning in cn.pixel_meanings:
        if meaning in key:
            out_pattern_without_pixel_meaning = re.sub(re.escape(meaning), '', key)

            return out_pattern_without_pixel_meaning, meaning

    # If none of the known pixel meanings are found, the key is returned as the pattern and the meaning is empty
    return key, ''


# Creates a dataframe from the attribute table of the 1x1 deg fishnet with GADM iso joined to it
def fishnet_with_GADM_iso(shapefile_uri):

    # Reads the 1x1 deg fishnet with GADM iso joined from S3 to extract "chunk_id" and "iso" fields
    gdf = gpd.read_file(shapefile_uri)

    # Creates a DataFrame of the 1x1def fishnet with "chunk_id" and "iso" fields
    fishnet_df = gdf[['chunk_id', 'iso']]

    return fishnet_df


def get_cluster_info(client, cluster):

    # Retrieves properties of the workers
    workers = client.scheduler_info()["workers"]

    # Retrieves the number of workers
    n_workers = len(workers)

    # Retrieves the number of threads per worker
    # https://chatgpt.com/share/e/672503f1-eef8-800a-9218-281624acf27e
    first_worker_address = next(iter(workers.keys()))
    nthreads = workers[first_worker_address]["nthreads"]

    # Retrieves scheduler info for other cluster properties
    scheduler_info = cluster.scheduler_info  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info['workers'][next(iter(scheduler_info['workers']))]['memory_limit']
        worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69c1440e-15b0-8329-a795-8b0d22800481
    dashboard_link = cluster.details_url

    return worker_memory, n_workers, nthreads, dashboard_link



# Write single GeoTIFF to S3 using in-memory buffer
def write_single_geotiff_to_s3(var, year, tile_id, data, no_data_val, transform, s3_path, logger_worker):

    fs = fsspec.filesystem("s3", anon=False)
    max_retries = 5
    base_wait_seconds = 2

    lu.print_and_log(f"  Writing {var} for year {year} for {tile_id} to {s3_path}: {timestr()}", False, logger_worker)
    upload_start_time = time.time()

    height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "LZW",
        "nodata": no_data_val,
        "tiled": True,
        "blockxsize": 400,
        "blockysize": 400,
        "BIGTIFF": "YES",  # For geotifs >4 GB
    }

    # Counts non-zero and non-NaN pixels for comparison with 1x1 deg geotifs
    # valid_pixel_count = int(np.count_nonzero(~np.isnan(data) & (data != 0)))
    valid_pixel_count = int(np.count_nonzero(~np.isnan(data)))
    # print("pixel count:", valid_pixel_count)

    # Writes to temporary file on disk
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmpfile:
        with rasterio.open(tmpfile.name, "w", **profile) as dst:
            dst.write(data, 1)
            dst.close()  # Ensures file is flushed before upload, per ChatGPT

        # Retries S3 upload
        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6980e192-5198-8332-8e69-b994315e91c0
        # Added this because I got a Content-Length HTTP header error during 10x10 deg aggregation uploads once
        # and didn't have any retries set ip.
        for attempt in range(1, max_retries + 1):
            try:
                fs.put_file(tmpfile.name, s3_path)
                break  # Success
            except (OSError, botocore.exceptions.BotoCoreError) as e:
                lu.print_and_log( f"  WARNING: S3 upload failed (attempt {attempt}/{max_retries}) for {tile_id}: {e}",True, logger_worker)
                if attempt == max_retries:
                    raise
                sleep_time = base_wait_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    upload_end_time = time.time()
    lu.print_and_log(f"  Upload completed for {var} for year {year} for {tile_id} to {s3_path} in {round(upload_end_time-upload_start_time)} seconds: {timestr()}", False, logger_worker)

    return valid_pixel_count


# To cache the pixel area zarr once,
# per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/693c4bed-09f0-832c-a6ef-66390c74aa60
PIXEL_AREA_STORE = None

def get_pixel_area_store():
    global PIXEL_AREA_STORE
    if PIXEL_AREA_STORE is None:
        fs = fsspec.filesystem("s3", anon=False)
        PIXEL_AREA_STORE = zarr.open_group(fs.get_mapper(cn.pixel_area_zarr_path), mode="r")
    return PIXEL_AREA_STORE


# Creates an empty txt file for each chunk in s3.
# Uses concurrent.futures to parallelize the txt creation. Otherwise, it's very slow.
# Based on https://chatgpt.com/share/e/67bf0fd9-7cb0-800a-8666-2becd97d45a7
# Uses ThreadPoolExecutor and `as_completed()` to avoid blocking.
def create_s3_task_files(stage, chunk_list):

    s3 = boto3.client("s3")

    # Uploads a single task file to S3.
    def upload_task_file(chunk):

        chunk_id_str = boundstr(chunk)  # Converts chunk ID to string
        tile_id = xy_to_tile_id(chunk[0], chunk[3])
        key = f"{cn.progress_tracking_path}pending_{tile_id}_{chunk_id_str}_{stage}.txt"

        try:
            s3.put_object(Bucket=cn.short_bucket_prefix, Key=key, Body="")
            return f"Created: {key}"
        except Exception as e:
            return f"Error creating task file {key}: {e}"

    # Uses ThreadPoolExecutor for parallel uploads
    max_workers = min(100, len(chunk_list))  # Limits workers to 100 or chunk count

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(upload_task_file, chunk): chunk for chunk in chunk_list}

        for future in concurrent.futures.as_completed(future_to_chunk):
            result = future.result()
            # print(result)  # Print each upload result in real-time

    elapsed_time = time.time() - start_time
    print(f"Created task tracking files in {cn.progress_tracking_path} in {elapsed_time:.2f} seconds")


# Renames the tracking file from 'pending' to 'in_progress' when a task starts
def rename_s3_task_file(stage, chunk_id, new_status, is_final, logger_worker):

    s3 = boto3.client("s3")
    chunk_id_str = boundstr(chunk_id)  # Converts chunk ID to string
    tile_id = xy_to_tile_id(chunk_id[0], chunk_id[3])

    # Iterates through the task status prefixes to find the status of the specific chunk .
    # Order of statuses matters: first one found is renamed.
    for prefix in cn.possible_task_statuses:
        old_key = f"{cn.progress_tracking_path}{prefix}{tile_id}_{chunk_id_str}_{stage}.txt"
        new_key = f"{cn.progress_tracking_path}{new_status}{tile_id}_{chunk_id_str}_{stage}.txt"

        # Retries renaming of task files in case there's a burst of renaming like at the start of the cluster
        # Per Claude session 'Quick task completion analysis'
        for attempt in range(5):
            try:
                s3.copy_object(Bucket=cn.short_bucket_prefix, CopySource={'Bucket': cn.short_bucket_prefix, 'Key': old_key}, Key=new_key)
                s3.delete_object(Bucket=cn.short_bucket_prefix, Key=old_key)
                return
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "NoSuchKey":
                    break  # File not at this prefix; try next prefix
                elif code in ("SlowDown", "503", "RequestLimitExceeded"):
                    sleep_time = min(30, 1.0 * (2 ** attempt)) + random.uniform(0.0, 1.0)
                    time.sleep(sleep_time)
                    continue  # Retry same prefix
                else:
                    print(f"Error renaming task file {old_key}: {e}")
                    return
            except Exception as e:
                print(f"Error renaming task file {old_key}: {e}")
                return

    lu.print_and_log(f"No existing task file found for chunk {chunk_id}. Skipping rename.", is_final, logger_worker)


# Deletes the tracking txt file from S3 when a task is completed
def delete_s3_task_file(stage, chunk_id, is_final, logger_worker):

    s3 = boto3.client("s3")
    chunk_id_str = boundstr(chunk_id)  # Converts chunk ID to string
    tile_id = xy_to_tile_id(chunk_id[0], chunk_id[3])

    # Iterates through the task status prefixes to find the status of the specific chunk
    for prefix in cn.possible_task_statuses:
        key = f"{cn.progress_tracking_path}{prefix}{tile_id}_{chunk_id_str}_{stage}.txt"

        try:
            s3.delete_object(Bucket=cn.short_bucket_prefix, Key=key)
            # print(f"Deleted: {key}")
            deleted = True  # Marks that at least one file was deleted
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue  # Moves to the next possible file if a file with that status doesn't exist
            else:
                # print(f"Error deleting task file {key}: {e}")
                return  # Exit if there's a real error

    # Logs if no files were deleted
    if not deleted:

        lu.print_and_log(f"No task file found for chunk {chunk_id}. Nothing to delete.", is_final, logger_worker)


def gdal_vrt_progress(pct, message, data):
    """
    GDAL progress callback.
    pct: 0.0–1.0
    message: current operation
    """
    pct_int = int(pct * 100)
    print(f" GDAL VRT build progress for {data}: {pct_int}%: {timestr()}", flush=True)
    return 1  # return 0 would cancel


def gdal_translate_progress(pct, message, data):
    """
    GDAL progress callback.
    pct: 0.0–1.0
    message: current operation
    """
    pct_int = int(pct * 100)
    print(f" GDAL.translate progress for {data}: {pct_int}%: {timestr()}", flush=True)
    return 1  # return 0 would cancel


# Mosaics geotif tiles to global geotif for a given variable
def mosaic_tiles_to_global(var_name, year_idx, first_tiles_to_process, base_path, model_version, model_type, model_path_description, no_upload, is_large_run):

    logger_worker = lu.setup_logging_worker()

    start_time = time.time()

    # Gets year based on the specific timeseries
    if "SOC_density" in var_name:
        year = cn.SOC_density_intervals[year_idx]
    elif "SOC_net" in var_name:
        year = cn.SOC_change_intervals[year_idx]
    elif "SOC_loss" in var_name:
        year = cn.SOC_change_intervals[year_idx]
    elif "SOC_gain" in var_name:
        year = cn.SOC_change_intervals[year_idx]
    else:  # Vegetation timeseries
        year = cn.interval_end_years_annual[year_idx]

    # Establishes year/year range and units for dataset
    if "density" in var_name:  # Vegetation or SOC
        units = cn.C_density_aggreg_pixel_meaning
    elif "change" in var_name:  # SOC density change
        units = cn.flux_aggreg_pixel_meaning
    elif "emis" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "removals" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "net" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "loss" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif "gain" in var_name:
        units = cn.flux_aggreg_pixel_meaning
    elif cn.land_state_pattern in var_name:
        units = ""
    else:
        units = ""

    # Input s3 folder for dataset and year
    base_path = base_path.replace("PATTERN", var_name)
    base_path = base_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    base_path = base_path.replace("START_END", str(year))
    base_path = base_path.replace("PER_HA_OR_PIXEL", units)

    # Hacky way to fix land_state and other unitless outputs that otherwise have in the path YYYY//40000_pixels.
    # This removes the extra / .
    base_path = base_path.replace("//CHUNK_SIZE_pixels", "/CHUNK_SIZE_pixels")

    input_path = base_path.replace("CHUNK_SIZE_pixels", f"{cn.global_aggregation_factor}_pixels")

    # Output s3 folder for dataset and year
    output_path = base_path.replace("CHUNK_SIZE_pixels", "global")

    # If processing an annual average map, it uses that for the year, e.g., avg_2016_2024. Otherwise, just uses the year.
    # Per Claude session 'LULUCF global geotif setup'
    avg_match = re.search(r'avg_\d{4}_\d{4}', base_path)
    year_for_name = avg_match.group(0) if avg_match else year
    output_name = f"{var_name}{units}_v{model_version}_{year_for_name}_global.tif"
    # print(output_name)

    # Collects s3 tiles for the dataset-year
    fs = fsspec.filesystem("s3", anon=False)
    if first_tiles_to_process == None:  # All tiles in folder
        tile_files = fs.glob(f"{input_path}*.tif")
    else:  # Specified first few tiles in folder (for testing)
        tile_files = fs.glob(f"{input_path}*.tif")[0:first_tiles_to_process]
    if len(tile_files) == 0:
        return f"No tiles found in {input_path}"
    lu.print_and_log(f"{len(tile_files)} tiles to be processed in {input_path}: {timestr()}", False, logger_worker)

    tile_files = [f"/vsis3/{fp}" for fp in tile_files]  # Faster for accessing than using vsis3_streaming, by experiment
    # print(tile_files)

    # Creates a temporary working directory for worker
    tmpdir = tempfile.mkdtemp(prefix="mosaic_")
    safe_name = re.sub(r'[^0-9a-zA-Z]+', '_', input_path.strip('/'))[-180:]  # Shortens name if too long, per Claude session 'LULUCF global geotif setup'
    list_path = os.path.join(tmpdir, f"tile_list_{safe_name}.txt")
    vrt_path = os.path.join(tmpdir, f"mosaic_{safe_name}.vrt")

    with open(list_path, "w") as f:
        f.write("\n".join(tile_files))

    # Builds VRT
    lu.print_and_log(f"Building VRT for {input_path} into {vrt_path}: {timestr()}", is_large_run, logger_worker)
    # Build VRT directly from list of files
    vrt = gdal.BuildVRT(vrt_path,
                        tile_files
                        # callback=gdal_vrt_progress,  # Can use progress tracking if vrt creation is taking a long time
                        # callback_data=os.path.basename(output_name)
                        )
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {input_path}")

    vrt = None  # flush to disk

    # Validates VRT
    try:
        info = gdal.Info(vrt_path, format="json")
        size = info.get("size", [])
        vrt_end_time = time.time()
        lu.print_and_log(f"{vrt_path} created successfully with size {size}, took {round(vrt_end_time - start_time)} seconds: {timestr()}",False, logger_worker)
    except Exception as e:
        lu.print_and_log(f"VRT validation failed: {e}", False, logger_worker)
        raise RuntimeError(f"VRT validation failed for {vrt_path}")

    # Translates VRT → GeoTIFF
    local_out = os.path.join(tmpdir, output_name)
    gtiff_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512"
        ],
        # callback=gdal_translate_progress,  # Can use progress tracking if gdal_translate is taking a long time
        # callback_data=os.path.basename(output_name)
    )
    lu.print_and_log(f"Writing vrt to geotif for {output_path}: {timestr()}", is_large_run, logger_worker)

    writing_start_time = time.time()
    gdal.Translate(local_out, vrt_path, options=gtiff_options)
    writing_end_time = time.time()
    lu.print_and_log(f"Wrote vrt to geotif for {output_path}, took {round(writing_end_time - writing_start_time)} seconds: {timestr()}", False, logger_worker)

    if not no_upload:
        lu.print_and_log(f"Uploading global geotif for {output_path}: {timestr()}", is_large_run, logger_worker)
        fs.put(local_out, output_path)
        lu.print_and_log(f"Uploaded global geotif to {output_path}: {timestr()}", False, logger_worker)

    end_time = time.time()
    lu.print_and_log(f"Total chunk processing for {output_path} took {round(end_time - start_time)} seconds: {timestr()}", False, logger_worker)

    return f"Global geotif written to {output_path}"



###################################################################################################
# Hansenize Functions
###################################################################################################
# Function to build a VRT using GDAL with vsis3 paths
    # raw_raster_paths_list_s3 = list of s3 paths (with "s3://" prefix) to all raw raster used as input for the build VRT step
    # output_vrt_s3 = s3 path (with "s3://" prefix) where vrt is created
def build_vrt_gdal_local(raw_raster_paths_list_s3, output_vrt_s3):
    # Set the environment variable to enable random writes for S3 using vsis3
    os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

    # Convert S3 paths to vsis3 format
    raw_raster_paths_list_vsis3 = [path.replace("s3://", "/vsis3/") for path in raw_raster_paths_list_s3]
    output_vrt_vsis3 = output_vrt_s3.replace("s3://", "/vsis3/")

    # Use GDAL to build the VRT
    gdal.BuildVRT(output_vrt_vsis3, raw_raster_paths_list_vsis3)

    #Check that s3 file exists
    check_s3_file_created(output_vrt_s3)

# Checks if a VRT already exists in s3
# https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67dc3f96-40f0-800a-9c89-2895c332bd01
def vrt_exists_in_s3(output_vrt_s3):

    s3 = boto3.client("s3")

    # Parse the S3 path
    s3_path_parts = output_vrt_s3.replace("s3://", "").split("/", 1)
    bucket_name = s3_path_parts[0]
    object_key = s3_path_parts[1]

    try:
        # Check if the file exists in S3
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True  # File exists
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False  # File does not exist
        else:
            raise  # Some other error occurred


# Function to build a VRT using GDAL using tmp dir as intermediate step to download input files and build VRT
# raw_raster_paths_list_s3 = list of s3 paths (with "s3://" prefix) to all raw raster used as input for the build VRT step
# output_vrt_s3 = s3 path (with "s3://" prefix) where vrt is saved to
def build_vrt_gdal_coiled(raw_raster_paths_list_s3, output_vrt_s3, local_vrt, main_logger):

    logger_worker = lu.setup_logging_worker()

    # Check if the VRT file already exists in S3
    if vrt_exists_in_s3(output_vrt_s3):
        return main_logger.info(f"VRT file already exists in S3: {output_vrt_s3}. Skipping creation.")
    vsis3_paths = []
    for s3_path in raw_raster_paths_list_s3:
        vsis3_path = s3_path.replace("s3://", "/vsis3/")
        vsis3_paths.append(vsis3_path)

    # Use GDAL to build the VRT
    # gdal.BuildVRT(local_vrt, "/vsis3/gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/AGB/raw/N00E010_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2015-fv5.0.tif")
    gdal.BuildVRT(local_vrt, vsis3_paths)
    lu.print_and_log(f"Built {local_vrt}: {timestr('time')}", True, logger_worker)

    # Various checks that vrt was created and has data in it
    try:
        vrt_dataset = rasterio.open(local_vrt)
    except rasterio.errors.RasterioIOError:
        print("Error: VRT file not found or invalid.")
        exit()

    if vrt_dataset.count == 0:
        print("VRT has no data or invalid sources.")
        exit()
    else:
        lu.print_and_log("VRT contains data.", True, logger_worker)

    if vrt_dataset.bounds:
        lu.print_and_log("VRT contains data or has valid metadata.", True, logger_worker)
    else:
        print("VRT has no data or invalid metadata.")
        exit()

    vrt_dataset.close()

    #Upload to s3
    upload_s3_file(output_vrt_s3, local_vrt)

    #If successfully uploaded, delete local vrt
    if check_s3_file_created(output_vrt_s3):
        #Delete local VRT file     #TODO create a microservice to do this instead of repeating code in multiple functions
        try:
            os.remove(local_vrt)
            if not os.path.exists(local_vrt):
                main_logger.info(f"Deleted local VRT file: {local_vrt}")
            else:
                main_logger.warning(f"Failed to delete local VRT file: {local_vrt}")
        except Exception as e:
            main_logger.warning(f"Error deleting local VRT file: {local_vrt} — {e}")


# Function to read a VRT from S3 using GDAL and vsis3
def warp_to_hansen_local(source_raster_s3_path, output_raster_s3_path, xmin, ymin, xmax, ymax, dt, no_data, tiled=True,
                   x_pixel_window=400, y_pixel_window=400):
    #Note: If tiled=False, set x_pixel_window=None, y_pixel_window=None

    # Set the environment variable to enable random writes for S3
    os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

    # Check that pixel window arguments are given if tiled = True
    if tiled and not (x_pixel_window and y_pixel_window):
        raise ValueError("If tiled = True, x_pixel_window and y_pixel_window must be passed as arguments")

    # Convert the S3 paths to GDAL's vsis3 paths
    source_gdal_path = source_raster_s3_path.replace("s3://", "/vsis3/")
    output_gdal_path = output_raster_s3_path.replace("s3://", "/vsis3/")

    # Open the VRT
    dataset = gdal.Open(source_gdal_path)

    if dataset:
        if tiled == True:
            # Warp the VRT to the new raster
            options = gdal.WarpOptions(
                dstSRS='EPSG:4326',  # Reproject to WGS84
                xRes=0.00025,  # X resolution (10 degrees)
                yRes=0.00025,  # Y resolution (10 degrees)
                targetAlignedPixels=True,  # Ensure target aligned pixels (-tap)
                outputBounds=[xmin, ymin, xmax, ymax],  # Output bounds
                dstNodata=no_data,  # Set no data to 0
                outputType=dt,  # Output data type
                creationOptions=['COMPRESS=DEFLATE', 'TILED=YES',   # Tiling with user-specified dimensions
                                 f'BLOCKXSIZE={x_pixel_window}',
                                 f'BLOCKYSIZE={y_pixel_window}'],
                format='GTiff'  # Output format
            )
        else:
            # Warp the VRT to the new raster
            options = gdal.WarpOptions(
                dstSRS='EPSG:4326',
                xRes=0.00025,
                yRes=0.00025,
                targetAlignedPixels=True,
                outputBounds=[xmin, ymin, xmax, ymax],
                dstNodata=no_data,
                outputType=dt,
                creationOptions=['COMPRESS=DEFLATE', 'TILED=NO'],  # No tiling (i.e. 40,000 x 1)
                format='GTiff'
            )

        gdal.Warp(output_gdal_path, source_gdal_path, options=options)

        # Check that file exists
        check_s3_file_created(output_raster_s3_path)

    else:
        raise RuntimeError(f"Failed to open VRT: {source_gdal_path}")

# Creates a 10x10 deg raster at 0.00025x0.00025 resolution from a VRT for a specified bounding box
def warp_to_hansen_coiled(source_vrt_path, filename, output_raster_s3_path_and_name, xmin, ymin, xmax, ymax,
                          dt, no_data, tiled=True, x_pixel_window=400, y_pixel_window=400, src_nodata=None, scale_factor=1, round_scaled=False):
    #Note: If tiled=False, set x_pixel_window=None, y_pixel_window=None

    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Creating {filename}: {timestr('time')}", False, logger_worker)

    startup_delay = random.uniform(0, 3)  # Slight delay before s3 is accessed so that request quota isn't exceeded
    time.sleep(startup_delay)

    # Check that pixel window arguments are given if tiled = True
    if tiled and not (x_pixel_window and y_pixel_window):
        raise ValueError("If tiled = True, x_pixel_window and y_pixel_window must be passed as arguments")

    # Open the VRT
    source_vrt_path = source_vrt_path.replace("s3://", "/vsis3/")
    # print(f"in hansen function, vrt path is {source_vrt_path}")
    dataset = gdal.Open(str(Path(source_vrt_path)))

    #Code to run gdal warp using Python API
    if dataset:
        warp_kwargs = dict(
            format="GTiff", # Output format
            dstSRS='EPSG:4326',  # Reproject to WGS84
            xRes=cn.resolution,  # X resolution (10 degrees)
            yRes=cn.resolution,  # Y resolution (10 degrees)
            targetAlignedPixels=True,  # Ensure target aligned pixels (-tap)
            outputBounds=[xmin, ymin, xmax, ymax],  # Output bounds
            dstNodata=no_data,  # Set no data
            outputType=dt,  # Output data type
        )

        if tiled == True:
            warp_kwargs["creationOptions"] = ['COMPRESS=DEFLATE', 'TILED=YES',  # Tiling with user-specified dimensions
                                             f'BLOCKXSIZE={x_pixel_window}',
                                             f'BLOCKYSIZE={y_pixel_window}']
        else:
            warp_kwargs["creationOptions"] = ['COMPRESS=DEFLATE', 'TILED=NO']  # No tiling (i.e. 40,000 x 1)

        if src_nodata is not None:
            warp_kwargs["srcNodata"] = src_nodata

        options = gdal.WarpOptions(**warp_kwargs)

        gdal.Warp(str(Path(filename)), str(Path(source_vrt_path)), options=options)
        lu.print_and_log(f"{filename} created: {timestr('time')}", True, logger_worker)

        # Fixing scale factor (divide by 10) for Ctrees AGB data
        if scale_factor != 1:
            np_dtype = map_to_numpy_dtype(gdal_to_string_dtype_mapping[dt])

            with rasterio.open(str(Path(filename)), "r+") as dst:
                data = dst.read(1)

                valid_mask = data != no_data

                scaled = data.astype(np.float32)
                scaled[valid_mask] = scaled[valid_mask] * scale_factor

                if round_scaled:
                    scaled[valid_mask] = np.rint(scaled[valid_mask])

                data_out = np.full(data.shape, no_data, dtype=np_dtype)
                data_out[valid_mask] = scaled[valid_mask].astype(np_dtype)

                dst.write(data_out, 1)
                dst.update_tags(1, scale_factor_applied=scale_factor, rounded=round_scaled)

        #Fixing greyscale colormap in GMWv3 data
        if "mangrove" in source_vrt_path:
            ds = gdal.Open(str(Path(filename)), gdal.GA_Update)
            if ds:
                band = ds.GetRasterBand(1)
                if band.GetColorTable():
                    band.SetColorTable(None)
                    band.SetRasterColorInterpretation(gdal.GCI_Undefined)
                    lu.print_and_log(f"Removed color table from {filename}", False, logger_worker)
                ds.FlushCache()
                ds = None

        #Checking if tile contains any data
        lu.print_and_log(f"Checking if {filename} contains data: {timestr('time')}", True, logger_worker)
        if check_geotiff_has_data(filename):
            lu.print_and_log(f"{filename} contains data. Uploading to s3: {timestr('time')}", True, logger_worker)

            # Uploads tile to s3
            upload_s3_file(output_raster_s3_path_and_name, filename)
            lu.print_and_log(f"{filename} uploaded to s3: {timestr('time')}", False, logger_worker)

            # Deletes rasters from cluster after uploading to s3
            os.remove(str(Path(filename)))

            success_message = f"Success Hansenizing {filename}: {timestr('time')}"
            return success_message  # Return both the success message and the statistics
        else:
            lu.print_and_log(f"{filename} is empty or contains only NoData values. Not uploading to s3: {timestr('time')}", False, logger_worker)
            return

    else:
        raise RuntimeError(f"Failed to open VRT: {source_vrt_path}")


def delete_build_vrt_input_files(raw_raster_paths_list_s3, vrt):
    # Delete local input files
    for s3_path in raw_raster_paths_list_s3:
        local_file = s3_path.split('/')[-1]
        os.remove(str(Path(local_file)))

    # Delete local vrt
    os.remove(str(Path(vrt)))


###################################################################################################
# 4km Map Function
###################################################################################################
def reaggregate_resolution(data, original_res, target_res):
    #Courtesy of ChatGPT
    """
    Reaggregates a numpy array by summing values within the target resolution window.

    Parameters:
        data (numpy array): The input array at high resolution.
        original_res (float): The resolution of the input data.
        target_res (float): The desired resolution for the output data.

    Returns:
        numpy array: The reaggregated array at the desired resolution.
    """
    factor = int(target_res / original_res)
    new_shape = (
        data.shape[0] // factor,
        factor,
        data.shape[1] // factor,
        factor
    )
    # Reshape and sum
    return data.reshape(new_shape).sum(axis=(1, 3))


###################################################################################################
# Zonal Stats Functions
###################################################################################################
# Function to calculate the number of bits needed to represent the maximum value in the array
def calculate_bits_needed(max_value):
    return int(np.ceil(np.log2(max_value + 1)))


# Convert numpy arrays to dask arrays if needed
def ensure_dask_array(array, chunks="auto"):
    if isinstance(array, np.ndarray):
        return da.from_array(array, chunks=chunks)
    return array


# Ensure all layers have a consistent data type (int16) for bit-shifting
def ensure_dtype(layer_array, dtype=np.int16):
    if layer_array.dtype != dtype:
        return layer_array.astype(dtype)
    return layer_array


# Dynamically combine layers using bit-shifting
def combine_zone_layers(sorted_layers):
    combined_array = None
    total_shift = 0

    # Loop through each layer
    for layer_name, layer_array in sorted_layers:
        # Convert to dask.array if it's a numpy array
        layer_array = ensure_dask_array(layer_array)

        # Convert layer to int16 if necessary for safe bit-shifting
        layer_array = ensure_dtype(layer_array)

        # Find the maximum value in the layer
        max_value = da.max(layer_array).compute()  # Compute to get the actual maximum value

        # Determine the number of bits needed to represent this layer
        bits_needed = calculate_bits_needed(max_value)

        # Print unique values in the current layer before shifting
        # print(f"Unique values in layer '{layer_name}' before shifting: {np.unique(layer_array.compute())}")

        # Shift the layer by the cumulative number of bits (based on previous layers)
        shifted_layer = layer_array << total_shift

        # Print unique values in the current layer after shifting
        # print(f"Unique values in layer '{layer_name}' after shifting: {np.unique(shifted_layer.compute())}")

        # If this is the first layer, initialize the combined array
        if combined_array is None:
            combined_array = shifted_layer
        else:
            # Use bitwise OR to combine the shifted layer with the previous layers
            combined_array = combined_array | shifted_layer

        # Update the total bit shift for the next layer
        total_shift += bits_needed

    return combined_array


# converts all numpy arrays in data dictionary to same type (default set to float32)
def to_numpy_type(input_dict, check_type=np.float32):
    out_dict = dict()
    for key, value in input_dict.items():
        if value.dtype != check_type:
            out_dict[key] = value.astype(check_type)
        else:
            out_dict[key] = value
    return out_dict


# converts a python dictionary to a numba dictionary (data dictionaries all need to be the same type)
def to_numba_dict(input_dict):
    from numba import from_dtype
    out = None

    for key, value in input_dict.items():
        if out is None:
            dict_type = from_dtype(value.dtype)
            ndim = value.ndim
            out = Dict.empty(key_type=types.unicode_type, value_type=types.Array(dict_type, ndim, "A"))
        out[key] = value
    return out


# calculates per-pixel carbon stocks/ fluxes by converting square meter pixel area rasters to hectares and multiplying densities/ factors by pixel area in hectares
@jit(nopython=True)
def calculate_total_mgc(numba_dict, input_units="MgC_ha", rep_str="MgC", pixel_area_name="pixel_area_m",
                        area_ha_name="pixel_area_ha"):
    output_dict = dict()
    dense_flux_arrays = dict()
    pixel_area = numba_dict[pixel_area_name]

    for key, value in numba_dict.items():
        if key != pixel_area_name:
            updated_key = key.replace(input_units, rep_str)
            output_dict[updated_key] = np.zeros_like(value)
            dense_flux_arrays[updated_key] = value
    output_dict[area_ha_name] = np.zeros_like(pixel_area)

    # Loop over each pixel
    for key, value in dense_flux_arrays.items():
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                # Convert pixel_area from square meters to hectares
                square_meters_to_hectares = np.float32(10000.0)
                area_in_hectares = pixel_area[i, j] / square_meters_to_hectares
                output_dict[key][i, j] = value[i, j] * area_in_hectares
                output_dict[area_ha_name][i, j] = area_in_hectares

    return output_dict


# Function to reverse the bit-shifting process
def reverse_bit_shifting(df, column_name, sorted_layers):
    # Calculate bits_needed_per_layer based on max values from Dask arrays in sorted_layers
    bits_needed_per_layer = []

    for layer_name, layer_array in sorted_layers:
        # Ensure the layer is a Dask array and calculate max value
        layer_array = ensure_dask_array(layer_array)
        max_value = da.max(layer_array).compute()  # Compute the maximum value

        # Determine the number of bits needed to represent this layer
        bits_needed = calculate_bits_needed(max_value)
        bits_needed_per_layer.append(bits_needed)

    total_shift = sum(bits_needed_per_layer)  # Start with the total bits used

    # Reverse bit-shifting: loop through each layer in reverse order
    layers = [layer_name for layer_name, _ in sorted_layers]  # Get the sorted layer names
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        bits_needed = bits_needed_per_layer[i]
        total_shift -= bits_needed
        # Create a mask for extracting the current layer
        mask = (1 << bits_needed) - 1
        # Shift right and apply the mask to extract the current layer's values
        df[layer] = df[column_name].apply(lambda x: (x >> total_shift) & mask)

    return df

"""
Run from src/LULUCF/
python -m scripts.preprocessing.hansenize -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --no_stats
-bb -180 -60 180 80 -cs 2   # entire world (12600 chunks) (60x 32GB r6i.2xlarge workers= 22 minutes; around 90 Coiled credits and $4 dollars of AWS costs)
"""

import argparse
import concurrent.futures
import coiled
import dask
import os
from osgeo import gdal
import numpy as np

from dask.distributed import Client
from dask.distributed import Client, LocalCluster
from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu

############################################################################################################
#TODO add to command line argument or have system time reformatted with rundate
run_date = "20241004"
is_final = False
process = 'drivers'
bounds =

############################################################################################################
# def hansenize_rasters(bounds, download_dict_with_data_types, is_final, no_upload):

#Step 1
# logger = lu.setup_logging()
# bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
# tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
# chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)
#
# # Stores the min, mean, and max chunks for inputs and outputs for the chunk
# chunk_stats = []

#Step 1: Create download dictionary
download_upload_dictionary ={}

#TODO add text input file or command line arguments to determine which inputs to preprocess
if process == 'drivers':
    download_upload_dictionary["drivers"] = {
        'raw_dir': cn.drivers_raw_dir,
        'raw_pattern': cn.drivers_pattern,
        'vrt': "drivers.vrt",
        'processed_dir': cn.drivers_processed_dir,
        'processed_pattern': cn.drivers_pattern
    }

if process == 'secondary_natural_forest':
    download_upload_dictionary["secondary_natural_forest_0_5"] = {
        'raw_dir': cn.secondary_natural_forest_raw_dir,
        'raw_pattern': cn.secondary_natural_forest_0_5_pattern,
        'vrt': "secondary_natural_forest_0_5.vrt",
        'processed_dir': cn.secondary_natural_forest_0_5_processed_dir,
        'processed_pattern': cn.secondary_natural_forest_0_5_pattern
    }

    download_upload_dictionary["secondary_natural_forest_6_10"] = {
        'raw_dir': cn.secondary_natural_forest_raw_dir,
        'raw_pattern': cn.secondary_natural_forest_6_10_pattern,
        'vrt': "secondary_natural_forest_6_10.vrt",
        'processed_dir': cn.secondary_natural_forest_6_10_processed_dir,
        'processed_pattern': cn.secondary_natural_forest_6_10_pattern
    }

    download_upload_dictionary["secondary_natural_forest_11_15"] = {
        'raw_dir': cn.secondary_natural_forest_raw_dir,
        'raw_pattern': cn.secondary_natural_forest_11_15_pattern,
        'vrt': "secondary_natural_forest_11_15.vrt",
        'processed_dir': cn.secondary_natural_forest_11_15_processed_dir,
        'processed_pattern': cn.secondary_natural_forest_11_15_pattern
    }

    download_upload_dictionary["secondary_natural_forest_16_20"] = {
        'raw_dir': cn.secondary_natural_forest_raw_dir,
        'raw_pattern': cn.secondary_natural_forest_16_20_pattern,
        'vrt': "secondary_natural_forest_16_20.vrt",
        'processed_dir': cn.secondary_natural_forest_16_20_processed_dir,
        'processed_pattern': cn.secondary_natural_forest_16_20_pattern
    }

    download_upload_dictionary["secondary_natural_forest_21_100"] = {
        'raw_dir': cn.secondary_natural_forest_raw_dir,
        'raw_pattern': cn.secondary_natural_forest_21_100_pattern,
        'vrt': "secondary_natural_forest_21_100.vrt",
        'processed_dir': cn.secondary_natural_forest_21_100_processed_dir,
        'processed_pattern': cn.secondary_natural_forest_21_100_pattern
    }

#Step 2: Create a VRT for each dataset that need to be hansenized
# Find all files that match the raw pattern
for item in download_upload_dictionary:
    download_upload_dictionary[item]

matching_files = uu.list_s3_files_with_pattern(path, pattern)
for item in download_upload_dictionary:


#TODO Add datatype to download dictionary

#Step 2: Build VRT of all input rasters from raw input folders

print("Creating vrt for natural forest biomass accumulation rates:")

rate_0_5_vrt = 'rate_0_5.vrt'
rate_6_10_vrt = 'rate_6_10.vrt'
rate_11_15_vrt = 'rate_11_15.vrt'
rate_16_20_vrt = 'rate_16_20.vrt'
rate_21_100_vrt = 'rate_21_100.vrt'


#TODO add text input file or command line arguments to determine which inputs to preprocess (if process == 'drivers' or process == 'all':)
#TODO add print/logs of which inputs are being processed

import boto3
import dask
from dask import delayed
import os
from osgeo import gdal


def split_s3_path(s3_path):
    # Remove the "s3://" prefix
    s3_path = s3_path.replace("s3://", "")

    # Split the remaining string by the first "/"
    bucket, key = s3_path.split("/", 1)

    return bucket, key


def list_s3_files_with_pattern(s3_path, pattern):
    """
    List files in an S3 bucket with a certain pattern.

    Parameters:
    - s3_path: The complete path to a folder (including s3://bucket-name/)
    - pattern: The file pattern to match

    Returns:
    A list of files that match the pattern.
    """
    s3 = boto3.client('s3')
    matching_files = []

    bucket_name, prefix = split_s3_path(s3_path)

    # List objects in the bucket with the given prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # Check if any contents are returned
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith(pattern):
                matching_files.append(f"s3://{bucket_name}/{key}")
    else:
        print(f"No files found in the bucket '{bucket_name}' with the prefix '{prefix}'")

    return matching_files


# Example usage
path = 's3://gfw2-data/drivers_of_loss/1_km/raw/20241004/'  # Replace with the prefix or folder in your bucket
pattern = 'drivers_of_TCL_1_km_20241004.tif'  # Replace with the pattern you want to match

matching_files = list_s3_files_with_pattern(path, pattern)

if matching_files:
    print(f"Files matching pattern '{pattern}':")
    for file in matching_files:
        print(file)
else:
    print(f"No files matching pattern '{pattern}' were found.")


###################################################################################################################
# Function to build a VRT using GDAL with vsis3 paths
@delayed
def build_vrt_gdal(s3_paths, output_vrt):
    # Convert S3 paths to GDAL's vsis3 format
    vsis3_paths = [path.replace("s3://", "/vsis3/") for path in s3_paths]
    output_s3_path = output_vrt.replace("s3://", "/vsis3/")

    # Use GDAL to build the VRT
    gdal.BuildVRT(output_s3_path, vsis3_paths)
    return output_vrt


# TODO maybe use dask futures instead

# Example usage
output_vrt = 's3://gfw2-data/drivers_of_loss/1_km/raw/20241004/check_vrt.vrt'  # This is an S3 path using /vsis3/

# Call the build function using Dask for distributed processing
vrt_task = build_vrt_gdal(matching_files, output_vrt)
dask.compute(vrt_task)

print(f"vrt created at: {output_vrt}")

##################################################################################################################
dt = gdal.GetDataTypeName(gdal.Open(output_vrt.replace("s3://", "/vsis3/")).GetRasterBand(
    1).DataType)  # Open vrt and read the datatype of the first band
tile_id = "00N_000E"
output_path = "s3://gfw2-data/drivers_of_loss/1_km/processed/20241004/"
out_pattern = "drivers_of_TCL_1_km_20241004.tif"
outfile = f"{output_path}{tile_id}_{out_pattern}"


# TODO add error handling if it can't open up vrt

# dataset = gdal.Open(output_vrt)

# if dataset is None:
#     print(f"Failed to open VRT: {output_vrt}")
# else:
#     band = dataset.GetRasterBand(1) # Get the first band (GDAL starts from 1, not 0)
#     data_type = gdal.GetDataTypeName(band.DataType) # Read the data type of the band
#     print(f"Data Type of the VRT: {data_type}")

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


print("Getting extent of", tile_id)

xmin, ymin, xmax, ymax = get_10x10_tile_bounds(tile_id)


# Function to read a VRT from S3 using GDAL and vsis3
@delayed
def warp_to_hansen(source_raster_s3_path, output_raster_s3_path, xmin, ymin, xmax, ymax, dt, no_data, tiled=False,
                   x_pixel_window=None, y_pixel_window=None):
    # #Check that pixel window arguments are given if tiled = True
    # if tiled and not (x_pixel_window and y_pixel_window):
    #     raise ValueError("If tiled = True, x_pixel_window and y_pixel_window must be passed as arguments")

    # Convert the S3 paths to GDAL's vsis3 paths
    source_gdal_path = source_raster_s3_path.replace("s3://", "/vsis3/")
    output_gdal_path = output_raster_s3_path.replace("s3://", "/vsis3/")

    # Open the VRT
    dataset = gdal.Open(source_gdal_path)

    if dataset is None:
        raise RuntimeError(f"Failed to open VRT: {source_gdal_path}")

    if tiled == True:
        # Warp the VRT to the new raster
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            xRes=0.00025,
            yRes=0.00025,
            targetAlignedPixels=True,
            outputBounds=[xmin, ymin, xmax, ymax],
            dstNodata=no_data,
            outputType=dt,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES',
                             f'BLOCKXSIZE={x_pixel_window}',
                             f'BLOCKYSIZE={y_pixel_window}'],
            # Add Deflate compression and tiling with block dimensions
            format='GTiff'
        )
    else:
        # Warp the VRT to the new raster
        options = gdal.WarpOptions(
            dstSRS='EPSG:4326',  # Reproject to WGS84
            xRes=0.00025,  # X resolution (10 degrees)
            yRes=0.00025,  # Y resolution (10 degrees)
            targetAlignedPixels=True,  # Ensure target aligned pixels (-tap)
            outputBounds=[xmin, ymin, xmax, ymax],  # Output bounds
            dstNodata=no_data,  # Set no data to 0
            outputType=dt,  # Output data type
            creationOptions=['COMPRESS=DEFLATE', 'TILED=NO'],  # Add Deflate compression and no tiling (40,000 x 1)
            format='GTiff'  # Output format
        )

    # print(options)
    gdal.Warp(output_gdal_path, source_gdal_path, options=options)

    print(f"Warped raster saved at: {output_gdal_path}")
    return output_gdal_path


# Step 3: Set the environment variable to enable random writes for S3
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'
# TODO move up

# Warp the VRT with a 10x10 degree resolution, set -dstnodata to 0, enable -tap, and overwrite any existing file
warp_task = warp_to_hansen(output_vrt, outfile, xmin, ymin, xmax, ymax, gdal.GDT_Byte, 0, True, '400', '400')
warp_task = warp_to_hansen(output_vrt, outfile, xmin, ymin, xmax, ymax, gdal.GDT_Byte, 0, False)
# TODO add GDAL datatype

# Execute the task using Dask
result = dask.compute(warp_task)

print(f"Warped raster is available at: {result[0]}")
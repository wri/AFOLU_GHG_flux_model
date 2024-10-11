"""
Run from src/LULUCF/
python -m scripts.preprocessing.hansenize -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --no_stats
-bb -180 -60 180 80 -cs 2   # entire world (12600 chunks) (60x 32GB r6i.2xlarge workers= 22 minutes; around 90 Coiled credits and $4 dollars of AWS costs)
"""
import boto3
import os
import argparse
import concurrent.futures
import coiled
import dask
from dask import delayed
import os
from osgeo import gdal
import numpy as np

from dask.distributed import Client
from dask.distributed import Client, LocalCluster
from dask.distributed import print
from numba import jit

# Project imports
# from ..utilities import constants_and_names as cn
# from ..utilities import universal_utilities as uu
# from ..utilities import log_utilities as lu
# from ..utilities import numba_utilities as nu
from src.LULUCF.scripts.utilities import constants_and_names as cn
from src.LULUCF.scripts.utilities import universal_utilities as uu
from src.LULUCF.scripts.utilities import log_utilities as lu
from src.LULUCF.scripts.utilities import numba_utilities as nu

#Set the environment variable to enable random writes for S3
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'

############################################################################################################
#TODO add to command line argument or have system time reformatted with rundate
run_date = "20241004"
is_final = False
process = 'drivers'
#bounds =
#TODO add text input file or command line arguments to determine which inputs to preprocess (if process == 'drivers' or process == 'all':)
#TODO add print/logs of which inputs are being processed
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

#Step 2: Create a VRT for each dataset that needs to be hansenized
for key,items in download_upload_dictionary.items():
    path = items["raw_dir"]
    pattern = items["raw_pattern"]
    vrt = items["vrt"]

    # Find all files that match the raw pattern
    raster_list  = uu.list_s3_files_with_pattern(path, pattern)
    if raster_list:
        print(f"Files matching pattern '{pattern}':")
        for file in raster_list:
            print(file)
    else:
        print(f"No files matching pattern '{pattern}' were found.")
    download_upload_dictionary[key]["raw_raster_list"] = raster_list

    #Create a vrt of all raw input rasters
    output_vrt = f"{path}{vrt}"
    vrt_task = uu.build_vrt_gdal(raster_list, output_vrt)
    dask.compute(vrt_task)
    print(f"vrt for {key} created at: {output_vrt}")

    #Add datatype to download_upload dictionary
    dt = gdal.GetDataTypeName(gdal.Open(output_vrt.replace("s3://", "/vsis3/")).GetRasterBand(1).DataType)  # Open vrt and read the datatype of the first band
    # TODO add error handling if it can't open up vrt
    gdal_dt = next(key for key, value in uu.gdal_dtype_mapping.items() if value == dt)
    download_upload_dictionary[key]["dt"] = gdal_dt
    print(f"vrt for {key} has data type: {dt} ({gdal_dt})")

#Step 3: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles
tile_id_list = []

tile_id = "00N_000E"
output_path = "s3://gfw2-data/drivers_of_loss/1_km/processed/20241004/"
out_pattern = "drivers_of_TCL_1_km_20241004.tif"
outfile = f"{output_path}{tile_id}_{out_pattern}"

# dataset = gdal.Open(output_vrt)

# if dataset is None:
#     print(f"Failed to open VRT: {output_vrt}")
# else:
#     band = dataset.GetRasterBand(1) # Get the first band (GDAL starts from 1, not 0)
#     data_type = gdal.GetDataTypeName(band.DataType) # Read the data type of the band
#     print(f"Data Type of the VRT: {data_type}")

xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)

# Warp the VRT with a 10x10 degree resolution, set -dstnodata to 0, enable -tap, and overwrite any existing file
warp_task = uu.warp_to_hansen(output_vrt, outfile, xmin, ymin, xmax, ymax, gdal.GDT_Byte, 0, False)
# TODO add GDAL datatype

# Execute the task using Dask
result = dask.compute(warp_task)

#TODO Add datatype to download dictionary?

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
from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu

############################################################################################################
#TODO add to command line argument or have system time reformatted
run_date = "10082024"
is_final = False

#bounds =

############################################################################################################
#START FUNCTION HERE:
#def hansenize_model_inputs():

#Step 1
# logger = lu.setup_logging()
# bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
# tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
# chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)
#
# # Stores the min, mean, and max chunks for inputs and outputs for the chunk
# chunk_stats = []



#Step
# : Build VRT of all input rasters from raw input folders
print("Creating vrt for natural forest biomass accumulation rates")

rate_0_5_vrt = 'rate_0_5.vrt'
rate_6_10_vrt = 'rate_6_10.vrt'
rate_11_15_vrt = 'rate_11_15.vrt'
rate_16_20_vrt = 'rate_16_20.vrt'
rate_21_100_vrt = 'rate_21_100.vrt'


#TODO add print/logs of which inputs are being processed)

#TODO add text input file or command line arguments to determine which inputs to preprocess

# def clip_raster_to_tile(input_path, output_path, bounds, nodata_value=None, dtype=None):
#     """
#     Clips a raster to specified bounds using GDAL's gdalwarp command.
#
#     Args:
#         input_path (str): Path to the input raster.
#         output_path (str): Path to save the clipped raster.
#         bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
#         nodata_value (float): NoData value to set for the output raster.
#         dtype (str): Data type for the output raster.
#     """
#     try:
#         minx, miny, maxx, maxy = bounds
#         gdalwarp_cmd = [
#             'gdalwarp',
#             '-te', str(minx), str(miny), str(maxx), str(maxy),
#             '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
#             '-co', 'COMPRESS=DEFLATE',
#             '-co', 'TILED=YES',
#             '-overwrite'
#         ]
#
#         if dtype:
#             gdalwarp_cmd.extend(['-ot', dtype])  # Correctly add the data type option
#
#         # Include any other options like -tr and -tap if needed
#         gdalwarp_cmd.extend([
#             '-tr', '0.00025', '0.00025',  # Set the output resolution explicitly
#             '-tap'                         # Align pixels
#         ])
#
#         # Add input and output paths
#         gdalwarp_cmd.extend([input_path, output_path])
#
#         logging.info(f"Clipping raster with command: {' '.join(gdalwarp_cmd)}")
#         subprocess.run(gdalwarp_cmd, check=True)
#         logging.info(f"Raster clipped successfully: {output_path}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"GDAL error during clipping: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error during raster clipping: {e}")
#
# def merge_and_clip_rasters_gdal(raster_paths, output_path, bounds, nodata_value=None, dtype=None):
#     """
#     Merges multiple rasters and clips to specified bounds using GDAL.
#
#     Args:
#         raster_paths (list): List of raster paths to merge.
#         output_path (str): Path to save the merged and clipped raster.
#         bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
#         nodata_value (float): NoData value to set for the output raster.
#         dtype (str): Data type for the output raster.
#     """
#     try:
#         minx, miny, maxx, maxy = bounds
#         temp_merged_path = output_path.replace('.tif', '_merged.tif')
#         gdal_merge_cmd = [
#             'gdalwarp',
#             '-te', str(minx), str(miny), str(maxx), str(maxy),
#             '-tr', '0.00025', '0.00025',  # Set the output resolution explicitly
#             '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
#             '-co', 'COMPRESS=DEFLATE',
#             '-co', 'TILED=YES',
#             '-overwrite'
#         ]
#
#         if dtype:
#             gdal_merge_cmd.extend(['-ot', dtype])  # Correctly add the data type option
#
#         # Add raster paths and output path
#         gdal_merge_cmd.extend(raster_paths + [temp_merged_path])
#
#         logging.info(f"Merging and clipping rasters with command: {' '.join(gdal_merge_cmd)}")
#         subprocess.run(gdal_merge_cmd, check=True)
#
#         # Move the merged output to final output path
#         os.rename(temp_merged_path, output_path)
#         logging.info(f"Rasters merged and clipped successfully: {output_path}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"GDAL error during merge and clip: {e}")
#     except Exception as e:
#         logging.error(f"Unexpected error during raster merging and clipping: {e}")
#     finally:
#         if os.path.exists(temp_merged_path):
#             os.remove(temp_merged_path)
#
# def hansenize_gdal(input_paths, output_path, bounds, nodata_value=None, dtype=None):
#     """
#     Main function for processing using GDAL.
#
#     Args:
#         input_paths (str or list): Input raster path or list of paths to process.
#         output_path (str): Path to save the processed raster.
#         bounds (tuple): Bounding box (minx, miny, maxx, maxy) for processing.
#         nodata_value (float): NoData value to set for the output raster.
#         dtype (str): Data type for the output raster.
#     """
#     if isinstance(input_paths, list):
#         merge_and_clip_rasters_gdal(input_paths, output_path, bounds, nodata_value, dtype)
#     else:
#         clip_raster_to_tile(input_paths, output_path, bounds, nodata_value, dtype)
#
#     gc.collect()
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Hansenize rasters.")
#     parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
#     parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
#     parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
#
#     parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
#     parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
#     parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
#
#     args = parser.parse_args()
#
#     # Create the cluster with command line arguments
#     main(args.cluster_name, args.bounding_box, args.chunk_size, args.run_local, args.no_stats, args.no_log)

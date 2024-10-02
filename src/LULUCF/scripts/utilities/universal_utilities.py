import os
import coiled
import boto3
import time
import math
import numpy as np
import pandas as pd
import pytz
import rasterio
import rasterio.transform
import rasterio.windows
import subprocess
import re
import requests
import concurrent.futures
from botocore.config import Config
from dask.distributed import print
from dask.distributed import Client
from datetime import datetime
from io import BytesIO
from osgeo import gdal

# Project imports
from . import constants_and_names as cn
from . import log_utilities as lu

# Time in Eastern US timezone as a string
def timestr():
    # return time.strftime("%Y%m%d_%H_%M_%S")

    # Define the Eastern Time timezone
    eastern = pytz.timezone('US/Eastern')

    # Get the current time in UTC and convert to Eastern Time
    eastern_time = datetime.now(eastern)

    # Format the time as a string
    return eastern_time.strftime("%Y%m%d_%H_%M_%S")


# Connects to a Coiled cluster of a specified name if the local flag isn't on
def connect_to_Coiled_cluster(cluster_name, run_local):

    # Runs locally without Dask or in a Coiled cluster using Dask
    if run_local:
        print("Running locally without Dask/Coiled.")
        return None, None
    else:   #TODO Make it so that this doesn't create a cluster if it doesn't exist. This will create a cluster.
        # Connects to the existing Coiled cluster
        cluster = coiled.Cluster(name=cluster_name)
        client = Client(cluster)

        return cluster, client



# Chunk bounds as a string
def boundstr(bounds):
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


# Chunk length in pixels
def calc_chunk_length_pixels(bounds):
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


# Maps GDAL data type to the appropriate string value
gdal_dtype_mapping = {
    gdal.GDT_Byte: 'Byte',
    gdal.GDT_UInt16: 'UInt16',
    gdal.GDT_Int16: 'Int16',
    gdal.GDT_UInt32: 'UInt32',
    gdal.GDT_Int32: 'Int32',
    gdal.GDT_Float32: 'Float32',
    gdal.GDT_Float64: 'Float64'
}

# Maps GDAL datatypes to numpy datatypes
def map_to_numpy_dtype(data_type):
    dtype_map = {
        'Float32': 'float32',
        'Float64': 'float64',
        'Byte': 'uint8',
        'Int32': 'int32',
        'Int16': 'int16',
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
def get_chunk_bounds(bounding_box, chunk_size):
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


# Calculates the elapsed time for a stage
def stage_duration(start_time_str, end_time_str, stage):

    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")
    end_time = datetime.strptime(end_time_str, "%Y%m%d_%H_%M_%S")

    print(f"Elapsed time for {stage}: {end_time - start_time}")


# Lazily opens tile within provided bounds (i.e. one chunk) and returns as a numpy array.
# If it can't open the uri for the chunk (tile does not exist), it creates a numpy array
# of the correct datatype for that input.
# The returned chunk needs to have the correct datatype because it'll eventually be used in a
# numba function, which is very particular about datatypes.
# For example, a dataset that's float32 can't have NoData chunks that are uint8 because
# the Numba functions won't be able to handle that (since they're so particular about datatypes).
# So, that is addressed here.
#TODO use coiled.cluster --mount_bucket argument to see if it improves performance when accessing s3
# (Here and other functions that use s3): https://chatgpt.com/share/e/1fe33655-3700-465c-8b5f-19b6b0444407
def get_tile_dataset_rio(uri, data_type, bounds, chunk_length_pixels, is_final, logger):

    # If the uri exists, the relevant window is opened and returned and returned as an array.
    # Note that this chunk could still just have NoData values, which would be downloaded.
    try:
        with rasterio.open(uri) as ds:
            window = rasterio.windows.from_bounds(*bounds, ds.transform)
            data = ds.read(1, window=window)

    # If the uri doesn't exist, a numpy array of the correct size and datatype populated with 0s is returned.
    except Exception as e:

        numpy_dtype = map_to_numpy_dtype(data_type)   # Translates the GDAL-style datatype to numpy-style datatype
        data = np.full((chunk_length_pixels, chunk_length_pixels), 0).astype(numpy_dtype)

        lu.print_and_log(f"flm: Error accessing the dataset. Returning array of all 0s: {e}", is_final, logger)

    return data


# Prepares list of chunks to download.
# Chunks are defined by a bounding box.
def prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger):

    futures = {}

    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])

    # Submit requests to S3 for input chunks but don't actually download them yet. This queueing of the requests before downloading them speeds up the downloading
    # Approach is to download all the input chunks up front for every year to make downloading more efficient, even though it means storing more upfront
    with concurrent.futures.ThreadPoolExecutor() as executor:
        lu.print_and_log(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)

        for key, value in updated_download_dict.items():
            futures[executor.submit(get_tile_dataset_rio, value[0], value[1], bounds, chunk_length_pixels, is_final, logger)] = key

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


# Saves array as a raster locally, then uploads it to s3. NoData value for outputs is optional
def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id,
                                     bounds_str, output_dict, is_final, logger, no_data_val=None):

    # Configures S3 client with increased retries; retries can max out for global analyses
    s3_config = Config(
        retries={
            'max_attempts': 10,  # Increases the number of retry attempts
            'mode': 'standard'
        }
    )
    s3_client = boto3.client("s3", config=s3_config)  # Uses the configured client with more retries

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)

    file_info = f'{tile_id}__{bounds_str}'

    if is_final:
        lu.print_and_log(f"Saving and uploading outputs for {bounds_str} in {tile_id}: {timestr()}", is_final, logger)

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():

        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        # Only prints if not a final run
        if not is_final:
            lu.print_and_log(f"Saving {bounds_str} in {tile_id} for {year_out}: {timestr()}", is_final, logger)

        # Includes NoData value in output raster
        if no_data_val is not None:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                               blockysize=400, nodata=no_data_val) as dst:
                dst.write(data_array, 1)

        # No NoData value in output raster
        else:
            with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels,
                               height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                               blockysize=400) as dst:
                dst.write(data_array, 1)

        s3_path = f"{cn.s3_out_dir}/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels/{time.strftime('%Y%m%d')}"

        # Only prints if not a final run
        if not is_final:
            lu.print_and_log(f"Uploading {bounds_str} in {tile_id} for {year_out} to {s3_path}: {timestr()}", is_final, logger)

        s3_client.upload_file(f"/tmp/{file_name}", "gfw2-data", Key=f"{s3_path}/{file_name}")

        # Deletes the local raster
        os.remove(f"/tmp/{file_name}")


# Lists rasters in an s3 folder and returns their names as a list
def list_rasters_in_folder(full_in_folder):

    cmd = ['aws', 's3', 'ls', full_in_folder]
    s3_contents_bytes = subprocess.check_output(cmd)

    # Converts subprocess results to useful string
    s3_contents_str = s3_contents_bytes.decode('utf-8')
    s3_contents_list = s3_contents_str.splitlines()
    rasters = [line.split()[-1] for line in s3_contents_list]
    rasters = [i for i in rasters if "tif" in i]

    return rasters


# Uploads a shapefile to s3
def upload_shp(in_folder, shp):

    print(f"flm: Uploading to {in_folder}{shp}: {timestr()}")

    shp_pattern = shp[:-4]

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(f"/tmp/{shp}", "gfw2-data", Key=f"{in_folder[15:]}{shp}")
    s3_client.upload_file(f"/tmp/{shp_pattern}.dbf", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.dbf")
    s3_client.upload_file(f"/tmp/{shp_pattern}.prj", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.prj")
    s3_client.upload_file(f"/tmp/{shp_pattern}.shx", "gfw2-data", Key=f"{in_folder[15:]}{shp_pattern}.shx")

    os.remove(f"/tmp/{shp}")
    os.remove(f"/tmp/{shp_pattern}.dbf")
    os.remove(f"/tmp/{shp_pattern}.prj")
    os.remove(f"/tmp/{shp_pattern}.shx")

    print(f"flm: Uploaded to {in_folder}{shp}: {timestr()}")


# Makes a shapefile of the footprints of rasters in a folder, for checking geographical completeness of rasters
def make_tile_footprint_shp(input_dict):

    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    # Task properties
    print(f"flm: Making tile index shapefile for: {in_folder}: {timestr()}")

    # Folder including s3 key
    s3_in_folder = in_folder
    vsis3_in_folder = f'/vsis3/{in_folder[5:]}' #[5] drops the s3:// at the front

    # List of all the filenames in the folder
    filenames = list_rasters_in_folder(s3_in_folder)

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

    # Uploads shapefile to s3
    upload_shp(s3_in_folder, shp)

    os.remove(f"/tmp/{file_paths_txt}")

    return(f"Completed: {timestr()}")


# Saves an xarray data array locally as a raster and then uploads it to s3
def save_and_upload_raster_10x10(**kwargs):

    s3_client = boto3.client("s3") # Needs to be in the same function as the upload_file call

    data_array = kwargs['data']   # The data being saved
    out_file_name = kwargs['out_file_name']   # The output file name
    out_folder = kwargs['out_folder']   # The output folder

    print(f"flm: Saving {out_file_name} locally")

    profile_kwargs = {'compress': 'lzw'}   # Adds attribute to compress the output raster
    # data_array.rio.to_raster(f"{out_file_name}", **profile_kwargs)
    data_array.rio.to_raster(f"/tmp/{out_file_name}", **profile_kwargs)

    print(f"flm: Saving {out_file_name} to {out_folder[10:]}{out_file_name}")

    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder[10:]}{out_file_name}")

    # Deletes the local raster
    os.remove(f"/tmp/{out_file_name}")


# Creates a list of 2x2 deg tiles to aggregate into 10x10 deg tiles, where the list is a list of dictionaries of the form
# [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__AGC_density_MgC_ha_2000.tif', '00N_120E__AGC_density_MgC_ha_2000.tif']},
# {'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/BGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__BGC_density_MgC_ha_2000.tif', '00N_120E__BGC_density_MgC_ha_2000.tif']}]
def create_list_for_aggregation(s3_in_folders):
    list_of_s3_names_total = []  # Final list of dictionaries of input s3 paths and output aggregated 10x10 raster names

    # Iterates through all the input s3 folders
    for s3_in_folder in s3_in_folders:

        simple_file_names = []  # List of output aggregatd output 10x10 rasters

        # Raw filenames in an input folder, e.g., ['00N_000E__6_-2_8_0__IPCC_classes_2020.tif', '00N_000E__6_-4_8_-2__IPCC_classes_2020.tif',...]
        filenames = list_rasters_in_folder(s3_in_folder)

        # Iterates through all the files in a folder and converts them to the output names.
        # Essentially [tile_id]__[pattern].tif. Drops the chunk bounds from the middle.
        for filename in filenames:
            result = filename[:10] + filename[filename.rfind("__") + len("__"):]  # Extracts the relevant parts of the raw file names
            simple_file_names.append(result)  # New list of simplified file names used for 10x10 degree outputs

        # Removes duplicate simplified file names.
        # There are duplicates because each 10x10 output raster has many constituent chunks, each of which have the same aggregated, final name
        # e.g., ['00N_000E__IPCC_classes_2020.tif', '00N_010E__IPCC_classes_2020.tif', ...]
        simple_file_names = np.unique(simple_file_names).tolist()

        # Makes nested lists of the file names. Nested for next step.
        # e.g., [['00N_110E__AGC_density_MgC_ha_2000.tif']]
        simple_file_names = [[item] for item in simple_file_names]

        # Makes a list of dictionaries, where the key is the input s3 path and the value is the output aggregated name
        # e.g., [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__AGC_density_MgC_ha_2000.tif']}]
        list_of_s3_name_dicts = [{key: value} for value in simple_file_names for key in [s3_in_folder]]

        # Adds the dictionary of s3 paths and output names for this folder to the list for all folders
        list_of_s3_names_total.append(list_of_s3_name_dicts)

    # Output of above is a nested list, where each input folder is its own inner list. Need to flatten to a list.
    # e.g., [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/AGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__AGC_density_MgC_ha_2000.tif', '00N_120E__AGC_density_MgC_ha_2000.tif']},
    # {'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/BGC_density_MgC_ha/2000/8000_pixels/20240821/': ['00N_110E__BGC_density_MgC_ha_2000.tif', '00N_120E__BGC_density_MgC_ha_2000.tif']}]
    list_of_s3_names_total = flatten_list(list_of_s3_names_total)

    print(
        f"flm: There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders.")

    return list_of_s3_names_total


# Flattens a nested list
def flatten_list(nested_list):
    return [x for xs in nested_list for x in xs]


# Merges rasters that are <10x10 degrees into 10x10 degree rasters in the standard grid.
# Approach is to merge rasters with gdal.Warp and then upload them to s3.
def merge_small_tiles_gdal(s3_name_dict, no_upload):
    in_folder = list(s3_name_dict.keys())[0]  # The input s3 folder for the small rasters
    out_file_name = list(s3_name_dict.values())[0][0]  # The output file name for the combined rasters

    s3_in_folder = in_folder  # The input s3 folder with s3:// prepended
    vsis3_in_folder = f'/vsis3/{in_folder[5:]}'  # The input s3 folder with /vsis3/ prepended

    # Lists all the rasters in the specified s3 folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # Gets the tile_id from the output file name in the standard format
    tile_id = out_file_name[:8]

    # Limits the input rasters to the specified tile_id (the relevant 10x10 area)
    filenames_in_focus_area = [i for i in filenames if tile_id in i]

    # Lists the tile paths for the relevant rasters
    tile_paths = []
    tile_paths = [vsis3_in_folder + filename for filename in filenames_in_focus_area]

    print(f"flm: Merging small rasters in {tile_id} in {vsis3_in_folder}")

    # Names the output folder. Same as the input folder but with the dimensions in pixels replaced
    out_folder = re.sub(r'\d+_pixels', f'{cn.full_raster_dims}_pixels', in_folder)

    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)

    # Dynamically sets the datatype for the merged raster based on the input rasters (courtesy of https://chatgpt.com/share/e/a91c4c98-b2b1-4680-a4a7-453f1a878052)
    # Determines the data type of the first raster
    first_raster_path = tile_paths[0]
    ds = gdal.Open(first_raster_path)
    raster_datatype = ds.GetRasterBand(1).DataType
    raster_nodata_value = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    # Defaults to Float32 if not found
    dtype_str = gdal_dtype_mapping.get(raster_datatype, 'Float32')

    # Merges the rasters (courtesy of ChatGPT: https://chatgpt.com/share/e/13158ebb-dd0a-41d8-8dfb-9ee12e4c804e)
    # This is the only system I found that maintains the extent of all the constituent rasters and doesn't change their resolution or pixel size or shift them.
    # I also tried various gdal_translate, build_vrt, and numpy padding approaches, none of which worked in all cases.
    merged_file = f"/tmp/merged_{out_file_name}"

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

    try:
        subprocess.check_call(merge_command)
        print(f"flm: Successfully merged rasters into {merged_file}")
    except subprocess.CalledProcessError as e:
        print(f"flm: Error merging rasters: {e}")
        return f"failure for {s3_name_dict}"

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call for uploading to work

    print(f"flm: Saving {out_file_name} to s3: {out_folder}{out_file_name}")

    if not no_upload:

        try:
            s3_client.upload_file(merged_file, "gfw2-data", Key=f"{out_folder[15:]}{out_file_name}")  #[15:] drops s3://gfw2-data/ from front
            print(f"flm: Successfully uploaded {out_file_name} to s3")
        except boto3.exceptions.S3UploadFailedError as e:
            print(f"flm: Error uploading file to s3: {e}")
            return f"failure for {s3_name_dict}"

    # Deletes the local merged raster
    os.remove(merged_file)

    return f"success for {s3_name_dict}"


# Creates numpy array of rates or ratios from a tab in an Excel spreadsheet, e.g., removal factors or carbon pool ratios
def convert_lookup_table_to_array(spreadsheet, sheet_name, fields_to_keep):
    # Fetches the file content. Courtesy of ChatGPT: https://chatgpt.com/share/e/aff31681-c9a7-40fe-85c1-73a1cab62066
    response = requests.get(spreadsheet)
    response.raise_for_status()  # Ensure we notice bad responses

    # Converts to Excel. Courtesy of ChatGPT: https://chatgpt.com/share/e/aff31681-c9a7-40fe-85c1-73a1cab62066
    excel_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    # Retains only the relevant columns
    filtered_data = excel_df[fields_to_keep]

    # Converts from dataframe to Numpy array
    filtered_array = filtered_data.to_numpy().astype(
        float)  # Need to convert Pandas dataframe to numpy array because Numba jit-decorated function can't use dataframes.
    filtered_array = filtered_array.astype(
        float)  # Convert from object dtype to float dtype-- necessary for numba to use it

    return filtered_array


# Creates arrays of 0s for any missing inputs and puts them in the corresponding typed dictionary
def complete_inputs(existing_input_list, typed_dict, datatype, chunk_length_pixels, bounds_str, tile_id, is_final, logger):
    for dataset_name in existing_input_list:
        if dataset_name not in typed_dict.keys():
            typed_dict[dataset_name] = np.full((chunk_length_pixels, chunk_length_pixels), 0, dtype=datatype)
            lu.print_and_log(f"Created {dataset_name} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
    return typed_dict


# Calculates stats for a chunk (numpy array)
# From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
def calculate_stats(array, name, bounds_str, tile_id, in_out):
    if array is None or not np.any(array):  # Check if the array is None or empty
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'data_type': 'no data'
        }
    else:    # Only calculates stats if there is data in the array
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': np.min(array),
            'mean_value': np.mean(array),
            'max_value': np.max(array),
            'data_type': array.dtype.name
        }


# Calculates chunk-level stats for all inputs and outputs and saves to Excel spreadsheet
# Also calculates the min and max value for each input and output across all chunks
# From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
def calculate_chunk_stats(all_stats, stage):

    print("Calculating tile stats...")

    # Convert accumulated statistics to a DataFrame
    df_all_stats = pd.DataFrame(all_stats)

    # Convert problematic non-numeric values to NaN
    df_all_stats['min_value'] = pd.to_numeric(df_all_stats['min_value'], errors='coerce')
    df_all_stats['max_value'] = pd.to_numeric(df_all_stats['max_value'], errors='coerce')

    # Sort the DataFrame by 'in_out' and 'layer_name'
    sorted_stats = df_all_stats.sort_values(by=['in_out', 'layer_name']).reset_index(drop=True)

    # Calculate the min and max values for each layer_name
    min_max_stats = df_all_stats.groupby('layer_name').agg(
        min_value=('min_value', 'min'),
        max_value=('max_value', 'max')
    ).reset_index()

    # Creates a dictionary to store separate DataFrames for each 'in_out' value
    # so that input and output layers can be reported on separate tabs.
    # That's necessary for full model runs, where Excel doesn't have enough rows to put all chunk stats in one tab.
    in_out_tables = {in_out_value: sorted_stats[sorted_stats['in_out'] == in_out_value]
                     for in_out_value in sorted_stats['in_out'].unique()}

    # Write the combined statistics to a single Excel file
    #TODO Create chunk_stats folder if it doesn't already exist

    # Should continue with model post-processing even if chunk stats don't work for some reason
    # (e.g., more many rows output than rows in an Excel spreadsheet)
    try:
        with pd.ExcelWriter(f'{cn.chunk_stats_path}{stage}_chunk_statistics_{timestr()}.xlsx') as writer:

            # Writes each 'in_out' DataFrame to its own sheet
            for in_out_value, table in in_out_tables.items():
                sheet_name = f"chunk_stats_{str(in_out_value)}"
                table.to_excel(writer, sheet_name=sheet_name, index=False)

            # Write the min and max statistics to the second sheet
            min_max_stats.to_excel(writer, sheet_name='min_max_for_layers', index=False)

        print(sorted_stats.head())  # Show first few rows of the stats DataFrame for inspection

    except Exception as e:
        print(f"Can't print chunk stats: {e}")


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
        if 'Contents' in response and len(response['Contents']) > 0:
            # Uses the first file in the folder (index 0 instead of 1)
            first_tiles[key] = cn.full_bucket_prefix + "/" + response['Contents'][1]['Key']
        else:
            first_tiles[key] = None  # In case no files are found

    return first_tiles


# Gets the datatype of a raster in s3.
# This seems much faster than the rasterio version that ChatGPT suggested later in the chat.
# From https://chatgpt.com/share/e/a48c768d-0331-43da-9fc6-ef8a84af586c
def get_dtype_from_s3(file_path):

    # Constructs the /vsis3/ path
    vsis3_path = f'/vsis3/{file_path[len("s3://"):]}'
    # print(f"Attempting to open: {vsis3_path}")

    dataset = gdal.Open(vsis3_path)
    if dataset:
        # print(f"Opened file: {vsis3_path}")
        band = dataset.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        # print(f"Data type: {data_type}")
        return data_type
    else:
        raise ValueError(f"Could not open file {vsis3_path}")


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
        dtype = get_dtype_from_s3(file_path)
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
import os
import boto3
import time
import math
import pandas as pd
import subprocess
import re
import concurrent.futures
from osgeo import gdal

# dask/parallelization libraries
import coiled
import dask
from dask.distributed import Client, LocalCluster
from dask.distributed import print
import distributed

# scipy basics
import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows
import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from rioxarray.merge import merge_arrays

# numba
from numba import jit
from numba.typed import Dict
from numba.core import types

# General paths and constants
LC_uri = 's3://gfw2-data/landcover'
s3_out_dir = 'climate/AFOLU_flux_model/LULUCF/outputs'
IPCC_class_max_val = 6

# IPCC codes
forest = 1
cropland = 2
settlement = 3
wetland = 4
grassland = 5
otherland = 6

first_year = 2000
last_year = 2020

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('gfw2-data')
s3_client = boto3.client("s3")

full_raster_dims = 40000
interval_years = 5  # number of years in interval. #TODO: calculate programmatically in numba function rather than coded here-- for greater flexibility.
sig_height_loss_threshold = 5  # meters
biomass_to_carbon = 0.47  # Conversion of biomass to carbon

# GLCLU codes
cropland = 244
builtup = 250

tree_wet_min_height_code = 27
tree_wet_max_height_code = 48
tree_dry_min_height_code = 127
tree_dry_max_height_code = 148

tree_threshold = 5  # Height minimum for trees (meters)


def timestr():
    return time.strftime("%Y%m%d_%H_%M_%S")


def boundstr(bounds):
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


def calc_chunk_length_pixels(bounds):
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


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
def get_chunk_bounds(chunk_params):
    min_x = chunk_params[0]
    min_y = chunk_params[1]
    max_x = chunk_params[2]
    max_y = chunk_params[3]
    chunk_size = chunk_params[4]

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

    lng = f"{str(lng_floor).zfill(3)}E" if (lng_floor >= 0) else f"{str(-lng_floor).zfill(3)}W"
    lat = f"{str(lat_ceil).zfill(2)}N" if (lat_ceil >= 0) else f"{str(-lat_ceil).zfill(2)}S"

    return f"{lat}_{lng}"


# Lazily opens tile within provided bounds (i.e. one chunk) and returns as a numpy array
# If it can't open the chunk (no data in it), it returns an array of the specified nodata value.
def get_tile_dataset_rio(uri, bounds, chunk_length_pixels, no_data_val):
    bounds_str = boundstr(bounds)

    # If the uri exists, the relevant window is opened and returned and returned as an array
    try:
        with rasterio.open(uri) as ds:
            window = rasterio.windows.from_bounds(*bounds, ds.transform)
            data = ds.read(1, window=window)

    # If the uri does not exist, an array of the correct data type and size is created and returned based on a tile that is known to exist
    except:
        uri_revised = re.sub("[0-9]{2}[A-Z][_][0-9]{3}[A-Z]", "00N_110E", uri)  # Substitutes the tile_id for the tile that doesn't exist with a tile_id that does exist
        with rasterio.open(uri_revised) as ds:
            data_type = ds.dtypes[0]  # Retrieves the datatype of the tile
        data = np.full((chunk_length_pixels, chunk_length_pixels), no_data_val).astype(data_type)  # Array of the right size and datatype

    return data


# Prepares list of chunks to download.
# Chunks are defined by a bounding box.
def prepare_to_download_chunk(bounds, download_dict, no_data_val):
    futures = {}

    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = calc_chunk_length_pixels(bounds)

    # Submit requests to S3 for input chunks but don't actually download them yet. This queueing of the requests before downloading them speeds up the downloading
    # Approach is to download all the input chunks up front for every year to make downloading more efficient, even though it means storing more upfront
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}")

        for key, value in download_dict.items():
            futures[executor.submit(get_tile_dataset_rio, value, bounds, chunk_length_pixels, no_data_val)] = key

    return futures


# Checks if tiles exist at all
def check_for_tile(download_dict, is_final):
    s3 = boto3.client('s3')
    i = 0

    while i < len(list(download_dict.values())):
        s3_key = list(download_dict.values())[i][15:]

        # Breaks the loop if the tile exists
        try:
            s3.head_object(Bucket='gfw2-data', Key=s3_key)
            if not is_final:
                print(f"Tile id {list(download_dict.values())[i][-12:-4]} exists. Proceeding.")
            return 1
        except:
            pass

        i += 1

    print(f"Tile id {list(download_dict.values())[0][-12:-4]} does not exist. Skipping chunk.")
    return 0


# Checks whether a chunk has data in it
def check_chunk_for_data(layers, item_to_check, bounds_str, tile_id, no_data_val, is_final):
    i = 0

    while i < len(list(layers.values())):
        # Checks if all the pixels have the nodata value
        min = np.min(list(layers.values())[i])  # Can't use np.all because it doesn't work in chunks that are mostly water; says nodata in chunk even if there is land

        # Breaks the loop if there is data in the chunk; don't need to keep checking chunk for data
        if min < no_data_val:
            if not is_final:
                print(f"Data in chunk {bounds_str}. Proceeding.")
            return 1

        i += 1

    print(f"No data in chunk {bounds_str} for any input.")
    return 0


# Saves array as a raster locally, then uploads it to s3
def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, output_dict, is_final):
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)
    file_info = f'{tile_id}__{bounds_str}'

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():
        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]

        if not is_final:
            print(f"Saving {bounds_str} in {tile_id} for {year_out}: {timestr()}")

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        with rasterio.open(f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels, height=chunk_length_pixels, count=1,
                           dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400, blockysize=400) as dst:
            dst.write(data_array, 1)

        s3_path = f"{s3_out_dir}/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels/{time.strftime('%Y%m%d')}"

        if not is_final:
            print(f"Uploading {bounds_str} in {tile_id} for {year_out} to {s3_path}: {timestr()}")

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
def upload_shp(full_in_folder, in_folder, shp):
    print(f"Uploading to {full_in_folder}{shp}: {timestr()}")

    shp_pattern = shp[:-4]
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(f"/tmp/{shp}", "gfw2-data", Key=f"{in_folder[10:]}{shp}")
    s3_client.upload_file(f"/tmp/{shp_pattern}.dbf", "gfw2-data", Key=f"{in_folder[10:]}{shp_pattern}.dbf")
    s3_client.upload_file(f"/tmp/{shp_pattern}.prj", "gfw2-data", Key=f"{in_folder[10:]}{shp_pattern}.prj")
    s3_client.upload_file(f"/tmp/{shp_pattern}.shx", "gfw2-data", Key=f"{in_folder[10:]}{shp_pattern}.shx")

    os.remove(f"/tmp/{shp}")
    os.remove(f"/tmp/{shp_pattern}.dbf")
    os.remove(f"/tmp/{shp_pattern}.prj")
    os.remove(f"/tmp/{shp_pattern}.shx")


# Makes a shapefile of the footprints of rasters in a folder, for checking geographical completeness of rasters
def make_tile_footprint_shp(input_dict):
    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    # Task properties
    print(f"Making tile index shapefile for: {in_folder}: {timestr()}")

    # Folder including s3 key
    s3_in_folder = f's3://{in_folder}'
    vsis3_in_folder = f'/vsis3/{in_folder}'

    # List of all the filenames in the folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # List of the tile paths in the folder
    tile_paths = [vsis3_in_folder + filename for filename in filenames]
    file_paths = 's3_paths.txt'

    with open(f"/tmp/{file_paths}", 'w') as file:
        for item in tile_paths:
            file.write(item + '\n')

    # Output shapefile name
    shp = f"raster_footprints_{pattern}.shp"
    cmd = ["gdaltindex", "-t_srs", "EPSG:4326", f"/tmp/{shp}", "--optfile", f"/tmp/{file_paths}"]
    subprocess.check_call(cmd)

    # Uploads shapefile to s3
    upload_shp(s3_in_folder, in_folder, shp)

    return f"Completed: {timestr()}"


# Saves an xarray data array locally as a raster and then uploads it to s3
def save_and_upload_raster_10x10(**kwargs):
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    data_array = kwargs['data']  # The data being saved
    out_file_name = kwargs['out_file_name']  # The output file name
    out_folder = kwargs['out_folder']  # The output folder

    print(f"Saving {out_file_name} locally")

    profile_kwargs = {'compress': 'lzw'}  # Adds attribute to compress the output raster
    # data_array.rio.to_raster(f"{out_file_name}", **profile_kwargs)
    data_array.rio.to_raster(f"/tmp/{out_file_name}", **profile_kwargs)

    print(f"Saving {out_file_name} to {out_folder[10:]}{out_file_name}")

    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder[10:]}{out_file_name}")

    # Deletes the local raster
    os.remove(f"/tmp/{out_file_name}")


# Creates a list of 2x2 deg tiles to aggregate into 10x10 deg tiles, where the list is a list of dictionaries of the form
# [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_000E__IPCC_classes_2020.tif', 0]},
# {'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_010E__IPCC_classes_2020.tif', 0]}, ... ]
def create_list_for_aggregation(s3_in_folder_dicts):
    s3_in_folders = [list(item.keys())[0] for item in s3_in_folder_dicts]
    no_data_values = [list(item.values())[0] for item in s3_in_folder_dicts]

    list_of_s3_name_dicts_total = []  # Final list of dictionaries of s3 paths and output aggregated 10x10 rasters

    # Iterates through all the desired s3 folders
    for s3_in_folder, no_data_value in zip(s3_in_folders, no_data_values):
        simple_file_names = []  # List of output aggregated 10x10 rasters

        # Raw filenames in a folder, e.g., ['00N_000E__6_-2_8_0__IPCC_classes_2020.tif', '00N_000E__6_-4_8_-2__IPCC_classes_2020.tif',...]
        filenames = list_rasters_in_folder(f"s3://{s3_in_folder}")

        # Iterates through all the files in a folder and converts them to the output names.
        # Essentially [tile_id]__[pattern].tif. Drops the chunk bounds from the middle.
        for filename in filenames:
            result = filename[:10] + filename[filename.rfind("__") + len("__"):]  # Extracts the relevant parts of the raw file names
            simple_file_names.append(result)  # New list of simplified file names used for 10x10 degree outputs

        # Removes duplicate simplified file names.
        # There are duplicates because each 10x10 output raster has many constituent chunks, each of which have the same aggregated, final name
        # e.g., ['00N_000E__IPCC_classes_2020.tif', '00N_010E__IPCC_classes_2020.tif', ...]
        simple_file_names = np.unique(simple_file_names).tolist()

        # Makes nested lists of the file names and no data values inside the list of all file names,
        # e.g., [['00N_000E__IPCC_classes_2020.tif', 0], ['00N_010E__IPCC_classes_2020.tif', 0], ... ]
        simple_file_names_and_no_data = [[item, no_data_value] for item in simple_file_names]

        # Makes a list of dictionaries, where the key is the input s3 path and the value is the output aggregated name
        # e.g., [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_000E__IPCC_classes_2020.tif', 0]},
        # {'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_010E__IPCC_classes_2020.tif', 0]}, ... ]
        list_of_s3_name_dicts = [{key: value} for value in simple_file_names_and_no_data for key in [s3_in_folder]]

        # Adds the dictionary of s3 paths and output names for this folder to the list for all folders
        list_of_s3_name_dicts_total.append(list_of_s3_name_dicts)

    # Output of above is a nested list, where each input folder is its own inner list. Need to flatten to a list.
    # e.g., [{'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_000E__IPCC_classes_2020.tif', 0]},
    # {'gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/8000_pixels/20240205/': ['00N_010E__IPCC_classes_2020.tif', 0]}, ... ]
    list_of_s3_name_dicts_total = flatten_list(list_of_s3_name_dicts_total)

    print(f"There are {len(list_of_s3_name_dicts_total)} chunks to process in {len(s3_in_folders)} input folders.")

    return list_of_s3_name_dicts_total


# Flattens a nested list
def flatten_list(nested_list):
    return [x for xs in nested_list for x in xs]


# Merges rasters that are <10x10 degrees into 10x10 degree rasters in the standard grid. Not currently using.
# Approach is to open rasters, convert to xarray, merge as xarray datarrays, then save as rasters locally and upload to s3.
def merge_small_tiles_xarray(s3_name_dict):
    in_folder = list(s3_name_dict.keys())[0]  # The input s3 folder for the small rasters
    out_file_name = list(s3_name_dict.values())[0]  # The output file name for the combined rasters

    s3_in_folder = f's3://{in_folder}'  # The input s3 folder with s3:// prepended
    vsis3_in_folder = f'/vsis3/{in_folder}'  # The input s3 folder with /vsis3/ prepended

    # Lists all the rasters in the specified s3 folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # Gets the tile_id from the output file name in the standard format
    tile_id = out_file_name[:8]

    # Limits the input rasters to the specified tile_id (the relevant 10x10 area)
    filenames_in_focus_area = [i for i in filenames if tile_id in i]

    # Lists the tile paths for the relevant rasters
    tile_paths = [s3_in_folder + filename for filename in filenames_in_focus_area]

    print(f"Opening small rasters in {tile_id} in {s3_in_folder}")

    # Opens the relevant rasters in a list of xarray data arrays
    small_rasters = [rioxarray.open_rasterio(tile_path, chunks=True) for tile_path in tile_paths]

    print(f"Merging {tile_id} in {s3_in_folder}")

    nodata_value = 255
    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)  # The bounding box for the output 10x10 deg tile

    # Merges the relevant small data arrays in the list
    # https://corteva.github.io/rioxarray/stable/examples/merge.html
    merged = merge_arrays(small_rasters, bounds=(min_x, min_y, max_x, max_y), nodata=nodata_value)  # Bounds of the output image (left, bottom, right, top))

    # Names the output folder. Same as the input folder but with the dimensions in pixels replaced
    out_folder = re.sub(r'\d+_pixels', f'{full_raster_dims}_pixels', in_folder)

    # Saves the merged xarray data array locally and then to s3
    save_and_upload_raster_10x10(data=merged, out_file_name=out_file_name, out_folder=out_folder)

    del merged

    return f"success for {s3_name_dict}"


# Merges rasters that are <10x10 degrees into 10x10 degree rasters in the standard grid.
# Approach is to merge rasters with gdal.Warp and then upload them to s3.
def merge_small_tiles_gdal(s3_name_no_data_dict):
    in_folder = list(s3_name_no_data_dict.keys())[0]  # The input s3 folder for the small rasters
    out_file_name_no_data = list(s3_name_no_data_dict.values())[0]  # The output file name for the combined rasters and their no data value
    out_file_name = out_file_name_no_data[0]  # The output file name
    no_data = out_file_name_no_data[1]  # The output no data value. Not currently using but it's available.

    s3_in_folder = f's3://{in_folder}'  # The input s3 folder with s3:// prepended
    vsis3_in_folder = f'/vsis3/{in_folder}'  # The input s3 folder with /vsis3/ prepended

    # Lists all the rasters in the specified s3 folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # Gets the tile_id from the output file name in the standard format
    tile_id = out_file_name[:8]

    # Limits the input rasters to the specified tile_id (the relevant 10x10 area)
    filenames_in_focus_area = [i for i in filenames if tile_id in i]

    # Lists the tile paths for the relevant rasters
    tile_paths = [vsis3_in_folder + filename for filename in filenames_in_focus_area]

    print(f"Merging small rasters in {tile_id} in {vsis3_in_folder}")

    # Names the output folder. Same as the input folder but with the dimensions in pixels replaced
    out_folder = re.sub(r'\d+_pixels', f'{full_raster_dims}_pixels', in_folder[10:])  # [10:] to remove the gfw2-data/ at the front

    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)
    output_extent = [min_x, min_y, max_x, max_y]  # Specify the extent in the order [xmin, ymin, xmax, ymax]

    warp_options = gdal.WarpOptions(outputBounds=output_extent, creationOptions=["COMPRESS=LZW"])
    # warp_options = gdal.WarpOptions(outputBounds=output_extent, creationOptions=["COMPRESS=LZW"], dstNodata=no_data)

    # Merges all output small rasters with the options above
    gdal.Warp(f"/tmp/{out_file_name}", tile_paths, options=warp_options)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    print(f"Saving {out_file_name} to s3: {out_folder}{out_file_name}")

    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder}{out_file_name}")

    # Deletes the local raster
    os.remove(f"/tmp/{out_file_name}")

    return f"success for {s3_name_no_data_dict}"


@jit(nopython=True)
def accrete_node(combo, new):
    combo = combo * 10 + new
    return combo

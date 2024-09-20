# utils/raster_utils.py

import os
import subprocess
import re
import numpy as np
import rasterio
from osgeo import gdal
from typing import List, Dict
import boto3
from numba import jit
from utils.logging_utils import timestr, print_and_log

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


def save_and_upload_raster_10x10(**kwargs):
    """
    Saves an xarray DataArray locally as a raster and then uploads it to S3.

    Args:
        **kwargs: Dictionary containing 'data', 'out_file_name', and 'out_folder'.
    """
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    data_array = kwargs['data']  # The data being saved
    out_file_name = kwargs['out_file_name']  # The output file name
    out_folder = kwargs['out_folder']  # The output folder

    print(f"flm: Saving {out_file_name} locally")

    profile_kwargs = {'compress': 'lzw'}  # Adds attribute to compress the output raster
    data_array.rio.to_raster(f"/tmp/{out_file_name}", **profile_kwargs)

    print(f"flm: Saving {out_file_name} to {out_folder[10:]}{out_file_name}")

    s3_client.upload_file(f"/tmp/{out_file_name}", "gfw2-data", Key=f"{out_folder[10:]}{out_file_name}")

    # Deletes the local raster
    os.remove(f"/tmp/{out_file_name}")


def create_list_for_aggregation(s3_in_folders: List[str]) -> List[Dict[str, List[str]]]:
    """
    Creates a list of dictionaries mapping input S3 folders to output aggregated raster names.

    Args:
        s3_in_folders (List[str]): List of input S3 folder paths.

    Returns:
        List[Dict[str, List[str]]]: List of dictionaries for aggregation.
    """
    from utils.s3_utils import list_rasters_in_folder  # Avoid circular imports

    list_of_s3_names_total = []  # Final list of dictionaries of input S3 paths and output aggregated 10x10 raster names

    # Iterates through all the input S3 folders
    for s3_in_folder in s3_in_folders:
        simple_file_names = []  # List of output aggregated 10x10 rasters

        # Raw filenames in an input folder
        filenames = list_rasters_in_folder(f"s3://{s3_in_folder}")

        # Iterates through all the files in a folder and converts them to the output names.
        # Essentially [tile_id]__[pattern].tif. Drops the chunk bounds from the middle.
        for filename in filenames:
            result = filename[:10] + filename[filename.rfind("__") + len("__"):]
            simple_file_names.append(result)  # New list of simplified file names used for 10x10 degree outputs

        # Removes duplicate simplified file names.
        simple_file_names = np.unique(simple_file_names).tolist()

        # Makes nested lists of the file names.
        simple_file_names = [[item] for item in simple_file_names]

        # Makes a list of dictionaries, where the key is the input S3 path and the value is the output aggregated name
        list_of_s3_name_dicts = [{s3_in_folder: value} for value in simple_file_names]

        # Adds the dictionary of S3 paths and output names for this folder to the list for all folders
        list_of_s3_names_total.append(list_of_s3_name_dicts)

    # Flatten the nested list
    list_of_s3_names_total = flatten_list(list_of_s3_names_total)

    print(
        f"flm: There are {len(list_of_s3_names_total)} 10x10 deg rasters to create across {len(s3_in_folders)} input folders.")

    return list_of_s3_names_total


def merge_small_tiles_gdal(s3_name_dict: Dict[str, List[str]], full_raster_dims: str) -> str:
    """
    Merges smaller rasters into a 10x10 degree raster using GDAL and uploads to S3.

    Args:
        s3_name_dict (Dict[str, List[str]]): Dictionary mapping input S3 folder to list of raster filenames.
        full_raster_dims (str): The raster dimensions to replace in the output folder path.

    Returns:
        str: Success or failure message.
    """
    in_folder = list(s3_name_dict.keys())[0]  # The input S3 folder for the small rasters
    out_file_name = list(s3_name_dict.values())[0][0]  # The output file name for the combined rasters

    s3_in_folder = f's3://{in_folder}'  # The input S3 folder with s3:// prepended
    vsis3_in_folder = f'/vsis3/{in_folder}'  # The input S3 folder with /vsis3/ prepended

    # Lists all the rasters in the specified S3 folder
    from utils.s3_utils import list_rasters_in_folder  # Avoid circular imports
    filenames = list_rasters_in_folder(s3_in_folder)

    # Gets the tile_id from the output file name in the standard format
    tile_id = out_file_name[:8]

    # Limits the input rasters to the specified tile_id (the relevant 10x10 area)
    filenames_in_focus_area = [i for i in filenames if tile_id in i]

    # Lists the tile paths for the relevant rasters
    tile_paths = [f"{vsis3_in_folder}{filename}" for filename in filenames_in_focus_area]

    print(f"flm: Merging small rasters in {tile_id} in {vsis3_in_folder}")

    # Names the output folder. Same as the input folder but with the dimensions in pixels replaced
    out_folder = re.sub(r'\d+_pixels', f'{full_raster_dims}_pixels',
                        in_folder[10:])  # [10:] to remove the 'gfw2-data/' at the front

    from utils.chunk_utils import get_10x10_tile_bounds  # Avoid circular imports
    min_x, min_y, max_x, max_y = get_10x10_tile_bounds(tile_id)

    output_extent = [min_x, min_y, max_x, max_y]  # Specify the extent in the order [xmin, ymin, xmax, ymax]

    # Dynamically sets the datatype for the merged raster based on the input rasters
    first_raster_path = tile_paths[0]
    ds = gdal.Open(first_raster_path)
    raster_datatype = ds.GetRasterBand(1).DataType
    raster_nodata_value = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    # Defaults to Float32 if not found
    dtype_str = gdal_dtype_mapping.get(raster_datatype, 'Float32')

    # Merges the rasters
    merged_file = f"/tmp/merged_{out_file_name}"

    merge_command = [
        'gdal_merge.py',
        '-o', merged_file,
        '-of', 'GTiff',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'TILED=YES',  # Internal tiling
        '-co', 'BLOCKXSIZE=400',
        '-co', 'BLOCKYSIZE=400',
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

    try:
        s3_client.upload_file(merged_file, "gfw2-data", Key=f"{out_folder}{out_file_name}")
        print(f"flm: Successfully uploaded {out_file_name} to s3")
    except boto3.exceptions.S3UploadFailedError as e:
        print(f"flm: Error uploading file to s3: {e}")
        return f"failure for {s3_name_dict}"

    # Deletes the local merged raster
    os.remove(merged_file)

    return f"success for {s3_name_dict}"


def save_and_upload_small_raster_set(bounds: List[float], chunk_length_pixels: int, tile_id: str, bounds_str: str,
                                     output_dict: Dict[str, List], is_final: bool, logger, s3_out_dir: str,
                                     no_data_val=None):
    """
    Saves arrays as rasters locally, then uploads them to S3. Optionally includes NoData values.

    Args:
        bounds (List[float]): Bounding box for the chunk.
        chunk_length_pixels (int): Length of the chunk in pixels.
        tile_id (str): Tile ID.
        bounds_str (str): String representation of chunk bounds.
        output_dict (Dict[str, List]): Dictionary containing output data.
        is_final (bool): Flag indicating if this is a final stage.
        logger (logging.Logger): Logger instance.
        s3_out_dir (str): S3 output directory.
        no_data_val: Optional NoData value for the raster.
    """
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)

    file_info = f'{tile_id}__{bounds_str}'

    for key, value in output_dict.items():
        data_array = value[0]
        data_type = value[1]
        data_meaning = value[2]
        year_out = value[3]

        if is_final:
            file_name = f"{file_info}__{key}.tif"
        else:
            file_name = f"{file_info}__{key}__{timestr()}.tif"

        print_and_log(f"Saving {bounds_str} in {tile_id} for {year_out}: {timestr()}", is_final, logger)

        # Includes NoData value in output raster
        if no_data_val is not None:
            with rasterio.open(
                    f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels, height=chunk_length_pixels,
                    count=1,
                    dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                    blockysize=400, nodata=no_data_val
            ) as dst:
                dst.write(data_array, 1)
        else:
            with rasterio.open(
                    f"/tmp/{file_name}", 'w', driver='GTiff', width=chunk_length_pixels, height=chunk_length_pixels,
                    count=1,
                    dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400,
                    blockysize=400
            ) as dst:
                dst.write(data_array, 1)

        s3_path = f"{s3_out_dir}/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels/{timestr()}"

        print_and_log(f"Uploading {bounds_str} in {tile_id} for {year_out} to {s3_path}: {timestr()}", is_final, logger)

        s3_client.upload_file(f"/tmp/{file_name}", "gfw2-data", Key=f"{s3_path}/{file_name}")

        # Deletes the local raster
        os.remove(f"/tmp/{file_name}")


def accrete_node(combo, new):
    """
    Accretes a new node value into the combo using Numba for performance.

    Args:
        combo (int): Existing combo value.
        new (int): New value to accrete.

    Returns:
        int: Updated combo value.
    """
    return combo * 10 + new


accrete_node = jit(nopython=True)(accrete_node)


def make_tile_footprint_shp(input_dict: Dict[str, str]) -> str:
    """
    Makes a shapefile of the footprints of rasters in a folder for checking geographical completeness.

    Args:
        input_dict (Dict[str, str]): Dictionary mapping input folders to patterns.

    Returns:
        str: Completion message with timestamp.
    """
    from utils.s3_utils import upload_shp  # Avoid circular imports

    in_folder = list(input_dict.keys())[0]
    pattern = list(input_dict.values())[0]

    # Task properties
    print(f"flm: Making tile index shapefile for: {in_folder}: {timestr()}")

    # Folder including S3 key
    s3_in_folder = f's3://{in_folder}'
    vsis3_in_folder = f'/vsis3/{in_folder}'

    # List of all the filenames in the folder
    filenames = list_rasters_in_folder(s3_in_folder)

    # List of the tile paths in the folder
    tile_paths = [f"{vsis3_in_folder}{filename}" for filename in filenames]

    file_paths = 's3_paths.txt'

    with open(f"/tmp/{file_paths}", 'w') as file:
        for item in tile_paths:
            file.write(item + '\n')

    # Output shapefile name
    shp = f"raster_footprints_{pattern}.shp"

    cmd = ["gdaltindex", "-t_srs", "EPSG:4326", f"/tmp/{shp}", "--optfile", f"/tmp/{file_paths}"]
    subprocess.check_call(cmd)

    # Uploads shapefile to S3
    upload_shp(s3_in_folder, in_folder, shp)

    return f"Completed: {timestr()}"

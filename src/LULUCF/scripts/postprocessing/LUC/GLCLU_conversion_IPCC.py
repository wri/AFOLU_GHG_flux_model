"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -bb 10 49.75 10.25 50 -cs 0.25 --run_local --no_upload --run_date YYYYMMDD

Coiled small tests (0.25x0.25 deg chunk):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn IPCC_land_use_change
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_change -bb 116.25 -2.25 116.5 -2 -cs 0.25 --run_date YYYYMMDD

Coiled small tests (1x1 deg chunk):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn IPCC_land_use_change
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_change -bb -64 -22 -63 -21 -cs 1 --create_zarr --run_date YYYYMMDD

Coiled test (10x10 deg chunk):

Full run:

"""

import argparse
import concurrent.futures
import gc
import os
import psutil
import time
import sys
import pandas as pd
import numpy as np

import fsspec
import xarray as xr

from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print
from numba import jit

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"



# Returns boolean values for whether a pixel is planted forest or tree crop
def get_sdpt_status(sdpt_type):
    sdpt_planted_forest = not np.isnan(sdpt_type) and int(sdpt_type) == 1
    sdpt_tree_crop      = not np.isnan(sdpt_type) and int(sdpt_type) == 2

    return sdpt_planted_forest, sdpt_tree_crop

# Returns boolean values for whether a pixel is oil palm using SDPT simpleName, Descals oil palm planting year, or pre-2000 plantation
def get_oil_palm_status(pre_2000_plantation, sdpt_name, descals_planting_year,  year):

    pre_2000 = not np.isnan(pre_2000_plantation) and int(pre_2000_plantation) == 1
    sdpt_oil_palm = not np.isnan(sdpt_name) and int(sdpt_name) == 1
    descals_oil_palm = not np.isnan(descals_planting_year) and 0 < int(descals_planting_year) <= int(year)

    oil_palm = (pre_2000 or sdpt_oil_palm or descals_oil_palm)

    return oil_palm


# Move general utilities from here up to UU
#######################################################################################################################
# IPCC Land use hierarchy: Settlements > Cropland > Forest Land > Grassland > Wetlands > Other
# GLAD LC numeric values --> Default IPCC LU assignment
settlement_lc   = {250}                                         # Built up
cropland_lc     = {244}                                         # Cropland
forest_lc       = set(range(27, 49)) | set(range(127, 149))     # Tall vegetation
grass_lc        = set(range(5, 27)) | set(range(105, 127))      # Short veg
wetland_lc      = set(range(200, 205))                          # Wetland
bare_lc         = set(range(0, 5)) | set(range(100, 105))       # Bare
water_lc        = set(range(205, 208))                          # Open water
ice_lc          = {241}                                         # Snow/ice

# Lookup table to go from GLAD LC -> LU token
lc_token_map = {
    **{v: "S" for v in settlement_lc},
    **{v: "C" for v in cropland_lc},
    **{v: "F" for v in forest_lc},
    **{v: "G" for v in grass_lc},
    **{v: "W" for v in wetland_lc},
    **{v: "B" for v in bare_lc},
    **{v: "O" for v in water_lc},
    **{v: "I" for v in ice_lc},
}

# Node codes describing which rule was used to determine final land use
node_code_map = {
    "built_glad": 10,

    "crop_glad": 20,
    "crop_oil_palm": 21,
    "crop_sdpt_tree_crop": 22,
    "crop_perm_ag_driver": 23,

    "forest_glad": 30,
    "forest_gmw_mangrove": 31,
    "forest_sdpt_planted_forest": 32,
    "forest_shift_cult_driver": 333,
    "forest_logging_driver": 334,
    "forest_wildfire_driver": 335,
    "forest_nat_dist_driver": 337,

    "grass_glad": 40,
    "grass_perm_ag_driver": 41,

    "wetland_glad": 50,

    "bare_glad": 60,
    "water_glad": 61,
    "ice_glad": 62,
}

# Default node codes before rules are applied
def default_node_code(token):
    if token == "S":
        return node_code_map["built_glad"]
    if token == "C":
        return node_code_map["crop_glad"]
    if token == "F":
        return node_code_map["forest_glad"]
    if token == "G":
        return node_code_map["grass_glad"]
    if token == "W":
        return node_code_map["wetland_glad"]
    if token == "B":
        return node_code_map["bare_glad"]
    if token == "O":
        return node_code_map["water_glad"]
    if token == "I":
        return node_code_map["ice_glad"]
    return None

# Function to get land use token per land cover numeric value (tokens used for regex exception rules)
def token_for_lc(v):
    return lc_token_map.get(v, "-")

def set_tokens(tokens, node_codes, indices, new_token, node_code):
    for i in indices:
        tokens[i] = new_token
        node_codes[i] = node_code




def apply_regex_rules(LC_timeseries, tcl_before_ts, driver, oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove):





    return LU_timeseries


# TODO: Does this need to use numba?
def IPCC_land_use(in_dict):

    # Dictionary for output arrays: IPCC land use class, land use node code, land use transition, and land use trajectory
    out_dict = {}

    # Input data
    LC_2015_block = in_dict[f"{cn.land_cover_pattern}_2015"]
    LC_2016_block = in_dict[f"{cn.land_cover_pattern}_2016"]
    LC_2017_block = in_dict[f"{cn.land_cover_pattern}_2017"]
    LC_2018_block = in_dict[f"{cn.land_cover_pattern}_2018"]
    LC_2019_block = in_dict[f"{cn.land_cover_pattern}_2019"]
    LC_2020_block = in_dict[f"{cn.land_cover_pattern}_2020"]
    LC_2021_block = in_dict[f"{cn.land_cover_pattern}_2021"]
    LC_2022_block = in_dict[f"{cn.land_cover_pattern}_2022"]
    LC_2023_block = in_dict[f"{cn.land_cover_pattern}_2023"]
    LC_2024_block = in_dict[f"{cn.land_cover_pattern}_2024"]

    tcl_block = in_dict[cn.tree_cover_loss_pattern]
    drivers_block = in_dict[cn.drivers_pattern]

    oil_palm_2000_extent_block = in_dict[cn.oil_palm_2000_extent_pattern]  # IDN/ MYS pre-2000 plantation
    oil_palm_first_year_block = in_dict[cn.oil_palm_first_year_pattern]    # Descals oil palm planting year

    planted_forest_tree_crop_block = in_dict[cn.planted_forest_tree_crop_pattern]
    planted_forest_type_block = in_dict[cn.planted_forest_type_pattern]


    # Mangrove extent
    # TODO: Read in as a union so only 1 tile set needed
    mangrove_extent_1996_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_1996"]
    mangrove_extent_2007_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2007"]
    mangrove_extent_2008_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2008"]
    mangrove_extent_2009_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2009"]
    mangrove_extent_2010_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2010"]
    mangrove_extent_2015_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2015"]
    mangrove_extent_2016_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2016"]
    mangrove_extent_2017_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2017"]
    mangrove_extent_2018_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2018"]
    mangrove_extent_2019_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2019"]
    mangrove_extent_2020_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2020"]

    # GPW cultivated grassland extent
    # TODO: Read in as a union so only 1 tile set needed


    # Filters tcl_block to only where tcl occurred before 2015 (ignoring 0s)
    #pre_2015_tcl_mask_block = ((tcl_block > 0) & (tcl_block < 15)).astype(np.uint8)

    # Add empty arrays for output datasets
    for year in cn.years_annual:
        out_dict[f"{cn.IPCC_class_pattern}_{year}"] =  np.zeros(LC_2015_block.shape, dtype=np.uint8)
        out_dict[f"{cn.IPCC_node_pattern}_{year}"] = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    for year in cn.years_annual[:-1]:
        out_dict[f"{cn.IPCC_change_pattern}_{year}_{year+1}"] = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    out_dict[f"{cn.IPCC_summary_pattern}"] = np.zeros(LC_2016_block.shape, dtype=np.uint32)

    # Iterates through all pixels in the chunk
    for row in range(LC_2015_block.shape[0]):
        for col in range(LC_2015_block.shape[1]):

            ### Reads input pixel values
            LC_2015 = LC_2015_block[row, col]
            LC_2016 = LC_2016_block[row, col]
            LC_2017 = LC_2017_block[row, col]
            LC_2018 = LC_2018_block[row, col]
            LC_2019 = LC_2019_block[row, col]
            LC_2020 = LC_2020_block[row, col]
            LC_2021 = LC_2021_block[row, col]
            LC_2022 = LC_2022_block[row, col]
            LC_2023 = LC_2023_block[row, col]
            LC_2024 = LC_2024_block[row, col]
            LC_timeseries = np.array([LC_2015, LC_2016, LC_2017, LC_2018, LC_2019, LC_2020, LC_2021, LC_2022, LC_2023, LC_2024]).astype('uint8')

            tcl_year = tcl_block[row, col]
            driver = drivers_block[row, col]
            try:
                tcl_before_ts = (int(tcl_year) <= 2015)
            except:
                tcl_before_ts = False

            planted_forest_tree_crop = planted_forest_tree_crop_block[row, col]     # simpleName
            sdpt_planted_forest, sdpt_tree_crop = get_sdpt_status(planted_forest_tree_crop)

            oil_palm_2000_extent = oil_palm_2000_extent_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]  # simpleType
            oil_palm_first_year = oil_palm_first_year_block[row, col]
            oil_palm = get_oil_palm_status(oil_palm_2000_extent, planted_forest_type, oil_palm_first_year, 2024)
            #TODO: Come back to this if allowing planting year logic during LU timeseries (i.e. F -> C in tall veg remaining tall veg)

            # Mangrove extent years (1 = mangrove, 0 = no mangrove)
            mang_1996 = mangrove_extent_1996_block[row, col]
            mang_2007 = mangrove_extent_2007_block[row, col]
            mang_2008 = mangrove_extent_2008_block[row, col]
            mang_2009 = mangrove_extent_2009_block[row, col]
            mang_2010 = mangrove_extent_2010_block[row, col]
            mang_2015 = mangrove_extent_2015_block[row, col]
            mang_2016 = mangrove_extent_2016_block[row, col]
            mang_2017 = mangrove_extent_2017_block[row, col]
            mang_2018 = mangrove_extent_2018_block[row, col]
            mang_2019 = mangrove_extent_2019_block[row, col]
            mang_2020 = mangrove_extent_2020_block[row, col]
            mang_timeseries = np.array([mang_1996, mang_2007, mang_2008, mang_2009, mang_2010, mang_2015, mang_2016, mang_2017, mang_2018, mang_2019, mang_2020]).astype('uint8')
            gmw_mangrove = bool(np.any(mang_timeseries == 1))

            # Pass in values for regex rules
            a, b, c, d, e, f = apply_regex_rules(LC_timeseries, tcl_before_ts, driver, oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove)





def calculate_and_upload_IPCC_land_use(bounds, download_dict_with_data_types, is_large_run, no_upload, output_folders, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []
    process = psutil.Process(os.getpid())
    logger_worker = lu.setup_logging_worker()
    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels

    ### Part 1: Downloads all inputs for chunk.
    # Replaces the placeholder tile_id in the download data dictionary with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_large_run, logger_worker, False)
    print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        data, status = future.result()
        if 'success' not in status:  # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
        layers[layer] = data

    ### Part 2: Calculates min, mode, and max for each input chunk.
    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_ipcc_stats(array, key, bounds_str, tile_id, 'input_layer'))
    print(chunk_stats)

    # Frees up a little memory 
    del updated_download_dict
    del futures
    gc.collect()
    
    

    ### Part 4: IPCC land use assignment

    lu.print_and_log(f"Determining IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}",False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)
    ipcc_start = time.time()

    out_dict = IPCC_land_use()
    print("out_dict:", out_dict)

    ipcc_end = time.time()
    lu.print_and_log(f"Done assigning IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after IPCC stage completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    lu.print_and_log(f"Assigned IPCC land use in {bounds_str} in {tile_id} in {round(ipcc_end - ipcc_start)} seconds: {uu.timestr()}",False, logger_worker)



    ### Part 6: Calculates chunk stats
    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers
    for key, array in out_dict.items():
        chunk_stats.append(uu.calculate_ipcc_stats(array, key, bounds_str, tile_id, 'output_layer'))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    ### Part 7: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if no_upload == False:
        out_no_data_val = 0
        print("output_folders:", output_folders)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict.items():
            data_type = value.dtype.name
            print("key", key)
            print("data_type:", data_type)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)
            print("out_pattern:", out_pattern)
            print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output
            matched_output_s3_folders = [item for item in output_folders if out_pattern_without_pixel_meaning in item]
            print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if year_range in item]
            print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict, is_large_run, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.outputs_path}: {uu.timestr()}", is_large_run, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_date, run_local, no_log, no_upload, chunk_shapefile_uri, bounding_box, chunk_size_deg, first_chunks, log_note):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'IPCC_land_use'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # batch_size = 3200   # 6 batches to cover all chunks
    batch_size = 5      # For testing batch processing

    # Determines if arguments for start and end year are valid
    start_year = cn.first_model_year_annual
    end_year = cn.last_model_year_annual
    # TODO: Delete?

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()  # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals, and the model output years for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(end_year, main_logger, start_year)
    # TODO: Delete?

    # Returns a dataframe of chunk_ids and iso code from the GADM4.1 1x1 deg fishnet used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Placeholder tile_id to obtain the datatype of each input tile set. Overwritten when chunks are assigned and analyzed.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to LU assignment).
    download_dict = {
        # TCL year
        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",

        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_dir}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_dir}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_dir}{sample_tile_id}.tif",
        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_dir}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
        # Global pasture watch
    }

    # GLCLU timeseries
    for year in cn.years_annual:
        download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"

    # GMW mangrove extent timeseries
    for year in cn.mangrove_extent_years:
        download_dict[f"{cn.mangrove_extent_processed_pattern}_{year}"] = f"{cn.mangrove_extent_processed_dir}{year}/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_{year}.tif"


    # Replaces the placeholder parts of the input paths with relevant values
    # download_dict = {key: value.replace("CHUNK_SIZE", '40000') for key, value in download_dict.items()}
    # download_dict = {key: value.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning) for key, value in download_dict.items()}
    #TODO: Delete?

    print("Download dictionary::")
    for key, item in download_dict.items():
        print(f"{key}: {item}")

    # Returns the first tile in each input so that the datatype can be determined per dataset
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # main_logger.info(f"download_dict_with_data_types for {stage}:")
    # for key, value in download_dict_with_data_types.items():
    #     main_logger.info(f"  {key}: {value}")
    # TODO: Delete?

    # Creates a list of output directories for all outputs
    output_dir_list_core_intermediate = [cn.IPCC_class_dir, cn.IPCC_node_dir, cn.IPCC_change_dir]
    output_dir_list = uu.create_output_dir_name_list(output_dir_list_core_intermediate, interval_type, start_year, chunk_size_pixels,
                            model_type, interval_end_years, interval_year_diff_list, run_date, False)
    output_dir_list.sort()  # Alphabetically order the outputs (modifies output_dir_list)

    main_logger.info(f"output_dir_list for {stage}:")
    for item in output_dir_list:
        main_logger.info(f"  {item}")



    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log" + "\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    all_results = []
    all_1x1_stats = []
    success_count = 0  # Count of successful chunks

    # TODO: Run locally or in coiled
    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        # This approach handles large task lists (graphs) better than [dask.delayed()]
        futures = []
        for chunk in chunk_batch:
            future = client.submit(calculate_and_upload_IPCC_land_use, )
            futures.append(future)
        batch_results = client.gather(futures)

        all_results.extend(batch_results)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, True, main_logger, batch_results)
        all_1x1_stats.extend(batch_stats)

        # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
        # That way there are some basic chunk stats (not sorted or anything) to fall back on.
        if len(chunk_batches) > 1:
            main_logger.info(f"Writing batch stats to spreadsheet: {uu.timestr()}")
            df_batch_stats = pd.DataFrame(batch_stats)
            out_spreadsheet = f'TEMP_BATCH_{stage}__batch_{i}_{uu.timestr()}.xlsx'
            local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"
            with pd.ExcelWriter(local_spreadsheet) as writer:
                df_batch_stats.to_excel(writer, sheet_name=f'stats__batch_{i}', index=False)

        del futures
        del batch_results
        client.run(gc.collect)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create IPCC land use classes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size_deg = args.chunk_size_deg
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, run_date, run_local, no_log, no_upload, chunk_shapefile_uri, bounding_box=bounding_box,
         chunk_size_deg=chunk_size_deg, first_chunks=first_chunks, log_note=log_note)





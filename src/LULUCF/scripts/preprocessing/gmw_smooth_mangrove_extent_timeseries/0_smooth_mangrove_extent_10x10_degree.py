"""
Run from src/LULUCF/

Local:
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -bb 110 -10 120 0 -cs 1 --run_local

5x5 degree chunks for testing:
#with mangroves
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -bb 110 -5 115 0 -cs 1 --run_local
#without mangroves
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -bb 60 -15 65 -10 -cs 1 --run_local

1x1 degree chunks for testing:
#with mangroves
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -bb 116 -3 117 -2 -cs 1 --run_local
#without mangroves
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -bb 60 -11 61 -10 -cs 1 --run_local

Coiled test run:
python -m scripts.utilities.create_cluster -n 4 -t 2 -m 64 -cn mangrove_smoothing_10x10deg
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_10x10_degree -cn mangrove_smoothing_10x10deg -bb 110 -10 120 0 -cs 1

todo:
- see if it can scale from 1 degree to 10 degrees after testing
- Go year by year instead of smoothing all years at once?

flm: Processing 10x10 tile 00N_110E with 100 1x1 deg chunks
/home/melrose94/miniforge3/envs/afolu/lib/python3.13/site-packages/distributed/client.py:3370: UserWarning: Sending large graph of size 16.39 GiB.
This may cause some slowdown.
Consider loading the data with Dask directly
 or using futures or delayed objects to embed the data into the graph without repetition.
See also https://docs.dask.org/en/stable/best-practices.html#load-data-with-dask for more information.
  warnings.warn(

"""
import argparse
from collections import defaultdict
import gc
import numpy as np
from numba import jit
import dask
from dask.distributed import print
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Project imports
from ...utilities import constants_and_names as cn
from ...utilities import universal_utilities as uu
from ...utilities import log_utilities as lu
from ...utilities import numba_utilities as nu
from ...utilities import resize_cluster

#Function to smooth the mangrove data:
    # fills in pixels where there is no mangrove extent (0) one year, but there is mangrove extent (1) in the previous year and the following year
    # then removes false positives (1) if there is no mangrove extent (0) in the previous year and the following year
@jit(nopython=True)
def smooth_mangrove_data(in_dict_uint8):
    # Dictionary for output numpy arrays
    out_dict_uint8 = {}

    #Input data blocks: hansenized mangrove extent
    # NOTE: You can't use f strings in a numba funciton to put together the dictionary key from cn constants.
    # These dictionary keys will need to be updated when we switch to GMWv4
    mangrove_extent_1996_block = in_dict_uint8["GMWv3_mangrove_extent_1996"]
    mangrove_extent_2007_block = in_dict_uint8["GMWv3_mangrove_extent_2007"]
    mangrove_extent_2008_block = in_dict_uint8["GMWv3_mangrove_extent_2008"]
    mangrove_extent_2009_block = in_dict_uint8["GMWv3_mangrove_extent_2009"]
    mangrove_extent_2010_block = in_dict_uint8["GMWv3_mangrove_extent_2010"]
    mangrove_extent_2015_block = in_dict_uint8["GMWv3_mangrove_extent_2015"]
    mangrove_extent_2016_block = in_dict_uint8["GMWv3_mangrove_extent_2016"]
    mangrove_extent_2017_block = in_dict_uint8["GMWv3_mangrove_extent_2017"]
    mangrove_extent_2018_block = in_dict_uint8["GMWv3_mangrove_extent_2018"]
    mangrove_extent_2019_block = in_dict_uint8["GMWv3_mangrove_extent_2019"]
    mangrove_extent_2020_block = in_dict_uint8["GMWv3_mangrove_extent_2020"]

    # Output data blocks: smoothed mangrove extent (empty blocks)
    # Initial and final year of consecutive years are not modified, only the years in between.
    # So smoothed data for those years will be identical to raw data and copied directly from input blocks
    mangrove_extent_2008_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2008"].shape).astype('uint8')
    mangrove_extent_2009_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2009"].shape).astype('uint8')
    mangrove_extent_2016_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2016"].shape).astype('uint8')
    mangrove_extent_2017_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2017"].shape).astype('uint8')
    mangrove_extent_2018_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2018"].shape).astype('uint8')
    mangrove_extent_2019_out_block = np.zeros(in_dict_uint8["GMWv3_mangrove_extent_2019"].shape).astype('uint8')

    # STEP 1: Fill in missing mangrove extent years (i.e. "false negatives") using raw data
    # Iterates through all pixels in the chunk
    for row in range(mangrove_extent_2007_block.shape[0]):
        for col in range(mangrove_extent_2007_block.shape[1]):

            # Input values for specific cell
            mangrove_extent_2007 = mangrove_extent_2007_block[row, col]
            mangrove_extent_2008 = mangrove_extent_2008_block[row, col]
            mangrove_extent_2009 = mangrove_extent_2009_block[row, col]
            mangrove_extent_2010 = mangrove_extent_2010_block[row, col]
            mangrove_extent_2015 = mangrove_extent_2015_block[row, col]
            mangrove_extent_2016 = mangrove_extent_2016_block[row, col]
            mangrove_extent_2017 = mangrove_extent_2017_block[row, col]
            mangrove_extent_2018 = mangrove_extent_2018_block[row, col]
            mangrove_extent_2019 = mangrove_extent_2019_block[row, col]
            mangrove_extent_2020 = mangrove_extent_2020_block[row, col]

            # Not modifying 2007 since it is the start of the consecutive time period (2007 - 2010)
            # Adding in mangrove extent in 2008 if there is mangrove extent in 2007 and 2009
            if mangrove_extent_2008 == 1:
                mangrove_extent_2008_out_block[row, col] = 1
            elif (mangrove_extent_2007 == 1 and mangrove_extent_2009 == 1):
                mangrove_extent_2008_out_block[row, col] = 1
            else:
                mangrove_extent_2008_out_block[row, col] = 0

            # Adding in mangrove extent in 2009 if there is mangrove extent in 2008 and 2010
            if mangrove_extent_2009 == 1:
                mangrove_extent_2009_out_block[row, col] = 1
            elif (mangrove_extent_2008 == 1 and mangrove_extent_2010 == 1):
                mangrove_extent_2009_out_block[row, col] = 1
            else:
                mangrove_extent_2009_out_block[row, col] = 0

            # Not modifying 2010 since it is the end of the consecutive time period (2007 - 2010)
            # Not modifying 2015 since it is the start of the consecutive time period (2015 - 2020)
            # Adding in mangrove extent in 2016 if there is mangrove extent in 2015 and 2017
            if mangrove_extent_2016 == 1:
                mangrove_extent_2016_out_block[row, col] = 1
            elif (mangrove_extent_2015 == 1 and mangrove_extent_2017 == 1):
                mangrove_extent_2016_out_block[row, col] = 1
            else:
                mangrove_extent_2016_out_block[row, col] = 0

            # Adding in mangrove extent in 2017 if there is mangrove extent in 2016 and 2018
            if mangrove_extent_2017 == 1:
                mangrove_extent_2017_out_block[row, col] = 1
            elif (mangrove_extent_2016 == 1 and mangrove_extent_2018 == 1):
                mangrove_extent_2017_out_block[row, col] = 1
            else:
                mangrove_extent_2017_out_block[row, col] = 0

            # Adding in mangrove extent in 2018 if there is mangrove extent in 2017 and 2019
            if mangrove_extent_2018 == 1:
                mangrove_extent_2018_out_block[row, col] = 1
            elif (mangrove_extent_2017 == 1 and mangrove_extent_2019 == 1):
                mangrove_extent_2018_out_block[row, col] = 1
            else:
                mangrove_extent_2018_out_block[row, col] = 0

            # Adding in mangrove extent in 2019 if there is mangrove extent in 2018 and 2020
            if mangrove_extent_2019 == 1:
                mangrove_extent_2019_out_block[row, col] = 1
            elif (mangrove_extent_2018 == 1 and mangrove_extent_2020 == 1):
                mangrove_extent_2019_out_block[row, col] = 1
            else:
                mangrove_extent_2019_out_block[row, col] = 0
            # Not modifying 2020 since it is the end of the consecutive time period (2015 - 2020)

    # STEP 2: Remove "false positive" in mangrove extent years using processed out_block data
    # Iterates through all pixels in the chunk
    for row in range(mangrove_extent_2007_block.shape[0]):
        for col in range(mangrove_extent_2007_block.shape[1]):

            # Input values for specific cell (overwrite smoothed data)
            mangrove_extent_2007 = mangrove_extent_2007_block[row, col]
            mangrove_extent_2008 = mangrove_extent_2008_out_block[row, col]
            mangrove_extent_2009 = mangrove_extent_2009_out_block[row, col]
            mangrove_extent_2010 = mangrove_extent_2010_block[row, col]
            mangrove_extent_2015 = mangrove_extent_2015_block[row, col]
            mangrove_extent_2016 = mangrove_extent_2016_out_block[row, col]
            mangrove_extent_2017 = mangrove_extent_2017_out_block[row, col]
            mangrove_extent_2018 = mangrove_extent_2018_out_block[row, col]
            mangrove_extent_2019 = mangrove_extent_2019_out_block[row, col]
            mangrove_extent_2020 = mangrove_extent_2020_block[row, col]

            # Removing mangrove extent in 2008 if there is no mangrove extent in 2007 and 2009
            if (mangrove_extent_2008 == 1 and mangrove_extent_2007 == 0 and mangrove_extent_2009 == 0):
                mangrove_extent_2008_out_block[row, col] = 0

            # Removing mangrove extent in 2009 if there is no mangrove extent in 2008 and 2010
            if (mangrove_extent_2009 == 1 and mangrove_extent_2008 == 0 and mangrove_extent_2010 == 0):
                mangrove_extent_2009_out_block[row, col] = 0

            # Removing mangrove extent in 2016 if there is no mangrove extent in 2015 and 2017
            if (mangrove_extent_2016 == 1 and mangrove_extent_2015 == 0 and mangrove_extent_2017 == 0):
                mangrove_extent_2016_out_block[row, col] = 0

            # Removing mangrove extent in 2017 if there is no mangrove extent in 2016 and 2018
            if (mangrove_extent_2017 == 1 and mangrove_extent_2016 == 0 and mangrove_extent_2018 == 0):
                mangrove_extent_2017_out_block[row, col] = 0

            # Removing mangrove extent in 2018 if there is no mangrove extent in 2017 and 2019
            if (mangrove_extent_2018 == 1 and mangrove_extent_2017 == 0 and mangrove_extent_2019 == 0):
                mangrove_extent_2018_out_block[row, col] = 0

            # Removing mangrove extent in 2019 if there is no mangrove extent in 2018 and 2020
            if (mangrove_extent_2019 == 1 and mangrove_extent_2018 == 0 and mangrove_extent_2020 == 0):
                mangrove_extent_2019_out_block[row, col] = 0

    # Adds the output arrays to the output data dictionary
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_1996"] = mangrove_extent_1996_block     #same as raw data
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2007"] = mangrove_extent_2007_block     #same as raw data
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2008"] = mangrove_extent_2008_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2009"] = mangrove_extent_2009_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2010"] = mangrove_extent_2010_block     #same as raw data
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2015"] = mangrove_extent_2015_block     #same as raw data
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2016"] = mangrove_extent_2016_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2017"] = mangrove_extent_2017_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2018"] = mangrove_extent_2018_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2019"] = mangrove_extent_2019_out_block
    out_dict_uint8["GMWv3_smoothed_mangrove_extent_2020"] = mangrove_extent_2020_block     #same as raw data

    return out_dict_uint8

# All steps for creating smoothed mangrove data: download chunks, smooth data, upload to s3
def process_smoothed_mangrove_data(bounds, download_dict_with_data_types, area_dict_with_data_types, tile_outdata_dict_10x10, bounds_10x10, is_final, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []
    logger_worker = lu.setup_logging_worker()
    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_final, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)


    ### Part 1: Downloads data chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed
    #todo: add step to check whether any of the years has data before downloading

    # Replaces the placeholder tile_id in the data dictionaries from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger_worker)
    #print(f"futures: {futures}")

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and download status (values)
    layers = {}

    # Ensures futures stores Future objects
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        data, status = future.result()
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
            lu.print_and_log(f"{status}", False, logger_worker)
        layers[layer] = data
    #print(f"layers: {layers}")

    # Get maximum values across all years to use as a check on whether to process data
    max_list = []
    for key, array in layers.items():
        max_list.append(layers[key].max())
        #print(f"layer: {key}, maximum value: {max_value}")
    max_value_all_years = max(max_list)
    #print(f"max_value_all_years: {max_value_all_years}")

    # Stops running if mangrove extent data is not binary
    if max_value_all_years != np.uint8(1) and max_value_all_years != np.uint8(0):
        raise ValueError(f"Maximum value of mangrove extent is not 0 or 1 in chunk {bounds_str} in {tile_id}")
    # todo: delete no data (0) chunks here?


    ### Part 2: Downloads pixel area chunk if there is mangrove extent in one of the raw years to calculate raw extent.

    # Download pixel area (m2) if chunk has mangrove extent to calculate mangrove extent in chunk stats
    if max_value_all_years == np.uint8(1):
        # Replaces the placeholder tile_id in the data dictionaries from main with the tile_id for this chunk
        updated_area_dict = uu.replace_tile_id_in_dict(area_dict_with_data_types, tile_id)

        # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
        area_futures = uu.prepare_to_download_chunk(bounds, updated_area_dict, chunk_length_pixels, is_final, logger_worker)
        # print(f"area_futures: {area_futures}")

        lu.print_and_log(f"Waiting for requests for pixel area in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Dictionary that stores the dataset name (key) and downloaded data and download status (values)
        area_layers = {}

        # Ensures futures stores Future objects
        for area_future in concurrent.futures.as_completed(area_futures):
            area_layer = area_futures[area_future]
            data, status = area_future.result()
            if 'success' not in status:  # Prints and logs any inputs that couldn't be accessed and are downloaded as all 0s
                lu.print_and_log(f"{status}", False, logger_worker)
            area_layers[area_layer] = data
        #print(f"area_layers: {area_layers}")
        #print(f"maximum pixel area: {area_layers['pixel_area_m2'].max()}")

    elif max_value_all_years == np.uint8(0):
        lu.print_and_log(f"No mangrove extent in chunk {bounds_str} in {tile_id}. Skipping pixel area data download: {uu.timestr()}", is_final, logger_worker)


    ### Part 3: Calculates mangrove extent in raw mangrove data if mangrove extent exists

    if max_value_all_years == np.uint8(1):
        lu.print_and_log(f"Calculating chunk stats for raw data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Calculates total mangrove extent in raw chunk by multiplying binary extent by per-pixel area (m2)
        for key, array in layers.items():
            mangrove_extent_area_m2 = array * area_layers['pixel_area_m2']
            chunk_stats.append(uu.calculate_stats(None, key, bounds_str, tile_id, 'output_layer', mangrove_extent_area_m2))
        #print(chunk_stats)

    elif max_value_all_years == np.uint8(0):
        lu.print_and_log(f"No mangrove extent in chunk {bounds_str} in {tile_id}. Skipping chunk stat calculation for raw data: {uu.timestr()}", is_final, logger_worker)


    ### Part 4: Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type

    lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    del futures
    del layers


    ### Part 5: Creates smoothed mangrove extent rasters

    if max_value_all_years == np.uint8(1):
        lu.print_and_log(f"Creating smoothed mangrove data in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker) # Prints during full runs
        uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

        # Create smoothed mangrove data
        out_dict_uint8 = smooth_mangrove_data(typed_dict_uint8)


        ### Part 6: Calculates mangrove extent in smoothed mangrove data if mangrove extent exists

        lu.print_and_log(f"Calculating chunk stats for smoothed data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Calculates total smoothed mangrove extent in chunk by multiplying binary extent by per-pixel area (m2)
        for key, array in out_dict_uint8.items():
            mangrove_extent_area_m2 = array * area_layers['pixel_area_m2']
            chunk_stats.append(uu.calculate_stats(None, key, bounds_str, tile_id, 'output_layer', mangrove_extent_area_m2))
        #print(chunk_stats)

        del area_futures
        del area_layers

    elif max_value_all_years == np.uint8(0):
        lu.print_and_log(f"No mangrove extent in chunk {bounds_str} in {tile_id}. Skipping smoothing step: {uu.timestr()}", False, logger_worker)
        lu.print_and_log(f"No mangrove extent in chunk {bounds_str} in {tile_id}. Skipping chunk stat calculation for smoothed data: {uu.timestr()}",False, logger_worker)

    del typed_dict_uint8


    ### Part 7: Convert Numba-constrained dictionary into normal Python arrays to which we can do anything in Python.

    if max_value_all_years == np.uint8(1):
        out_dict_all_dtypes = {}

        # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
        out_dicts = [out_dict_uint8]

        # Loop through each dictionary and update out_dict_all_dtypes
        for out_dict in out_dicts:
            for key, value in out_dict.items():
                out_dict_all_dtypes[key] = value

        del out_dicts
        del out_dict_uint8


        ### Part 8: Add smoothed data to 10 x 10 degree rasters if there is mangrove extent

        # calculate pixel offset in tile_arrays
        min_x_tile, min_y_tile, max_x_tile, max_y_tile = bounds_10x10
        min_x_chunk, min_y_chunk, max_x_chunk, max_y_chunk = bounds

        x_offset = int(min_x_chunk - min_x_tile)
        y_offset = int(max_y_tile - max_y_chunk)
        row_start = y_offset * chunk_length_pixels
        col_start = x_offset * chunk_length_pixels

        for key, array in out_dict_all_dtypes.items():
            tile_outdata_dict_10x10[key][row_start:row_start + chunk_length_pixels, col_start:col_start + chunk_length_pixels] = array

        del out_dict_all_dtypes

    ### Part 9: Collect garbage after all deletions
    gc.collect()

    ### Part 10: Return the success message and chunk stats
    return_message = f"Success creating smoothed mangrove extent raster for {bounds_str}: {uu.timestr()}"
    #todo: print out how many tasks left

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)
    #todo: should this be added inside main for 10 x 10 degree tile completion?

    return return_message, chunk_stats

def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload= False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_mangrove_smoothing_10x10_deg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) >= 100:
        is_final = True
        main_logger.info("Running as final model.")
    #todo: move this to tile for loop?

    # This is just a placeholder tile_id to obtain the datatype. It is overwritten when chunks are assigned and analyzed.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download
    download_dict = {}
    for year in cn.mangrove_extent_years:
        download_dict[f"{cn.mangrove_extent_hansenized_pattern}_{year}"] = f"{cn.mangrove_extent_hansenized_dir}{year}/{sample_tile_id}_{cn.mangrove_extent_hansenized_pattern}_{year}.tif"
    #print(f"download_dict: {download_dict}")

    # Dictionary for pixel area array to calculate pre- and post-smoothing mangrove extent chunk stats
    area_dict = {}
    area_dict["pixel_area_m2"] = f"{cn.pixel_area_dir}/{cn.pixel_area_pattern}_{sample_tile_id}.tif"
    #print(f"area_dict: {area_dict}")

    # List of output directories for smoothed mangrove extent data
    output_dir_list = []
    for year in cn.mangrove_extent_years:
        output_dir_list.append(f"{cn.mangrove_extent_processed_dir}{year}/")
    #print(f"output_dir_list: {output_dir_list}")

    # Returns the first tile in each input so that the datatype can be determined.
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)
    first_area_tiles = uu.first_file_name_in_s3_folder(area_dict)

    # Creates a download dictionary with the datatype of each input which is supplied to each chunk being analyzed.
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)
    area_dict_with_data_types = uu.add_file_type_to_dict(first_area_tiles)
    #print(f"download_dict_with_data_types: {download_dict_with_data_types}")
    #print(f"area_dict_with_data_types: {area_dict_with_data_types}")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)


    ### Step 2: Create 1x1 degree outputs and merges into 10x10 degree output tiles

    # Creates list of tasks to run (1 task = 1 1x1 degree chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log" + "\n")

    # Groups 1x1 deg chunks into 10x10 deg tiles
    chunk_dict_by_10x10_tile = defaultdict(list)
    for chunk in chunk_list:
        tile_10x10 = uu.xy_to_tile_id(chunk[0], chunk[3])
        chunk_dict_by_10x10_tile[tile_10x10].append(chunk)
    #print(f"chunk_dict_by_10x10_tile: {chunk_dict_by_10x10_tile}")
    #todo: if length of chunk list is not 100 and is_final is True, throw error that not all 1x1 degree chunks are present

    # Begins processing 1x1 chunks according to 10x10 tile
    for tile, chunk_list in chunk_dict_by_10x10_tile.items():
        main_logger.info(f"Processing 10x10 tile {tile} with {len(chunk_list)} 1x1 deg chunks")

        bounds_10x10 = uu.get_10x10_tile_bounds(tile)
        chunk_length_pixels_10x10 = uu.calc_chunk_length_pixels(bounds_10x10)
        bounds_str_10x10 = uu.boundstr(bounds_10x10)

        # Initialize blank 10x10 arrays for each year in the smoothed extent time series
        tile_outdata_dict = {}
        for year in cn.mangrove_extent_years:
            pattern = f"{cn.mangrove_extent_processed_pattern}_{year}"
            tile_outdata_dict[pattern] = np.zeros((cn.full_raster_dims, cn.full_raster_dims), dtype=np.uint8)
        #print(f"tile_outdata_dict: {tile_outdata_dict}")
        #todo this may not work in the numba function because these are numpy arrays. check

        mangrove_1x1_deg_delayed_results = [dask.delayed(process_smoothed_mangrove_data)
                    (chunk, download_dict_with_data_types, area_dict_with_data_types, tile_outdata_dict, bounds_10x10, is_final, stage)
                    for chunk in chunk_list]

        # Runs analysis and gathers results
        mangrove_1x1_deg_results = dask.compute(*mangrove_1x1_deg_delayed_results)
        success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, mangrove_1x1_deg_results)

        # Only saves arrays to geotifs and uploads them to s3 if enabled
        if not no_upload:

            main_logger.info(f"Uploading smoothed mangrove data for {tile}: {uu.timestr()}")
            out_no_data_val = 0  # NoData value for output raster (optional)
            tile_out_dict_all_dtypes = {}

            # Adds metadata used for uploading outputs to s3 to the dictionary
            for key, value in tile_outdata_dict.items():
                data_type = value.dtype.name
                out_pattern, year_range = uu.strip_and_extract_years(key)
                matched_output_s3_folder = [item for item in output_dir_list if year_range in item][0]
                s3_path_without_bucket = f"{matched_output_s3_folder[cn.full_bucket_prefix_length:]}"

                # Dictionary with metadata for each array
                tile_out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

            # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
            upload_tasks = uu.save_and_upload_raster_10x10(bounds_10x10, chunk_length_pixels_10x10, tile, bounds_str_10x10,
                                                           tile_out_dict_all_dtypes, is_final, main_logger, out_no_data_val)

            # Only prints if not a final run
            lu.print_and_log(f"Upload tasks created for {bounds_str_10x10} in {tile}. Uploading now: {uu.timestr()}", is_final, main_logger)

            # Executes uploads in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

            # Only prints if not a final run
            lu.print_and_log(f"Uploads completed for {bounds_str_10x10} in {tile}: {uu.timestr()}", is_final, main_logger)

            # Clears memory of unneeded arrays
            del tile_out_dict_all_dtypes
            gc.collect()


    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    #TODO: start cleaning the code from here
    ### Step 3: Chunk stats for 1x1 degree outputs, aggregates logs
    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")
            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count_1x1 > 0):
        chunk_stats_path = uu.compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats, and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="smooth mangrove data")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 10x10 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)


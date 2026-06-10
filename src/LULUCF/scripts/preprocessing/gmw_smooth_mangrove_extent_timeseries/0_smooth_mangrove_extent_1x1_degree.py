"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local:
1x1 deg data chunk:
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -bb 116 -3 117 -2 -cs 1 --run_local

1x1 deg no data chunk:
python -m scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -bb 60 -11 61 -10 -cs 1 --run_local

Coiled test run:
python -m src.utilities.create_cluster -n 2 -t 2 -m 8 -cn mangrove_smoothing_1x1deg
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -cn mangrove_smoothing_1x1deg  -bb 110 -10 120 0 -cs 1
Note: Took 8 minutes to smooth all 11 mangrove extent years in 00N_110E (100 1x1 deg chunks: 54 data chunks, 46 skipped chunks) with 2 workers, 2 thread per worker and 8 GB per worker (vs 17 min locally).

Coiled full run:
python -m src.utilities.create_cluster -n 20 -t 8 -m 16 -cn mangrove_smoothing_1x1deg_full_run
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -cn mangrove_smoothing_1x1deg_full_run -ln "This is intended to be the definitive global run for mangrove extent smoothing using GADM v4.1."
Note: Took 40 minutes (13 coiled credits) to smooth all 11 mangrove extent years globally (1,519 data chunks, 17,313 skipped chunks) with 20 workers, 8 thread per worker and 16 GB per worker.
Note: Try with 1 thread per worker

todo:
- see if it can scale from 1 degree to 10 degrees

"""
import argparse
import sys
import gc
import numpy as np
from numba import jit
import dask
from dask.distributed import print
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Project imports
from src.utilities import constants_and_names as cn, log_utilities as lu, universal_utilities as uu, numba_utilities as nu, resize_cluster


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
def preprocess_and_upload_1x1_deg_smoothed_mangrove_data(bounds, download_dict_with_data_types, area_dict_with_data_types, is_final, no_upload, output_folders, stage):

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
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger_worker, False)
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

    # Delete downloaded data and return if there is no mangrove extent in any of the years
    elif max_value_all_years == np.uint8(0):

        # Returns empty chunk stats in no data chunks
        for key, array in layers.items():
            chunk_stats.append(uu.calculate_stats(None, key, bounds_str, tile_id, 'input_layer', None))
        # print(chunk_stats)

        del futures
        del layers
        gc.collect()

        return_message = f"No mangrove extent in chunk {bounds_str} in {tile_id}. Skipped chunk smoothing step: {uu.timestr()}"

        # Removes task tracking file from S3 once task is finished
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

        return return_message, chunk_stats  # Return both the success message and the statistics

    # Proceed with the rest of the script if there is mangrove extent in any of the years
    elif max_value_all_years == np.uint8(1):


        ### Part 2: Downloads pixel area chunk if there is mangrove extent in one of the raw years to calculate raw extent.

        # Replaces the placeholder tile_id in the data dictionaries from main with the tile_id for this chunk
        updated_area_dict = uu.replace_tile_id_in_dict(area_dict_with_data_types, tile_id)

        # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
        area_futures = uu.prepare_to_download_chunk(bounds, updated_area_dict, chunk_length_pixels, is_final, logger_worker, False)
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


        ### Part 3: Calculates mangrove extent in raw mangrove data if mangrove extent exists

        lu.print_and_log(f"Calculating chunk stats for raw data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Calculates total mangrove extent in raw chunk by multiplying binary extent by per-pixel area (m2)
        for key, array in layers.items():
            mangrove_extent_area_m2 = array * area_layers['pixel_area_m2']
            chunk_stats.append(uu.calculate_stats(None, key, bounds_str, tile_id, 'output_layer', mangrove_extent_area_m2))
        #print(chunk_stats)


        ### Part 4: Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type

        lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Creates the typed dictionaries for all input layers (including those that originally had no data)
        typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

        del futures
        del layers


        ### Part 5: Creates smoothed mangrove extent rasters

        lu.print_and_log(f"Creating smoothed mangrove data in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker) # Prints during full runs
        uu.rename_s3_task_file(stage, bounds, "calculating_", is_final, logger_worker)

        # Create smoothed mangrove data
        out_dict_uint8 = smooth_mangrove_data(typed_dict_uint8)


        ### Part 6: Calculates mangrove extent in smoothed mangrove data if mangrove extent exists

        lu.print_and_log(f"Calculating chunk stats for smoothed data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Calculates total smoothed mangrove extent in chunk by multiplying binary extent by per-pixel area (m2)
        #todo: convert m2 to hectares
        for key, array in out_dict_uint8.items():
            mangrove_extent_area_m2 = array * area_layers['pixel_area_m2']
            chunk_stats.append(uu.calculate_stats(None, key, bounds_str, tile_id, 'output_layer', mangrove_extent_area_m2))
        #print(chunk_stats)

        del area_futures
        del area_layers
        del typed_dict_uint8


        ### Part 7: Convert Numba-constrained dictionary into normal Python arrays to which we can do anything in Python.

        out_dict_all_dtypes = {}

        # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
        out_dicts = [out_dict_uint8]

        # Loop through each dictionary and update out_dict_all_dtypes
        for out_dict in out_dicts:
            for key, value in out_dict.items():
                out_dict_all_dtypes[key] = value

        # Clear memory of unneeded arrays
        del out_dicts
        del out_dict_uint8


        ### Part 9: Saves numpy arrays as rasters and uploads to s3

        uu.rename_s3_task_file(stage, bounds, "uploading_", is_final, logger_worker)

        # Only saves arrays to geotifs and uploads them to s3 if enabled
        if not no_upload:

            out_no_data_val = 0  # NoData value for output raster (optional)

            # Adds metadata used for uploading outputs to s3 to the dictionary
            for key, value in out_dict_all_dtypes.items():

                data_type = value.dtype.name
                out_pattern, year_range = uu.strip_and_extract_years(key)
                matched_output_s3_folder = [item for item in output_folders if year_range in item][0]

                # Output paths without bucket (s3://gfw2-data)
                s3_path_without_bucket = f"{matched_output_s3_folder[cn.full_bucket_prefix_length:]}"

                # Dictionary with metadata for each array
                out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

            # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
            upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes, is_final, logger_worker, out_no_data_val)

            # Only prints if not a final run
            lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", is_final, logger_worker)

            # Executes uploads in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

            # Only prints if not a final run
            lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger_worker)

        # Clears memory of unneeded arrays
        del out_dict_all_dtypes
        gc.collect()

        return_message = f"Success creating smoothed mangrove extent raster for {bounds_str}: {uu.timestr()}"

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_final, logger_worker)

        return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload= False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_mangroves_1x1_deg'
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
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

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
        output_dir_list.append(f"{cn.mangrove_1x1deg_smoothed_dir}{year}/")
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


    ### Step 2: Create 1x1 degree outputs
    #todo: aggregate 1 x 1 degree chunks then combine into 10x10 degree chunks and upload (separate script)

    # Creates list of tasks to run (1 task = 1 1x1 degree chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log" + "\n")

    mangrove_1x1_deg_delayed_results = [dask.delayed(preprocess_and_upload_1x1_deg_smoothed_mangrove_data)
                (chunk, download_dict_with_data_types, area_dict_with_data_types, is_final, no_upload,output_dir_list, stage)
                for chunk in chunk_list]

    # Runs analysis and gathers results
    mangrove_1x1_deg_results = dask.compute(*mangrove_1x1_deg_delayed_results)
    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, mangrove_1x1_deg_results)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


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
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
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


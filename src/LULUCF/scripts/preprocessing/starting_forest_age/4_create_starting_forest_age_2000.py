"""
Creates starting forest age in 2000 from interpolated 2010 age map and associated classification of the age pixels.
This matches a slide in the AFOLU model diagram deck.

age_2000 source flag key:
1 = Not tall vegetation in 2000 (age assigned 0)
2 = Not tall vegetation in 2005 or 2010 (i.e. disturbance) (assign Besnard age at disturbance map)
3 = Age <= 10 years in 2010 (assign Besnard age at disturbance map)
4 = Forest between 2000 and 2010 and age >10 in 2010 -> age_2000 = age_2010 - 10

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local:
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.4_create_starting_forest_age_2000 -bb 10 49 11 50 -cs 1 --run_local --no_upload --no_stats --no_log

Coiled test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 2 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.4_create_starting_forest_age_2000 -cn LULUCF_preprocessing -bb 10 49 11 50 -cs 1

Full Coiled run:
python -m src.utilities.create_cluster -n 100 -t 2 -m 4 -cn LULUCF_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.4_create_starting_forest_age_2000 -cn LULUCF_preprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "This is intended to be the definitive forest age in 2000 run."
https://cloud.coiled.io/clusters/1018509/account/wri-forest-research/information?organization=wri
Peak memory per worker: ~X GB
Time for total processing for each task: X-X seconds (based on scanning the console)
Time until chunk stats: X
Time after chunk stats: X
Coiled credits: X (101/hr for 200 m8g.medium workers, according to dashboard)
AWS cost: $X (2.28/hr for 200 m8g.medium workers, according to dashboard)
-t 2 is based on testing different threads/worker combinations:
https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210731476254597?focus=true

"""

import argparse
import concurrent.futures
import numpy as np
import os
import rasterio
import rasterio.merge
from rasterio.windows import from_bounds
import dask
import fsspec
import psutil
import time

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


def create_starting_forest_age_2000(bounds, download_dict_with_data_types, year, chunk_size_pixels,
                                    is_large_run, no_upload, output_dir_list, stage):

    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)


    ### Part 1: Downloads chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    download_dict_with_data_types[cn.forest_age_2010_gap_filled_pattern] = [f"{cn.forest_age_2010_gap_filled_dir}{tile_id}__{bounds_str}__{cn.forest_age_2010_gap_filled_pattern}.tif", 'Byte']
    for key, value in download_dict_with_data_types.items():
        value[0] = value[0].replace("CHUNK_SIZE", "4000")

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)
    print(updated_download_dict)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_large_run, logger_worker, False)
    # print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and download status (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]  # Gets the corresponding key
        data, status = future.result()  # Unpacks the tuple result
        if 'success' not in status: # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
        layers[layer] = data

    # # Test prints
    # print(layers)
    # print(layers[f"{cn.vegetation_height_pattern}_2000"].max())
    # print(layers[f"{cn.vegetation_height_pattern}_2000"].dtype)
    # print(layers[cn.forest_age_2010_pattern].max())
    # print(layers[cn.forest_age_2010_pattern].dtype)


    ### Part 2: Age calculation in 2000

    # Get the 2010 forest age layer
    age_2010 = layers[cn.forest_age_2010_gap_filled_pattern].astype(np.uint16)

    # Read the 1x1 deg global raster (once per chunk)
    with rasterio.open(cn.global_age_at_disturbance_file) as src:
        # Read the relevant window into memory at the resolution of the working chunk
        window = from_bounds(*bounds, transform=src.transform)
        global_age_at_disturbance_resampled = src.read(1, window=window, out_shape=age_2010.shape,
                                    resampling=rasterio.enums.Resampling.nearest)

    # Convert global_age_at_disturbance_resampled to int16 in case of nodata
    global_age_at_disturbance_resampled = global_age_at_disturbance_resampled.astype(np.uint16)

    # Initialize a binary disturbance mask for 2000–2010
    disturbance_mask = np.zeros_like(age_2010, dtype=bool)

    # Disturbance criteria 1: Low vegetation height (<5 m) in 2000, 2005, or 2010
    for year in [2005, 2010]:
        veg_height_key = f"{cn.vegetation_height_pattern}_{year}"
        disturbance_mask |= layers[veg_height_key] < 5

    # # Disturbance criteria 2: Vegetation height drop >= 5 m from one time to another.
    # # Define year pairs to compare.
    # # Not currently using as a reason to reset the age in 2000,
    # # per https://app.asana.com/1/25496124013636/task/1209335072194767/comment/1210698775382719?focus=true
    # # Essentially, although there has been a disturbance resulting in height decrease, this was tree cover
    # # the entire time, so nothing has clearly interrupted the age progression from 2000 to 2010.
    # # In other words, there has been no decision interruption in stand age just because of a height decrease.
    # height_drop_pairs = [(2000, 2005), (2005, 2010), (2000, 2010)]
    # for start_year, end_year in height_drop_pairs:
    #     key_start = f"{cn.vegetation_height_pattern}_{start_year}"
    #     key_end = f"{cn.vegetation_height_pattern}_{end_year}"
    #     height_diff = layers[key_start].astype(np.int16) - layers[key_end].astype(np.int16)
    #     disturbance_mask |= height_diff >= 5

    # # Disturbance criteria 3: Check all annual disturbance rasters from 2001 to 2010
    # # Not currently using as a reason to reset the age in 2000,
    # # per https://app.asana.com/1/25496124013636/task/1209335072194767/comment/1210698775382719?focus=true
    # # Essentially, although there has been an annual disturbance flag, this was tree cover
    # # the entire time, so nothing has clearly interrupted the age progression from 2000 to 2010.
    # # In other words, there has been no decision interruption in stand age just because of a height decrease.
    # for year in range(2001, 2011):
    #     disturbance_key = f"{cn.forest_disturbance_layer_name}_{year}"
    #     disturbance_mask |= layers[disturbance_key] > 0

    # Age=0 criteria: Low vegetation height (<5m) in 2000 → age = 0
    veg_height_2000_key = f"{cn.vegetation_height_pattern}_2000"
    low_height_2000_mask = layers[veg_height_2000_key] < 5

    # Subtract without clamping yet
    raw_subtract = age_2010.astype(np.int16) - 10

    # Mask where age would be <= 0
    underflow_mask = (raw_subtract <= 0)

    # Clamp for the fallback case
    safe_subtract = np.maximum(raw_subtract, 0)

    # Final logic for calculating age
    age_2000 = np.where(
        low_height_2000_mask,
        0,  # 1 = Not tall vegetation in 2000 -> age assigned 0
        np.where(
            disturbance_mask,
            global_age_at_disturbance_resampled,  # 2 = Not tall vegetation in 2005 or 2010 (i.e. disturbance) -> assign Besnard age at disturbance map
            np.where(
                underflow_mask,
                global_age_at_disturbance_resampled,  # 3 = Age <= 10 years in 2010 -> assign Besnard age at disturbance map
                safe_subtract  # 4 = Forest between 2000 and 2010 and age >10 in 2010 -> age_2000 = age_2010 - 10
            )
        )
    ).astype(np.uint16)

    # Final logic for assigning the source_flag
    age_2000_source_flag = np.where(
        low_height_2000_mask,
        1,  # 1 = Not tall vegetation in 2000 -> age assigned 0
        np.where(
            disturbance_mask,
            2,  # 2 = Not tall vegetation in 2005 or 2010 (i.e. disturbance) -> assign Besnard age at disturbance map
            np.where(
                underflow_mask,
                3,  # 3 = Age <= 10 years in 2010 -> assign Besnard age at disturbance map
                4   # 4 = Forest between 2000 and 2010 and age >10 in 2010 -> age_2000 = age_2010 - 10
            )
        )
    ).astype(np.uint8)

    numpy_end = time.time()
    lu.print_and_log(f"Done calculating forest age in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after age in 2000 creation for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    lu.print_and_log(f"Calculated forest age in {bounds_str} in {tile_id} in {round(numpy_end-chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)


    ### Part 3: Saves and uploads the output rasters

    fs = fsspec.filesystem("s3")

    forest_age_2010_path = updated_download_dict[cn.forest_age_2010_gap_filled_pattern][0]

    with rasterio.open(forest_age_2010_path) as src:
        profile = src.profile.copy()

    lu.print_and_log(f" Saving and uploading {bounds_str}: {uu.timestr()}", is_large_run, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    if is_large_run:
        age_2000_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2000_gap_filled_pattern}.tif"
        age_2000_source_flag_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2000_gap_filled_source_flag_pattern}.tif"
    else:
        age_2000_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2000_gap_filled_pattern}__{uu.timestr()}.tif"
        age_2000_source_flag_name = f"{tile_id}__{bounds_str}__{cn.forest_age_2000_gap_filled_source_flag_pattern}__{uu.timestr()}.tif"

    # Saves filled in focal chunk locally
    if run_local and no_upload:
        age_2000_tmp_path = f"/mnt/c/GIS/AFOLU_flux_model/forest_age/year_2000/{age_2000_name}"
        age_2000_source_flag_tmp_path = f"/mnt/c/GIS/AFOLU_flux_model/forest_age/year_2000/{age_2000_source_flag_name}"
    else:
        age_2000_tmp_path = f"/tmp/{age_2000_name}"
        age_2000_source_flag_tmp_path = f"/tmp/{age_2000_source_flag_name}"

    # The age_2000 raster continues to be uint16
    profile.update({
        "dtype": "uint16"
    })
    with rasterio.open(age_2000_tmp_path, "w", **profile) as dst:
        dst.write(age_2000, 1)

    # The source flag raster is uint8
    profile.update({
        "dtype": "uint8"
    })
    with rasterio.open(age_2000_source_flag_tmp_path, "w", **profile) as dst:
        dst.write(age_2000_source_flag, 1)

    # Optional: Uploads to S3
    age_2000_s3_path = f"{output_dir_list[0]}{age_2000_name}"
    age_2000_source_flag_s3_path = f"{output_dir_list[1]}{age_2000_source_flag_name}"

    if not no_upload:
        with fs.open(age_2000_s3_path, "wb") as f_out:
            with open(age_2000_tmp_path, "rb") as f_in:
                f_out.write(f_in.read())
        with fs.open(age_2000_source_flag_s3_path, "wb") as f_out:
            with open(age_2000_source_flag_tmp_path, "rb") as f_in:
                f_out.write(f_in.read())

    return_message = f"Success creating forest age 2000 for {bounds_str}: {uu.timestr()}"

    chunk_stats.append(uu.calculate_stats(age_2000, cn.forest_age_2000_gap_filled_pattern, bounds_str, tile_id, 'output_layer'))
    chunk_stats.append(uu.calculate_stats(age_2000_source_flag, cn.forest_age_2000_gap_filled_source_flag_pattern, bounds_str, tile_id, 'output_layer'))

    if not run_local:
        os.remove(age_2000_tmp_path)
        os.remove(age_2000_source_flag_tmp_path)

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    return return_message, chunk_stats


def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    year = 2000
    stage = f'create_starting_forest_age_{year}__1x1_deg'
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
    main_logger.info(f"Year for age map: {year}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info("Running as final model.")

    # This is just a placeholder tile_id that is used to obtain the datatype of each tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model). Forest age 2010 is added to the download dictionary at the chunk level
    # since that is a 4000x4000-pixel input.
    download_dict = {}

    for year in range(2000, 2011, cn.five_year_interval_duration):
        download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_5_year_path}{year}/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_{year}.tif"

    # # Not currently using annual disturbance rasters for age_2000 creation, but keeping here just in case we change that
    # for year in range(2001, 2011):
    #     download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.forest_disturbance_annual_dir}{year}/{year}_{sample_tile_id}.tif"
    main_logger.info(f"download_dict without age_2010 input: {download_dict}")

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # print(download_dict_with_data_types)

    # Creates list of output directories specific to the run
    output_dir_list = [cn.forest_age_2000_gap_filled_dir, cn.forest_age_2000_gap_filled_source_flag_dir]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]
    main_logger.info(f"output_dir_list: {output_dir_list}")


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)

    delayed_results_1x1_deg = [dask.delayed(create_starting_forest_age_2000)
                       (chunk, download_dict_with_data_types, year, chunk_size_pixels, is_large_run, no_upload, output_dir_list, stage)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    results_1x1_deg = dask.compute(*delayed_results_1x1_deg)

    success_count_1x1, all_1x1_stats = uu.count_successful_chunks(chunk_list, is_large_run, main_logger, results_1x1_deg)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")

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
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create starting forest age in 2000.")
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

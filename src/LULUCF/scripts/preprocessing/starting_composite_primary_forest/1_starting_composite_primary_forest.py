"""
Maps composite primary forest in 2015 (model start).
Uses the same code to create 2015 composite primary forest as 1_calculate_veg_fluxes.py does.
This is currently used only as a contextual layer for zonal stats (via zarr), not as an input to the vegetation model.
The vegetation model generates a 2015 composite primary forest map at the top of the numba function and
then iterates on that.
The vegetation model creates geotifs of composite primary forest in 2015 but doesn't put them in the vegetation mega-zarr
because that would be an extra year of data for one variable in the zarr, which throws off the time dimension.
This script is essentially just to make a composite primary forest zarr for 2015 to use for zonal stats and anything
else a zarr is needed for.

In future vegetation model runs, this could be used as an input to the vegetation model.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -bb 10 49.75 10.25 50 -cs 0.25 --run_local --no_upload

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn vegetation_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -cn vegetation_preprocessing -bb 116.25 -2.25 116.5 -2 -cs 0.25

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn vegetation_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -cn vegetation_preprocessing -bb -64 -22 -63 -21 -cs 1 --create_zarr

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 4 -cn vegetation_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -cn vegetation_preprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --create_zarr

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 4 -cn vegetation_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -cn vegetation_preprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --create_zarr

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 4 -cn vegetation_preprocessing
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.1_starting_composite_primary_forest -cn vegetation_preprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --log_note "Creating starting composite primary forest for 2015 for model v1.0.5 (2016-2024)."
"""

import argparse
import concurrent.futures
import dask
import os
import psutil
import time
import numpy as np
import fsspec
import xarray as xr
import resource

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


# All steps for creating starting composite primary forest: download chunks, calculate, upload to s3
def create_and_upload_starting_composite_primary_forest(bounds, download_dict_with_data_types, year,
                                                        is_large_run, no_upload, create_zarr,
                                                        output_folders, stage, zarr_path=None, outputs_to_zarr=None):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    ### Part 1: Downloads chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

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
    # print("max of GFW2024:", layers[cn.tree_cover_loss_pattern].max())
    # print("max of primary_2001:", layers['primary_2001'].max())
    # print("max of forest_age_gap_filled_start_year:", layers['forest_age_gap_filled_start_year'].max())
    # print("max of ifl_2016:", layers['ifl_2016'].max())


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))


    ### Part 3: Creates starting composite primary forest

    lu.print_and_log(f"Creating starting composite primary forest for {year} in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker) # Prints during full runs
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)
    calc_start = time.time()

    # Filters tcl_block to only where tcl occurred before 2015 (ignoring 0s)
    pre_2015_tcl_mask = ((layers[cn.tree_cover_loss_pattern] > 0) & (layers[cn.tree_cover_loss_pattern] < 15)).astype(np.uint8)

    # Masks out any primary forest where TCL occurred before 2015
    primary_2015 = (layers[cn.primary_2001_pattern] * (1 - pre_2015_tcl_mask)).astype(np.uint8)

    # Merges together IFL 2016 and primary 2015 so that if either is 1, it will be in the merged block
    # composite_primary_block = np.maximum(ifl_2016_block, primary_2015).astype(np.uint8)
    composite_primary = np.where((layers[cn.ifl_2016_pattern] > 0) | (primary_2015 > 0) | (layers[cn.forest_age_start_year_pattern] >= cn.primary_age_threshold), 1, 0).astype(np.uint8)

    calc_end = time.time()
    lu.print_and_log(f"Done starting composite primary forest in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after numba calculations completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", is_large_run, logger_worker)
    lu.print_and_log(f"Calculated starting composite primary forest for {bounds_str} in {tile_id} in {round(calc_end-calc_start)} seconds: {uu.timestr()}", False, logger_worker)


    ### Part 4: Writes outputs to pre-existing global mega-zarr (only if activated)

    out_dict_all_dtypes = {
        cn.starting_composite_primary_forest_pattern : composite_primary
    }

    zu.populate_zarr(bounds, bounds_str, create_zarr, [1], is_large_run, logger_worker, zarr_path,
                     out_dict_all_dtypes, outputs_to_zarr, stage, tile_id)


    ### Part 5: Calculates min, mean, max, and count for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
    for key, array_per_ha in out_dict_all_dtypes.items():

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', None))


    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)
        upload_start_time = time.time()

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():

            data_type = value.dtype.name
            # print("key:", key)
            # print("output_folders:", output_folders)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output  (list of one element)
            matched_output_s3_folder = [item for item in output_folders if out_pattern_without_pixel_meaning in item][0]
            # print("matched_output_s3_folder:", matched_output_s3_folder)

            # Output paths without bucket (s3://gfw2-data)
            s3_path_without_bucket = f"{matched_output_s3_folder[cn.full_bucket_prefix_length:]}"

            # Dictionary with metadata for each array
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                           out_dict_all_dtypes, is_large_run, logger_worker, out_no_data_val)

        # Only prints if not a final run
        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", is_large_run, logger_worker)

        # Executes uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        # Only prints if not a final run
        upload_end_time = time.time()
        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} {round(upload_end_time - upload_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    # Clears memory of unneeded arrays
    del out_dict_all_dtypes

    chunk_end_time = time.time()
    lu.print_and_log(f"Total chunk processing for {bounds_str} in {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return_message = f"Success creating starting composite primary forest for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"Peak memory for {bounds_str} in {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name,
         run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'starting_composite_primary_forest'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local,'standard', stage)

    year = 2015
    run_date = cn.starting_composite_primary_forest_run_date

    start_time = uu.timestr()  # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Vegetation model version: {cn.veg_model_version}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Year: {year}")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"Tolerance for comparison between model and zarr chunk stat metrics: {cn.zarr_difference_tolerance}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # large-scale testing
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # Whenever the run is large-scale (final), force zarr creation
    if is_large_run:
        create_zarr = True
    main_logger.info(f"Create and populate global mega-zarr: {create_zarr}")

    # This is just a placeholder tile_id that is used to obtain the datatype of each input tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    # It shouldn't really matter what the sample_tile_id is.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model).
    # These inputs don't depend on the starting year of the model.
    download_dict = {
        cn.primary_2001_pattern : f"{cn.primary_2001_dir}{sample_tile_id}.tif",
        cn.ifl_2016_pattern: f"{cn.ifl_2016_dir}{sample_tile_id}.tif",
        cn.tree_cover_loss_pattern: f"{cn.tree_cover_loss_dir}{cn.tree_cover_loss_pattern}_{sample_tile_id}.tif",
        cn.forest_age_start_year_pattern: f"{cn.forest_age_2015_gap_filled_dir}{sample_tile_id}__{cn.forest_age_2015_gap_filled_pattern}.tif"
    }

    # Replaces the placeholder parts of the input paths with relevant values
    download_dict = {
        key: value.replace("CHUNK_SIZE", '40000')
        for key, value in download_dict.items()
    }

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

    if is_large_run:
        main_logger.info(f"download_dict_with_data_types for {stage}:")
        for key, value in download_dict_with_data_types.items():
            main_logger.info(f"  {key}: {value}")


    # Creates a list of output directories (core and intermediates) for all outputs and intervals based on specifics of the model run
    output_dir_list = [cn.starting_composite_primary_forest_dir]
    output_dir_list = [path.replace('CHUNK_SIZE', str(chunk_size_pixels)) for path in output_dir_list]
    main_logger.info(f"output_dir_list for {stage}:")
    for item in output_dir_list:
        main_logger.info(f"  {item}")


    ### Step 2: Create empty (metadata-only), global mega-zarr in s3.
    ### Zarr approach from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc

    # Only creates the global mega-zarr if needed (large runs or otherwise specified)
    if create_zarr:

        # Creates s3 paths for the raw mega-zarr
        zarr_path = zu.create_zarr_path(cn.starting_composite_primary_forest_zarr_path, chunk_size_pixels, str(year),
                                                  'standard', cn.veg_model_version_underscore, 'NA',
                                        run_date, main_logger)
        outputs_to_zarr = [cn.starting_composite_primary_forest_pattern]

        # Creates the global mega-zarr with metadata only
        zu.initialize_global_zarr(zarr_path, outputs_to_zarr, 1,
                                  ((len(cn.interval_end_years_annual)), chunk_size_pixels, chunk_size_pixels), main_logger)

        # Checks the zarr coordinates and extent
        fs = fsspec.filesystem("s3", anon=False)
        mapper = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(mapper, consolidated=False)
        main_logger.info(f"mega-zarr coords: {ds.coords}")
        main_logger.info(f"y range: {ds.y.values.min()}, {ds.y.values.max()}")
        main_logger.info(f"x range: {ds.x.values.min()}, {ds.x.values.max()}")
        main_logger.info(f"mega-zarr chunk size (years, y, x): {ds.chunksizes}")

    else:
        zarr_path = None
        outputs_to_zarr = False


    ### Step 3: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    delayed_results_1x1deg = [dask.delayed(create_and_upload_starting_composite_primary_forest)
                       (chunk, download_dict_with_data_types, year,
                        is_large_run, no_upload, create_zarr, output_dir_list, stage,
                        zarr_path, outputs_to_zarr)
                       for chunk in chunk_list]

    # Runs analysis and gathers results
    results_1x1deg = dask.compute(*delayed_results_1x1deg)

    success_count, all_stats = uu.count_successful_chunks(chunk_list, is_large_run, main_logger, results_1x1deg)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 4: Consolidate chunk stats and export

    if not no_stats:
        model_chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stats", main_logger)


    ### Step 5: Compare model output chunk stats to zarr chunk stats for each variable (only if chunk stats and zarr created)
    ### Not running zarr chunk stats comparison. I was having trouble getting it to work because of problems with
    ### variable names and years, and I don't think it's worth fiddling with more.
    ### Leaving the code in here just in case I do want to revisit it, but for now I'm not worried about zarr population.

    # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successful (wasn't skipped).
    if (not no_stats) and create_zarr:

        main_logger.info(f"Starting zarr chunk stats comparison: {uu.timestr()}")

        # Text added to output chunk stats table name(s) (Excel or Parquet)
        comparison_insert = "_original_zarr_comparison"

        # The name of the chunk stats table from the model
        model_chunk_stats_table_name = os.path.basename(model_chunk_stats_path)
        # print(model_chunk_stats_table_name)

        tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
            comparison_insert, main_logger, model_chunk_stats_path)

        # List of dataframes with original and zarr chunk stats and their difference for each dataset-year combination
        all_merged_tables = []

        # Number of chunks with differences between original and zarr exceeding tolerance
        chunks_count_exceeding_total = 0

        # Number of chunks that have model chunk stats but not corresponding zarr chunk stats
        chunks_without_zarr_stats_total = 0

        # Iterates through select variables/datasets for chunk stats comparison. Can modify as needed.
        for var_name_with_pattern_year, var_name in zip(outputs_to_zarr, outputs_to_zarr):

            main_logger.info(f"Starting {var_name_with_pattern_year}: {uu.timestr()}")
            var_start_time = time.time()

            # Runs chunk stats for a dataset (all years) in the zarr in parallel
            chunk_stats_variable_year_zarr = zu.run_parallel_stats(
                client=client,
                chunk_list=chunk_list,
                var=var_name_with_pattern_year,
                zarr_path=zarr_path,
                interval_end_years=[year]
            )
            # print("chunk_stats_variable_year_zarr:", chunk_stats_variable_year_zarr)

            # After all zarr chunk stats is done for the dataset-year combination,
            # the chunk stats from the zarr are compared to the chunk stats from the model.
            # This is done with Pandas dataframes and is not parallelized because it's just table manipulation
            # for each dataset-year combination.
            # The model output vs. zarr comparison is done after each dataset-year combination
            # to get more real-time feedback on how the datasets compare (rather than waiting until after
            # all zarr chunk stats have been calculated to do the metric comparisons).
            chunks_count_exceeding, chunks_without_zarr_stats = zu.compare_dataset_year_chunk_stats(all_merged_tables,
                                                                                    chunk_stats_variable_year_zarr,
                                                                                    main_logger,
                                                                                    tables_to_compare_dict,
                                                                                    var_name,
                                                                                    zarr_comparison_stats_path)

            # Total number of chunks that have differences in metrics between the model and zarr
            # that exceed the tolerance
            chunks_count_exceeding_total += chunks_count_exceeding
            chunks_without_zarr_stats_total += chunks_without_zarr_stats

            var_end_time = time.time()
            main_logger.info(f"  Processed {var_name_with_pattern_year} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")

        # Counts up chunks that had differences exceeding the tolerance and uploads chunk stats comparisons.
        zu.upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                              main_logger, model_chunk_stats_table_name,
                                              stage, start_time, zarr_comparison_stats_name, zarr_comparison_stats_path)


    ### Step 6: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)


    ### Step 7: Resize cluster down to 1 worker for remaining steps since they only need a minimal remainder of the
    ### cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 8: Count output geotifs in s3

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    if not no_upload and is_large_run:
        for output_folder in output_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 9: Merge compiled worker log and main log
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)


    # Closes the Dask client if not running locally
    if not run_local:
        client.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate vegetation fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('--create_zarr', action='store_true', help='Create and populate global mega-zarr with model outputs')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_size_deg = args.chunk_size_deg
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    create_zarr = args.create_zarr

    # Create the cluster with command line arguments
    main(cluster_name, run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size_deg=chunk_size_deg, first_chunks=first_chunks, log_note=log_note)
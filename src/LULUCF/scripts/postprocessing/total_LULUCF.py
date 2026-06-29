"""
Creates 1x1 deg outputs for summative soil and LULUCF fluxes for every interval.
Vegetation is annual fluxes but both soil datasets (organic and mineral) are in 5-year intervals.
Vegetation starts in 2016 but soil datasets start in 2000. Summative LULUCF starts in 2016 as well (covers only overlapping years).
So, the five-year soil values are added to the annual vegetation to get annual LULUCF fluxes.
Separate output folders and zarrs are created for the summative soil outputs and summative LULUCF outputs,
just for simplicity.

Can only run on 1x1 degree chunks that do not have the run timestamp in the file name.
The way this builds the input file names, it can't handle filenames with the run timestamp.
It also can't handle chunks smaller than 1x1 degree.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.postprocessing.total_LULUCF -bb 10 49 11 50 -cs 1 --no_upload -vid YYYYMMDD -oid YYYYMMDD -mid YYYYMMDD

Coiled small tests (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.postprocessing.total_LULUCF -cn vegetation_postprocessing -bb 10 49 11 50 -cs 1 --create_zarr -vid YYYYMMDD -oid YYYYMMDD -mid YYYYMMDD

Coiled large shapefile test (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.postprocessing.total_LULUCF -cn vegetation_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --create_zarr  -vid YYYYMMDD -oid YYYYMMDD -mid YYYYMMDD -ln "Summative outputs for 1884-feature shapefile for model v1.0.3."

Full run (1x1 deg chunk needs a 32GB worker):
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.postprocessing.total_LULUCF -cn vegetation_postprocessing -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --create_zarr  -vid YYYYMMDD -oid YYYYMMDD -mid YYYYMMDD -ln "This is intended to be the definitive summative output run."

Optimization notes: https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210788116876878?focus=true

Summation error notes (originally from vegetation summative output script (since deleted), but
still relevant for summing soils and vegetation): The summative outputs don't exactly match the sums of the individual components at the pixel
or chunk level when summed independently on a calculator because of floating point errors.
For example, gross emissions from AGC, BGC, deadwood C, and litter C don't exactly match CO2-only emissions.
I tried a bit to fix this but (https://chatgpt.com/c/681244d9-83dc-800a-b397-0706e79391c0)
but it really wasn't worth it. The pixel-level and chunk level summative outputs deviate too little from the sums of
the individual components to make dealing with float64 worth it.
Analysis of how little the independently summed components differ from the summative outputs for 1884-chunk results are at
tab "net_vs_gross_floating_pt_error" in
https://onewri-my.sharepoint.com/:x:/g/personal/david_gibbs_wri_org/EX4w0jshE5ZIt8Yg0gERfRIB83OWJCfSf5gDbF7SgBHkzQ?e=psKFxQ&nav=MTVfe0NFMjcwREFELUFDMkYtNEUzQi1BMTA4LTVBQTREMThCOEExM30
This also carries over to the all-interval totals, where the independently summed fluxes from each interval do not
exactly match the all-interval totals. I don't have a saved analysis for that, but it's also a small difference.
"""

import argparse
import dask
import concurrent.futures
import os
import psutil
import sys
import time
import fsspec
import xarray as xr

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster
from src.utilities import zarr_utilities as zu

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"


def create_LULUCF_outputs(bounds, interval_type, interval_year_diff_list,
                          interval_end_years, is_large_run, no_upload, create_zarr,
                          organic_soil_input_date, mineral_soil_input_date,
                          summative_inputs_by_interval_dir_list, summative_outputs_by_interval_dir_list,
                          stage, raw_mega_zarr_path, outputs_to_zarr):
    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    ### Part 1: Downloads all inputs for chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # Dictionary of data to download (inputs to model)
    download_dict = {}

    # Iterates through inputs and creates the dictionary of patterns and download paths for vegetation model outputs
    for summative_input_by_interval in summative_inputs_by_interval_dir_list:

        # All the components of the input path
        parts = summative_input_by_interval.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"{interval_type}_intervals")
        interval_segment = parts[interval_idx + 1]

        # Gets the segment for the pixel meaning. Different possibilities for carbon pools, fluxes, and everything else.
        if "_ha_yr" in parts:
            pix_meaning_idx = parts.index("_ha_yr")
            pix_meaning_segment = parts[pix_meaning_idx]
        elif "_ha" in parts:
            pix_meaning_idx = parts.index("_ha")
            pix_meaning_segment = parts[pix_meaning_idx]
        else:
            pix_meaning_segment = ''

        # Constructs the dictionary entry.
        # Value has to be a list because prepare_to_download_chunk expects the download dictionary keys to be lists.
        download_dict[f"{pattern_segment}_{interval_segment}"] = [
            f"{summative_input_by_interval}{tile_id}__{bounds_str}__{pattern_segment}{pix_meaning_segment}_{interval_segment}.tif"]

    # Adds organic soil outputs to the download dictionary: burned and drained totals
    for interval in ["2001_2005", "2006_2010", "2011_2015", "2016_2020", "2021_2024"]:
        drained_path = cn.drained_organic_soils_total_dir.replace('START_END', interval)
        drained_path = drained_path.replace('CHUNK_SIZE', str(chunk_length_pixels))
        drained_path = drained_path.replace('RUN_DATE', organic_soil_input_date)
        download_dict[f"{cn.drained_organic_soils_total_pattern}_{interval}"] = [f"{drained_path}{tile_id}__{bounds_str}__{cn.drained_organic_soils_total_pattern}__{interval}.tif"]

        burned_path = cn.burned_organic_soils_total_dir.replace('START_END', interval)
        burned_path = burned_path.replace('CHUNK_SIZE', str(chunk_length_pixels))
        burned_path = burned_path.replace('RUN_DATE', organic_soil_input_date)
        download_dict[f"{cn.burned_organic_soils_total_pattern}_{interval}"] = [f"{burned_path}{tile_id}__{bounds_str}__{cn.burned_organic_soils_total_pattern}__{interval}.tif"]

    # print(download_dict)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_large_run, logger_worker,True)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]  # Gets the corresponding key
        data, status = future.result()  # Unpacks the tuple result
        if 'success' not in status:  # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
            raise ValueError(f"Expected input folders not found for {bounds_str}. Check inputs.")  # Quits if s3 folder not found. All input folders should be found for this stage.
        layers[layer] = data

    # print(layers)
    # print(layers['burned_total_Mg_CO2e_ha_yr_2016_2020'].max())
    # print(layers['drained_total_Mg_CO2e_ha_yr_2016_2020'].max())
    # print(layers['drained_total_Mg_CO2e_ha_yr_2011_2015'].max())


    ### Part 2: Creates summative outputs

    lu.print_and_log(f"Summing LULUCF outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)

    # Everything in out_dict also needs to be in cn.LULUCF_summative_output_dirs
    # because that has the list of basic output directories which are customized for this run
    out_dict = {}

    # Summative outputs for outputs with year ranges in their name (i.e. fluxes)
    for i, interval_end_year in enumerate(interval_end_years):

        # Determines what organic soil interval to use for each vegetation year
        if interval_end_year >= 2015 and interval_end_year < 2021:
            org_soil_interval = "2016_2020"
        elif interval_end_year >= 2021 and interval_end_year < 2025:
            org_soil_interval = "2021_2024"
        else:
            sys.exit("Interval end year not found")

        # Gross emissions across all carbon pools and gases
        out_dict[f"{cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                layers[f"{cn.gross_emis_all_C_pools_all_gases_pattern}_{interval_end_year}"] +
                layers[f"{cn.drained_organic_soils_total_pattern}_{org_soil_interval}"] +
                layers[f"{cn.burned_organic_soils_total_pattern}_{org_soil_interval}"])

        out_dict[f"{cn.net_flux_all_C_pools_all_gases_LULUCF_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                layers[f"{cn.net_flux_all_C_pools_all_gases_pattern}_{interval_end_year}"] +
                layers[f"{cn.drained_organic_soils_total_pattern}_{org_soil_interval}"] +
                layers[f"{cn.burned_organic_soils_total_pattern}_{org_soil_interval}"])

        # print(f"{cn.drained_organic_soils_total_pattern}_{org_soil_interval}")
        # print(layers[f"{cn.drained_organic_soils_total_pattern}_{org_soil_interval}"])

    lu.print_and_log(f"Done summing LULUCF outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"After creating LULUCF outputs for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)


    ### Part 3: Writes outputs to pre-existing global mega-zarr (only if requested)

    zu.populate_zarr(bounds, bounds_str, create_zarr, interval_end_years, is_large_run, logger_worker, raw_mega_zarr_path,
                     out_dict, outputs_to_zarr, stage, tile_id)

    ### Part 4: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, logger_worker,'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers as a dictionary with chunk attributes
    # NOTE: The full-interval chunk sums don't exactly match the sums of the individual intervals' chunk sums
    # because of float32 rounding errors. However the output full-model rasters are definitely close enough
    # at the pixel level, so I'm fine with this slight difference.
    # Worked on it in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/681244d9-83dc-800a-b397-0706e79391c0
    # but never implemented the fix because the very slight rounding results in <0.01% difference.

    for key, array_per_ha in out_dict.items():
        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)


    ### Part 4: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict.items():
            data_type = value.dtype.name
            print("key:", key)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, interval_year_range = uu.strip_and_extract_years(key)
            print("out_pattern:", out_pattern)
            print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output (list of one element).
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in summative_outputs_by_interval_dir_list if
                                         out_pattern_without_pixel_meaning in item]
            print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if interval_year_range in item]
            print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict[key] = [value, data_type, out_pattern, interval_year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                           out_dict, is_large_run, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.veg_outputs_path}: {uu.timestr()}",
                         is_large_run, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False,
                     logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_date, veg_input_date, organic_soil_input_date, mineral_soil_input_date,
         run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_summative_output_calculation'
    model_type = 'standard_model'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_year = cn.first_model_year_annual
    end_year = cn.last_model_year_annual

    start_time = uu.timestr()  # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Vegetation flux run date: {veg_input_date}")
    main_logger.info(f"Organic soil flux run date: {organic_soil_input_date}")
    main_logger.info(f"Mineral soil flux run date: {mineral_soil_input_date}")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"Tolerance for comparison between model and zarr chunk stat metrics: {cn.zarr_difference_tolerance}")

    # Calculates the interval type, difference between start and end years of intervals,
    # and the model output years for the model run
    interval_type_veg, interval_year_diff_list_veg, interval_length_list_veg, interval_end_years_list_veg = uu.get_interval_info(
        cn.first_model_year_annual, cn.last_model_year_annual, main_logger)

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks,
                                                         fishnet_iso_df, main_logger)

    # Can only run on 1x1 degree chunks
    if chunk_size_pixels != 4000:
        sys.exit("This stage can only be run on 1x1 degree (4000 pixel) chunks.")

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info("Running as final model.")

    # Unlike numba-based scripts, this one doesn't construct the download dictionary in the main function.
    # Instead, it creates a list of input folders, from which a download dictionary is created for each chunk (in the chunk-level function).
    # It's a little simpler this way. Since the datatypes of the inputs don't need to be specified in advance for this script
    # (since it's not using numba), there's no need to centrally create a download dictionary with each input's datatype
    # just once on the scheduler, as is more efficient for scripts that use numba.
    # Creates a list of input directories used in summative output creation based on specifics of the model run
    veg_inputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.veg_summative_for_LULUCF_output_dirs, "annual",
                                                                           cn.first_model_year_annual,
                                                                           chunk_size_pixels, cn.veg_model_version_underscore, model_type,
                                                                           interval_end_years_list_veg,
                                                                           interval_year_diff_list_veg, veg_input_date, False,
                                                                           "per_ha")

    # print("veg_inputs_by_interval_dir_list:", veg_inputs_by_interval_dir_list)
    if is_large_run:
        main_logger.info(f"veg_inputs_by_interval_dir_list:")
        for item in veg_inputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # # Creates a list of output directories for soil totals
    # soil_summative_outputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.soil_output_dirs, "annual",
    #                                                                         cn.first_model_year_annual,
    #                                                                         chunk_size_pixels, model_type,
    #                                                                         interval_end_years_list_veg,
    #                                                                         interval_year_diff_list_veg, veg_input_date,False,
    #                                                                         "per_ha")
    #
    # print(soil_summative_outputs_by_interval_dir_list)
    # if is_large_run:
    #     main_logger.info(f"outputs_dir_list:")
    #     for item in soil_summative_outputs_by_interval_dir_list:
    #         main_logger.info(f"  {item}")

    # Creates a list of output directories for LULUCF totals
    LULUCF_summative_outputs_by_interval_dir_list = uu.create_output_dir_name_list(cn.LULUCF_output_dirs, "annual",
                                                                            cn.first_model_year_annual,
                                                                            chunk_size_pixels, model_type, cn.veg_model_version_underscore,
                                                                            interval_year_diff_list_veg, veg_input_date,False,
                                                                            "per_ha")

    # print("LULUCF_summative_outputs_by_interval_dir_list:", LULUCF_summative_outputs_by_interval_dir_list)
    if is_large_run:
        main_logger.info(f"LULUCF_summative_outputs_by_interval_dir_list:")
        for item in LULUCF_summative_outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Makes a txt for each task in the list. These are deleted as tasks are completed.
    main_logger.info("Creating task txts in s3...")
    uu.create_s3_task_files(stage, chunk_list)

    # Establishes the location in s3 of the soil and LULUCF flux mega-zarrs
    if create_zarr:
        # # Creates s3 paths for the raw mega-zarr
        # raw_mega_zarr_path = zu.create_mega_zarr_path(cn.soil_outputs_path_mega_zarr, chunk_size_pixels, interval_type_veg, model_type, run_date,
        #                                               main_logger)
        # outputs_to_zarr = cn.soil_output_patterns  # [0:2] # For testing
        #
        # # Creates the global mega-zarr with metadata only
        # zu.initialize_global_mega_zarr(raw_mega_zarr_path, outputs_to_zarr, len(interval_year_diff_list),
        #                             ((len(cn.interval_end_years_annual)), chunk_size_pixels, chunk_size_pixels), main_logger)
        #
        # # Checks the zarr coordinates and extent
        # fs = fsspec.filesystem("s3", anon=False)
        # mapper = fs.get_mapper(raw_mega_zarr_path)
        # ds = xr.open_zarr(mapper, consolidated=False)
        # main_logger.info(f"mega-zarr coords: {ds.coords}")
        # main_logger.info(f"y range: {ds.y.values.min()}, {ds.y.values.max()}")
        # main_logger.info(f"x range: {ds.x.values.min()}, {ds.x.values.max()}")
        # main_logger.info(f"mega-zarr chunk size (years, y, x): {ds.chunksizes}")

        raw_mega_zarr_path = zu.create_zarr_path(cn.LULUCF_outputs_path_mega_zarr, chunk_size_pixels, interval_type_veg,
                                                 model_type, model_version, run_date, main_logger)
        outputs_to_zarr = cn.LULUCF_output_patterns  # [0:2] # For testing

        # Creates the global mega-zarr with metadata only
        zu.initialize_global_zarr(raw_mega_zarr_path, outputs_to_zarr, len(interval_year_diff_list_veg),
                                  ((len(cn.interval_end_years_annual)), chunk_size_pixels, chunk_size_pixels), main_logger)

        # Checks the zarr coordinates and extent
        fs = fsspec.filesystem("s3", anon=False)
        mapper = fs.get_mapper(raw_mega_zarr_path)
        ds = xr.open_zarr(mapper, consolidated=False)
        main_logger.info(f"mega-zarr coords: {ds.coords}")
        main_logger.info(f"y range: {ds.y.values.min()}, {ds.y.values.max()}")
        main_logger.info(f"x range: {ds.x.values.min()}, {ds.x.values.max()}")
        main_logger.info(f"mega-zarr chunk size (years, y, x): {ds.chunksizes}")

    else:
        raw_mega_zarr_path = None
        outputs_to_zarr = False


    ### Step 2: Create 1x1 degree outputs

    summative_output_tasks = [dask.delayed(create_LULUCF_outputs)
                              (chunk, interval_type_veg, interval_year_diff_list_veg, interval_end_years_list_veg,
                               is_large_run, no_upload, create_zarr, organic_soil_input_date, mineral_soil_input_date,
                               veg_inputs_by_interval_dir_list, LULUCF_summative_outputs_by_interval_dir_list, stage,
                               raw_mega_zarr_path, outputs_to_zarr)
                              for chunk in chunk_list]

    # Runs analysis and gathers results
    summative_output_results = dask.compute(*summative_output_tasks)

    success_count, all_stats = uu.count_successful_chunks(chunk_list, is_large_run, main_logger,
                                                          summative_output_results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    # ### Step 3: Counts files in output folders, aggregates chunk stats for 1x1 degree outputs
    #
    # # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # # cluster, not all the workers.
    # if not run_local:
    #     workers = client.scheduler_info()["workers"]
    #     n_workers = len(workers)
    #
    #     # Reduces number of workers in the cluster down to 1 if there is more than 10
    #     if n_workers > 10:
    #         main_logger.info("Resizing cluster to 1 worker")
    #
    #         resize_cluster.resize_coiled_cluster(cluster_name, 1)
    #
    # # Iterates through output folders and counts the number of output rasters (only if uploads enabled)
    # if not no_upload and is_large_run:
    #     for output_folder in summative_outputs_by_interval_dir_list:
    #         geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
    #         main_logger.info(f"Output rasters in {output_folder}: {file_count}")
    #         # print(geotiff_files)
    #
    # # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # # and min and max values across all chunks for all inputs and outputs
    # # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    # if (not no_stats) and (success_count > 0):
    #     model_chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload,
    #                                                         main_logger)
    #
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)
    #
    # ### Step 4: Compares model output chunk stats to zarr chunk stats for each variable-year (only if chunk stats created)
    #
    # if (not no_stats) and create_zarr:
    #
    #     main_logger.info(f"Starting zarr chunk stats comparison: {uu.timestr()}")
    #
    #     # Text added to output chunk stats table name(s) (Excel or Parquet)
    #     comparison_insert = "_summative_zarr_comparison"
    #
    #     # The name of the chunk stats table from the model
    #     model_chunk_stats_table_name = os.path.basename(model_chunk_stats_path)
    #     # print(model_chunk_stats_table_name)
    #
    #     tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
    #         comparison_insert, main_logger, model_chunk_stats_path)
    #
    #     # Resizes cluster up to 50 workers for zarr chunk stat comparison only if a large-scale run
    #     if (not run_local) and (is_large_run == True):
    #         main_logger.info("Resizing cluster to 50 workers")
    #
    #         resize_cluster.resize_coiled_cluster(cluster_name, 50)
    #
    #     # List of dataframes with original and zarr chunk stats and their difference for each dataset-year combination
    #     all_merged_tables = []
    #
    #     # Number of chunks with differences between original and zarr exceeding tolerance
    #     chunks_count_exceeding_total = 0
    #
    #     # Iterates through variables/datasets. Each chunk=10000x10000 is transferred by just one task/worker
    #     # so that multiple workers aren't touching the same zarr chunk at the same time.
    #     for var_name in outputs_to_zarr:
    #
    #         main_logger.info(f"Starting {var_name}: {uu.timestr()}")
    #         var_start_time = time.time()
    #
    #         # Iterates through years
    #         for year_idx in range(len(cn.interval_end_years_annual)):
    #             year = cn.interval_end_years_annual[year_idx]
    #
    #             # Gets stats for selected 1x1 deg chunks in raw and rechunked zarrs
    #             main_logger.info(f"  Starting zarr stats for {var_name} for year {year}: {uu.timestr()}")
    #             year_start_time = time.time()
    #
    #             # Runs chunk stats for a dataset-year in the zarr in parallel
    #             chunk_stats_variable_year_zarr = zu.run_parallel_stats(
    #                 client=client,
    #                 chunk_list=chunk_list,
    #                 var=var_name,
    #                 year_idx=year_idx,
    #                 zarr_path=raw_mega_zarr_path,
    #             )
    #             year_end_time = time.time()
    #             main_logger.info(
    #                 f"    Got zarr stats for {var_name} for year {year} in {round(year_end_time - year_start_time)} seconds: {uu.timestr()}")
    #
    #             # After all zarr chunk stats is done for the dataset-year combination,
    #             # the chunk stats from the zarr are compared to the chunk stats from the model.
    #             # This is done with Pandas dataframes and is not parallelized because it's just table manipulation
    #             # for each dataset-year combination.
    #             # The model output vs. zarr comparison is done after each dataset-year combination
    #             # to get more real-time feedback on how the datasets compare (rather than waiting until after
    #             # all zarr chunk stats have been calculated to do the metric comparisons).
    #             all_merged_tables, chunks_count_exceeding, chunks_without_zarr_stats = zu.compare_dataset_year_chunk_stats(
    #                 all_merged_tables,
    #                 chunk_stats_variable_year_zarr,
    #                 main_logger,
    #                 tables_to_compare_dict,
    #                 var_name, year,
    #                 zarr_comparison_stats_path)
    #
    #             # Total number of chunks that have differences in metrics between the model and zarr
    #             # that exceed the tolerance
    #             chunks_count_exceeding_total += chunks_count_exceeding
    #
    #         var_end_time = time.time()
    #         main_logger.info(
    #             f"  Processed {var_name}  in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")
    #
    # ### Step 5: All chunk stats comparison iterations done.
    # ### Counts up chunks that had differences exceeding the tolerance and uploads chunk stats comparisons.
    #
    # zu.upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, main_logger, model_chunk_stats_table_name,
    #                                       stage, start_time, zarr_comparison_stats_name, zarr_comparison_stats_path)
    #
    # ### Step 6: Aggregates logs
    #
    # # Sets it so that no worker logs are created if doing a local run
    # if not run_local:
    #     main_logger.info("Resizing cluster to 1 worker")
    #     resize_cluster.resize_coiled_cluster(cluster_name, 1)
    #
    #     # Creates combined log from all workers if not deactivated
    #     worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats, and worker log compilation", main_logger)
    #
    #     # Adds the workers' logs to the main log and uploads to s3
    #     lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
    #
    # # Closes the Dask client if not running locally
    # if not run_local:
    #     client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate summative outputs for soil and LULUCF (vegetation, organic soil, mineral soil).")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run (used for output folders), in YYYYMMDD')
    parser.add_argument('-vid', '--veg_input_date', help='Date of run for vegetation inputs, in YYYYMMDD')
    parser.add_argument('-oid', '--organic_soil_input_date', help='Date of run for organic soil inputs, in YYYYMMDD')
    parser.add_argument('-mid', '--mineral_soil_input_date', help='Date of run for mineral soil inputs, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('--create_zarr', action='store_true',help='Create soil and LULUCF mega-zarrs and populate with outputs')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    veg_input_date = args.veg_input_date
    organic_soil_input_date = args.organic_soil_input_date
    mineral_soil_input_date = args.mineral_soil_input_date
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    create_zarr = args.create_zarr

    # Create the cluster with command line arguments
    main(cluster_name, run_date, veg_input_date, organic_soil_input_date, mineral_soil_input_date,
         run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)
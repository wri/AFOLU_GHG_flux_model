"""
Calculates carbon densities (Mg C/ha) and annual gross gain, gross loss, and net stock changes (Mg CO2/ha/yr,
accounting for shorter interval length in the last interval) in 0-30 cm topsoil.
Like for vegetation, gross and net loss (emissions) is positive and gross and net gain (removals) is negative.

Density remains in Mg C/ha; stock changes are converted to Mg CO2/ha/yr for ease of downstream use.

Calling gross values gain and loss instead of emissions and removals to differentiate them from vegetation emissions and removals (which are in CO2(e).)
Loss is positive and gain is negative, to match the signs for emissions and removals

NoData value is np.nan.
NoData used for:
density- pixels without a value;
net change- pixels without a value;
loss and gain- pixels without a value in the relevant direction (i.e. a loss pixel with gain gets NaN)
0 is reserved for net, loss, and gain pixels that had no change in density.
Thus, when consecutive densities are the same, net, loss, and gain will all have 0s.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.1_SOC_stock_and_stock_change -bb 110 -1 111 0 -cs 1 -mt standard -mpd test_box

Coiled small test (1x1 deg):
python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.1_SOC_stock_and_stock_change -cn mineral_soil -bb 110 -1 111 0 -cs 1 -mt standard -mpd test_box --create_zarr

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 100 -t 1 -m 8 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.1_SOC_stock_and_stock_change -cn mineral_soil -mt standard -mpd 1884_features-cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp -ln "SOC timeseries for 1884-feature shapefile."

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 8 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.1_SOC_stock_and_stock_change -cn mineral_soil -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "This is intended to be the definitive SOC timeseries creation for 2000-2022."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6877a34b-02cc-800a-88cc-a123cdc9ed1b
"""

import argparse
import gc
import os
import concurrent.futures
import numpy as np
import fsspec
import pandas as pd
import psutil
import rasterio
from datetime import date
import xarray as xr
import resource
import time
import re
import zarr
from dask.distributed import print
from dask import config
from concurrent.futures import ThreadPoolExecutor

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


def create_soil_C_density_and_change(bounds, is_large_run, stage, no_upload, create_zarr, nodata_val, outputs_by_interval_dir_list,
                                     mega_zarr_path=None, outputs_to_zarr=None):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats_combined = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    # Download dictionary is the SOC global COGs
    download_dict = cn.SOC_COGS

    # Converts the raw COG's kg C/m^3 (top 30 cm) that is rescaled by 10 -> Mg C/ha without the rescaling.
    # OGH rescaled the global COGs by 10 to make them ints instead of floats to save storage.
    SOC_CONVERSION_FACTOR = 3.0 / 10.0  # = 0.3

    # Report the number of retries for the task. Untested.
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3
    retry_env_var = os.environ.get("DASK_TASK_RETRIES", "0")
    retry_count = int(retry_env_var)

    if retry_count > 0:
        msg = f"Running task for {bounds_str} in {tile_id} (retry #{retry_count}: {uu.timestr()})"
        lu.print_and_log(msg, False, logger_worker)


    ### Part 1: Downloads all inputs for chunk
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, download_dict, chunk_length_pixels, is_large_run, logger_worker, False)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    # Revised with Claude session 'SOC stock script hang at task 3799'
    try:
        for future in concurrent.futures.as_completed(futures, timeout=cn.download_timeout):
            layer = futures[future]
            data, status = future.result()
            if 'success' not in status:
                lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
            layers[layer] = data
    except concurrent.futures.TimeoutError:
        raise RuntimeError(f"Download timed out after {cn.download_timeout}s for {bounds_str} ({tile_id}): {uu.timestr()}")

    organic_soil_mask_uri = f"{cn.organic_soil_extent_dir}{tile_id}__{cn.organic_soil_extent_pattern}.tif"

    organic_soil_mask = uu.get_tile_dataset_rio(organic_soil_mask_uri, bounds, chunk_length_pixels, logger_worker,'uint8')
    organic_soil_mask = organic_soil_mask[0]  # Converts downloaded tuple (array, status) to just the array


    ### Part 2: Calculate carbon densities at each interval (Mg C/ha for 0-30 cm)

    # Output dictionaries for full extent and mineral soil extents.
    # Combined into single dict at end for uploading but separate now because calculating deltas is easier with them separate.
    out_dict_full_extent = {}
    out_dict_min_soil_extent = {}
    calc_start = time.time()

    for end_year in list(layers.keys()):
        interval_array_full_extent = layers[end_year]

        # Replace COG int16 NoData with nan
        interval_array_full_extent = np.where(interval_array_full_extent == nodata_val, np.nan, interval_array_full_extent)

        # Convert units from kg C/m³ * 10 for 0-30 cm depth -> Mg C/ha for 0-30 cm depth
        converted_array_full_extent = (interval_array_full_extent * SOC_CONVERSION_FACTOR).astype(np.float32)

        # print(f"\n--- Chunk {bounds_str} ---")
        # print(f"SOC density array shape for {bounds_str} for {end_year}: {converted_array_full_extent.shape}")
        # print(f"Organic soil mask shape for {bounds_str} for {end_year}: {organic_soil_mask.shape}")

        # Masks extent to just mineral soil (excludes pixels with high chance of being organic soil, per OpenGeoHub analysis)
        converted_array_min_soil_extent = np.where(organic_soil_mask != cn.organic_soil_mask_val, converted_array_full_extent, np.nan)

        # Save back to output dicts with the converted unit arrays
        out_dict_full_extent[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}_{end_year}"] = converted_array_full_extent
        out_dict_min_soil_extent[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}_{end_year}"] = converted_array_min_soil_extent

    lu.print_and_log(f"After calculating densities for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",False, logger_worker)

    # Need to put the SOC layers in chronological order so they can be differenced later for full extent and mineral soil extent
    out_dict_full_extent_ordered = dict(sorted(out_dict_full_extent.items()))
    out_dict_min_soil_extent_ordered = dict(sorted(out_dict_min_soil_extent.items()))

    # print("out_dict_full_extent_ordered:", out_dict_full_extent_ordered)
    # print("out_dict_min_soil_extent_ordered:", out_dict_min_soil_extent_ordered)


    ### Part 3: Calculate density changes between adjacent intervals (Mg CO2/ha/yr for 0-30 cm)

    # Computes and save deltas. Iterates through both full extent and mineral soil extent
    # print(year_ranges)
    lu.print_and_log(f"Calculating consecutive SOC changes for {bounds_str}: {uu.timestr()}", False, logger_worker)
    for i, start_year in enumerate(cn.SOC_density_intervals[:-1]):  # Stops iterating at year before last because year_diff is based on the next year
        end_year = cn.SOC_density_intervals[i+1]
        year_diff = end_year-start_year
        # print(f"start_year: {start_year}; end_year: {end_year}; year_diff: {year_diff}")

        # Final interval is 3.5 years (2015-2020 vs. 2020-2022, midpoints are 2017.5 and 2021, difference is 3.5)
        if end_year == cn.SOC_density_intervals[-1]:
            year_diff = 3.5

        lu.print_and_log(f"Calculating SOC change for {end_year} to {start_year} using {year_diff} years for {bounds_str}: {uu.timestr()}", is_large_run, logger_worker)

        # Multiplies difference by -1 to make net loss positive and net gain negative (as for vegetation)
        # Interval arrays must be unsigned so difference can be negative
        net_full_extent = (out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}_{end_year}"] -
                             out_dict_full_extent_ordered[f"{cn.SOC_density_full_extent_pattern}{cn.C_density_pixel_meaning}_{start_year}"]) / year_diff * -1 * cn.C_to_CO2
        net_min_soil = (out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}_{end_year}"] -
                          out_dict_min_soil_extent_ordered[f"{cn.SOC_density_min_soil_extent_pattern}{cn.C_density_pixel_meaning}_{start_year}"]) / year_diff * -1 * cn.C_to_CO2

        # Multiplying by -1 creates -0s, so need to force all -0s back to 0. Per Claude session 'Fix negative zero in stock net change calculation'
        net_full_extent[net_full_extent == 0] = np.float32(0)
        net_min_soil[net_min_soil == 0] = np.float32(0)

        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69d50592-48b8-8329-b529-2babe02f7f27
        # Gross loss (positive, like for vegetation)
        SOC_loss_full_extent = np.full_like(net_full_extent, np.nan, dtype=np.float32)
        SOC_loss_full_extent[net_full_extent > 0] = net_full_extent[net_full_extent > 0]
        SOC_loss_full_extent[net_full_extent == 0] = 0

        SOC_loss_min_soil = np.full_like(net_min_soil, np.nan, dtype=np.float32)
        SOC_loss_min_soil[net_min_soil > 0] = net_min_soil[net_min_soil > 0]
        SOC_loss_min_soil[net_min_soil == 0] = 0

        # Gross gain (negative, like for vegetation)
        SOC_gain_full_extent = np.full_like(net_full_extent, np.nan, dtype=np.float32)
        SOC_gain_full_extent[net_full_extent < 0] = net_full_extent[net_full_extent < 0]
        SOC_gain_full_extent[net_full_extent == 0] = 0

        SOC_gain_min_soil = np.full_like(net_min_soil, np.nan, dtype=np.float32)
        SOC_gain_min_soil[net_min_soil < 0] = net_min_soil[net_min_soil < 0]
        SOC_gain_min_soil[net_min_soil == 0] = 0

        # Saves back to output dicts with the converted unit arrays
        out_dict_full_extent_ordered[f"{cn.SOC_net_full_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = net_full_extent
        out_dict_min_soil_extent_ordered[f"{cn.SOC_net_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = net_min_soil

        out_dict_full_extent_ordered[f"{cn.SOC_loss_full_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = SOC_loss_full_extent
        out_dict_min_soil_extent_ordered[f"{cn.SOC_loss_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = SOC_loss_min_soil

        out_dict_full_extent_ordered[f"{cn.SOC_gain_full_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = SOC_gain_full_extent
        out_dict_min_soil_extent_ordered[f"{cn.SOC_gain_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}_{end_year}"] = SOC_gain_min_soil

    calc_end = time.time()
    lu.print_and_log(f"After calculating SOC change for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",False, logger_worker)
    lu.print_and_log(f"Calculated {bounds_str} in {tile_id} in {round(calc_end-calc_start)} seconds: {uu.timestr()}", False, logger_worker)

    # print("out_dict_full_extent_ordered:", out_dict_full_extent_ordered)
    # print("out_dict_min_soil_extent_ordered:", out_dict_min_soil_extent_ordered)


    ### Part 5: Writes outputs to pre-existing global mega-zarr (only if activated)

    # Combine the full extent and mineral soil extent dictionaries into a single dictionary
    out_dict_combined = out_dict_full_extent_ordered | out_dict_min_soil_extent_ordered
    # print(out_dict_combined)

    zu.populate_zarr(bounds, bounds_str, create_zarr, cn.SOC_density_intervals, is_large_run, logger_worker, mega_zarr_path,
                  out_dict_combined, outputs_to_zarr, stage, tile_id)


    ### Part 6: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, logger_worker, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers as a dictionary with chunk attributes.
    # NOTE: The full-interval chunk sums don't exactly match the sums of the individual intervals' chunk sums
    # because of float32 rounding errors. However the output full-model rasters are definitely close enough
    # at the pixel level, so I'm fine with this slight difference.
    # Worked on it in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/681244d9-83dc-800a-b397-0706e79391c0
    # but never implemented the fix because the very slight rounding results in <0.01% difference.

    # Lists of pixel counts for full extent and mineral soil extent for all years, to check if any pixels are being lost in any years.
    # All years for density for a given extent should have the same number of pixels.
    # I noticed in a global run on 2025-12-24 that some chunks dropped 0.1x0.1 deg areas for some years, maybe because I had too many workers.
    # This would catch sub-chunks being dropped from individual years.
    full_extent_density_pixel_count_list = []
    mineral_extent_density_pixel_count_list = []

    for key, array_per_ha in out_dict_combined.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats = uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel)

        # Populates lists of pixel counts for each chunk across years to make sure they're the same
        if chunk_stats['pattern'] == f"{cn.SOC_density_full_extent_pattern}_ha":
            full_extent_density_pixel_count_list.append(chunk_stats['count_value'])
        if chunk_stats['pattern'] == f"{cn.SOC_density_min_soil_extent_pattern}_ha":
            mineral_extent_density_pixel_count_list.append(chunk_stats['count_value'])

        chunk_stats_combined.append(chunk_stats)

    # Extent chunks for all years should have the same number of pixels.
    # Sometimes they don't, and I think that's because the underlying geotifs have different numbers of pixels, so it's not a problem with this script.
    # Still, I want to be aware of when the pixel counts are different across years.
    all_same_full_extent = len(set(full_extent_density_pixel_count_list)) <= 1
    all_same_mineral = len(set(mineral_extent_density_pixel_count_list)) <= 1
    lu.print_and_log(f"Pixel count in full extent chunk for {bounds_str} in {tile_id}: {full_extent_density_pixel_count_list}. All the same: {all_same_full_extent}.", False, logger_worker)
    lu.print_and_log(f"Pixel count in mineral soil extent chunk for {bounds_str} in {tile_id}: {mineral_extent_density_pixel_count_list}. All the same: {all_same_mineral}.", False, logger_worker)

    if not all_same_full_extent or not all_same_mineral:
        msg = (
            f"WARNING: Pixel count mismatch in chunk {bounds_str} ({tile_id}). "
            f"Full extent counts: {full_extent_density_pixel_count_list}, "
            f"Mineral extent counts: {mineral_extent_density_pixel_count_list}"
        )
        lu.print_and_log(msg, False, logger_worker)


    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)


    ### Part 7: Saves numpy arrays as rasters and uploads to s3

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    # print("outputs_by_interval_dir_list:", outputs_by_interval_dir_list)
    if not no_upload:

        out_no_data_val = np.nan  # NoData value for output raster (optional)
        upload_start_time = time.time()

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_combined.items():
            data_type = value.dtype.name
            # print("key:", key)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, interval_year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("interval_year_range:", interval_year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output (list of one element).
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in outputs_by_interval_dir_list if
                                         out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if interval_year_range in item]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)
            # print("")

            # Dictionary with metadata for each array
            out_dict_combined[key] = [value, data_type, out_pattern, interval_year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                           out_dict_combined, is_large_run, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False,logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        upload_end_time = time.time()
        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {outputs_by_interval_dir_list[0]} in {round(upload_end_time - upload_start_time)} seconds: {uu.timestr()}", is_large_run, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"  Total chunk processing for {bounds_str} in {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}",False, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"Peak memory for {bounds_str} in {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    # return return_message  # Return both the success message and the statistics
    return return_message, chunk_stats_combined  # Return both the success message and the statistics


def main(cluster_name, model_type,
         run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None,
         run_date=None, model_path_description=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'soil_carbon_densities_and_changes'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 3800
    # batch_size = 2  # For testing batch processing

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Theoretically, prevents indefinite retries. Untested.
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3
    config.set({
        "distributed.scheduler.allowed-failures": 2
    })

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)
    
    # Sets date as today if it's not supplied
    if not run_date:
        today = date.today()
        run_date = today.strftime("%Y%m%d")

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"SOC model version: {cn.SOC_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
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
    # is_large_run = True  # For simulating a large run
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # Whenever the run is large-scale (final), force zarr creation
    if is_large_run:
        create_zarr = True
    main_logger.info(f"Create and populate global mega-zarr: {create_zarr}")

    # List of output paths in s3 before each interval is added, with placeholders replaced
    outputs_dir_list = [cn.SOC_density_full_extent_dir, cn.SOC_loss_full_extent_dir, cn.SOC_gain_full_extent_dir, cn.SOC_net_full_extent_dir,
                        cn.SOC_density_min_soil_extent_dir, cn.SOC_loss_min_soil_extent_dir, cn.SOC_gain_min_soil_extent_dir, cn.SOC_net_min_soil_extent_dir]
    outputs_dir_list = [path.replace("CHUNK_SIZE_pixels", f"{chunk_size_pixels}_pixels") for path in outputs_dir_list]
    outputs_dir_list = [path.replace("RUN_DATE", run_date) for path in outputs_dir_list]
    outputs_dir_list = [path.replace(cn.model_version_type_description_placeholder, f"version_{cn.SOC_model_version_underscore}__{model_type}__{model_path_description}") for path in outputs_dir_list]
    # print(outputs_dir_list)

    # List of output paths by interval in s3
    outputs_by_interval_dir_list = []

    # Replaces the placeholder intervals with the actual intervals
    for output_dir in outputs_dir_list:
        if "density" in output_dir:
            for SOC_density_interval in cn.SOC_density_intervals:
                output_dir_interval = output_dir.replace("START_END", str(SOC_density_interval))
                output_dir_interval = output_dir_interval.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning)
                outputs_by_interval_dir_list = outputs_by_interval_dir_list + [output_dir_interval]

        if ("net" in output_dir) or ("loss" in output_dir) or ("gain" in output_dir):
            for SOC_change_interval in cn.SOC_change_intervals:
                output_dir_interval = output_dir.replace("START_END", str(SOC_change_interval))
                output_dir_interval = output_dir_interval.replace("PER_HA_OR_PIXEL", cn.flux_density_pixel_meaning)
                outputs_by_interval_dir_list = outputs_by_interval_dir_list + [output_dir_interval]

    # print(outputs_by_interval_dir_list)
    if is_large_run:
        main_logger.info(f"outputs_dir_list ({len(outputs_dir_list)} folders):")
        for item in outputs_by_interval_dir_list:
            main_logger.info(f"  {item}")

    # Gets the first COG's URL and gets the NoData value
    first_url = list(cn.SOC_COGS.values())[0][0]
    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(first_url) as src:
            nodata_val = src.nodata


    ### Step 2: Create empty (metadata-only), global mega-zarr in s3.
    ### Zarr approach from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc

    # Only creates the global mega-zarr if needed (large runs or otherwise specified)
    if create_zarr:

        # Creates s3 paths for the raw mega-zarr
        zarr_path = zu.create_zarr_path(cn.SOC_path_zarr, chunk_size_pixels, 'N/A',
                                        model_type, cn.SOC_model_version_underscore, model_path_description,
                                        run_date, main_logger)

        # These variables are added to the mega-zarr
        # Adds the unit to the zarr variable names (uses re.sub to apply to end of string only so that these don't overwrite each other).
        outputs_to_zarr = cn.SOC_outputs_to_zarr
        outputs_to_zarr_with_unit = [
            re.sub(r'^(SOC_net.*)$', r'\1_ha_yr', pattern)
            for pattern in outputs_to_zarr
        ]
        outputs_to_zarr_with_unit = [
            re.sub(r'^(SOC_loss.*)$', r'\1_ha_yr', pattern)
            for pattern in outputs_to_zarr_with_unit
        ]
        outputs_to_zarr_with_unit = [
            re.sub(r'^(SOC_gain.*)$', r'\1_ha_yr', pattern)
            for pattern in outputs_to_zarr_with_unit
        ]
        outputs_to_zarr_with_unit = [
            re.sub(r'^(SOC_density.*)$', r'\1_ha', pattern)
            for pattern in outputs_to_zarr_with_unit
        ]

    #     # Creates the global mega-zarr with metadata only
    #     zu.initialize_global_zarr(zarr_path, outputs_to_zarr_with_unit, len(cn.SOC_density_intervals),
    #                               ((cn.end_year_count), chunk_size_pixels, chunk_size_pixels), main_logger)
    #
    #     fs = fsspec.filesystem("s3", anon=False)
    #     mapper = fs.get_mapper(zarr_path)
    #     z = zarr.open_group(mapper, mode="r")
    #     test_var_name = list(z.array_keys())[0]  # Chooses first dataset just to check properties
    #     arr = z[test_var_name]
    #     main_logger.info(f"Inspecting variable: {test_var_name}")
    #     main_logger.info(f"Zarr dtype: {arr.dtype}")
    #     main_logger.info(f"Zarr fill_value: {arr.fill_value}")
    #     main_logger.info(f"Zarr shape: {arr.shape}")
    #     main_logger.info(f"Zarr chunks: {arr.chunks}")
    #
    # else:
    #     zarr_path = None
    #     outputs_to_zarr = False
    #
    #
    # ### Step 3: Create 1x1 deg outputs
    #
    # # Creates list of tasks to run (1 task = 1 chunk)
    # main_logger.info("Workers' logs to be appended after main function log"+ "\n")
    #
    # chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    # main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")
    #
    # # Accumulates all output messages and statistics across batches
    # # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    # all_results = []
    # all_stats = []
    # success_count = 0  # Count of successful chunks
    #
    # # Iterates through the batches
    # for i, chunk_batch in enumerate(chunk_batches):
    #     main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
    #     main_logger.info("Creating batch task txts in s3...")
    #     uu.create_s3_task_files(stage, chunk_batch)
    #
    #     # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_LULUCF_fluxes ... )]
    #     futures = []
    #     for chunk in chunk_batch:
    #
    #         future = client.submit(create_soil_C_density_and_change, chunk,
    #                                is_large_run, stage, no_upload, create_zarr, nodata_val, outputs_by_interval_dir_list,
    #                                zarr_path, outputs_to_zarr)
    #         futures.append(future)
    #
    #     batch_results = client.gather(futures)
    #
    #     all_results.extend(batch_results)
    #
    #     success_count, batch_stats = uu.count_successful_chunks(chunk_batch, is_large_run, main_logger, batch_results)
    #     all_stats.extend(batch_stats)
    #
    #     # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
    #     # That way there are some basic chunk stats (not sorted or anything) to fall back on.
    #     if len(chunk_batches) > 1:
    #
    #         main_logger.info(f"Writing batch stats to disk: {uu.timestr()}")
    #         df_batch_stats = pd.DataFrame(batch_stats)
    #
    #         timestamp = uu.timestr()
    #
    #         # Writes batch output to parquet file if output is large
    #         if len(df_batch_stats) > 900_000:
    #             out_file = f"TEMP_BATCH_{stage}__batch_{i}_{timestamp}.parquet"
    #             local_path = f"{cn.local_chunk_stats_path}{out_file}"
    #
    #             # Coerce output to string so there aren't mismatched types
    #             # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694c44d0-19e8-8330-8098-a7ec93366e44
    #             for col in ['min_value', 'max_value', 'mean_value', 'sum_value', 'count_value']:
    #                 if col in df_batch_stats.columns:
    #                     df_batch_stats[col] = df_batch_stats[col].astype(str)
    #
    #             df_batch_stats.to_parquet(
    #                 local_path,
    #                 engine="pyarrow",
    #                 index=False
    #             )
    #
    #         # Otherwise, writes output to spreadsheet
    #         else:
    #             out_file = f"TEMP_BATCH_{stage}__batch_{i}_{timestamp}.xlsx"
    #             local_path = f"{cn.local_chunk_stats_path}{out_file}"
    #
    #             with pd.ExcelWriter(local_path) as writer:
    #                 df_batch_stats.to_excel(
    #                     writer,
    #                     sheet_name=f"stats__batch_{i}",
    #                     index=False
    #                 )
    #
    #     del futures
    #     del batch_results
    #     client.run(gc.collect)
    #
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)
    #
    #
    # ### Step 4: Gather worker logs (preliminary, just in case later step goes awry)
    #
    # # Collects worker logs before moving to processing that doesn't need the cluster
    # if not run_local:
    #
    #     # Creates combined log from all workers if not deactivated
    #     worker_log_local_path_prelim = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage} with preliminary worker log compilation", main_logger)
    #
    #
    # ### Step 5: Consolidate chunk stats and export
    #
    # # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # # and min and max values across all chunks for all inputs and outputs
    # # only if not suppressed by the --no_stats flag and at least one chunk was successful (wasn't skipped).
    # if (not no_stats) and (success_count > 0):
    #     model_chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
    #     uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)


    ### Step 6: Compares model output chunk stats to zarr chunk stats for each variable-year (only if chunk stats created)

    model_chunk_stats_path = 'chunk_stats/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260611_20_36_25__with_pivot__KEEP.xlsx' #TODO for testing

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

        # Number of chuinks that have model chunk stats but not corresponding zarr chunk stats
        chunks_without_zarr_stats_total = 0

        # Iterates through variables/datasets.
        for test_var_name in outputs_to_zarr:

            main_logger.info(f"Starting {test_var_name}: {uu.timestr()}")
            var_start_time = time.time()

            # Runs chunk stats for a dataset (all years) in the zarr in parallel
            chunk_stats_variable_year_zarr = zu.run_parallel_stats(
                client=client,
                chunk_list=chunk_list,
                var=test_var_name,
                zarr_path=zarr_path,
                interval_end_years=cn.SOC_density_intervals
            )

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
                                                                   test_var_name,
                                                                   zarr_comparison_stats_path)

            # Total number of chunks that have differences in metrics between the model and zarr
            # that exceed the tolerance
            chunks_count_exceeding_total += chunks_count_exceeding
            chunks_without_zarr_stats_total += chunks_without_zarr_stats

            var_end_time = time.time()
            main_logger.info(f"  Processed {test_var_name} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")

        # Counts up chunks that had differences exceeding the tolerance and uploads chunk stats comparisons.
        zu.upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                              main_logger, model_chunk_stats_table_name,
                                              stage, start_time, zarr_comparison_stats_name, zarr_comparison_stats_path)


    ### Step 7: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)


    ### Step 8: Resize cluster down to 1 worker for remaining steps since they only need a minimal remainder of the
    ### cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 9: Count output geotifs in s3
    # Iterates through select output folders and counts the number of output rasters (only if uploads enabled and a large run (to save console space))

    main_logger.info(f"Counting geotifs in select output folders. Expecting {len(chunk_list)} in each: {uu.timestr()}")

    if not no_upload and is_large_run:
        for output_folder in outputs_by_interval_dir_list:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            if file_count != len(chunk_list):
                main_logger.warning(f"WARNING: Output file count in {output_folder} does not match expectations!")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 10: Merge compiled worker log and main log
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)


    # Closes the Dask client if not running locally
    if not run_local:
        client.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and compute SOC stock + change.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('--create_zarr', action='store_true', help='Create and populate global mega-zarr with model outputs')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size_deg = args.chunk_size_deg
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    create_zarr = args.create_zarr

    # Create the cluster with command line arguments
    main(cluster_name, model_type, run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size_deg=chunk_size_deg, first_chunks=first_chunks,
         run_date=run_date, model_path_description=model_path_description, log_note=log_note)

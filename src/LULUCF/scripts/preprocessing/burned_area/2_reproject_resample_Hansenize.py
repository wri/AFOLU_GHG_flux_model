"""
Burned area preprocessing Step 2:

This script converts annual burned area rasters to final Hansen rasters (10x10 deg, 0.00025x0.00025 deg, WGS84).
Chunk sizes are 10x10 deg, not 1x1 deg (unlike some other processes).

Run this preprocessing code on the outputs of Step 1.
Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e
This code aligns with the core model (as of 3/5/25) in terms of logging, chunk stats, etc.
It does not currently track task progress through the s3 txt file system because that's designed to track progress
when all chunks in a list are expected to be uploaded to s3. Here, only outputs with burned area are uploaded,
so there'd be lots of chunks left over in s3 that were processed but just didn't have data.

python -m src.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m src.LULUCF.scripts.preprocessing.burned_area.2_reproject_resample_Hansenize -cn AFOLU_flux_model_scripts -yr 2001 -bb 40 50 50 60 -cs 10 -yr 2000 2024

python -m src.utilities.create_cluster -n 30 -t 5 -cn AFOLU_flux_model_scripts
python -m src.LULUCF.scripts.preprocessing.burned_area.2_reproject_resample_Hansenize -cn AFOLU_flux_model_scripts -bb -180 -60 180 80 -cs 10 -yr 2000 2024
Max memory usage: ~20 GB/worker
Time: 1.5 hours through calculation, 1.5 hours with tile stats; Credits: 190; Cost: $9.30

For the full run, -n 30 is good for workers because each year is processed separately, so there's always going to be
laggard tasks at the end of a year and the more workers there are, the greater the cost from waiting around for the laggards.
So, it's best to let it run with fewer workers but not have ~100 workers waiting for a few lagging tasks.

FOR NEXT TIME: -n 25 -t 7   because there seems to be enough memory to use more threads at once
"""

import argparse
import io
import boto3
import rasterio
import re
import numpy as np
import dask
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
from dask.distributed import print
from shapely.geometry import box
from pyproj import Transformer

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

# CRS Definitions
MODIS_SINUSOIDAL = "+proj=sinu +lon_0=0 +datum=WGS84 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
WGS84_CRS = "EPSG:4326"

# Output resolution (0.00025° ≈ 28m)
GRID_SIZE = 10  # 10x10-degree tiles

# Set Block Size for Efficient Processing
BLOCK_SIZE = 400


### Step 2: Identify MODIS tiles intersecting the 10x10 deg chunk
def list_modis_rasters_for_tile(year, bounds, logger_worker, is_final):
    """
    Finds all MODIS rasters in S3 that intersect the given 10x10° bounding box.
    Checks intersections in both Sinusoidal and WGS84.
    When it only checks intersection from WGS84 perspective, it misses some important MODIS-10x10 overlaps.
    ChatGPT came up with this system for checking overlap from both directions.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    intersecting_modis_files = []

    # Set up bidirectional transformation
    to_sinusoidal = Transformer.from_crs(WGS84_CRS, MODIS_SINUSOIDAL, always_xy=True)
    to_wgs84 = Transformer.from_crs(MODIS_SINUSOIDAL, WGS84_CRS, always_xy=True)

    west, south, east, north = bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])

    # Convert the 10x10 WGS84 grid to Sinusoidal
    sin_west, sin_south = to_sinusoidal.transform(west, south)
    sin_east, sin_north = to_sinusoidal.transform(east, north)

    expanded_grid_box_sinu = box(sin_west, sin_south, sin_east, sin_north)

    # Convert the WGS84 10x10 box to a Shapely Polygon for WGS84 intersection check
    grid_box_wgs84 = box(west, south, east, north)

    # Iterate through MODIS rasters in S3
    for page in paginator.paginate(Bucket=cn.short_bucket_prefix, Prefix=cn.burned_area_hdf_converted_to_raw_raster_dir):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if str(year) in key and key.endswith(".tif"):
                    # Read raster metadata from S3
                    file_stream = io.BytesIO()
                    s3.download_fileobj(cn.short_bucket_prefix, key, file_stream)
                    file_stream.seek(0)

                    with rasterio.open(file_stream) as src:
                        # Step 1: Get MODIS raster bounding box in **Sinusoidal**
                        raster_box_sinu = box(*src.bounds)

                        # Step 2: Convert MODIS raster bounds to WGS84
                        raster_west, raster_south = to_wgs84.transform(src.bounds.left, src.bounds.bottom)
                        raster_east, raster_north = to_wgs84.transform(src.bounds.right, src.bounds.top)
                        raster_box_wgs84 = box(raster_west, raster_south, raster_east, raster_north)

                        # Step 3: Check intersection in both projections (WGS84 or MODIS sinusoidal)
                        if expanded_grid_box_sinu.intersects(raster_box_sinu) or grid_box_wgs84.intersects(raster_box_wgs84):
                            intersecting_modis_files.append(key)

    lu.print_and_log(f"Found {len(intersecting_modis_files)} MODIS rasters for {year} in {tile_id} ({bounds}): {intersecting_modis_files}: {uu.timestr()}", is_final, logger_worker)
    return intersecting_modis_files


### Step 3: Merge, Reproject, Resample, and Clip to 10x10 degree Hansenized tile
def process_10x10_tile(year, bounds, is_final, fishnet_iso_df):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()

    try:

        s3 = boto3.client("s3")
        west, south, east, north = bounds
        bounds_str = uu.boundstr(bounds)
        tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
        lu.print_and_log(f"Processing {tile_id} for {year} ({bounds}): {uu.timestr()}", is_final, logger_worker)
        if is_final:
            print(f"Processing {tile_id} for {year} ({bounds}): {uu.timestr()}")

        # Get relevant MODIS rasters. Theoretically this can be done once in main() rather than for every year to be more efficient
        # because a 10x10 tile should intersect the same MODIS tiles every year.
        # But it seems safer to redo this every year just in case some MODIS tiles are missing in whichever year I choose
        # as the template for 10x10 vs. MODIS intersection.
        modis_rasters = list_modis_rasters_for_tile(year, bounds, logger_worker, is_final)
        if not modis_rasters:
            return_message = f"No MODIS rasters found for {bounds} in {year}. Skipping: {uu.timestr()}"
            lu.print_and_log(return_message, is_final, logger_worker)
            if is_final:
                print(return_message)
            return return_message, chunk_stats  # No Hansenized tif uploaded because no burned area in it

        # Download and merge MODIS rasters
        sources = []
        for s3_key in modis_rasters:
            file_stream = io.BytesIO()
            s3.download_fileobj(cn.short_bucket_prefix, s3_key, file_stream)
            file_stream.seek(0)
            sources.append(rasterio.open(file_stream))

        # Merge all relevant MODIS rasters before reprojecting
        merged_array, merged_transform = merge(sources)

        # Force correct resolution & grid alignment
        width = int(GRID_SIZE / cn.resolution)  # Should be 40000
        height = int(GRID_SIZE / cn.resolution)  # Should be 40000
        dst_transform = from_bounds(west, south, east, north, width, height)

        # Prepare output profile
        profile = sources[0].profile
        profile.update(
            crs=WGS84_CRS,
            transform=dst_transform,
            width=width,
            height=height,
            dtype="uint8",
            compress="DEFLATE",
            blockxsize=BLOCK_SIZE,
            blockysize=BLOCK_SIZE,
            tiled=True
        )

        # Create reprojected array
        reprojected_array = np.zeros((height, width), dtype=np.uint8)

        # Perform reprojection
        reproject(
            source=merged_array[0],
            destination=reprojected_array,
            src_transform=merged_transform,
            src_crs=MODIS_SINUSOIDAL,
            dst_transform=dst_transform,
            dst_crs=WGS84_CRS,
            resampling=Resampling.nearest
        )

        # Check if raster contains valid data. Does not save and upload if no valid data
        if np.count_nonzero(reprojected_array) == 0:
            return_message = f"{tile_id} {year} is empty (no burned area pixels). Skipping upload: {uu.timestr()}"
            lu.print_and_log(return_message, is_final, logger_worker)
            if is_final:
                print(return_message)
            return return_message, chunk_stats

        # Save to S3
        s3_key = f"{cn.burned_area_final_dir}{year}/{tile_id}_{cn.burned_area_final_pattern}_{year}.tif"
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(reprojected_array, 1)

            s3.upload_fileobj(memfile, cn.short_bucket_prefix, s3_key)
            lu.print_and_log(f"Saved {s3_key}: {uu.timestr()}", is_final, logger_worker)

        # Calculates stats for the output layer
        chunk_stats.append(uu.calculate_stats(reprojected_array, f"burned_area_{year}", bounds_str, tile_id, 'output_layer', fishnet_iso_df))
        # print(chunk_stats)

        return_message = f"Success for {bounds} for {year}: {uu.timestr()}"

        return return_message, chunk_stats

    except Exception as e:

        return_message = f"Error processing chunk {bounds} for {year}: {e}: {uu.timestr()}"

        lu.print_and_log(return_message, is_final, logger_worker)
        print(return_message)

        return return_message, chunk_stats


### Main Function
def main(cluster_name, year_range, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    start_year = year_range[0]
    end_year = year_range[1]
    processed_years = list(range(start_year, end_year+1))

    # Model stage being run
    stage = f'Hansenize_burned_area_{start_year}_{end_year}'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Years for burned area Hansenization: {processed_years}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    # chunk_list = get_10x10_grid()
    # chunk_list = [[40, 50, 50, 60]]  # Test area in central Russia that originally wasn't getting all the MODIS tiles within this 10x10 deg area
    # chunk_list = [[40, 50, 50, 60], [50, 50, 60, 60]]  # Test area in central Russia that originally wasn't getting all the MODIS tiles within this 10x10 deg area
    # chunk_list = [[-120, 30, -110, 40]]  # Test area in southwest US/northwest Mexico

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunk_list) > 20:
        is_final = True
        main_logger.info("Running as final model.")

    # Accumulates all statistics and output messages from chunk analysis
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_stats = []
    return_messages = []
    success_count = 0

    # Populates a list of output folders, one for each year of Hansenized rasters
    output_folders = []
    for year in processed_years:
        output_folders.append(f"{cn.burned_area_final_dir}{year}/")

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    # Iterates through selected years
    for year in processed_years:


        main_logger.info(f"Processing {year}: {uu.timestr()}")
        delayed_results = [dask.delayed(process_10x10_tile)(year, chunk, is_final, fishnet_iso_df) for chunk in chunk_list]

        # Runs analysis and gathers results
        results = dask.compute(*delayed_results)
        success_count = uu.count_successful_chunks(all_stats, chunk_list, is_final, main_logger, results, return_messages)
        main_logger.info(f"Finished processing {year}: {uu.timestr()}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 1)


    # Iterates through output folders and counts the number of output rasters.
    for output_folder in output_folders:
        output_folder = re.sub('RES_pixels', '4000_pixels', output_folder)
        output_folder = re.sub('RUN_DATE', uu.timestr()[:8], output_folder)  # Converts YYYYMMDD_HH_MM_SS to YYYYMMDD
        output_folder = f"{cn.full_bucket_prefix}/{output_folder}"   # Need to prepend s3 and bucket name for counting

        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
        main_logger.info(f"Output rasters in {output_folder}: {file_count}")

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', action='store_true', help='Use shapefile to determine chunks')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, required=True, help='Starting and ending years for burned area processing')
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
    year_range = args.year_range
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, year_range, run_local, no_stats, no_log, no_upload, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size=chunk_size,
         first_chunks=first_chunks, log_note=log_note)

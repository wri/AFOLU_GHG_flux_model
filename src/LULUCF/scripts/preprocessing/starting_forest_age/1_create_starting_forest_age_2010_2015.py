"""
Maps forest age in 2010 and 2015 in 1x1 deg geotifs using GAMI v3.0.
Requires Python library zarr v2.x; can't use zarr v3.x, which is what my more recent conda environments are using (for zonal stats purposes).
If I try running this with zarr3, I get errors about not being able to access the files.

So, start with:
conda activate coiled_20250203

https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8f5974e7-3ece-11ef-967a-4ffbfe06208e
https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/
Metadata for older version: https://datapub.gfz-potsdam.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/2023-006_Besnard-et-al_Data-Description-v2.1.pdf
Also needs zarr package

Mini GEE app for age viewing from Simon Besnard (email 6/27/25):
https://besnardsim.users.earthengine.app/view/globalforestage
GEE asset for older version of GAMI: projects/ee-besnardsim/assets/GAMI_v2_0_mean_100m

This preprocessing step doesn't scale quite like others, as far as I can tell.
It starts by reading in the relevant ZARR pieces for the chunks being processed, but I don't know how that scales.
Reading the ZARR chunks involves dozens of tasks and takes much longer than all the subsequent processing.
When I was developing this script, I was able to quickly get something that worked on a single 1x1 chunk but then
slowed down in proportion to the number of chunks, even when there was an ample number of workers.
Obviously, that's not how it should go with Dask.
It took a lot longer to work out how to have the script not slow down as I scaled.
I tried several things (in conjunction with ChatGPT) but ultimately settled on having the script read the
ZARR pieces into an array at original resolution, then process the arrays real-time (not lazily).
The original approach was to have everything occur lazily and only compute at the very end, at the time of upload--
but that simply did not scale.
Note that I also tried processing 10x10 degree chunks but the problem there was that the processing of the chunks
once downloaded took too much memory and would've required really large workers.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local (won't run Dask locally because of usage of submit):
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.1_create_starting_forest_age_2010_2015 -bb 10 49 11 50 -cs 1 --run_local --no_upload

Coiled tiny test:
python -m src.utilities.create_cluster -cn starting_forest_age -n 1 -t 1 -m 8
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.1_create_starting_forest_age_2010_2015 -cn starting_forest_age -bb 10 49 11 50 -cs 1

Coiled larger test (because this doesn't always scale beyond 1 chunk well):
python -m src.utilities.create_cluster -cn starting_forest_age -n 4 -t 4 -m 8
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.1_create_starting_forest_age_2010_2015 -cn starting_forest_age -bb 10 47 13 50 -cs 1

Full run:
python -m src.utilities.create_cluster -n 80 -m 8 -cn starting_forest_age
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.1_create_starting_forest_age_2010_2015 -cn starting_forest_age -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "Global forest age for 2015 using GAMI v3.1."

In the four times I've run this, I've used more workers each time. When I first tried with 10 workers,
it kept running out of memory after a few hundred chunks. With 20 workers, it ran for much longer.
With about 40 workers, it ran all the way through.
I don't know what memory per worker is best to use. It doesn't seem to use more than about 3 GB/worker,
so maybe 8GB/worker is enough next time.
There's definitely more performance testing I could do with this, but it's not worth it because I run it very infrequently.

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67dcb99b-edb8-800a-abd8-f718de76043c
https://chatgpt.com/share/e/67e1b7f6-c0d4-800a-a945-3133de9bf3a0

"""

import argparse
import sys

import gc
import numpy as np
import rasterio
import os
import boto3
import uuid
import shutil
import traceback
import xarray as xr
from affine import Affine
import fsspec
from fsspec.implementations.cached import CachingFileSystem

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu


def try_open_zarr(url, cache_path, consolidated=True):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path, exist_ok=True)

    fs = CachingFileSystem(
        fs=fsspec.filesystem("s3", anon=True, endpoint_url="https://s3.gfz-potsdam.de"),
        cache_storage=cache_path,
        block_size=0,
        check_files=False
    )
    store = fs.get_mapper(url)
    return xr.open_zarr(store, consolidated=consolidated)


def calculate_forest_age(bounds, is_large_run, no_upload, output_dir_list, stage):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    logger_worker = lu.setup_logging_worker()
    s3 = boto3.client("s3")

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    zarr_url = "s3://dog.atlaseo-glm.eo-gridded-data/collections/GAMI/GAMI_v3.1.zarr"

    base_cache_dir = os.path.expanduser("~/zarr_cache")
    os.makedirs(base_cache_dir, exist_ok=True)

    tile_uuid = uuid.uuid4().hex[:6]
    cache_dir = os.path.join(base_cache_dir, f"{tile_id}_{tile_uuid}")

    try:
        uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)
        lu.print_and_log(f"Processing chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

        try:
            ds = try_open_zarr(zarr_url, cache_dir, consolidated=True)
        except ValueError as e:
            if "mmap length is greater than file size" in str(e):
                lu.print_and_log(f"Zarr mmap error in {tile_id}, retrying with consolidated=False...", False, logger_worker)
                ds = try_open_zarr(zarr_url, cache_dir, consolidated=False)
            else:
                raise
        # Gets the forest age dimension of the ZARR
        forest_age = ds["forest_age"]

        # Bounding box for the chunk
        lon_min, lon_max = bounds[0], bounds[2]
        lat_min, lat_max = bounds[1], bounds[3]

        # The chunk needs to be buffered by +/- double the pixel resolution so that the resampled output extends all the way
        # to the edge of the bounding box. Without cn.resolution*2, there was a one pixel-wide border around the edge with
        # no values in it. I believe that's because the resampling was only for pixels with their centers in the targeted
        # slice, so the slice has to be extended a bit so that the outermost resampled pixels have their center within
        # the (expanded) bounding box. The output raster is still the exact right size.
        buffer = cn.resolution * 2

        uu.rename_s3_task_file(stage, bounds, "loading_", is_large_run, logger_worker)

        # Loads only selected chunk. Loads into memory so that subsequent steps are "eager", not "lazy"
        lu.print_and_log(f"Loading data into memory {bounds_str}: {uu.timestr()}", False, logger_worker)  # Prints even during full run
        da_chunk = forest_age.sel(
            time="2010-01-01",
            latitude=slice(lat_max + buffer, lat_min - buffer),
            longitude=slice(lon_min - buffer, lon_max + buffer)
        ).load()

        # Deletes cache as soon as the data are loaded into memory
        shutil.rmtree(cache_dir, ignore_errors=True)

        uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)

        lu.print_and_log(f"Cleaning {bounds_str}: {uu.timestr()}", False, logger_worker) # Prints even during full run
        # da_cleaned = da_chunk.where(da_chunk != -9999, 0)
        da_cleaned = da_chunk.where((da_chunk >= 0))
        da_median = da_cleaned.median(dim="members")  # Median of the 20 ESA AGB estimates upon which age is based

        # Target high-res lat/lon grid
        new_lat = np.arange(lat_max - cn.resolution / 2, lat_min, -cn.resolution)
        new_lon = np.arange(lon_min + cn.resolution / 2, lon_max, cn.resolution)

        # Essentially, resamples from original to final resolution
        lu.print_and_log(f"Interpolating {bounds_str}: {uu.timestr()}", is_large_run, logger_worker)
        da_resampled = da_median.interp(latitude=new_lat, longitude=new_lon, method="nearest")

        # Rounds from float to int and makes NoData = 0
        da_2010 = da_resampled.round().fillna(0).astype("int16")
        arr_2010 = da_2010.values

        # Creates the 2015 age map by adding 5 to the 2010 age map where it does not equal 0
        arr_2015 = np.where(arr_2010 != 0, arr_2010 + 5, 0).astype("int16")

        transform = Affine.translation(lon_min, lat_max) * Affine.scale(cn.resolution, -cn.resolution)
        crs = "EPSG:4326"

        # Output paths
        file_2010 = f"/tmp/{tile_id}__{bounds_str}__{cn.forest_age_2010_pattern}.tif"
        file_2015 = f"/tmp/{tile_id}__{bounds_str}__{cn.forest_age_2015_pattern}.tif"

        # Writes 2010 age to raster
        lu.print_and_log(f"Saving 2010 raster {file_2010}: {uu.timestr()}", is_large_run, logger_worker)
        with rasterio.open(file_2010, 'w', driver='GTiff',
            height=arr_2010.shape[0], width=arr_2010.shape[1],
            count=1, dtype="int16", crs=crs, transform=transform,
            compress="LZW", tiled=True, blockxsize=400, blockysize=400
        ) as dst:
            dst.write(arr_2010, 1)

        # Writes 2015 age to raster
        lu.print_and_log(f"Saving 2015 raster {file_2015}: {uu.timestr()}", is_large_run, logger_worker)
        with rasterio.open(file_2015, 'w', driver='GTiff',
            height=arr_2015.shape[0], width=arr_2015.shape[1],
            count=1, dtype="int16", crs=crs, transform=transform,
            compress="LZW", tiled=True, blockxsize=400, blockysize=400
        ) as dst:
            dst.write(arr_2015, 1)

        chunk_stats.append(uu.calculate_stats(arr_2010, cn.forest_age_2010_pattern, bounds_str, tile_id, 'output_layer'))
        chunk_stats.append(uu.calculate_stats(arr_2015, cn.forest_age_2015_pattern, bounds_str, tile_id, 'output_layer'))

        uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

        if not no_upload:

            # Uploads to S3
            s3_key_2010 = f"{output_dir_list[0][cn.full_bucket_prefix_length:]}{os.path.basename(file_2010)}"
            s3_key_2015 = f"{output_dir_list[1][cn.full_bucket_prefix_length:]}{os.path.basename(file_2015)}"

            lu.print_and_log(f"Uploading to S3: {s3_key_2010}, {s3_key_2015}", is_large_run, logger_worker)
            s3.upload_file(file_2010, cn.short_bucket_prefix, s3_key_2010)
            s3.upload_file(file_2015, cn.short_bucket_prefix, s3_key_2015)

        lu.print_and_log(f"Finished chunk {bounds_str}: {uu.timestr()}", is_large_run, logger_worker)

        return_message = f"Success creating 2010/2015 age maps for {bounds_str}: {uu.timestr()}"

        # Cleans up the worker
        os.remove(file_2010)
        os.remove(file_2015)

        # Removes task tracking file from S3 once task is successful
        uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    except Exception as e:

        error_trace = traceback.format_exc()
        return_message = f"Error creating 2010/2015 age maps for chunk {bounds}: {e}---{error_trace}: {uu.timestr()}"

        lu.print_and_log(return_message, False, logger_worker)
        uu.rename_s3_task_file(stage, bounds, "error_", is_large_run, logger_worker)

        shutil.rmtree(cache_dir, ignore_errors=True)

    finally:
        # Always try to remove cache dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    return return_message, chunk_stats  # Returns both the success message and the chunk statistics



def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    age_years = [2010, 2015]
    stage = f'create_forest_age_{age_years[0]}_{age_years[1]}__1x1_deg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Starting time for stage
    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Years for age maps: {age_years}")

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size, first_chunks, fishnet_iso_df, main_logger)

    # # To restart part way through the chunk_list
    # chunk_list = chunk_list[11400:]
    # chunk_list = chunk_list[18000:]



    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info("Running as final model.")

    # Creates list of output directories specific to the run
    output_dir_list = [cn.forest_age_2010_dir, cn.forest_age_2015_dir]
    output_dir_list = [path.replace("CHUNK_SIZE", str(chunk_size_pixels)) for path in output_dir_list]


    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")


    # Runs in batches of specified size. This may help with managing zarr access/caches.
    batch_size = 2000
    # batch_size = 5  # For testing
    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")
    all_forest_age_results = []
    all_1x1_stats = []

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        # Clear cache at the start of each batch, just in case something is left over from the
        # previous batch
        shutil.rmtree(os.path.expanduser("~/zarr_cache"), ignore_errors=True)

        futures = [client.submit(calculate_forest_age, chunk, is_large_run, no_upload, output_dir_list, stage)
                   for chunk in chunk_batch]

        try:
            forest_age_results = client.gather(futures)
        except Exception as e:
            main_logger.error(f"Batch {i + 1} failed: {e}: {uu.timestr()}")
            sys.exit()

        all_forest_age_results.extend(forest_age_results)

        success_count_1x1, batch_stats = uu.count_successful_chunks(chunk_batch, is_large_run, main_logger, forest_age_results)
        all_1x1_stats.extend(batch_stats)

        del futures
        del forest_age_results
        client.run(gc.collect)

        if not no_upload:
            for output_folder in output_dir_list:
                geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
                main_logger.info(f"Output rasters in {output_folder}: {file_count}")

        main_logger.info(f"Batch {i + 1}/{len(chunk_batches)} complete: {success_count_1x1} succeeded: {uu.timestr()}")
        main_logger.info("Clearing base Zarr cache after batch...")
        shutil.rmtree(os.path.expanduser("~/zarr_cache"), ignore_errors=True)
        uu.stage_duration(start_time, uu.timestr(), f"{stage}_batch_{i}", main_logger)

        # Prepares 1x1 deg chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
        # and min and max values across all chunks for all inputs and outputs
        # only if not suppressed by the --no_stats flag and at least one chunk was successfully (wasn't skipped).
        if not no_stats:
            chunk_stats_path = uu.compile_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)

        if not run_local:
            # Creates combined log from all workers if not deactivated
            worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
            uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats and worker log compilation", main_logger)

            # Adds the workers' logs to the main log and uploads to s3
            lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create starting forest age in 2010 and 2015.")
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

"""
Creates 10x10 deg geotifs from the starting composite primary forest zarr (2015, single uint8 mask).
It creates a task list for all 10x10 deg tiles in the area of interest, then runs that task list in parallel.

Providing a bounding box with -bb or a chunk shapefile limits the 10x10 deg creation
to the 10x10 deg tiles that contain the bounding box or shapefile.
The entire 10x10 deg tile that contains the selected chunks will be processed (not just the parts with the selected chunks).

The chunk stats table argument (xlsx or Parquet) allows the pixel counts in the 10x10 deg tiles to be compared to
the pixel counts in the constituent 1x1 deg tiles to make sure that pixels aren't being lost during 10x10 deg tile
creation.

model_type, model_path_description, and the zarr run date are not command line arguments here (unlike the
starting_carbon_pools and vegetation_model 10x10 scripts) because 1_starting_composite_primary_forest.py hardcodes
model_type='standard', model_path_description='NA', and the run date to cn.starting_composite_primary_forest_run_date
when it builds the zarr. Hardcoding the same values here guarantees this script points at that same zarr.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.2_starting_composite_primary_forest_to_10x10deg -bb 23 -4 24 -3 -mt standard -mpd test_box --run_local --no_upload -mcstn starting_composite_primary_forest_1x1_chunk_statistics_20260210_17_37_50__KEEP.xlsx -ft 1

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn starting_composite_primary_forest
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.2_starting_composite_primary_forest_to_10x10deg -cn starting_composite_primary_forest -bb 23 -4 24 -3 -mt standard -mpd test_box -mcstn starting_composite_primary_forest_1x1_chunk_statistics_20260210_17_37_50__KEEP.xlsx

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn starting_composite_primary_forest
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.2_starting_composite_primary_forest_to_10x10deg -cn starting_composite_primary_forest -mt standard -mpd Cerrado -mcstn starting_composite_primary_forest_1x1_chunk_statistics_20260210_17_37_50__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn starting_composite_primary_forest
python -m src.LULUCF.scripts.preprocessing.starting_composite_primary_forest.2_starting_composite_primary_forest_to_10x10deg -cn starting_composite_primary_forest -mt standard -mpd global -mcstn starting_composite_primary_forest_1x1_chunk_statistics_20260210_17_37_50__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --log_note "Global 10x10 deg creation for starting composite primary forest (2015)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
and then Claude session 'Starting composite primary forest 10x10 geotifs'
"""

import argparse
import pandas as pd
import os
from dask.distributed import print
from distributed import KilledWorker

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


def main(cluster_name, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name,
         chunk_shapefile_uri=False, bounding_box=None,
         first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'starting_composite_primary_forest_aggregation_to_10x10_deg'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    run_date = cn.starting_composite_primary_forest_run_date

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # This dataset only ever covers the model start year
    year = cn.LC_first_year

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Vegetation model version: {cn.veg_model_version}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Year: {year}")
    main_logger.info(f"no_upload: {no_upload}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    # Gets a list of unique tile_ids from the chunk list
    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = sorted(list(set(tile_ids)))

    # Single variable to turn into 10x10 tiles
    var_name = cn.starting_composite_primary_forest_pattern

    # Output directory for the 10x10 deg outputs. This is the same structure as cn.starting_composite_primary_forest_dir
    # (the 1x1 deg output directory) but with cn.full_raster_dims (10x10 deg tile size) instead of cn.chunk_dims (1x1 deg tile size).
    output_dir = f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/{var_name}/{year}/{cn.full_raster_dims}_pixels/{run_date}/"
    main_logger.info(f"Output directory for 10x10 deg tiles: {output_dir}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used.
    # Uses vegetation model version so that the starting C pool run can be associated with the vegetation model easily.
    zarr_path = zu.create_zarr_path(cn.starting_composite_primary_forest_zarr_path, source_zarr_chunk_size, run_date, main_logger,
                                    cn.veg_model_version_underscore, model_type, model_path_description)
    main_logger.info(f"Aggregating from zarr ({source_zarr_chunk_size} pixel chunks): {zarr_path}")

    # Output directory
    output_base = cn.starting_composite_primary_forest_dir
    output_base = output_base.replace(f"{cn.chunk_dims}_pixels", f"{str(cn.full_raster_dims)}_pixels")
    main_logger.info(f"Core output path for aggregation: {output_base}")


    ### Step 2: Prepare model chunk stats for comparison with zarr chunk stats

    main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
    model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

    # Text added to output chunk stats table name(s) (Excel or Parquet)
    comparison_insert = "_10x10_deg_aggregation_comparison"

    tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
        comparison_insert, main_logger, model_chunk_stats_path)

    model_10x10_counts_df = tables_to_compare_dict[cn.counts_1x1_in_10x10]

    # Limits the pixel counts in the model output df to just this variable, so that pixel count differences
    # between the core model and the aggregation aren't reported for anything else that happens to be in the table.
    model_10x10_counts_df = model_10x10_counts_df[model_10x10_counts_df["layer_name"].str.contains(var_name, regex=False, na=False)]


    ### Step 3: Create 10x10 deg outputs

    all_task_count = len(tile_ids_to_process)

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    if all_task_count > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    main_logger.info(f"There are {all_task_count} tasks to process (1 variable x {len(tile_ids_to_process)} tiles)")

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for tile_id in tile_ids_to_process:

        future = client.submit(zu.create_10x10_deg_geotif_from_zarr,
                               var_name, 0, tile_id, zarr_path, output_base,
                               cn.veg_model_version_underscore, model_type, model_path_description, no_upload,
                               True, 0, True)

        futures.append(future)

    main_logger.info(f"There are {len(futures)} tiles to aggregate")

    # Results is a list of tuples, where each tuple is the per-ha and per-pixel chunk stats, each of which is a dictionary
    # for this variable-tile. per-pixel count_value is always None here because this variable is a uint8 mask,
    # not a float32 numeric output (see create_10x10_deg_geotif_from_zarr).
    try:
        results = client.gather(futures)
    except KilledWorker as e:
        main_logger.error(
            f"FAILED: A task was killed after repeated worker deaths — almost certainly out of memory. "
            f"Check the peak memory logs above for the affected tile. "
            f"Consider re-running with larger worker memory. "
            f"Dask error: {e}"
        )
        raise

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 4: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} worker log compilation", main_logger)


    ### Step 5: Resize cluster down to 1 worker for pixel count comparison, output counts, and log merging since those
    ### only needs a minimal remainder of the cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 6: Compare pixel counts in original 1x1 deg geotifs to pixel counts in 10x10 deg geotifs

    # Extracts the per-ha dictionaries from the returned tile stats so they are a separate flat list
    counts_per_ha_10x10_stats_list = [ha_dict for (ha_group, pixel_group) in results for ha_dict in ha_group]

    # Converts the pixel counts for the 10x10s into a dataframe
    counts_per_ha_10x10_df = pd.DataFrame(counts_per_ha_10x10_stats_list)

    # Merges the pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_per_ha_df = model_10x10_counts_df.merge(counts_per_ha_10x10_df, on='tile_name', how='left')

    # Gets the difference between pixel counts in 10x10s and 1x1s for each tile
    merged_10x10_counts_per_ha_df['pixel_count_diff'] = merged_10x10_counts_per_ha_df['total_count'] - merged_10x10_counts_per_ha_df['count_value']
    max_pixel_count_diff = merged_10x10_counts_per_ha_df['pixel_count_diff'].abs().max()

    if max_pixel_count_diff > 0:
        main_logger.warning(f"WARNING: at least one tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff}: {uu.timestr()}")
    else:
        main_logger.info(f"No tiles have a difference in pixel counts between 1x1s and 10x10s.")

    # Number of rows from model output without matching 10x10 aggregation pixel counts
    main_logger.info(f"Rows without pixel count comparison for output: {merged_10x10_counts_per_ha_df['pixel_count_diff'].isna().sum()}")

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_per_ha_df, stage, no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stat comparison", main_logger)


    ### Step 7: Count output geotifs in s3

    # Only one output folder exists for this variable (no per-pixel or 0.04x0.04 deg aggregated outputs,
    # since those are only created for float32 numeric outputs).
    if not no_upload and is_large_run:
        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_dir)
        main_logger.info(f"Output rasters in {output_dir}: {file_count}")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 8: Merge worker and local logs
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create 10x10 deg output geotifs for starting composite primary forest (2015)")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=True, help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard)')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area)')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description, log_note=log_note)

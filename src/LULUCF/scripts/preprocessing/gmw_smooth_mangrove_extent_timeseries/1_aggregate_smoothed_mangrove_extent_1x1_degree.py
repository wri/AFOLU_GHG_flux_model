"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.1_aggregate_smoothed_mangrove_extent_1x1_degree --first_10x10s_to_process 2

Coiled test:
python -m src.utilities.create_cluster -n 2 -m 2 -cn mangrove_10x10_tiles
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.1_aggregate_smoothed_mangrove_extent_1x1_degree -cn mangrove_10x10_tiles --first_10x10s_to_process 2

Full Coiled run:
python -m src.utilities.create_cluster -n 20 -t 8 -m 8 -cn mangrove_10x10_tiles
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.1_aggregate_smoothed_mangrove_extent_1x1_degree -cn mangrove_10x10_tiles
Time: 35 for all years; Cost: 24 credits; peak memory: x GB/worker
Note: Try with 1 thread per worker with 4 GB workers next time

todo:
- have step 0 and step 1 do pixel count instead of chunk stats for comparison

"""

import argparse
import dask
import re
import sys

# Project imports
from src.utilities import constants_and_names as cn, log_utilities as lu, universal_utilities as uu, resize_cluster

def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload= False,
         first_10x10s_to_process=None, log_note=None):


    ### Step 1: Preparation
    # Model stage being run
    stage = f'smoothed_mangrove_extent_10x10_deg_aggreg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # chunk stats for all years
    all_years_stats = []

    for year in cn.mangrove_extent_years:

        stage = f'smoothed_mangrove_extent_{year}_10x10_deg_aggreg'
        start_time = uu.timestr()
        main_logger.info(f"Stage {stage} started at: {start_time}")

        # Directories to process
        input_dir_list = [f"{cn.mangrove_1x1deg_smoothed_dir}{year}/"]
        output_dir = f"{cn.mangrove_extent_processed_dir}{year}/"


        ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

        # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
        # These are the basis for the aggregation tasks.
        list_of_s3_name_dicts_total = uu.create_list_for_aggregation(input_dir_list, main_logger)

        # For testing. Limits the number of output rasters to that given in the command line
        if first_10x10s_to_process:
            list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_10x10s_to_process]
        # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific tile

        # Extracts and lists unique tile_ids, the target for aggregation
        tile_ids = set()
        for entry in list_of_s3_name_dicts_total:
            for key, filenames in entry.items():
                for filename in filenames:
                    match = re.search(cn.tile_id_pattern, filename)
                    if match:
                        tile_ids.add(match.group())
        print(f"tile_ids: {tile_ids}")

        # Converts set of tile_ids to sorted list of tile_ids
        chunk_list = sorted(tile_ids)
        main_logger.info(f"tile_ids to process: {chunk_list}")
        main_logger.info(f"Number of tile_ids to process: {len(chunk_list)}")

        # Determines if the output file names for final versions of outputs should be used
        is_final = False
        if len(chunk_list) > 20:
            is_final = True
            main_logger.info("Running as final model.")

        main_logger.info(f"Aggregating 1x1 deg outputs to 10x10 deg outputs: {uu.timestr()}")

        # Each task is a single 10x10 deg aggregated geotif
        delayed_results_10x10_deg = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_final, no_upload, output_dir, 'extent')
                                            for s3_name_dict in list_of_s3_name_dicts_total]

        results_10x10_deg = dask.compute(*delayed_results_10x10_deg)

        success_count_10x10, all_10x10_stats = uu.count_successful_chunks(chunk_list, is_final, main_logger, results_10x10_deg)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}", main_logger)

        # Add chunk stats for this year to all_years_stats
        if (not no_stats) and (success_count_10x10 > 0):
            #uu.aggregate_10x10_chunk_stats(all_10x10_stats, stage, no_upload, main_logger)
            all_years_stats.extend(all_10x10_stats)


    ### Step 3: Chunk stats for 10x10 degree outputs, aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Prepares 10x10 deg chunk stats spreadsheet: mangrove extent
    stage = 'smoothed_mangrove_extent_all_years_10x10_deg_aggreg'
    if (not no_stats) and (len(all_years_stats) > 0):
        uu.aggregate_10x10_chunk_stats(all_years_stats, stage, no_upload, main_logger)

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
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-f', '--first_10x10s_to_process', type=int, help='Number of chunks to process from input list')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    first_10x10s_to_process = args.first_10x10s_to_process
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, run_local, no_stats, no_log, no_upload, first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)
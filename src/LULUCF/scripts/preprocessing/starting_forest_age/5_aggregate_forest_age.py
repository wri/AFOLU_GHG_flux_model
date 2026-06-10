"""
Aggregates 1x1 deg forest age geotifs into 10x10 deg forest age geotifs.
Can be used on forest age maps for 2000 (model start year), 2010 (GAMI v2.1 year), or 2015 (alternative model start year).

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Local:
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.5_aggregate_forest_age --years 2000 2010 2015 --first_10x10s_to_process 2 --run_local

Coiled test:
python -m src.utilities.create_cluster -cn starting_forest_age -n 1 -t 1 -m 4
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.5_aggregate_forest_age -cn starting_forest_age --years 2000 2010 2015 --first_10x10s_to_process 2

Full Coiled run
python -m src.utilities.create_cluster -cn starting_forest_age -n 30 -t 4 -m 4
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.5_aggregate_forest_age -cn starting_forest_age --years 2000 2010 2015
Cluster: https://cloud.coiled.io/clusters/1018902/account/wri-forest-research/information?organization=wri
Peak memory per worker: ~2 GB
Time until chunk stats: 27:48 for 2000, 15:13 for 2010, 13:05 for 2015
Time after chunk stats: 27:49 for 2000, 15:14 for 2010, 13:05 for 2015
Coiled credits: 15.51 for 2000, 8.2 for 2010, 8.5 for 2015 (31/hr for 30 m8g.medium workers, according to dashboard)
AWS cost: $0.37 for 2000, $0.18 for 2010, $0.20 for 2015 ($0.72/hr for 30 m8g.medium workers, according to dashboard)

python -m src.utilities.create_cluster -cn starting_forest_age -n 50 -t 4 -m 4
python -m src.LULUCF.scripts.preprocessing.starting_forest_age.5_aggregate_forest_age -cn starting_forest_age --years 2010 2015
2010 and 2015 redone at https://cloud.coiled.io/clusters/1019308/account/wri-forest-research/information?organization=wri
Peak memory per worker: ~2 GB
Time until chunk stats: 14:26 for 2010, 12:20 for 2015
Time after chunk stats: 14:26 for 2010, 12:21 for 2015
Coiled credits: 14.9 for 2010, 10 for 2015 (51/hr for 50 m8g.medium workers, according to dashboard)
AWS cost: $0.35 for 2010, $0.25 for 2015 ($1.16/hr for 50 m8g.medium workers, according to dashboard)
Using 50 workers used more credits than 30 workers but didn't save much time, presumably because a few
tasks take a long time and are stragglers, making all the workers wait around and incurring charges.

Cluster configuration experimentation at https://cloud.coiled.io/clusters/1018902/account/wri-forest-research/information?organization=wri.
It indicated that using 4 threads/worker and 4 GB workers was best if I'm not in a rush because it used
the least Coiled credits.

But subsequent experimenting with LULUCF aggregation (https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210803828525318?focus=true)
suggested otherwise:
Tests of LULUCF output aggregation show that 1 thread/worker with 4GB workers is low in Coiled credit usage
and runs quickly compared to other configurations.
So, if I run this again, consider changing the workers to -t 1 -m 4.

"""

import argparse
import dask
import re
import sys

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu


def main(cluster_name, years_to_aggregate, run_local=False, no_stats=False, no_log=False, no_upload= False,
         first_10x10s_to_process=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'starting_forest_age__{years_to_aggregate}__10x10_deg_aggreg'
    model_type = 'standard'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")

    for year in years_to_aggregate:

        year_start_time = uu.timestr()
        main_logger.info(f"Aggregating 1x1 deg forest age maps for {year}: {uu.timestr()}")

        if year == 2000:  # Aggregates age and age source flag geotifs
            output_dir_list = [cn.forest_age_2000_gap_filled_dir, cn.forest_age_2000_gap_filled_source_flag_dir]
        elif year == 2010:
            output_dir_list = [cn.forest_age_2010_gap_filled_dir]
        elif year == 2015:
            output_dir_list = [cn.forest_age_2015_gap_filled_dir]
        else:
            sys.exit("Year not in list")

        # Creates list of output directories specific to the run
        output_dir_list = [path.replace("CHUNK_SIZE", str(4000)) for path in output_dir_list]
        output_dir_list.sort()  # Sorts in-place
        main_logger.info(f"Directories to aggregate for {year}: {output_dir_list}")


        ### Step 2: Aggregates 1x1 degree outputs to 10x10 degree outputs

        # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
        # These are the basis for the aggregation tasks.
        list_of_s3_name_dicts_total = uu.create_list_for_aggregation(output_dir_list, main_logger)

        # For testing. Limits the number of output rasters to that given in the command line
        if first_10x10s_to_process:
            list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:first_10x10s_to_process]
        # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[338:339]  # To limit it to a specific 10x10 deg tile

        # Extracts and lists unique tile_ids, the target for aggregation
        tile_ids = set()
        for entry in list_of_s3_name_dicts_total:
            for key, filenames in entry.items():
                for filename in filenames:
                    match = re.search(cn.tile_id_pattern, filename)
                    if match:
                        tile_ids.add(match.group())

        # Converts set of tile_ids to sorted list of tile_ids
        chunk_list = sorted(tile_ids)
        main_logger.info(f"tile_ids to aggregate within: {chunk_list} ({len(chunk_list)}) tile_ids")

        # Determines if the output file names for final versions of outputs should be used
        is_large_run = False
        if len(chunk_list) > 20:
            is_large_run = True
            main_logger.info("Running as final model.")

        main_logger.info(f"Aggregating 1x1 deg outputs to 10x10 deg outputs for {year}: {uu.timestr()}")

        # Each task is a single 10x10 deg aggregated geotif
        delayed_results_10x10_deg = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, is_large_run, no_upload)
                                            for s3_name_dict in list_of_s3_name_dicts_total]

        results_10x10_deg = dask.compute(*delayed_results_10x10_deg)

        success_count_10x10, all_10x10_stats = uu.count_successful_chunks(chunk_list, is_large_run, main_logger, results_10x10_deg)

        uu.stage_duration(year_start_time, uu.timestr(), f"{stage} for {year}", main_logger)


        ### Step 3: Chunk stats (i.e. pixel counts) for 10x10 degree outputs, aggregates logs

        # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
        if (not no_stats) and (success_count_10x10 > 0):
            uu.aggregate_10x10_chunk_stats(all_10x10_stats, stage, no_upload, main_logger)

        uu.stage_duration(year_start_time, uu.timestr(), f"{stage} with tile stats for {year}", main_logger)

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
    parser = argparse.ArgumentParser(description="Aggregate starting forest age to 10x10 deg geotifs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-f', '--first_10x10s_to_process', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('--years', nargs="*", type=int, required=True, help='Year(s) to process, as a list: [2000], [2000, 2010], [2000, 2010, 2015], etc')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    first_10x10s_to_process = args.first_10x10s_to_process
    years = args.years
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    main(cluster_name, years, run_local, no_stats, no_log, no_upload, first_10x10s_to_process=first_10x10s_to_process, log_note=log_note)


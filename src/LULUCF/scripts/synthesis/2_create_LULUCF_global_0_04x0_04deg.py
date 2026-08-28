"""
Creates global geotifs at 0.04x0.04 deg resolution (approximately 4x4 km at the equator) for specified inputs.
Units are Mg CO2(e)/0.04x0.04 deg pixel/year for annual data and annual averages.
The geotifs can be used for presentations and other static displays.
They are not to be used for calculations or statistics.
This does not create display jpegs.

Can only run on 10x10 degree tiles already in 0.04x0.04 deg resolution.

For testing, it can be run on a specified number of datasets, years, and/or tile_ids.
It can't be run based on the extent of a shapefile or bounding box; the only way to geographically limit this
is by telling it to run on only the X first tiles with -ft argument.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.synthesis.scripts.2_create_LULUCF_global_0_04x0_04deg -mt standard -mpd global -fy 1 -fv 1 -ft 1 --run_local --no_upload --input_date 20260614

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn LULUCF_summation
python -m src.LULUCF.synthesis.scripts.2_create_LULUCF_global_0_04x0_04deg -cn LULUCF_summation -mt standard -mpd global -fy 1 -fv 1 -ft 1 --input_date 20260614

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn LULUCF_summation
python -m src.LULUCF.synthesis.scripts.2_create_LULUCF_global_0_04x0_04deg -cn LULUCF_summation -mt standard -mpd global -fy 2 -fv 2 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date 20260614 -ln "This is intended to be the definitive 1884-chunk 0.04x0.04 deg output run."

Full run:
python -m src.utilities.create_cluster -n 10 -t 1 -m 4 -cn LULUCF_summation
python -m src.LULUCF.synthesis.scripts.2_create_LULUCF_global_0_04x0_04deg -cn LULUCF_summation --input_date 20260614 -mt standard -mpd global --log_note "This is a global run for LULUCF v1.0.0: veg v1.0.5 + SOC v1.0.1 + org soil v1.0.1, 2016-2024."

Based on corresponding vegetation script, but with Claude session 'LULUCF global geotif setup'
#TODO Output combined organic soil emissions + mineral soil net change geotif at 0.04x0.04 deg resolution
"""

import argparse
import os
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"

def main(cluster_name, input_date, model_type, run_local, no_log, no_upload,
         first_variables_to_process=None, first_years_to_process=None, first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'LULUCF_0_04deg_output_global'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"LULUCF model version: {cn.LULUCF_model_version}")
    main_logger.info(f"Veg: {cn.veg_model_version};  SOC: {cn.SOC_model_version}; Organic soil: {cn.organic_soil_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    LULUCF_run = cn.LULUCF_full_version_underscore.replace("MODEL_TYPE", model_type)
    LULUCF_run = LULUCF_run.replace("MODEL_PATH_DESCRIPTION", model_path_description)

    base_path = (
            cn.LULUCF_outputs_path
            .replace(cn.model_version_type_description_placeholder, LULUCF_run)
            + f"PATTERN/annual_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    )

    # base_path = f"{cn.LULUCF_outputs_path}PATTERN/annual_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    # base_path = base_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    main_logger.info(f"Core output path for aggregation: {base_path}")

    # Outputs to create global maps for
    full_list_of_vars = [
        cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern,
        cn.gross_removals_all_C_pools_LULUCF_pattern,
        cn.net_flux_all_C_pools_all_gases_LULUCF_pattern,
    ]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars
    main_logger.info(f"Variables to create 10x10 deg tiles for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars)})")

    # Limits the processed years to the supplied number (for testing)
    if first_years_to_process:
        years_to_process = first_years_to_process
    else:
        years_to_process = cn.veg_end_year_count
    main_logger.info(f"Years to aggregate to 10x10 deg and compare chunk stats for: {years_to_process} out of {cn.veg_end_year_count}")

    # Determines if large run parameters should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(vars_to_process * years_to_process) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")


    ### Step 2: Creates outputs
    ### Separate submissions for timeseries and annual average maps because of the differing formats of the input years,
    ### But they still run in parallel because they are all part of the same futures.
    ### Per Claude session 'LULUCF global geotif setup'

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")
    # Timeseries output submission
    for var_name in vars_to_process:
        for year_idx in range(years_to_process):

            future = client.submit(uu.mosaic_tiles_to_global,
                                   var_name, year_idx, first_tiles_to_process, base_path,
                                   cn.LULUCF_model_version_underscore, model_type, model_path_description,
                                   no_upload, is_large_run)
            futures.append(future)

    # Annual average submission
    base_path_avg = base_path.replace("START_END", f"avg_{cn.veg_year_range_str}")
    for var_name in vars_to_process:

        future = client.submit(uu.mosaic_tiles_to_global,
                               var_name, 0, first_tiles_to_process, base_path_avg,
                               cn.LULUCF_model_version_underscore, model_type, model_path_description,
                               no_upload, is_large_run)
        futures.append(future)

    results = client.gather(futures)
    print(results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Aggregates logs

    # Resizes cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    # cluster, not all the workers.
    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 8
        if n_workers > 8:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Sets it so that no worker logs are created if doing a local run
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global 0.04x0.04 deg output maps.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, model_type, run_local, no_log, no_upload,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description, log_note=log_note)


"""
Creates global outputs at 0.04x0.04 deg resolution (approximately 4x4 km at the equator) for specified inputs.
Net change, loss, and gain are Mg CO2/0.04x0.04 deg pixel/year for interval-level outputs.
Density is Mg C/0.04x0.04 deg pixel/year.
Negative is SOC gain and positive is SOC loss (same signs as vegetation).
These are for presentations and other static displays.
They are not to be used for calculations or statistics.

Can only run on 10x10 degree tiles already in 0.04x0.04 deg resolution.

For testing, it can be run on a specified number of datasets, years, and/or tile_ids.
It can't be run based on the extent of a shapefile or bounding box; the only way to geographically limit this
is by telling it to run on only the X first tiles with -ft argument.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.3_create_veg_global_0_04x0_04deg -mt standard -mpd global -fy 1 -fv 1 -ft 1 --run_local --no_upload --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 4 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.3_create_SOC_global_0_04x0_04deg -cn mineral_soil -mt standard -mpd global -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled large shapefile test:
python -m src.utilities.create_cluster -n 5 -t 1 -m 4 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.3_create_SOC_global_0_04x0_04deg -cn mineral_soil -mt standard -mpd global -fy 2 -fv 2 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD -ln "Test 1884 chunk run"

Full run:
python -m src.utilities.create_cluster -n 5 -t 1 -m 4 -cn mineral_soil
python -m src.LULUCF.scripts.mineral_soil_organic_carbon.3_create_SOC_global_0_04x0_04deg -cn mineral_soil --input_date YYYYMMDD -mt standard -mpd global --log_note "This is a global run for SOC v1.0.1 (2000-2022, revised organic/mineral soil split)."

# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant
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
    stage = 'soil_0_04deg_output_global'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"SOC model version: {cn.SOC_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Start year: 2000; end year: {cn.SOC_density_intervals[-1]}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")

    # Outputs to create global maps for
    # Separate lists for density and change because they have different numbers of years, so they need to be handled separately
    full_list_of_vars_density = [
        cn.SOC_density_full_extent_pattern,
        cn.SOC_density_min_soil_extent_pattern
    ]

    full_list_of_vars_change = [
        cn.SOC_net_full_extent_pattern,
        cn.SOC_net_min_soil_extent_pattern,
        cn.SOC_loss_full_extent_pattern,
        cn.SOC_loss_min_soil_extent_pattern,
        cn.SOC_gain_full_extent_pattern,
        cn.SOC_gain_min_soil_extent_pattern
    ]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process_density = full_list_of_vars_density[0:first_variables_to_process]
        vars_to_process_change = full_list_of_vars_change[0:first_variables_to_process]
    else:
        vars_to_process_density = full_list_of_vars_density
        vars_to_process_change = full_list_of_vars_change
    main_logger.info(
        f"Variables to create density global maps for: {vars_to_process_density} ({len(vars_to_process_density)} out of {len(full_list_of_vars_density)})")
    main_logger.info(
        f"Variables to create change global maps for: {vars_to_process_change} ({len(vars_to_process_change)} out of {len(full_list_of_vars_change)})")

    # Limits the processed years to the supplied number (for testing)
    if first_years_to_process:
        years_to_process_density = first_years_to_process
        years_to_process_change = first_years_to_process
    else:
        years_to_process_density = len(cn.SOC_density_intervals)
        years_to_process_change = len(cn.SOC_change_intervals)
    main_logger.info(
        f"Years to create global density maps for: {years_to_process_density} out of {len(cn.SOC_density_intervals)}")
    main_logger.info(
        f"Years to create global change maps for: {years_to_process_change} out of {len(cn.SOC_change_intervals)}")

    # Determines if large run parameters should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if ((len(vars_to_process_density)*2) * (years_to_process_density*2)) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    base_path = f"{cn.SOC_outputs_path}PATTERN/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {base_path}")


    ### Step 2: Creates outputs
    ### Separate submissions for density and change because of the different numbers of years,
    ### But they still run in parallel because they are all part of the same futures.

    main_logger.info(f"Starting processing: {uu.timestr()}")
    futures = []

    # Density output processing
    for var_name in vars_to_process_density:
        for year_idx in range(years_to_process_density):

            future = client.submit(uu.mosaic_tiles_to_global,
                                   var_name, year_idx, first_tiles_to_process, base_path,
                                   cn.SOC_model_version_underscore, model_type, model_path_description,
                                   no_upload, is_large_run)
            futures.append(future)

    # Change output processing
    for var_name in vars_to_process_change:
        for year_idx in range(years_to_process_change):

            future = client.submit(uu.mosaic_tiles_to_global,
                                   var_name, year_idx, first_tiles_to_process, base_path,
                                   cn.SOC_model_version_underscore, model_type, model_path_description,
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

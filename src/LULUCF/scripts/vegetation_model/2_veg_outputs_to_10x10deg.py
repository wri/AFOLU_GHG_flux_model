"""
Creates 10x10 deg per-hectare and per-pixel geotifs from global zarr for numeric model outputs.
Coded to run for summative outputs + land state nodes but can change code to run for more or less variables.
Limited to just summative + land state nodes for now because it is quite expensive for just these variables.
It creates a task list for all datasets, years, and 10x10 deg tiles for the variables, years, and area of interest,
then runs that giant task list in parallel in batches (as a safeguard against failure during a large task list).

Providing a bounding box with -bb or a chunk shapefile limits the 10x10 deg creation
to the 10x10 deg tiles that contain the bounding box or shapefile.
The entire 10x10 deg tile that contains the selected chunks will be processed (not just the parts with the selected chunks).

The chunk stats table argument (xlsx or Parquet) allows the pixel counts in the 10x10 deg tiles to be compared to
the pixel counts in the constituent 1x1 deg tiles to make sure that pixels aren't being lost during 10x10 deg tile
creation.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -bb 10 49 11 50 --run_local --no_upload -mt standard -mpd global -fy 1 -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 32 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -bb 10 49 11 50 -fy 2 -fv 2 -ft 2 -mt standard -mpd global -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx  --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -bb 10 49 11 50 -fy 2 -fv 2 -ft 2 -mt standard -mpd global -mcstn parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -bb -64 -22 -63 -21 -fy 3 -fv 3 -ft 3 -mt standard -mpd global -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -bb -64 -22 -63 -21 -fy 3 -fv 3 -ft 3 -mt standard -mpd global -mcstn parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 --input_date YYYYMMDD

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_postprocessing
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn vegetation_fluxes_1x1_chunk_statistics_XYZ.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD -log_note "10x10 deg tile creation for vegetation model v1.0.5 (2016-2024)."
python -m src.LULUCF.scripts.vegetation_model.2_veg_outputs_to_10x10deg -cn vegetation_postprocessing -mt standard -mpd global -mcstn parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "10x10 deg tile creation for vegetation model v1.0.5 (2016-2024)."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3

#TODO change NoData in flux outputs to something besides 0 because 0 has a meaning for fluxes
#TODO Parallelize 10x10 deg tile uploads in create_10x10_deg_geotif_from_zarr, per Claude session 'LULUCF 30-m outputs script'. Applies to veg, SOC, and LULUCF. Haven't tried at all.
"""

import argparse
import pandas as pd
import os
from distributed import KilledWorker
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


def main(cluster_name, input_date, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None,
         first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_aggregation_to_10x10_deg'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    batch_size = 5000  # 8 batches to cover all tasks
    # batch_size = 4  # large-scale testing

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Vegetation model version: {cn.veg_model_version}")
    main_logger.info(f"Vegatation model path descriptor: {model_path_description}")
    main_logger.info(f"Start year: {cn.LC_first_year}; end year: {cn.LC_last_year}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"Batch size: {batch_size} tasks")

    #TODO hopefully change NoData for veg model in next run
    no_data_val = 0

    # Calculates the interval type, difference between start and end years of intervals, and the model output years
    # for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(cn.LC_first_year, cn.LC_last_year, main_logger)

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

    # Outputs to turn into 10x10 tile
    # full_list_of_vars = cn.full_outputs_to_zarr   # If all variables are to be made into 10x10s (but very expensive)
    # full_list_of_vars = [cn.net_flux_all_C_pools_all_gases_pattern] # For testing
    full_list_of_vars = cn.veg_summative_output_patterns + [cn.land_state_pattern, cn.composite_primary_forest] # Summative outputs + land state nodes + composite primary forest

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

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used
    zarr_path = zu.create_zarr_path(cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, 'annual',
                                         model_type, cn.veg_model_version_underscore, model_path_description,
                                         input_date, main_logger)
    main_logger.info(f"Aggregating from zarr ({source_zarr_chunk_size} pixel chunks): {zarr_path}")

    output_base = f"{cn.veg_outputs_path}PATTERN/annual_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{input_date}/"
    main_logger.info(f"Core output path for aggregation: {output_base}")


    ### Step 2: Prepare model chunk stats for comparison with zarr chunk stats

    main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
    model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

    # Text added to output chunk stats table name(s) (Excel or Parquet)
    comparison_insert = "_10x10_deg_aggregation_comparison"

    tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
        comparison_insert, main_logger, model_chunk_stats_path)

    model_10x10_counts_df = tables_to_compare_dict[cn.counts_1x1_in_10x10]

    # Limits the pixel counts in the model output df to just the model outputs that are being aggregated in this step.
    # That way, pixel count differences between the core model and the aggregation aren't being reported at the end for
    # model outputs that aren't being run here.
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69437ae5-a94c-8326-b6d4-ad0dab6bb903
    pattern = "|".join(vars_to_process)
    model_10x10_counts_df = model_10x10_counts_df[model_10x10_counts_df["layer_name"].str.contains(pattern, regex=True, na=False)]


    ### Step 3: Create 10x10 deg outputs

    # Creates full list of tasks to run (var_name, year_idx, tile_id)
    all_tasks = [
        (var_name, year_idx, tile_id)
        for var_name in vars_to_process
        for year_idx in range(years_to_process)
        for tile_id in tile_ids_to_process
    ]

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if len(all_tasks) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    task_batches = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]

    main_logger.info(f"There are {len(all_tasks)} tasks to process ({len(vars_to_process)} variables x {years_to_process} years x {len(tile_ids_to_process)} tiles)")

    main_logger.info(f"Tasks divided into {len(task_batches)} batches of up to {batch_size} tasks: {uu.timestr()}")

    all_results = []

    # Iterates through the batches
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6980e192-5198-8332-8e69-b994315e91c0
    for i, task_batch in enumerate(task_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(task_batches)} ({len(task_batch)} tasks): {uu.timestr()}")

        futures = []

        for var_name, year_idx, tile_id in task_batch:  # Must be the same order as the tuple in all_tasks

            future = client.submit(zu.create_10x10_deg_geotif_from_zarr,
                                   var_name, year_idx, tile_id, zarr_path, output_base,
                                   cn.veg_model_version_underscore, model_type, model_path_description,
                                   no_upload, False, no_data_val, retries=3)
            futures.append(future)

        # Results is a list of tuples, where each tuple is the per-ha and per-pixel chunk stats, each of which is a dictionary
        # for a single variable-year-tile
        # e.g., ([{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_ha_yr_2016.tif',
        # 'tile_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_ha_yr_2016.tif', 'in_out': 'output_layer',
        # 'pattern': 'gross_emissions__all_C_pools__CO2_only__MgCO2', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data', 'max_value': 'no data',
        # 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}],
        # [{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_pixel_yr_2016.tif',
        # 'tile_name': '00N_050W__gross_emissions__all_C_pools__CO2_only__MgCO2_pixel_yr_2016.tif', 'in_out': 'output_layer',
        # 'pattern': 'gross_emissions__all_C_pools__CO2_only__MgCO2', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data',
        # 'max_value': 'no data', 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}]),
        # ([{'chunk_id': 'N/A', ... 'data_type': 'no data'}])]
        # Theoretically, catches cluster failure due to exceeded memory so that failures are clearer. Not fully tested.
        # Per Claude session 'Cluster task retry failures in vegetation outputs'
        try:
            batch_results = client.gather(futures)
        except KilledWorker as e:
            main_logger.error(
                f"BATCH {i + 1} FAILED: A task was killed after 4 worker deaths — "
                f"almost certainly out of memory. Check peak memory logs for the affected tile. "
                f"Consider re-running with larger worker memory (-m 64). "
                f"Dask error: {e}"
            )
            raise
        # print(batch_results)

        all_results.extend(batch_results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} worker log compilation", main_logger)


    ### Step 4: Resize cluster down to 1 worker for chunk stats and log aggregation since that only needs a minimal remainder of the
    ### cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 5: Compare pixel counts in original 1x1 deg geotifs to pixel counts in 10x10 deg geotifs

    # Extracts the per-ha and per-pixel dictionaries from the returned tile stats so they are separate flat lists
    # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/693c28bf-ade0-8325-b28b-de531cad2408
    counts_per_ha_10x10_stats_list = [ha_dict for (ha_group, pixel_group) in all_results for ha_dict in ha_group]
    counts_per_pixel_10x10_stats_list = [pixel_dict for (ha_group, pixel_group) in all_results for pixel_dict in pixel_group]
    # print(counts_per_ha_10x10_stats_list)
    # print(counts_per_pixel_10x10_stats_list)

    # Converts the pixel counts from per-ha and per-pixel in the 10x10s into dataframes
    counts_per_ha_10x10_df = pd.DataFrame(counts_per_ha_10x10_stats_list)
    counts_per_pixel_10x10_df = pd.DataFrame(counts_per_pixel_10x10_stats_list)

    # Merges the per-ha pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_per_ha_df = model_10x10_counts_df.merge(counts_per_ha_10x10_df, on='tile_name', how='left')

    # Renames the counts in the 1x1 df from ha to pixel so that their tile names match the per-pixel output
    # and they can be joined. Otherwise, the per-pixel tile names won't match the pixel counts from the 1x1s (since they say ha).
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6945fa31-cd14-8333-bf8a-f71cc6918ae6
    model_10x10_counts_df.loc[:, 'tile_name'] = (
        model_10x10_counts_df['tile_name'].str.replace('ha', 'pixel', regex=False)
    )

    merged_10x10_counts_per_pixel_df = model_10x10_counts_df.merge(counts_per_pixel_10x10_df, on='tile_name', how='left')

    # Gets the difference between pixel counts in 10x10s and 1x1s for each tile
    merged_10x10_counts_per_ha_df['pixel_count_diff'] = merged_10x10_counts_per_ha_df['total_count'] - merged_10x10_counts_per_ha_df['count_value']
    max_pixel_count_diff_per_ha = merged_10x10_counts_per_ha_df['pixel_count_diff'].abs().max()

    merged_10x10_counts_per_pixel_df['pixel_count_diff'] = merged_10x10_counts_per_pixel_df['total_count'] - merged_10x10_counts_per_pixel_df['count_value']
    max_pixel_count_diff_per_pixel = merged_10x10_counts_per_pixel_df['pixel_count_diff'].abs().max()

    if max_pixel_count_diff_per_ha > 0:
        main_logger.warning(f"WARNING: at least one per-hectare tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff_per_ha}: {uu.timestr()}")
    else:
        main_logger.info(f"No per-hectare tiles have a difference in pixel counts between 1x1s and 10x10s.")

    if max_pixel_count_diff_per_pixel > 0:
        main_logger.warning(f"WARNING: at least one per-pixel tile has a difference in pixel counts between 1x1s and 10x10s! Max difference is {max_pixel_count_diff_per_pixel}: {uu.timestr()}")
    else:
        main_logger.info(f"No per-pixel tiles have a difference in pixel counts between 1x1s and 10x10s.")

    # Number of rows from model output without matching 10x10 aggregation pixel counts
    main_logger.info(f"Rows without pixel count comparison for per-ha output: {merged_10x10_counts_per_ha_df['pixel_count_diff'].isna().sum()}")
    main_logger.info(f"Rows without pixel count comparison for per-pixel output: {merged_10x10_counts_per_pixel_df['pixel_count_diff'].isna().sum()}")

    # Prepares 10x10 deg chunk stats spreadsheet: pixel count for outputs
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_per_ha_df, f"{stage}_per_ha", no_upload, main_logger)
    uu.aggregate_10x10_chunk_stats(merged_10x10_counts_per_pixel_df, f"{stage}_per_pixel", no_upload, main_logger)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stat comparison", main_logger)


    ### Step 6: Count output geotifs in s3

    # Counts per-hectare outputs
    output_dir_list_per_ha = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, 'annual', cn.LC_first_year,
                                                            cn.full_raster_dims, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
                                                            interval_year_diff_list, input_date, False, "per_ha")
    output_dir_list_per_ha.sort()  # Alphabetically order the outputs (modifies output_dir_list_per_ha)
    if is_large_run:
        main_logger.info(f"output_dir_list_per_ha for {stage}:")
        for item in output_dir_list_per_ha:
            main_logger.info(f"  {item}")
    # print(output_dir_list_per_ha)

    # Iterates through output folders and counts the number of output rasters (only if uploads enabled and a large run (to save console space))
    if not no_upload and is_large_run:
        for output_folder in output_dir_list_per_ha:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output per-ha rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    # Counts per-pixel outputs
    output_dir_list_per_pixel = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, 'annual', cn.LC_first_year,
                                                               cn.full_raster_dims, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
                                                               interval_year_diff_list, input_date, False, "per_pixel")
    output_dir_list_per_pixel.sort()
    if is_large_run:
        main_logger.info(f"output_dir_list_per_pixel for {stage}:")
        for item in output_dir_list_per_pixel:
            main_logger.info(f"  {item}")
    # print(output_dir_list_per_pixel)

    if not no_upload and is_large_run:
        for output_folder in output_dir_list_per_pixel:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output per-pixel rasters in {output_folder}: {file_count}")
            # print(geotiff_files)

    # Counts 0.04x0.04 deg outputs
    output_dir_list_aggreg = uu.create_output_dir_name_list(cn.veg_summative_output_dirs, 'annual', cn.LC_first_year,
                                                            cn.global_aggregation_factor, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
                                                            interval_year_diff_list, input_date, False, "_0_04deg_yr")
    output_dir_list_aggreg.sort()
    if is_large_run:
        main_logger.info(f"output_dir_list_aggreg for {stage}:")
        for item in output_dir_list_aggreg:
            main_logger.info(f"  {item}")
    # print(output_dir_list_aggreg)

    if not no_upload and is_large_run:
        for output_folder in output_dir_list_aggreg:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output aggregated rasters in {output_folder}: {file_count}")
            if file_count != len(tile_ids_to_process):
                main_logger.warning(f"WARNING: Output file count in {output_folder} does not match expectations!")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 7: Merge worker and local logs
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create 10x10 deg per-ha and per-pixel output geotifs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-fy', '--first_years_to_process', type=int, help='Number of years to process from raw mega-zarr (for testing)')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=True, help='s3 path for model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    first_years_to_process = args.first_years_to_process
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_log = args.no_log
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri, bounding_box=bounding_box,
         first_variables_to_process=first_variables_to_process, first_years_to_process=first_years_to_process,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description, log_note=log_note)

"""
Creates 10x10 deg per-hectare and per-pixel geotifs from global zarr for numeric model outputs.
Coded to run for all starting carbon densities but can change code to run for more or less variables.
It creates a task list for all datasets and 10x10 deg tiles for the variables and area of interest,
then runs that giant task list in parallel.

Providing a bounding box with -bb or a chunk shapefile limits the 10x10 deg creation
to the 10x10 deg tiles that contain the bounding box or shapefile.
The entire 10x10 deg tile that contains the selected chunks will be processed (not just the parts with the selected chunks).

The chunk stats table argument (xlsx or Parquet) allows the pixel counts in the 10x10 deg tiles to be compared to
the pixel counts in the constituent 1x1 deg tiles to make sure that pixels aren't being lost during 10x10 deg tile
creation.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -bb 23 -4 24 -3 --run_local --no_upload -mt standard -mpd global -fv 1 -ft 1 --input_date YYYYMMDD

Coiled small tests (needs 32 GB because of per-ha and per-pixel outputs):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn starting_carbon_pools
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools -bb 23 -4 24 -3 -fv 2 -ft 2 -mt standard -mpd global -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260129_14_36_42__KEEP.xlsx  --input_date YYYYMMDD

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn starting_carbon_pools_10x10__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools_10x10__Ctrees -bb -80 30 -70 40 -mpd global --input_date 20260629 -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260630_00_52_24.xlsx

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn starting_carbon_pools
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools -mt standard -mpd global -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260129_14_36_42__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --input_date YYYYMMDD

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn starting_carbon_pools
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools -mt standard -mpd global -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260129_14_36_42__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --input_date YYYYMMDD

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn starting_carbon_pools
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools -mt standard -mpd global -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260129_14_36_42__KEEP.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --input_date YYYYMMDD --log_note "Global 10x10 deg creation for starting carbon pools using ESA CCI AGB v6, with starting carbon pool adjustments."

For sensitivity anlysis:
Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn starting_carbon_pools_10x10__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools_10x10__Ctrees -mt ctrees_starting_AGC -bb -60 -20 -50 -10 -mpd test_box -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260630_00_52_24.xlsx

python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn starting_carbon_pools_10x10__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools_10x10__Ctrees -mt ctrees_starting_AGC -bb -80 30 -70 40 -mpd test_box -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260630_00_52_24.xlsx

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn starting_carbon_pools_10x10__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.2_starting_C_outputs_to_10x10deg -cn starting_carbon_pools_10x10__Ctrees -mt ctrees_starting_AGC -mpd global -mcstn starting_carbon_pools_2015_1x1_deg_1x1_chunk_statistics_20260630_00_52_24.xlsx -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --log_note "Global 10x10 deg creation for starting carbon pools using C-Trees AGB."

Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/690a21cd-2ea0-8333-9c7f-7091f8016fb3
"""

import argparse
import pandas as pd
import os
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


def main(cluster_name, input_date, model_type, run_local, no_log, no_upload, model_chunk_stats_table_name, chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_years_to_process=None, first_tiles_to_process=None, model_path_description=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'starting_carbon_pool_aggregation_to_10x10_deg'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Assuming I'm only going to run this on carbon pools for 2015 for now
    year = cn.LC_first_year

    if model_type == cn.alt_AGB:
        biomass_source = "Ctrees"
        zarr_root = cn.starting_C_densities_2015_ctrees_path_mega_zarr
        output_base = f"{cn.full_bucket_prefix}/climate/Ctrees_biomass/{year}/year_2015_derived_carbon_pools/PATTERN/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{cn.ctrees_run_date}/"
        output_dir_list = [cn.agc_2015_ctrees_raw_dir, cn.bgc_2015_ctrees_raw_dir, cn.deadwood_c_2015_ctrees_raw_dir,
                           cn.litter_c_2015_ctrees_raw_dir, cn.non_soil_c_2015_ctrees_raw_dir, cn.agc_2015_ctrees_LC_masked_dir,
                           cn.bgc_2015_ctrees_LC_masked_dir, cn.deadwood_c_2015_ctrees_LC_masked_dir, cn.litter_c_2015_ctrees_LC_masked_dir,
                           cn.non_soil_c_2015_ctrees_LC_masked_dir, cn.starting_C_pools_ctrees_LC_masked_state_dir]
    else:
        biomass_source = "ESA_CCI"
        zarr_root = cn.starting_C_densities_2015_path_mega_zarr
        output_base = f"{cn.full_bucket_prefix}/climate/ESA_CCI_biomass/{cn.esa_AGB_v}/{year}/year_2015_derived_carbon_pools/PATTERN/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{cn.carbon_2015_creation_date}/"
        output_dir_list = [cn.agc_2015_raw_dir, cn.bgc_2015_raw_dir, cn.deadwood_c_2015_raw_dir, cn.litter_c_2015_raw_dir,
                           cn.non_soil_c_2015_raw_dir, cn.agc_2015_LC_masked_dir, cn.bgc_2015_LC_masked_dir, cn.deadwood_c_2015_LC_masked_dir,
                           cn.litter_c_2015_LC_masked_dir, cn.non_soil_c_2015_LC_masked_dir, cn.starting_C_pools_LC_masked_state_dir]

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    if biomass_source == "Ctrees":
        main_logger.info("Starting biomass source: Ctrees 2015 AGB sensitivity analysis")
        main_logger.info(f"Ctrees AGB input pattern: {cn.ctrees_agb_2015_pattern}")
    elif biomass_source == "ESA_CCI":
        main_logger.info("Starting biomass source: ESA CCI 2015 AGB standard run")
        main_logger.info(f"ESA CCI AGB version: {cn.esa_AGB_v}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Year for carbon pools: {year}")
    main_logger.info(f"Input date: {input_date}")
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

    # Outputs to turn into 10x10 tile
    # These variables are added to the mega-zarr
    full_list_of_vars = [cn.agc_raw_dens_pattern, cn.bgc_raw_dens_pattern,
                       cn.deadwood_c_raw_dens_pattern, cn.litter_c_raw_dens_pattern, cn.non_soil_c_raw_dens_pattern,
                       cn.agc_LC_masked_dens_pattern, cn.bgc_LC_masked_dens_pattern,
                       cn.deadwood_c_LC_masked_dens_pattern, cn.litter_c_LC_masked_dens_pattern, cn.non_soil_c_LC_masked_dens_pattern,
                       cn.starting_C_pools_LC_masked_source_flag_pattern]
    # full_list_of_vars = [cn.starting_C_pools_LC_masked_source_flag_pattern]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars
    main_logger.info(f"Variables to create 10x10 deg tiles for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars)})")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to aggregate to 10x10 deg and compare chunk stats for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The zarr path that's being used
    mega_zarr_path = zu.create_zarr_path(zarr_root, chunk_size_pixels, str(year), model_type, cn.veg_model_version_underscore,
                                         model_path_description, input_date, main_logger)
    main_logger.info(f"Aggregating from zarr ({source_zarr_chunk_size} pixel chunks): {mega_zarr_path}")
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

    all_task_count = len(vars_to_process) * len(tile_ids_to_process)

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # For simulating a large run
    if all_task_count > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    main_logger.info(f"There are {all_task_count} tasks to process ({len(vars_to_process)} variables x {len(tile_ids_to_process)} tiles)")

    futures = []

    main_logger.info(f"Starting processing: {uu.timestr()}")

    for var_name in vars_to_process:

        for tile_id in tile_ids_to_process:

            append_start_year_to_var = model_type == cn.alt_AGB #TODO: Not needed after fixing populate_zarr (see Step 5 TODO in 1_create_starting_carbon_pools.py)

            future = client.submit(zu.create_10x10_deg_geotif_from_zarr,
                                   var_name, 0, tile_id, mega_zarr_path, output_base, biomass_source,
                                   model_type, model_path_description, no_upload, True, 0, append_start_year_to_var)
            futures.append(future)

    main_logger.info(f"There are {len(futures)} tiles to aggregate ({len(tile_ids_to_process)} tiles x {len(vars_to_process)} variables)")

    # Results is a list of tuples, where each tuple is the per-ha and per-pixel chunk stats, each of which is a dictionary
    # for a single variable-year-tile
    # e.g., ([{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__carbon_density__AGC__landcover_masked__MgC.tif',
    # 'tile_name': '00N_050W__carbon_density__AGC__landcover_masked__MgC.tif', 'in_out': 'output_layer',
    # 'pattern': 'carbon_density__AGC__landcover_masked__MgC', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data', 'max_value': 'no data',
    # 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}],
    # [{'chunk_id': 'N/A', 'tile_id': '00N_050W', 'layer_name': '00N_050W__carbon_density__BGC__landcover_masked__MgC.tif',
    # 'tile_name': '00N_050W__carbon_density__BGC__landcover_masked__MgC.tif', 'in_out': 'output_layer',
    # 'pattern': 'carbon_density__BGC__landcover_masked__MgC', 'years': 2016, 'min_value': 'no data', 'mean_value': 'no data',
    # 'max_value': 'no data', 'count_value': 7912448, 'sum_value': 'no data', 'data_type': 'no data'}]),
    # ([{'chunk_id': 'N/A', ... 'data_type': 'no data'}])]
    results = client.gather(futures)
    # print(results)

    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)


    ### Step 3: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} worker log compilation", main_logger)


    ### Step 4: Resize cluster down to 1 worker for pixel count comparison, output counts, and log merging since those
    ### only needs a minimal remainder of the cluster, not all the workers.

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
    counts_per_ha_10x10_stats_list = [ha_dict for (ha_group, pixel_group) in results for ha_dict in ha_group]
    counts_per_pixel_10x10_stats_list = [pixel_dict for (ha_group, pixel_group) in results for pixel_dict in pixel_group]
    # print(counts_per_ha_10x10_stats_list)
    # print(counts_per_pixel_10x10_stats_list)

    # Converts the pixel counts from per-ha and per-pixel in the 10x10s into dataframes
    counts_per_ha_10x10_df = pd.DataFrame(counts_per_ha_10x10_stats_list)
    counts_per_pixel_10x10_df = pd.DataFrame(counts_per_pixel_10x10_stats_list)

    # Merges the per-ha pixel counts for the 10x10 tiles against the pixel counts for the 1x1s
    merged_10x10_counts_per_ha_df = model_10x10_counts_df.merge(counts_per_ha_10x10_df, on='tile_name', how='left')
    # print("model_10x10_counts_df:", model_10x10_counts_df)
    print("model_10x10_counts_df:", model_10x10_counts_df['tile_name'].iloc[0])
    # print("counts_per_ha_10x10_df:", counts_per_ha_10x10_df)
    print("counts_per_ha_10x10_df:", counts_per_ha_10x10_df['tile_name'].iloc[0])
    # print("merged_10x10_counts_per_ha_df:", merged_10x10_counts_per_ha_df)

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
    output_dir_list_per_ha = [path.replace("CHUNK_SIZE", str(cn.full_raster_dims)) for path in output_dir_list]
    output_dir_list_per_ha = [path.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning) for path in output_dir_list_per_ha]

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
    output_dir_list_per_pixel = [path.replace("CHUNK_SIZE", str(cn.full_raster_dims)) for path in output_dir_list]
    output_dir_list_per_pixel = [path.replace("PER_HA_OR_PIXEL", cn.C_per_pixel_pixel_meaning) for path in output_dir_list_per_pixel]
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
    output_dir_list_aggreg = [path.replace("CHUNK_SIZE", str(cn.global_aggregation_factor)) for path in output_dir_list]
    output_dir_list_aggreg = [path.replace("PER_HA_OR_PIXEL", cn.C_density_aggreg_pixel_meaning) for path in output_dir_list_aggreg]
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
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard). Not currently using.')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area). Not currently using.')
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

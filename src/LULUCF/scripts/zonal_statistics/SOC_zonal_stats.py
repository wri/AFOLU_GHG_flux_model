"""
Zonal stats for soil organic carbon.

Density in Mg C; loss, gain and net change are in Mg CO2/yr.
For net stock change, negative is SOC gain and positive is SOC loss (same as signs for vegetation).
Gross gain is negative and gross loss is positive (same as signs for vegetation).
Calling gross values gain and loss instead of emissions and removals to differentiate them from vegetation emissions and removals (which are in CO2(e).)

Area to analyze can be specified with a shapefile or a bounding box.
If a shapefile is supplied, all 10x10 deg tiles that intersect the shapefile are analyzed iteratively.
If a bounding box is supplied:
   1. If the bounding box is <6x6 deg, the exact area of the bounding box is analyzed. This is to allow tests in areas smaller than 6x6 deg.
   2. If the bounding box is >6x6 deg, all 10x10 deg tiles that intersect the bounding box are analyzed iteratively.

Model chunk stats are used to determine if each 10x10 deg tile has any pixels in it (per the 1x1_counts_in_10x10 tab).
If no pixels, the tile is skipped to save time.

Requires zarr v3.1.3 on workers, which is set up using a pre-uploaded software package.
Usage of that is triggered with the --zonal_stats flag.
Without --zonal_stats, Coiled will use zarr v3.1.6, which will either not let the zarr be read or fail during compute.
Warnings about mismatches being client and worker/cluster versions for some packages are expected but doesn't seem to be a problem.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -m 32 -cn SOC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.SOC_zonal_stats -cn SOC_zonal_stats -bb 10 49 11 50 -fv 2 -ft 2 -mt standard -mpd global --input_date YYYYMMDD -zd test_box --veg_input_date 20260130 --veg_model_version 1.0.05 --veg_model_type standard --veg_model_path_description global -mcstn KEEP_definitive_runs/SOC_density/v1_0_1__2000_2022_revised_org_soil__20260526/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260526_15_43_37__KEEP.xlsx

Coiled 8-tile test (Central and East Africa):
python -m src.utilities.create_cluster -n 50 -m 32 -cn SOC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.SOC_zonal_stats -cn SOC_zonal_stats -bb 13 -14 44 -3 -fv 3 -ft 3 -mt standard -mpd global --input_date YYYYMMDD -zd Central_Africa_test  -mcstn KEEP_definitive_runs/SOC_density/v1_0_1__2000_2022_revised_org_soil__20260526/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260526_15_43_37__KEEP.xlsx

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 50 -m 32 -cn SOC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.SOC_zonal_stats -cn SOC_zonal_stats -mt standard -mpd global --input_date YYYYMMDD -zd Cerrado_test -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp
-mcstn KEEP_definitive_runs/SOC_density/v1_0_1__2000_2022_revised_org_soil__20260526/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260526_15_43_37__KEEP.xlsx

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 50 -m 32 -cn SOC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.SOC_zonal_stats -cn SOC_zonal_stats -mt standard -mpd global --input_date YYYYMMDD -zd 1884_chunk_test -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp
-mcstn KEEP_definitive_runs/SOC_density/v1_0_1__2000_2022_revised_org_soil__20260526/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260526_15_43_37__KEEP.xlsx

Full run:
python -m src.utilities.create_cluster -n 50 -m 32 -cn SOC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.SOC_zonal_stats -cn SOC_zonal_stats -mt standard -mpd global --input_date YYYYMMDD -zd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --log_note "Zonal stats for vegetation model v1.0.5 (2016-2024)."
-mcstn KEEP_definitive_runs/SOC_density/v1_0_1__2000_2022_revised_org_soil__20260526/soil_carbon_densities_and_changes_1x1_chunk_statistics_20260526_15_43_37__KEEP.xlsx

#TODO Add permafrost as contextual layer
"""

import argparse
import pandas as pd
import os
import gc
from pathlib import Path
import time
from dask.distributed import print
import xarray as xr
import numpy as np
from flox.xarray import xarray_reduce
from flox import ReindexArrayType, ReindexStrategy

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import zonal_stats_utilities as zsu
from src.utilities import terminate_cluster


def main(cluster_name, input_date, model_type, no_upload, zonal_stats_description,
         # veg_input_date, veg_model_version, veg_model_type, veg_model_path_description,
         chunk_shapefile_uri=False, bounding_box=None,
         first_variables_to_process=None, first_tiles_to_process=None, model_path_description=None,
         model_chunk_stats_table_name=None, log_note=None):


    ### Step 1: Preparation

    # Model stage being run
    stage = 'SOC_zonal_statistics'
    model_version = cn.SOC_model_version_underscore
    output_path = cn.SOC_outputs_path

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # If an area smaller than 6x6 deg is given (for testing), the sub_tile_test flag is activated
    # and the analysis extent changes further down
    print(not chunk_shapefile_uri)
    if (not chunk_shapefile_uri) and ((abs(bounding_box[0]-bounding_box[2]) < 6) and (abs(bounding_box[1]-bounding_box[3]) < 6)):
        sub_tile_test = True
    else:
        sub_tile_test = False

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    main_logger.info(f"Stage {stage} started at: {uu.timestr()}")
    main_logger.info(f"SOC model version: {cn.SOC_model_version}")
    main_logger.info(f"SOC model path descriptor: {model_path_description}")
    main_logger.info(f"SOC zonal stats descriptor: {zonal_stats_description}")
    main_logger.info(f"Start year: 2000; end year: {cn.SOC_density_intervals[-1]}")
    main_logger.info(f"Input date: {input_date}")
    main_logger.info(f"no_upload: {no_upload}")
    # main_logger.info(f"Veg model input date (for forest age): {veg_input_date}")
    # main_logger.info(f"Veg model version (for forest age): {veg_model_version}")
    # main_logger.info(f"Veg model type (for forest age): {veg_model_type}")
    # main_logger.info(f"Veg model path description (for forest age): {veg_model_path_description}")
    main_logger.info(f"Running sub-tile test area: {sub_tile_test}")

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

    # Outputs to performs zonal stats on
    full_list_of_vars = [
                        cn.SOC_density_full_extent_pattern, cn.SOC_density_min_soil_extent_pattern,
                        cn.SOC_net_full_extent_pattern, cn.SOC_net_min_soil_extent_pattern,
                        cn.SOC_loss_full_extent_pattern, cn.SOC_loss_min_soil_extent_pattern,
                        cn.SOC_gain_full_extent_pattern, cn.SOC_gain_min_soil_extent_pattern
                        ]

    # Adds units to patterns because variables in zarr have units
    full_list_of_vars_with_units = [
        zu.add_units_year_to_pattern(var_name, 9999)[0]  # Dummy year since we don't need the year to access the datasets in the zarr, just add the units
        for var_name in full_list_of_vars
    ]

    # Limits the processed variables to the supplied number (for testing)
    if first_variables_to_process:
        vars_to_process = full_list_of_vars_with_units[0:first_variables_to_process]
    else:
        vars_to_process = full_list_of_vars_with_units
    main_logger.info(f"Variables to run zonal stats on for: {vars_to_process} ({len(vars_to_process)} out of {len(full_list_of_vars_with_units)})")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to perform zonal stats on: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  #4000x4000

    # The SOC zarr path that's being used
    SOC_zarr_path = zu.create_zarr_path(cn.SOC_path_zarr, source_zarr_chunk_size, input_date, main_logger,
                                        cn.SOC_model_version_underscore, model_type, model_path_description)
    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {SOC_zarr_path}")

    # Creates dataframe of state_node codes and meanings
    state_node_df = zsu.create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, cn.sheet)
    node_codes = np.array(list(state_node_df['land_state']), dtype=np.uint32)
    main_logger.info(f"State nodes are from: {cn.state_node_lookup_table_local} or {cn.state_node_lookup_table_s3}, sheet {cn.sheet}")
    main_logger.info(f"State nodes are: {node_codes}")

    local_zonal_stats_folder = Path(cn.SOC_local_zonal_stats_table_folder)
    local_zonal_stats_folder.mkdir(parents=True, exist_ok=True)

    if model_chunk_stats_table_name:
        main_logger.info(f"Reading local model chunk stats tables: {uu.timestr()}")
        model_chunk_stats_path = os.path.join(cn.local_chunk_stats_path, model_chunk_stats_table_name)

        tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
            "", main_logger, model_chunk_stats_path)
        counts_in_10x10 = tables_to_compare_dict['1x1_counts_in_10x10']
        # print("counts_in_10x10:", counts_in_10x10)
    else:
        counts_in_10x10 = None


    ### Step 2: Prepare input zarrs

    prep_start_time = time.time()

    adm0_xr = xr.open_zarr(cn.adm0_zarr_path, consolidated=False).rename_vars(band_data=cn.adm0_pattern)
    pixel_area_xr = xr.open_zarr(cn.pixel_area_zarr_path, consolidated=False).rename_vars(band_data=cn.pixel_area_zstats_pattern)
    WDPA_xr = xr.open_zarr(cn.WDPA_zarr_path, consolidated=False).rename_vars(band_data=cn.WDPA_pattern)
    cont_eco_xr = xr.open_zarr(cn.cont_eco_zarr_path, consolidated=False).rename_vars(band_data=cn.cont_eco_zstats_pattern)
    landmark_xr = xr.open_zarr(cn.landmark_zarr_path, consolidated=False).rename_vars(band_data=cn.landmark_pattern)
    composite_primary_xr = xr.open_zarr(cn.starting_composite_primary_forest_zarr_path, consolidated=False)  # No rename because it's created by a different process where the variable is named starting_composite_primary_forest
    KBA_xr = xr.open_zarr(cn.KBA_zarr_path, consolidated=False).rename_vars(band_data=cn.KBA_pattern)
    watersheds_xr = xr.open_zarr(cn.watersheds_zarr_path, consolidated=False).rename_vars(band_data=cn.watersheds_pattern)
    drivers_xr = xr.open_zarr(cn.drivers_of_loss_zarr_path, consolidated=False).rename_vars(band_data=cn.drivers_of_loss_pattern)
    # BRA_biomes_xr = xr.open_zarr(cn.BRA_biomes_zarr_path, consolidated=False).rename_vars(band_data=cn.BRA_biomes_pattern)
    # managed_land_CAN_xr = xr.open_zarr(cn.managed_land_CAN_zarr_path, consolidated=False).rename_vars(band_data=cn.managed_land_CAN_pattern)  # Alignment issues below, so not using it.     # Tried in https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69c09184-06dc-8332-a90f-7bf0e803ea16
    # managed_land_USA_xr = xr.open_zarr(cn.managed_land_USA_zarr_path, consolidated=False).rename_vars(band_data=cn.managed_land_USA_pattern)

    # # Need the veg zarr path for forest age
    # # Forest age categorical variable from Claude, session 'Forest age categorization in zonal stats'
    # veg_zarr_path = zu.create_zarr_path(
    #     cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, 'annual',
    #     veg_model_type, cn.veg_model_version_underscore, veg_model_path_description,
    #     veg_input_date, main_logger
    # )
    # veg_ds = xr.open_zarr(veg_zarr_path, consolidated=False)
    # forest_age_xr = veg_ds[cn.forest_age_output_pattern]


    ds = xr.open_zarr(SOC_zarr_path, consolidated=False)
    ds_selected_analysis_vars = ds[vars_to_process]

    main_logger.info(f"Rounding coordinates: {uu.timestr()}")
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])
    ds_selected_analysis_vars = zsu.round_coords(ds_selected_analysis_vars)
    adm0_xr = zsu.round_coords(adm0_xr)
    WDPA_xr = zsu.round_coords(WDPA_xr)
    cont_eco_xr = zsu.round_coords(cont_eco_xr)
    landmark_xr = zsu.round_coords(landmark_xr)
    composite_primary_xr = zsu.round_coords(composite_primary_xr)
    KBA_xr = zsu.round_coords(KBA_xr)
    watersheds_xr = zsu.round_coords(watersheds_xr)
    drivers_xr = zsu.round_coords(drivers_xr)
    # forest_age_xr = zsu.round_coords(forest_age_xr)
    # BRA_biomes_xr = zsu.round_coords(BRA_biomes_xr)
    # managed_land_CAN_xr = zsu.round_coords(managed_land_CAN_xr)
    # managed_land_USA_xr = zsu.round_coords(managed_land_USA_xr)
    # land_state_node = zsu.round_coords(ds[cn.land_state_pattern])

    main_logger.info(f"Cropping: {uu.timestr()}")
    pixel_area_aligned = reference
    adm0_aligned = zsu.safe_crop(adm0_xr, reference)
    WDPA_aligned = zsu.safe_crop(WDPA_xr, reference)
    cont_eco_aligned = zsu.safe_crop(cont_eco_xr, reference)
    landmark_aligned = zsu.safe_crop(landmark_xr, reference)
    composite_primary_aligned = zsu.safe_crop(composite_primary_xr, reference)
    KBA_aligned = zsu.safe_crop(KBA_xr, reference)
    watersheds_aligned = zsu.safe_crop(watersheds_xr, reference)
    drivers_aligned = zsu.safe_crop(drivers_xr, reference)
    # forest_age_aligned = zsu.safe_crop(forest_age_xr, reference)
    # BRA_biomes_aligned = zsu.safe_crop(BRA_biomes_xr, reference)
    # managed_land_CAN_aligned = zsu.safe_crop(managed_land_CAN_xr, reference)
    # managed_land_USA_aligned = zsu.safe_crop(managed_land_USA_xr, reference)
    # land_state_node_aligned = zsu.safe_crop(land_state_node, reference)
    ds_selected_analysis_vars_aligned = zsu.safe_crop(ds_selected_analysis_vars, reference)

    # # Age category contextual layer
    # # NOTE: Not using in this script because age doesn't mean anything without some information about land cover or land state node.
    # # So, not including until there are also contextual layers about land cover and/or land state.
    # # Each year of the SOC density zarr gets a forest age
    # # Per Claude session 'Forest age categorization in zonal stats'
    # # Hardcoded mapping: SOC year coordinate → age zarr year coordinate to use for categorization.
    # # Key is index of SOC zarr; value is index of forest age zarr
    # # First 3 SOC years use the 2016 age; last 2 use the 2020 age.
    # soc_year_to_age_year = {
    #     0: 0,  # 0: Density in 2005 and change for 2005 vs. 2010 -> 0: age in 2016
    #     1: 0,  # 1: Density in 2010 and change for 2010 vs. 2015 -> 0: age in 2016
    #     2: 0,  # 2: Density in 2015 and change for 2015 vs. 2020 -> 0: age in 2016
    #     3: 4,  # 3: Density in 2020 and change for 2020 vs. 2022 -> 4: age in 2020
    #     4: 4,  # 4: Density in 2022 and no change data -> 4: age in 2020
    # }
    #
    # # Pre-categorize the two unique age years needed
    # unique_age_years = set(soc_year_to_age_year.values())
    # categorized_age_by_year = {
    #     age_idx: zsu.categorize_age(forest_age_aligned.isel(year=age_idx, drop=True)).astype(np.uint8)
    #     for age_idx in unique_age_years
    # }
    #
    # # Build 3D (year, y, x) array with year coords matching the SOC zarr
    # soc_years = ds_selected_analysis_vars_aligned.year.values
    # # print("soc_year:", soc_years)
    # age_slices = [categorized_age_by_year[soc_year_to_age_year[int(yr)]] for yr in soc_years]
    # # print("age_slices:", age_slices)
    # forest_age_cat_3d = xr.concat(age_slices, dim=xr.DataArray(soc_years, dims='year', name='year'))
    # # print("forest_age_cat_3d:", forest_age_cat_3d)
    # forest_age_cat_xr = forest_age_cat_3d.rename(cn.forest_age_category_pattern).to_dataset()
    # # print("forest_age_cat_xr:", forest_age_cat_xr)


    main_logger.info(f"Selecting datasets: {uu.timestr()}")
    # List of selected variable names (already aligned and cropped)
    selected_datasets = list(ds_selected_analysis_vars_aligned.data_vars)

    # Expand pixel_area to match shape of flux variables
    pixel_area_expanded = pixel_area_aligned.expand_dims(year=ds_selected_analysis_vars_aligned.year)

    # Use the exact same x/y coordinates for both
    x_coords = reference.coords['x']
    y_coords = reference.coords['y']

    # Replace coords in both sources
    main_logger.info(f"Replacing coordinates: {uu.timestr()}")
    ds_selected_analysis_vars_aligned = ds_selected_analysis_vars_aligned.assign_coords(x=x_coords, y=y_coords)

    # Convert pixel_area from m² to hectares, then adds to the list of layers to analyze (to get area of contextual layers)
    main_logger.info(f"Calculating pixel area: {uu.timestr()}")
    pixel_area_layer = (pixel_area_expanded * cn.m2_to_ha).astype("float32")

    # Multiply each flux var by pixel_area
    main_logger.info(f"Calculating per-pixel values for analysis layers: {uu.timestr()}")
    flux_layers = []
    for var in selected_datasets:
        flux_scaled = (ds_selected_analysis_vars_aligned[var] * pixel_area_layer).astype("float32")
        flux_layers.append(flux_scaled)

    flux_layers.append(pixel_area_layer)

    # Also updates the list of analysis layer names
    selected_datasets.append("pixel_area_ha")

    # Stack into one flux cube: shape (analysis_layer, year, y, x)
    main_logger.info(f"Stacking analysis layers into flux cube: {uu.timestr()}")
    flux_cube = xr.concat(flux_layers, dim="analysis_layer")

    # Set the analysis_layer coordinate names
    flux_cube = flux_cube.assign_coords(
        analysis_layer=("analysis_layer", selected_datasets)
    )
    flux_cube = zsu.round_coords(flux_cube)
    main_logger.info(f"flux_cube: {flux_cube}")

    prep_end_time = time.time()
    main_logger.info(f"  Finished zonal stats prep, took {round(prep_end_time - prep_start_time)} seconds: {uu.timestr()}")


    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    tiles_processed = 0  # The number of tiles actually processed (since some are skipped)

    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} (tile {i+1} of {len(tile_ids_to_process)}): {uu.timestr()}")
        tile_start_time = time.time()

        # Skips if any existing file already contains this tile_id (to not repeat that tile if restarting the zonal stats)
        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6998a64b-e568-8329-8a19-e10423d00669
        existing = set(os.listdir(local_zonal_stats_folder))
        if any(tile_id in fname for fname in existing):
            main_logger.info(f"  Skipping {tile_id}; output already exists")
            continue

        if model_chunk_stats_table_name:
            tile_count = counts_in_10x10.loc[counts_in_10x10["tile_id"] == tile_id, "total_count"].sum()
            main_logger.info(f"  Pixel count across all analysis layers is {tile_count}")
            if tile_count == 0:
                main_logger.info(f"  Skipping {tile_id}; no pixels in it for any analysis layers")
                continue

        # Count of tiles actually processed (not skipped)
        tiles_processed += 1

        # If the bounding box is less than 6x6 deg, the exact bounding box is used (to enable small tests)
        if sub_tile_test == True:
            west, south, east, north = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            main_logger.info("  Running test area")
        else: # Otherwise, the tiles that intersect the bounding box or shapefile are used
            west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

        # Subset the flux cube by x/y coordinates
        main_logger.info(f"  Subsetting: {uu.timestr()}")
        flux_cube_subset = flux_cube.sel(
            x=slice(west, east),
            y=slice(north, south)  # Note: y typically decreases from top to bottom
        )

        adm0_aligned_subset = adm0_aligned.sel(x=slice(west, east), y=slice(north, south))
        WDPA_aligned_subset = WDPA_aligned.sel(x=slice(west, east), y=slice(north, south))
        cont_eco_aligned_subset = cont_eco_aligned.sel(x=slice(west, east), y=slice(north, south))
        landmark_aligned_subset = landmark_aligned.sel(x=slice(west, east), y=slice(north, south))
        composite_primary_aligned_subset = composite_primary_aligned.sel(x=slice(west, east), y=slice(north, south))
        KBA_aligned_subset = KBA_aligned.sel(x=slice(west, east), y=slice(north, south))
        watersheds_aligned_subset = watersheds_aligned.sel(x=slice(west, east), y=slice(north, south))
        drivers_aligned_subset = drivers_aligned.sel(x=slice(west, east), y=slice(north, south))
        # forest_age_cat_subset = forest_age_cat_xr.sel(x=slice(west, east), y=slice(north, south))
        # BRA_biomes_aligned_subset = BRA_biomes_aligned.sel(x=slice(west, east), y=slice(north, south))
        # managed_land_CAN_aligned_subset = managed_land_CAN_aligned.sel(x=slice(west, east), y=slice(north, south))  # Alignment issue below, so not using it
        # managed_land_USA_aligned_subset = managed_land_USA_aligned.sel(x=slice(west, east), y=slice(north, south))
        # land_state_node_aligned_subset = land_state_node_aligned.sel(x=slice(west, east), y=slice(north, south))
        pixel_area_expanded_subset = pixel_area_expanded.sel(x=slice(west, east), y=slice(north, south))

        # Creates xarrays of 0s if contextual layer doesn't extend to the current tile.
        # Don't need to do with land_state_nodes because those should exist everywhere there are model outputs.
        # Also, don't need to do with pixel_area because that should exist everywhere.
        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6995f836-b304-8333-8b92-cf24f95d812f
        if adm0_aligned_subset[cn.adm0_pattern].sizes.get("x", 0) == 0 or adm0_aligned_subset[cn.adm0_pattern].sizes.get("y", 0) == 0:
            adm0_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.adm0_pattern)
            main_logger.info(f"  {cn.adm0_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            adm0_da = adm0_aligned_subset[cn.adm0_pattern]

        if WDPA_aligned_subset[cn.WDPA_pattern].sizes.get("x", 0) == 0 or WDPA_aligned_subset[cn.WDPA_pattern].sizes.get("y", 0) == 0:
            WDPA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.WDPA_pattern)
            main_logger.info(f"  {cn.WDPA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            WDPA_da = WDPA_aligned_subset[cn.WDPA_pattern]

        if cont_eco_aligned_subset[cn.cont_eco_zstats_pattern].sizes.get("x", 0) == 0 or cont_eco_aligned_subset[cn.cont_eco_zstats_pattern].sizes.get("y", 0) == 0:
            cont_eco_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.cont_eco_zstats_pattern)
            main_logger.info(f"  {cn.cont_eco_zstats_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            cont_eco_da = cont_eco_aligned_subset[cn.cont_eco_zstats_pattern]

        if landmark_aligned_subset[cn.landmark_pattern].sizes.get("x", 0) == 0 or landmark_aligned_subset[cn.landmark_pattern].sizes.get("y", 0) == 0:
            landmark_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.landmark_pattern)
            main_logger.info(f"  {cn.landmark_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            landmark_da = landmark_aligned_subset[cn.landmark_pattern]

        if KBA_aligned_subset[cn.KBA_pattern].sizes.get("x", 0) == 0 or KBA_aligned_subset[cn.KBA_pattern].sizes.get("y", 0) == 0:
            KBA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.KBA_pattern)
            main_logger.info(f"  {cn.KBA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            KBA_da = KBA_aligned_subset[cn.KBA_pattern]

        if watersheds_aligned_subset[cn.watersheds_pattern].sizes.get("x", 0) == 0 or watersheds_aligned_subset[cn.watersheds_pattern].sizes.get("y", 0) == 0:
            watersheds_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.watersheds_pattern)
            main_logger.info(f"  {cn.watersheds_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            watersheds_da = watersheds_aligned_subset[cn.watersheds_pattern]

        if drivers_aligned_subset[cn.drivers_of_loss_pattern].sizes.get("x", 0) == 0 or drivers_aligned_subset[cn.drivers_of_loss_pattern].sizes.get("y", 0) == 0:
            drivers_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.drivers_of_loss_pattern)
            main_logger.info(f"  {cn.drivers_of_loss_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            drivers_da = drivers_aligned_subset[cn.drivers_of_loss_pattern]

        # if forest_age_cat_subset[cn.forest_age_category_pattern].sizes.get("x", 0) == 0 or forest_age_cat_subset[cn.forest_age_category_pattern].sizes.get("y", 0) == 0:
        #     forest_age_cat_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.forest_age_category_pattern)
        #     main_logger.info(f"  {cn.forest_age_category_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     forest_age_cat_da = forest_age_cat_subset[cn.forest_age_category_pattern]

        # if BRA_biomes_aligned_subset[cn.BRA_biomes_pattern].sizes.get("x", 0) == 0 or BRA_biomes_aligned_subset[cn.BRA_biomes_pattern].sizes.get("y", 0) == 0:
        #     bra_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.BRA_biomes_pattern)
        #     main_logger.info(f"  {cn.BRA_biomes_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     bra_da = BRA_biomes_aligned_subset[cn.BRA_biomes_pattern]

        # if managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern].sizes.get("x", 0) == 0 or managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern].sizes.get("y", 0) == 0:
        #     managed_land_CAN_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.managed_land_CAN_pattern)
        #     main_logger.info(f"  {cn.managed_land_CAN_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     managed_land_CAN_da = managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern]

        # if managed_land_USA_aligned_subset[cn.managed_land_USA_pattern].sizes.get("x", 0) == 0 or managed_land_USA_aligned_subset[cn.managed_land_USA_pattern].sizes.get("y", 0) == 0:
        #     managed_land_USA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.managed_land_USA_pattern)
        #     main_logger.info(f"  {cn.managed_land_USA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     managed_land_USA_da = managed_land_USA_aligned_subset[cn.managed_land_USA_pattern]

        if (composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern].sizes.get("x", 0) == 0 or
                composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern].sizes.get("y", 0) == 0):
            composite_primary_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.starting_composite_primary_forest_pattern)
            main_logger.info(f"  {cn.starting_composite_primary_forest_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            composite_primary_da = composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern]

        # Turns the composite primary forest zarr (which has chunks of 1x4000x4000) into something without a year dimension at all (4000x4000).
        # That allows it to be used with the other contextual layers, which are also just 4000x4000 (no year dimension).
        # Note: composite primary forest has chunks of 1x4000x4000 because of how it's made; it uses the same function as the zarr for the vegetation model,
        # rather than the script of the other contextual layers.
        if "year" in composite_primary_da.dims:
            composite_primary_da = composite_primary_da.isel(year=0, drop=True)

        # Final alignment
        main_logger.info(f"  Aligning {tile_id}: {uu.timestr()}")
        (flux_cube_subset,
         pixel_area_expanded_subset,
         adm0_da,
         WDPA_da,
         cont_eco_da,
         landmark_da,
         composite_primary_da,
         KBA_da,
         watersheds_da,
         drivers_da,
         # forest_age_cat_da,
         # bra_da,
         # managed_land_CAN_da,
         # managed_land_USA_da,
         # land_state_node_aligned_subset,
         ) = xr.align(
            flux_cube_subset,
            pixel_area_expanded_subset,
            adm0_da,
            WDPA_da,
            cont_eco_da,
            landmark_da,
            composite_primary_da,
            KBA_da,
            watersheds_da,
            drivers_da,
            # forest_age_cat_da,
            # bra_da,
            # managed_land_CAN_da,
            # managed_land_USA_da,
            # land_state_node_aligned_subset,
            join="override"
        )

        main_logger.info(f"  Computing {tile_id}: {uu.timestr()}")
        results = xarray_reduce(
            flux_cube_subset,
            *(
                adm0_da,
                # land_state_node_aligned_subset,
                WDPA_da,
                cont_eco_da,
                landmark_da,
                composite_primary_da,
                KBA_da,
                watersheds_da,
                drivers_da,
                # forest_age_cat_da,
                # bra_da,
                # managed_land_CAN_da,
                # managed_land_USA_da,
                flux_cube_subset["year"]
            ),
            func='sum',
            expected_groups=(
                cn.gadm_adm0_ids,
                # node_codes,
                cn.WDPA_codes,
                cn.cont_eco_codes,
                cn.landmark_codes,
                cn.composite_primary_codes,
                cn.KBA_codes,
                cn.watershed_codes,
                cn.drivers_codes,
                # cn.forest_age_category_codes,
                # cn.BRA_biomes_codes,
                # cn.managed_land_codes,  # For Canada
                # cn.managed_land_codes,  # For USA
                flux_cube_subset.year.values,
            ),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        # Contextual layers to use to merge pixel_area against other analysis layers (to calculate flux/ha)
        contextual_layers = [
            cn.adm0_pattern,
            # cn.land_state_pattern,
            cn.WDPA_pattern,
            cn.cont_eco_zstats_pattern,
            cn.landmark_pattern,
            cn.starting_composite_primary_forest_pattern,
            cn.KBA_pattern,
            cn.watersheds_pattern,
            cn.drivers_of_loss_pattern,
            # cn.forest_age_category_pattern,
            # cn.BRA_biomes_pattern,
            # cn.managed_land_CAN_pattern,
            # cn.managed_land_USA_pattern,
            'year'
        ]

        main_logger.info(f"  Done computing {tile_id}: {uu.timestr()}")
        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)
        df = zsu.create_df(coord_dict, state_node_df, contextual_layers, tile_id, 'SOC', main_logger)
        main_logger.info(f"  Rows in {tile_id} dataframe: {len(df.index)}: {uu.timestr()}")

        main_logger.info(f"  Saving {tile_id} output table: {uu.timestr()}")
        tile_df_name = f'SOC_zonal_stats_{tile_id}_v{cn.SOC_model_version_underscore}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}'
        df.to_parquet(f"{local_zonal_stats_folder}/{tile_df_name}.parquet")

        # Clean up at end of tile
        # del results, coord_dict, df, flux_cube_subset
        gc.collect()

        tile_end_time = time.time()
        main_logger.info(f"  Done with {tile_id}, took {round(tile_end_time) - round(tile_start_time)} seconds: {uu.timestr()}")


    ### After all tiles completed

    all_tiles_end_time = time.time()
    main_logger.info(f"Finished tile analyses, took {round(all_tiles_end_time - prep_start_time)} seconds: {uu.timestr()}")
    if tiles_processed > 0:
        average_time = (all_tiles_end_time - prep_start_time)/tiles_processed
        main_logger.info(f"Average time per tile (excluding skipped tiles): {round(average_time)} seconds (for {tiles_processed} tiles)")
    else:
        main_logger.info("No tiles processed")

    # # Terminates cluster because all further processing is done locally
    # terminate_cluster.terminate_cluster(cluster_name)

    # Collects all tile parquet files
    parquet_files = sorted(
        str(local_zonal_stats_folder / f)
        for f in os.listdir(local_zonal_stats_folder)
        if f.endswith(".parquet") and "zonal_stats_" in f
    )

    if not parquet_files:
        main_logger.info("No tile parquet files found.")
        return

    # List of dataframes from each tile, to be combined
    df_list = []

    # Rows in all output tables
    total_rows = 0

    # Iterates through all tiles to sum rows
    main_logger.info(f"Counting rows in all output tables: {uu.timestr()}")
    for parquet_output in parquet_files:
        main_logger.info(f"Getting row count in {parquet_output}: {uu.timestr()}")
        df = pd.read_parquet(parquet_output)
        total_rows += len(df.index)

    main_logger.info(f"Total rows: {total_rows}")

    # Only tries to combine tables into one table if less than specified number of rows
    if total_rows > 30_000_000:
        main_logger.info("Too many rows to aggregate into global df. Skipping.")
    else:
        main_logger.info(f"Combining all parquets: {uu.timestr()}")
        for parquet_output in parquet_files:
            main_logger.info(f"Reading {parquet_output}: {uu.timestr()}")
            df = pd.read_parquet(parquet_output)
            df_list.append(df)
            total_rows += len(df.index)

        # Combines all the tile-level dfs in the list into a single df
        main_logger.info(f"Combining dataframes: {uu.timestr()}")
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)

        main_logger.info(f"Rows in combined dataframe: {len(combined_df.index)}")
        if len(combined_df.index) != total_rows:
            main_logger.warning("Sum of row count from individual tables and row count in combined table do not match!")
        main_logger.info(combined_df.head())

        combined_df_name = f'SOC_model_zonal_stats_v{model_version}_{time.strftime('%Y%m%d_%H_%M_%S')}'
        combined_df.to_parquet(f"{local_zonal_stats_folder}/{combined_df_name}.parquet")
        if len(combined_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
            combined_df.to_csv(f"{local_zonal_stats_folder}/{combined_df_name}.csv", index=False)

        # Converts combined df from long to wide
        combined_wide_df = zsu.create_wide_df(combined_df, main_logger)

        combined_wide_df_name = f'SOC_model_zonal_stats_v{model_version}_wide_{time.strftime('%Y%m%d_%H_%M_%S')}'
        combined_wide_df.to_parquet(f"{local_zonal_stats_folder}/{combined_wide_df_name}.parquet")
        if len(combined_wide_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
            combined_wide_df.to_csv(f"{local_zonal_stats_folder}/{combined_wide_df_name}.csv", index=False)

    # Converts each parquet to csv
    main_logger.info(f"Converting parquet files to csvs: {uu.timestr()}")
    for parquet_output in parquet_files:
        main_logger.info(f"Converting {parquet_output} to csv: {uu.timestr()}")
        df = pd.read_parquet(parquet_output)
        csv_output = parquet_output.replace('parquet', 'csv')
        df.to_csv(csv_output, index=False)

    # Uploads outputs to s3 if the run is large enough
    zsu.upload_zstats_to_s3(stage, local_zonal_stats_folder, output_path, main_logger,
                        model_path_description, model_type, model_version, tiles_processed)

    end_time = time.time()
    main_logger.info(f"Finished zonal stats, took {round(end_time - prep_start_time)} seconds: {uu.timestr()}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run zonal statistics on global zarrs against contextual layers")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-ft', '--first_tiles_to_process', type=int, help='Number of tiles to process (for testing)')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-zd', '--zonal_stats_description', help='Description of zonal stats run (e.g., global, Brazil, test area).')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=False, help='local path for model chunk stats to check if tile had any pixels in it, and skip if empty')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    # # Used to specify the forest age contextual layer
    # parser.add_argument('--veg_input_date', help='Date of vegetation model run, YYYYMMDD')
    # parser.add_argument('--veg_model_version', help='Version of vegetation model')
    # parser.add_argument('--veg_model_type', help='Vegetation model type (e.g., standard)')
    # parser.add_argument('--veg_model_path_description', help='Vegetation model path description (e.g., global)')

    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_tiles_to_process = args.first_tiles_to_process
    first_variables_to_process = args.first_variables_to_process
    model_type = args.model_type
    model_path_description = args.model_path_description
    zonal_stats_description = args.zonal_stats_description
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    log_note = args.log_note

    # veg_input_date = args.veg_input_date
    # veg_model_version = args.veg_model_version
    # veg_model_type = args.veg_model_type
    # veg_model_path_description = args.veg_model_path_description

    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, input_date, model_type, no_upload, zonal_stats_description,
         # veg_input_date, veg_model_version, veg_model_type, veg_model_path_description,
         chunk_shapefile_uri, bounding_box=bounding_box, first_variables_to_process=first_variables_to_process,
         first_tiles_to_process=first_tiles_to_process, model_path_description=model_path_description,
         model_chunk_stats_table_name=model_chunk_stats_table_name, log_note=log_note)

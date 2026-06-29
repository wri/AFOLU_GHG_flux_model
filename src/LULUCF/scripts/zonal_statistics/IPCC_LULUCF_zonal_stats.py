"""
Purpose: Calculates zonal statistics for combined LULUCF flux outputs using IPCC land-use classifications and other contextual datasets. 
The script reads vegetation, LULUCF, and IPCC land-use zarrs, converts per-hectare fluxes to per-pixel totals, and summarizes emissions, removals, 
net flux, and area by IPCC land-use class, transition, summary, land state, administrative unit, and other contextual layers for each 10×10° tile. 
Tile-level outputs are written as parquet tables and optionally combined into global long- and wide-format zonal statistics tables.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

10x10:
python -m src.utilities.create_cluster -n 10 -m 128 -cn IPCC_LULUCF_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_LULUCF_zonal_stats -cn IPCC_LULUCF_zonal_stats -bb 100 10 110 20 -vid 20260130 -tid 20260614 -lid 20260617 -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

Central and East Africa (8, 10x10 tiles):
python -m src.utilities.create_cluster -n 50 -m 32 -cn IPCC_LULUCF_zonal_stats__Central_Africa --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_LULUCF_zonal_stats -cn IPCC_LULUCF_zonal_stats__Central_Africa -bb 13 -14 44 -3 -vid 20260130 -tid 20260614 -lid 20260617 -zd Central_Africa_test

Global run:
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_LULUCF_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_LULUCF_zonal_stats -cn IPCC_LULUCF_zonal_stats -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -vid 20260130 -tid 20260614 -lid 20260617 -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."


global_south
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_LULUCF_zonal_stats_global_south --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_LULUCF_zonal_stats -cn IPCC_LULUCF_zonal_stats_global_south --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_south.txt -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -vid 20260130 -tid 20260614 -lid 20260617 -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

global_north
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_LULUCF_zonal_stats_global_north --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_LULUCF_zonal_stats -cn IPCC_LULUCF_zonal_stats_global_north --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_north.txt -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -vid 20260130 -tid 20260614 -lid 20260617 -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

Notes:
    - Took 5 minutes to run for 10x10 degree area (00N_110E) with 25 workers (17 credits, $1).
        Analysis layers = LULUCF emissions, LULUCF removals, and LULUCF net flux, vegetation net flux, pixel area
        Contextual layers = IPCC class, node, change and summary, land state node, continent/ecozone, driver, primary/IFL
    - Toook 19 minutes to run for Central Africa (8 tiles) with 50 workers (128 credits, $7)

"""

import argparse
import gc
import os
import time
from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from dask.distributed import print
from flox.xarray import xarray_reduce
from flox import ReindexArrayType, ReindexStrategy

from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import zonal_stats_utilities as zsu
from src.utilities import terminate_cluster

def read_tile_ids_file(tile_ids_file):
    with open(tile_ids_file, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def main(cluster_name, lulucf_input_date, veg_input_date, lu_input_date, model_type="standard", lulucf_model_path_description="global",
         veg_model_path_description="global", lu_model_path_description="global", zonal_stats_description="global",
         chunk_shapefile_uri=False, bounding_box=None, first_variables_to_process=None, first_tiles_to_process=None,
         model_chunk_stats_table_name=None, log_note=None, tile_ids_file=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = "IPCC_land_use_zonal_statistics"
    veg_model_version = cn.veg_model_version_underscore
    lulucf_model_version = cn.LULUCF_model_version_underscore
    lu_model_version = cn.IPCC_LU_version_underscore

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # If an area smaller than 6x6 deg is given (for testing), the sub_tile_test flag is activated
    # and the analysis extent changes further down
    if (not chunk_shapefile_uri) and ((abs(bounding_box[0] - bounding_box[2]) < 6) and (abs(bounding_box[1] - bounding_box[3]) < 6)):
        sub_tile_test = True
    else:
        sub_tile_test = False

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    main_logger.info(f"Stage {stage} started at: {uu.timestr()}")
    main_logger.info(f"Zonal stats description: {zonal_stats_description}")
    main_logger.info(f"Vegetation version: {cn.veg_model_version}")
    main_logger.info(f"Vegetation model path description: {veg_model_path_description}")
    main_logger.info(f"Vegetation input date: {veg_input_date}")
    main_logger.info(f"LULUCF version: {cn.LULUCF_model_version}")
    main_logger.info(f"LULUCF model path description: {lulucf_model_path_description}")
    main_logger.info(f"LULUCF input date: {lulucf_input_date}")
    main_logger.info(f"IPCC land use model version: {cn.IPCC_LU_version}")
    main_logger.info(f"IPCC land use path description: {lu_model_path_description}")
    main_logger.info(f"IPCC land use input date: {lu_input_date}")
    main_logger.info(f"Running sub-tile test area: {sub_tile_test}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1  # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    # Gets a list of unique tile_ids from the chunk list
    tile_ids = []
    for chunk in chunk_list:
        tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])  # tile_id in YYN/S_XXXE/W
        tile_ids.append(tile_id)

    unique_tile_ids = sorted(list(set(tile_ids)))

    if tile_ids_file:
        tile_ids_to_keep = read_tile_ids_file(tile_ids_file)

        original_count = len(unique_tile_ids)

        unique_tile_ids = [
            tile_id
            for tile_id in unique_tile_ids
            if tile_id in tile_ids_to_keep
        ]

        main_logger.info(
            f"Filtered tile list from {original_count} to {len(unique_tile_ids)} using {tile_ids_file}"
        )

    # Outputs to performs zonal stats on
    lulucf_vars = [
        f"LULUCF_{cn.gross_emis_all_C_pools_all_gases_pattern}",
        f"LULUCF_{cn.gross_removals_all_C_pools_pattern}",
        f"LULUCF_{cn.net_flux_all_C_pools_all_gases_pattern}",
    ]

    veg_vars = [
        # cn.gross_emis_all_C_pools_all_gases_pattern,
        # cn.gross_removals_all_C_pools_pattern,
        cn.net_flux_all_C_pools_all_gases_pattern,
    ]

    full_list_of_vars = lulucf_vars + veg_vars

    full_list_of_vars_with_units = [
        zu.add_units_year_to_pattern(var_name, 9999)[0]
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
    source_zarr_chunk_size = cn.chunk_dims  # 4000x4000

    # Vegetation zarr
    veg_zarr_path = zu.create_zarr_path( cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, "annual", model_type,
                                         veg_model_version, veg_model_path_description, veg_input_date, main_logger)

    main_logger.info(f"Vegetation zarr path: {veg_zarr_path}")


    # LULUCF zarr
    lulucf_zarr_path = (
        f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/"
        f"outputs_LULUCF_totals/"
        f"LULUCF_version_{lulucf_model_version}_standard__{lulucf_model_path_description}"
        f"__veg_v{veg_model_version}"
        f"__org_soil_v{cn.organic_soil_model_version_underscore}"
        f"__min_soil_v{cn.SOC_model_version_underscore}"
        f"/zarr/annual_intervals/{source_zarr_chunk_size}_pixels/{lulucf_input_date}/LULUCF_annual.zarr"
    )

    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {lulucf_zarr_path}")

    lu_zarr_path = zu.create_zarr_path(cn.IPCC_outputs_path_mega_zarr, source_zarr_chunk_size, "annual", model_type,
                                       lu_model_version, lu_model_path_description, lu_input_date, main_logger)

    # Creates dataframe of state_node codes and meanings
    state_node_df = zsu.create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, cn.sheet)
    node_codes = [0] + np.array(list(state_node_df['land_state']), dtype=np.uint32)
    # main_logger.info(f"State nodes are from: {cn.state_node_lookup_table_local} or {cn.state_node_lookup_table_s3}, sheet {cn.sheet}")
    # main_logger.info(f"State nodes are: {node_codes}")

    local_zonal_stats_folder = Path(cn.land_use_zonal_stats_table_folder)
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
    # WDPA_xr = xr.open_zarr(cn.WDPA_zarr_path, consolidated=False).rename_vars(band_data=cn.WDPA_pattern)
    cont_eco_xr = xr.open_zarr(cn.cont_eco_zarr_path, consolidated=False).rename_vars(band_data=cn.cont_eco_zstats_pattern)
    # landmark_xr = xr.open_zarr(cn.landmark_zarr_path, consolidated=False).rename_vars(band_data=cn.landmark_pattern)
    composite_primary_xr = xr.open_zarr(cn.starting_composite_primary_forest_zarr_path,consolidated=False)  # No rename because it's created by a different process where the variable is named starting_composite_primary_forest
    # KBA_xr = xr.open_zarr(cn.KBA_zarr_path, consolidated=False).rename_vars(band_data=cn.KBA_pattern)
    # watersheds_xr = xr.open_zarr(cn.watersheds_zarr_path, consolidated=False).rename_vars(band_data=cn.watersheds_pattern)
    drivers_xr = xr.open_zarr(cn.drivers_of_loss_zarr_path, consolidated=False).rename_vars(band_data=cn.drivers_of_loss_pattern)
    # # BRA_biomes_xr = xr.open_zarr(cn.BRA_biomes_zarr_path, consolidated=False).rename_vars(band_data=cn.BRA_biomes_pattern)
    # # managed_land_CAN_xr = xr.open_zarr(cn.managed_land_CAN_zarr_path, consolidated=False).rename_vars(band_data=cn.managed_land_CAN_pattern)
    # # managed_land_USA_xr = xr.open_zarr(cn.managed_land_USA_zarr_path, consolidated=False).rename_vars(band_data=cn.managed_land_USA_pattern)

    # Open LULUCF, vegetation, and land use zarrs
    lulucf_ds = xr.open_zarr(lulucf_zarr_path, consolidated=False)
    veg_ds = xr.open_zarr(veg_zarr_path, consolidated=False)
    lu_ds = xr.open_zarr(lu_zarr_path, consolidated=False)

    lulucf_vars_to_process = [v for v in vars_to_process if v in lulucf_ds.data_vars]
    veg_vars_to_process = [v for v in vars_to_process if v in veg_ds.data_vars]

    main_logger.info(f"LULUCF variables to process: {lulucf_vars_to_process}")
    main_logger.info(f"Vegetation variables to process: {veg_vars_to_process}")

    lulucf_selected = lulucf_ds[lulucf_vars_to_process]
    veg_selected = veg_ds[veg_vars_to_process]

    ds_selected_analysis_vars = xr.merge([lulucf_selected, veg_selected], compat="override")

    # Round and align
    main_logger.info(f"Rounding coordinates: {uu.timestr()}")

    adm0_xr = zsu.round_coords(adm0_xr)
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])
    # WDPA_xr = zsu.round_coords(WDPA_xr)
    cont_eco_xr = zsu.round_coords(cont_eco_xr)
    # landmark_xr = zsu.round_coords(landmark_xr)
    composite_primary_xr = zsu.round_coords(composite_primary_xr)
    # KBA_xr = zsu.round_coords(KBA_xr)
    # watersheds_xr = zsu.round_coords(watersheds_xr)
    drivers_xr = zsu.round_coords(drivers_xr)
    # # BRA_biomes_xr = zsu.round_coords(BRA_biomes_xr)
    # # managed_land_CAN_xr = zsu.round_coords(managed_land_CAN_xr)
    # # managed_land_USA_xr = zsu.round_coords(managed_land_USA_xr)
    land_state_node = zsu.round_coords(veg_ds[cn.land_state_pattern])
    # forest_age = zsu.round_coords(ds[cn.forest_age_output_pattern])
    lu_ds = zsu.round_coords(lu_ds)
    ds_selected_analysis_vars = zsu.round_coords(ds_selected_analysis_vars)


    main_logger.info(f"Cropping: {uu.timestr()}")
    adm0_aligned = zsu.safe_crop(adm0_xr, reference)
    pixel_area_aligned = reference
    # WDPA_aligned = zsu.safe_crop(WDPA_xr, reference)
    cont_eco_aligned = zsu.safe_crop(cont_eco_xr, reference)
    # landmark_aligned = zsu.safe_crop(landmark_xr, reference)
    composite_primary_aligned = zsu.safe_crop(composite_primary_xr, reference)
    # KBA_aligned = zsu.safe_crop(KBA_xr, reference)
    # watersheds_aligned = zsu.safe_crop(watersheds_xr, reference)
    drivers_aligned = zsu.safe_crop(drivers_xr, reference)
    # # BRA_biomes_aligned = zsu.safe_crop(BRA_biomes_xr, reference)
    # # managed_land_CAN_aligned = zsu.safe_crop(managed_land_CAN_xr, reference)
    # # managed_land_USA_aligned = zsu.safe_crop(managed_land_USA_xr, reference)
    land_state_node_aligned = zsu.safe_crop(land_state_node, reference)
    # forest_age_aligned = zsu.safe_crop(forest_age, reference)
    lu_ds = zsu.safe_crop(lu_ds, reference)
    ds_selected_analysis_vars_aligned = zsu.safe_crop(ds_selected_analysis_vars, reference)

    # Categorizes age into specified categories
    # forest_age_cat_xr = zsu.categorize_age(forest_age_aligned).astype(np.uint8).rename(cn.forest_age_category_pattern).to_dataset()

    # List of selected variable names (already aligned and cropped)
    main_logger.info(f"Selecting datasets: {uu.timestr()}")
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

    # Multiply each flux var by pixel_area and create flux-specific area layers
    main_logger.info(f"Calculating per-pixel values for analysis layers: {uu.timestr()}")

    flux_layers = []
    analysis_layer_names = []

    for var in selected_datasets:
        flux_scaled = (ds_selected_analysis_vars_aligned[var] * pixel_area_layer).astype("float32")
        flux_layers.append(flux_scaled)
        analysis_layer_names.append(var)

    # Total IPCC-context area
    flux_layers.append(pixel_area_layer)
    analysis_layer_names.append("pixel_area_ha")

    selected_datasets = analysis_layer_names

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

    # IPCC land use contextusl layers aligned to LULUCF analysis years.
    # LULUCF flux outputs have 9 interval end years (2016-2024)
    # IPCC outputs have 10 annual positions (2015-2024)
        # IPCC class/node: positions 0-9 are annual outputs
        # IPCC change: position 0 is empty, positions 1-9 are change interval end years
        # IPCC summary: position 0 is static 2015_2024 summary, positions 1-9 are empty
    main_logger.info(f"Selecting IPCC land use contextual layers: {uu.timestr()}")
    analysis_years = ds_selected_analysis_vars_aligned.year

    # IPCC contextual layers aligned to LULUCF flux years, 2016-2024
    ipcc_class = lu_ds[cn.IPCC_class_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_node = lu_ds[cn.IPCC_node_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_change = lu_ds[cn.IPCC_change_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_summary = (lu_ds[cn.IPCC_summary_pattern].isel(year=0, drop=True).expand_dims(year=analysis_years).assign_coords(year=analysis_years))


    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    tiles_processed = 0  # The number of tiles actually processed (since some are skipped)

    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} ({i + 1}/{len(tile_ids_to_process)}): {uu.timestr()}")
        tile_start_time = time.time()

        # Skips if any existing file already contains this tile_id (to not repeat that tile if restarting the zonal stats)
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
        # Otherwise, the tiles that intersect the bounding box or shapefile are used
        else:
            west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

        # Subset the flux cube by x/y coordinates
        main_logger.info(f"  Subsetting: {uu.timestr()}")
        flux_cube_subset = flux_cube.sel(
            x=slice(west, east),
            y=slice(north, south)  # Note: y typically decreases from top to bottom
        )

        adm0_aligned_subset = adm0_aligned.sel(x=slice(west, east), y=slice(north, south))
        pixel_area_expanded_subset = pixel_area_expanded.sel(x=slice(west, east), y=slice(north, south))
        # WDPA_aligned_subset = WDPA_aligned.sel(x=slice(west, east), y=slice(north, south))
        cont_eco_aligned_subset = cont_eco_aligned.sel(x=slice(west, east), y=slice(north, south))
        # landmark_aligned_subset = landmark_aligned.sel(x=slice(west, east), y=slice(north, south))
        composite_primary_aligned_subset = composite_primary_aligned.sel(x=slice(west, east), y=slice(north, south))
        # KBA_aligned_subset = KBA_aligned.sel(x=slice(west, east), y=slice(north, south))
        # watersheds_aligned_subset = watersheds_aligned.sel(x=slice(west, east), y=slice(north, south))
        drivers_aligned_subset = drivers_aligned.sel(x=slice(west, east), y=slice(north, south))
        # # BRA_biomes_aligned_subset = BRA_biomes_aligned.sel(x=slice(west, east), y=slice(north, south))
        # # managed_land_CAN_aligned_subset = managed_land_CAN_aligned.sel(x=slice(west, east), y=slice(north, south))
        # # managed_land_USA_aligned_subset = managed_land_USA_aligned.sel(x=slice(west, east), y=slice(north, south))
        land_state_node_aligned_subset = land_state_node_aligned.sel(x=slice(west, east), y=slice(north, south))
        # forest_age_cat_subset = forest_age_cat_xr.sel(x=slice(west, east), y=slice(north, south))
        ipcc_class_subset = ipcc_class.sel(x=slice(west, east), y=slice(north, south))
        ipcc_node_subset = ipcc_node.sel(x=slice(west, east), y=slice(north, south))
        ipcc_change_subset = ipcc_change.sel(x=slice(west, east), y=slice(north, south))
        ipcc_summary_subset = ipcc_summary.sel(x=slice(west, east), y=slice(north, south))

        # Creates xarrays of 0s if contextual layer doesn't extend to the current tile.
        # Don't need to do with land_state_nodes and pixel_area because those should exist everywhere there are model outputs.
        if adm0_aligned_subset[cn.adm0_pattern].sizes.get("x", 0) == 0 or adm0_aligned_subset[cn.adm0_pattern].sizes.get("y", 0) == 0:
            adm0_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.adm0_pattern)
            main_logger.info(f"  {cn.adm0_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            adm0_da = adm0_aligned_subset[cn.adm0_pattern]

        # if WDPA_aligned_subset[cn.WDPA_pattern].sizes.get("x", 0) == 0 or WDPA_aligned_subset[
        #     cn.WDPA_pattern].sizes.get("y", 0) == 0:
        #     WDPA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.WDPA_pattern)
        #     main_logger.info(f"  {cn.WDPA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     WDPA_da = WDPA_aligned_subset[cn.WDPA_pattern]

        if cont_eco_aligned_subset[cn.cont_eco_zstats_pattern].sizes.get("x", 0) == 0 or cont_eco_aligned_subset[
            cn.cont_eco_zstats_pattern].sizes.get("y", 0) == 0:
            cont_eco_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.cont_eco_zstats_pattern)
            main_logger.info(f"  {cn.cont_eco_zstats_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            cont_eco_da = cont_eco_aligned_subset[cn.cont_eco_zstats_pattern]

        # if landmark_aligned_subset[cn.landmark_pattern].sizes.get("x", 0) == 0 or landmark_aligned_subset[
        #     cn.landmark_pattern].sizes.get("y", 0) == 0:
        #     landmark_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.landmark_pattern)
        #     main_logger.info(f"  {cn.landmark_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     landmark_da = landmark_aligned_subset[cn.landmark_pattern]
        #
        # if KBA_aligned_subset[cn.KBA_pattern].sizes.get("x", 0) == 0 or KBA_aligned_subset[cn.KBA_pattern].sizes.get(
        #         "y", 0) == 0:
        #     KBA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.KBA_pattern)
        #     main_logger.info(f"  {cn.KBA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     KBA_da = KBA_aligned_subset[cn.KBA_pattern]
        #
        # if watersheds_aligned_subset[cn.watersheds_pattern].sizes.get("x", 0) == 0 or watersheds_aligned_subset[
        #     cn.watersheds_pattern].sizes.get("y", 0) == 0:
        #     watersheds_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
        #         cn.watersheds_pattern)
        #     main_logger.info(f"  {cn.watersheds_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     watersheds_da = watersheds_aligned_subset[cn.watersheds_pattern]

        if drivers_aligned_subset[cn.drivers_of_loss_pattern].sizes.get("x", 0) == 0 or drivers_aligned_subset[
            cn.drivers_of_loss_pattern].sizes.get("y", 0) == 0:
            drivers_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.drivers_of_loss_pattern)
            main_logger.info(f"  {cn.drivers_of_loss_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            drivers_da = drivers_aligned_subset[cn.drivers_of_loss_pattern]

        # if forest_age_cat_subset[cn.forest_age_category_pattern].sizes.get("x", 0) == 0 or forest_age_cat_subset[
        #     cn.forest_age_category_pattern].sizes.get("y", 0) == 0:
        #     forest_age_cat_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
        #         cn.forest_age_category_pattern)
        #     main_logger.info(f"  {cn.forest_age_category_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # else:
        #     forest_age_cat_da = forest_age_cat_subset[cn.forest_age_category_pattern]
        #
        # # if BRA_biomes_aligned_subset[cn.BRA_biomes_pattern].sizes.get("x", 0) == 0 or BRA_biomes_aligned_subset[
        # #     cn.BRA_biomes_pattern].sizes.get("y", 0) == 0:
        # #     bra_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.BRA_biomes_pattern)
        # #     main_logger.info(f"  {cn.BRA_biomes_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # # else:
        # #     bra_da = BRA_biomes_aligned_subset[cn.BRA_biomes_pattern]
        # #
        # # if managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern].sizes.get("x", 0) == 0 or managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern].sizes.get("y", 0) == 0:
        # #     managed_land_CAN_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.managed_land_CAN_pattern)
        # #     main_logger.info(f"  {cn.managed_land_CAN_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # # else:
        # #     managed_land_CAN_da = managed_land_CAN_aligned_subset[cn.managed_land_CAN_pattern]
        # #
        # # if managed_land_USA_aligned_subset[cn.managed_land_USA_pattern].sizes.get("x", 0) == 0 or managed_land_USA_aligned_subset[cn.managed_land_USA_pattern].sizes.get("y", 0) == 0:
        # #     managed_land_USA_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(cn.managed_land_USA_pattern)
        # #     main_logger.info(f"  {cn.managed_land_USA_pattern} not in {tile_id}. Creating xarray of all 0s.")
        # # else:
        # #     managed_land_USA_da = managed_land_USA_aligned_subset[cn.managed_land_USA_pattern]

        if (composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern].sizes.get("x", 0) == 0 or
                composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern].sizes.get("y", 0) == 0):
            composite_primary_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.starting_composite_primary_forest_pattern)
            main_logger.info(
                f"  {cn.starting_composite_primary_forest_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            composite_primary_da = composite_primary_aligned_subset[cn.starting_composite_primary_forest_pattern]

        if ipcc_class_subset.sizes.get("x", 0) == 0 or ipcc_class_subset.sizes.get("y", 0) == 0:
            ipcc_class_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.IPCC_class_pattern)
        else:
            ipcc_class_da = ipcc_class_subset.rename(cn.IPCC_class_pattern)

        if ipcc_node_subset.sizes.get("x", 0) == 0 or ipcc_node_subset.sizes.get("y", 0) == 0:
            ipcc_node_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.IPCC_node_pattern)
        else:
            ipcc_node_da = ipcc_node_subset.rename(cn.IPCC_node_pattern)

        if ipcc_change_subset.sizes.get("x", 0) == 0 or ipcc_change_subset.sizes.get("y", 0) == 0:
            ipcc_change_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.IPCC_change_pattern)
        else:
            ipcc_change_da = ipcc_change_subset.rename(cn.IPCC_change_pattern)

        if ipcc_summary_subset.sizes.get("x", 0) == 0 or ipcc_summary_subset.sizes.get("y", 0) == 0:
            ipcc_summary_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.IPCC_summary_pattern)
        else:
            ipcc_summary_da = ipcc_summary_subset.rename(cn.IPCC_summary_pattern)

        # Turns the composite primary forest zarr (which has chunks of 1x4000x4000) into something without a year dimension at all (4000x4000).
        # That allows it to be used with the other contextual layers, which are also just 4000x4000 (no year dimension).
        # Note: composite primary forest has chunks of 1x4000x4000 because of how it's made; it uses the same function as the zarr for the LULUCF model,
        # rather than the script of the other contextual layers.
        if "year" in composite_primary_da.dims:
            composite_primary_da = composite_primary_da.isel(year=0, drop=True)

        # Final alignment
        main_logger.info(f"  Aligning {tile_id}: {uu.timestr()}")
        (flux_cube_subset,
         pixel_area_expanded_subset,
         adm0_da,
         # WDPA_da,
         cont_eco_da,
         # landmark_da,
         composite_primary_da,
         # KBA_da,
         # watersheds_da,
         drivers_da,
         # forest_age_cat_da,
         # # bra_da,
         # # managed_land_CAN_da,
         # # managed_land_USA_da,
         land_state_node_aligned_subset,
         ipcc_class_da,
         ipcc_node_da,
         ipcc_change_da,
         ipcc_summary_da,
         ) = xr.align(
            flux_cube_subset,
            pixel_area_expanded_subset,
            adm0_da,
            # WDPA_da,
            cont_eco_da,
            # landmark_da,
            composite_primary_da,
            # KBA_da,
            # watersheds_da,
            drivers_da,
            # forest_age_cat_da,
            # # bra_da,
            # # managed_land_CAN_da,
            # # managed_land_USA_da,
            land_state_node_aligned_subset,
            ipcc_class_da,
            ipcc_node_da,
            ipcc_change_da,
            ipcc_summary_da,
            join="override"
        )

        main_logger.info(f"  Computing {tile_id}: {uu.timestr()}")
        results = xarray_reduce(
            flux_cube_subset,
            *(
                adm0_da,
                land_state_node_aligned_subset,
                # WDPA_da,
                cont_eco_da,
                # landmark_da,
                composite_primary_da,
                # KBA_da,
                # watersheds_da,
                drivers_da,
                # forest_age_cat_da,
                # # bra_da,
                # # managed_land_CAN_da,
                # # managed_land_USA_da,
                ipcc_class_da,
                ipcc_node_da,
                ipcc_change_da,
                ipcc_summary_da,
                flux_cube_subset["year"]
            ),
            func='sum',
            expected_groups=(
                cn.gadm_adm0_ids,
                node_codes,
                # cn.WDPA_codes,
                cn.cont_eco_codes,
                # cn.landmark_codes,
                cn.composite_primary_codes,
                # cn.KBA_codes,
                # cn.watershed_codes,
                cn.drivers_codes,
                # cn.forest_age_category_codes,
                # # cn.BRA_biomes_codes,
                # # cn.managed_land_codes,  # For Canada
                # # cn.managed_land_codes,  # For USA
                cn.ipcc_class_codes,
                cn.ipcc_node_codes,
                cn.ipcc_change_codes,
                cn.ipcc_change_codes,
                flux_cube_subset.year.values,
            ),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        # Contextual layers to use to merge pixel_area against other analysis layers (to calculate flux/ha)
        contextual_layers = [
            cn.adm0_pattern,
            cn.land_state_pattern,
            # cn.WDPA_pattern,
            cn.cont_eco_zstats_pattern,
            # cn.landmark_pattern,
            cn.starting_composite_primary_forest_pattern,
            # cn.KBA_pattern,
            # cn.watersheds_pattern,
            cn.drivers_of_loss_pattern,
            # cn.forest_age_category_pattern,
            # # cn.BRA_biomes_pattern,
            # # cn.managed_land_CAN_pattern,
            # # cn.managed_land_USA_pattern,
            cn.IPCC_class_pattern,
            cn.IPCC_node_pattern,
            cn.IPCC_change_pattern,
            cn.IPCC_summary_pattern,
            'year'
        ]

        main_logger.info(f"  Done computing {tile_id}: {uu.timestr()}")
        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)

        del results
        gc.collect()

        df = zsu.create_df(coord_dict, state_node_df, contextual_layers, tile_id, 'vegetation', main_logger)

        del coord_dict
        gc.collect()

        main_logger.info(f"  Rows in {tile_id} dataframe: {len(df.index)}: {uu.timestr()}")
        main_logger.info(f"  Saving {tile_id} output table: {uu.timestr()}")
        tile_df_name = (f"ipcc_lulucf_zonal_stats_{tile_id}_v{lu_model_version}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}")
        df.to_parquet(f"{local_zonal_stats_folder}/{tile_df_name}.parquet")

        # Clean up at end of tile
        #del results, coord_dict, df, flux_cube_subset
        del df, flux_cube_subset
        gc.collect()

        tile_end_time = time.time()
        main_logger.info(f"  Done with {tile_id}, took {round(tile_end_time) - round(tile_start_time)} seconds: {uu.timestr()}")

        ### After all tiles completed

    all_tiles_end_time = time.time()
    main_logger.info(f"Finished tile analyses, took {round(all_tiles_end_time - prep_start_time)} seconds: {uu.timestr()}")
    if tiles_processed > 0:
        average_time = (all_tiles_end_time - prep_start_time) / tiles_processed
        main_logger.info(f"Average time per tile (excluding skipped tiles): {round(average_time)} seconds (for {tiles_processed} tiles)")
    else:
        main_logger.info("No tiles processed")

    # Terminates cluster because all further processing is done locally
    terminate_cluster.terminate_cluster(cluster_name)

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

        combined_df_name = f'ipcc_lulucf_zonal_stats_v{lu_model_version}_{time.strftime("%Y%m%d_%H_%M_%S")}'
        combined_df.to_parquet(f"{local_zonal_stats_folder}/{combined_df_name}.parquet")
        if len(combined_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
            combined_df.to_csv(f"{local_zonal_stats_folder}/{combined_df_name}.csv", index=False)

        # Converts combined df from long to wide
        combined_wide_df = zsu.create_wide_df(combined_df, main_logger)

        combined_wide_df_name = f'ipcc_lulucf_zonal_stats_v{lu_model_version}_wide_{time.strftime("%Y%m%d_%H_%M_%S")}'
        combined_wide_df.to_parquet(f"{local_zonal_stats_folder}/{combined_wide_df_name}.parquet")
        if len(combined_wide_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
            combined_wide_df.to_csv(f"{local_zonal_stats_folder}/{combined_wide_df_name}.csv", index=False)

    # # Converts each parquet to csv
    # main_logger.info(f"Converting parquet files to csvs: {uu.timestr()}")
    # for parquet_output in parquet_files:
    #     main_logger.info(f"Converting {parquet_output} to csv: {uu.timestr()}")
    #     df = pd.read_parquet(parquet_output)
    #     csv_output = parquet_output.replace('parquet', 'csv')
    #     df.to_csv(csv_output, index=False)
    #
    # # Uploads outputs to s3 if the run is large enough
    # zsu.upload_zstats_to_s3(stage, local_zonal_stats_folder, output_path, main_logger,
    #                     model_path_description, model_type, lulucf_model_version, tiles_processed)

    end_time = time.time()
    main_logger.info(f"Finished zonal stats, took {round(end_time - prep_start_time)} seconds: {uu.timestr()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ccombined LULUCF flux zonal stats by IPCC land use zarr context.")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")

    parser.add_argument("-vid", "--veg_input_date", required=True, help="Vegetation model run date")
    parser.add_argument("-tid", "--lulucf_input_date", required=True, help="Total LULUCF model run date")
    parser.add_argument("-lid", "--lu_input_date", required=True, help="IPCC land use run date")

    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="1x1 fishnet shapefile")
    parser.add_argument("-ft", "--first_tiles_to_process", type=int)
    parser.add_argument("-mt", "--model_type", default="standard")

    parser.add_argument("-vmpd", "--veg_model_path_description", default="global")
    parser.add_argument("-tmpd", "--lulucf_model_path_description", default="global")
    parser.add_argument("-lmpd", "--lu_model_path_description", default="global")
    parser.add_argument("-zd", "--zonal_stats_description", required=True)

    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=False, help='local path for model chunk stats to check if tile had any pixels in it, and skip if empty')
    parser.add_argument("-ln", "--log_note")
    parser.add_argument( "--tile_ids_file", required=False, help="Path to txt file containing 10x10 tile IDs to process, one per line.")

    args = parser.parse_args()

    main(
        cluster_name=args.cluster_name,
        lulucf_input_date=args.lulucf_input_date,
        veg_input_date=args.veg_input_date,
        lu_input_date=args.lu_input_date,
        model_type=args.model_type,
        lulucf_model_path_description=args.lulucf_model_path_description,
        veg_model_path_description=args.veg_model_path_description,
        lu_model_path_description=args.lu_model_path_description,
        zonal_stats_description=args.zonal_stats_description,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        first_variables_to_process=args.first_variables_to_process,
        first_tiles_to_process=args.first_tiles_to_process,
        model_chunk_stats_table_name=args.model_chunk_stats_table_name,
        log_note=args.log_note,
        tile_ids_file=args.tile_ids_file,
    )

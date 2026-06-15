"""
Run:
python -m src.utilities.create_cluster -n 1 -m 64 -cn IPCC_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.IPCC_zonal_stats -cn IPCC_zonal_stats -vid 20260130 -lid 20268888 -bb 119 -6 120 -5 -lmpd 1x1_test -zd 1x1_test
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import print
from flox.xarray import xarray_reduce
from flox import ReindexArrayType, ReindexStrategy

from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import zonal_stats_utilities as zsu
from src.utilities import terminate_cluster


def create_ipcc_lu_context_df(coord_dict, tile_id, main_logger):
    main_logger.info(f"  Creating {tile_id} land-use-context zonal stats df: {uu.timestr()}")

    df = pd.DataFrame(coord_dict)
    df["tile_id"] = tile_id

    # net flux zarr year index 0-8 corresponds to reporting years 2016-2024
    df["year"] = df["year"] + cn.interval_end_years_annual[0]

    # Split area and flux rows
    merge_keys = [ cn.IPCC_class_pattern, cn.IPCC_node_pattern, cn.IPCC_change_pattern, cn.IPCC_summary_pattern, "year"]

    df_area = (df[df["analysis_layer"] == "pixel_area_ha"].rename(columns={"value": "area_ha"})[merge_keys + ["area_ha"]])
    df_flux = df[df["analysis_layer"] != "pixel_area_ha"].copy()
    df_flux["analysis_layer"] = df_flux["analysis_layer"].str.replace("_ha_yr", "", regex=False)

    df_out = df_flux.merge(df_area, on=merge_keys, how="left")
    df_out["density__Mg_ha"] = df_out["value"] / df_out["area_ha"].replace(0, pd.NA)
    df_out["gas"] = "all gases"

    return df_out


def main(cluster_name, veg_input_date, lu_input_date, model_type = "standard", veg_model_path_description = "global", lu_model_path_description = "global",
         zonal_stats_description = "global", chunk_shapefile_uri=False, bounding_box=None, first_variables_to_process=None, first_tiles_to_process=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = "IPCC_land_use_zonal_statistics"
    model_version = cn.veg_model_version_underscore

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # If an area smaller than 6x6 deg is given (for testing), the sub_tile_test flag is activated
    # and the analysis extent changes further down
    print(not chunk_shapefile_uri)
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
    main_logger.info(f"Vegetation model version: {cn.veg_model_version}")
    main_logger.info(f"Vegetation model path descriptor: {veg_model_path_description}")
    main_logger.info(f"IPCC land use path descriptor: {lu_model_path_description}")
    main_logger.info(f"Zonal stats descriptor: {zonal_stats_description}")
    main_logger.info(f"Vegetation input date: {veg_input_date}")
    main_logger.info(f"Land-use input date: {lu_input_date}")
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

    # Outputs to performs zonal stats on
    full_list_of_vars = [cn.gross_emis_all_C_pools_all_gases_pattern,
                         cn.gross_removals_all_C_pools_pattern,
                         cn.net_flux_all_C_pools_all_gases_pattern]

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

    # The zarr path that's being used
    veg_zarr_path = zu.create_zarr_path(cn.veg_outputs_path_mega_zarr, source_zarr_chunk_size, "annual", model_type,
                                         model_version, veg_model_path_description, veg_input_date, main_logger)
    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {veg_zarr_path}")

    #STATE NODE CODES/ MEANINGS

    local_zonal_stats_folder = Path(cn.veg_local_zonal_stats_table_folder)
    local_zonal_stats_folder.mkdir(parents=True, exist_ok=True)

    #MODEL CHUNK STATS TABLE


    ### Step 2: Prepare input zarrs

    prep_start_time = time.time()

    lu_zarr_path = zu.create_zarr_path(cn.IPCC_outputs_path_mega_zarr, source_zarr_chunk_size, "annual", model_type,
                                        cn.IPCC_LU_version.replace(".", "_"), lu_model_path_description, lu_input_date, main_logger)
    main_logger.info(f"Land-use zarr: {lu_zarr_path}")

    # Open zarrs
    veg_ds = xr.open_zarr(veg_zarr_path, consolidated=False)
    lu_ds = xr.open_zarr(lu_zarr_path, consolidated=False)
    pixel_area_xr = xr.open_zarr(cn.pixel_area_zarr_path, consolidated=False).rename_vars(band_data=cn.pixel_area_zstats_pattern)

    # Select only these vegetation model variables from the zarr
    flux_ds = veg_ds[vars_to_process]

    # Round and align
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])
    flux_ds = zsu.round_coords(flux_ds)
    lu_ds = zsu.round_coords(lu_ds)

    flux_ds = zsu.safe_crop(flux_ds, reference)
    lu_ds = zsu.safe_crop(lu_ds, reference)

    pixel_area = reference
    pixel_area_expanded = pixel_area.expand_dims(year=flux_ds.year)

    # Convert pixel_area from m² to hectares, then adds to the list of layers to analyze (to get area of contextual layers)
    main_logger.info(f"Calculating pixel area: {uu.timestr()}")
    pixel_area_layer = (pixel_area_expanded * cn.m2_to_ha).astype("float32")

    # Multiply each flux var by pixel_area
    main_logger.info(f"Calculating per-pixel values for analysis layers: {uu.timestr()}")
    flux_layers = []
    for var in vars_to_process:
        flux_scaled = (flux_ds[var] * pixel_area_layer).astype("float32")
        flux_layers.append(flux_scaled)

    flux_layers.append(pixel_area_layer)

    # Also updates the list of analysis layer names
    selected_datasets = vars_to_process + ["pixel_area_ha"]

    # Stack into one flux cube: shape (analysis_layer, year, y, x)
    main_logger.info(f"Stacking analysis layers into flux cube: {uu.timestr()}")
    flux_cube = xr.concat(flux_layers, dim="analysis_layer").assign_coords(analysis_layer=("analysis_layer", selected_datasets))

    # LU context layers aligned to vegetation net-flux years.
    # Vegetation net flux years are 2016-2024, index 0-8.
    # IPCC class/node end-year context is LU indices 1-9.
    ipcc_class = lu_ds[cn.IPCC_class_pattern].isel(year=slice(1, 10))
    ipcc_node = lu_ds[cn.IPCC_node_pattern].isel(year=slice(1, 10))
    ipcc_change = lu_ds[cn.IPCC_change_pattern].isel(year=slice(1, 10))

    # Summary is static at index 0; expand to match flux years.
    ipcc_summary_static = lu_ds[cn.IPCC_summary_pattern].isel(year=0, drop=True)
    ipcc_summary = ipcc_summary_static.expand_dims(year=flux_ds.year)

    # Force all year coords to match flux cube.
    ipcc_class = ipcc_class.assign_coords(year=flux_ds.year)
    ipcc_node = ipcc_node.assign_coords(year=flux_ds.year)
    ipcc_change = ipcc_change.assign_coords(year=flux_ds.year)
    ipcc_summary = ipcc_summary.assign_coords(year=flux_ds.year)




    prep_end_time = time.time()
    main_logger.info(f"  Finished zonal stats prep, took {round(prep_end_time - prep_start_time)} seconds: {uu.timestr()}")



    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    tiles_processed = 0  # The number of tiles actually processed (since some are skipped)

    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} ({i + 1}/{len(tile_ids_to_process)}): {uu.timestr()}")

        # Skip tile if output parquet already exists. Useful when restarting.
        existing = set(os.listdir(local_zonal_stats_folder))
        if any(tile_id in fname and "ipcc_lu_zonal_stats" in fname for fname in existing):
            main_logger.info(f"  Skipping {tile_id}; output already exists")
            continue

        if sub_tile_test:
            west, south, east, north = bounding_box
        else:
            west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

        flux_subset = flux_cube.sel(
            x=slice(west, east),
            y=slice(north, south)
        )

        # Use flux_subset as the exact reference grid for LU layers
        reference_subset = flux_subset.isel(analysis_layer=0, drop=True)

        class_subset = zsu.safe_crop(
            ipcc_class.sel(x=slice(west, east), y=slice(north, south)),
            reference_subset,
        )
        node_subset = zsu.safe_crop(
            ipcc_node.sel(x=slice(west, east), y=slice(north, south)),
            reference_subset,
        )
        change_subset = zsu.safe_crop(
            ipcc_change.sel(x=slice(west, east), y=slice(north, south)),
            reference_subset,
        )
        summary_subset = zsu.safe_crop(
            ipcc_summary.sel(x=slice(west, east), y=slice(north, south)),
            reference_subset,
        )

        # Force coordinates to match exactly
        class_subset = class_subset.assign_coords(x=reference_subset.x, y=reference_subset.y)
        node_subset = node_subset.assign_coords(x=reference_subset.x, y=reference_subset.y)
        change_subset = change_subset.assign_coords(x=reference_subset.x, y=reference_subset.y)
        summary_subset = summary_subset.assign_coords(x=reference_subset.x, y=reference_subset.y)

        main_logger.info(f"  Computing {tile_id}: {uu.timestr()}")

        results = xarray_reduce(
            flux_subset, class_subset, node_subset, change_subset, summary_subset, flux_subset["year"], func="sum",
            expected_groups=(
                np.arange(0, 9, dtype=np.uint8),       # IPCC class
                np.arange(0, 400, dtype=np.uint16),    # node code
                np.arange(0, 89, dtype=np.uint8),      # change code
                np.arange(0, 89, dtype=np.uint16),     # summary code, assuming 2-digit summaries
                flux_subset.year.values,
            ),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0,
        ).compute()

        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)

        # Rename flox group dims into useful names
        coord_dict[cn.IPCC_class_pattern] = coord_dict.pop(class_subset.name)
        coord_dict[cn.IPCC_node_pattern] = coord_dict.pop(node_subset.name)
        coord_dict[cn.IPCC_change_pattern] = coord_dict.pop(change_subset.name)
        coord_dict[cn.IPCC_summary_pattern] = coord_dict.pop(summary_subset.name)

        df = create_ipcc_lu_context_df(coord_dict, tile_id, main_logger)

        tile_df_name = (f"ipcc_lu_zonal_stats_{tile_id}_v{model_version}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}")
        df.to_parquet(local_zonal_stats_folder / f"{tile_df_name}.parquet")

        tiles_processed += 1

        del results, coord_dict, df, flux_subset
        gc.collect()

    if cluster_name:
        terminate_cluster.terminate_cluster(cluster_name)

    main_logger.info(f"Finished {tiles_processed} tiles: {uu.timestr()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vegetation net flux zonal stats by IPCC land-use zarr context.")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")
    parser.add_argument("-vid", "--veg_input_date", required=True, help="Vegetation model run date")
    parser.add_argument("-lid", "--lu_input_date", required=True, help="IPCC land use run date")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="1x1 fishnet shapefile")
    parser.add_argument("-ft", "--first_tiles_to_process", type=int)
    parser.add_argument("-mt", "--model_type", default="standard")
    parser.add_argument("-vmpd", "--veg_model_path_description", default="global")
    parser.add_argument("-lmpd", "--lu_model_path_description", default="global")
    parser.add_argument("-zd", "--zonal_stats_description", required=True)
    parser.add_argument("-ln", "--log_note")

    args = parser.parse_args()

    main(cluster_name=args.cluster_name, veg_input_date=args.veg_input_date, lu_input_date=args.lu_input_date, model_type=args.model_type,
         veg_model_path_description=args.veg_model_path_description, lu_model_path_description=args.lu_model_path_description,
         zonal_stats_description=args.zonal_stats_description, chunk_shapefile_uri=args.chunk_shapefile_uri, bounding_box=args.bounding_box,
         first_variables_to_process=args.first_variables_to_process, first_tiles_to_process=args.first_tiles_to_process, log_note=args.log_note)
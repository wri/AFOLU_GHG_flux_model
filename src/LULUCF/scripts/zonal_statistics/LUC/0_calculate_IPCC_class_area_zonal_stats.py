"""
Purpose: Calculates the total area of IPCC land use classes for each 10x10 deg tile, by country, primary forest, land state,
IPCC class, IPCC change, IPCC summary and IPCC node, for each year 2016-2024. Every pixel is counted whether or not it has flux.
Use these tables for total country/class areas. Flux zonal stats are calculated separately by
0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats.py.

Tile tables are written to <land_use_zonal_stats_table_folder>/ipcc_class_area/ and combined into
ipcc_class_area_by_country_v<version>_<description>_<timestamp> (.parquet, .csv, _wide.csv) in <land_use_zonal_stats_table_folder>.

Output columns: adm0, country_name, region_L1, region_L2_L3, starting_composite_primary_forest, land_state_broad_class,
                tall_veg_type, IPCC_class, IPCC_class_name, IPCC_change, IPCC_change_name, IPCC_summary, IPCC_summary_name,
                IPCC_node_code, IPCC_node_code_name, year, tile_id, area_ha

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

10x10:
python -m src.utilities.create_cluster -n 10 -m 128 -cn IPCC_area_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_IPCC_class_area_zonal_stats -cn IPCC_area_zonal_stats -bb 100 10 110 20 -zd 20N_100W

Sweden:
python -m src.utilities.create_cluster -n 15 -m 32 -cn IPCC_area_zonal_stats__Sweden --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_IPCC_class_area_zonal_stats -cn IPCC_area_zonal_stats__Sweden -bb 10 55 25 70 -zd Sweden

Global run:
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_area_zonal_stats__global --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_IPCC_class_area_zonal_stats -cn IPCC_area_zonal_stats__global -zd global
"""

import argparse
import gc
import math
import os
import re
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

########################################################################################################################
# Run settings: defaults used when the corresponding command line argument isn't given.
########################################################################################################################
# 1x1 deg fishnet of chunks to process (used when no bounding box is given)
CHUNK_SHAPEFILE_URI = "s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp"

# Model versions of the inputs. Every zarr path, output file name and output folder uses these, not the versions in
# constants_and_names. Keep the same as in the flux script.
VEG_MODEL_VERSION = "1.0.5"
IPCC_LU_VERSION = "1.0.0"

VEG_INPUT_DATE = "20260130"       # Vegetation model run date (land state)
LU_INPUT_DATE = "20260617"        # IPCC land use run date

# Starting composite primary forest zarr. Primary forest is also a filter (pixels with values other than 0 or 1 are left out),
# so it changes totals, not only the primary/non-primary split.
#   - PRIMARY_FOREST_ZARR_PATH: read this zarr. The default is the one the validated IPCC_LULUCF_zonal_stats.py run read
#     (Sweden -38.3 Mt CO2e/yr). It's in a different folder from the path cn builds now.
#   - Set PRIMARY_FOREST_ZARR_PATH = None to build the path from cn with PRIMARY_FOREST_RUN_DATE instead.
PRIMARY_FOREST_ZARR_PATH = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/starting_composite_primary_forest/version_1_0_5__standard__global/2015/presence/zarr/4000_pixels/20260210/starting_composite_primary_forest.zarr"
# TODO: Not in sync with cn
PRIMARY_FOREST_RUN_DATE = "20260210"

# Sheet of the state node lookup table. Its land state codes are the expected groups for flox, so pixels with land states that
# aren't in the sheet are left out of the zonal stats (except 0, which this script counts). It has to match the vegetation zarr being summarized.
#   - Default: v105_20260601, the sheet cn had for the validated IPCC_LULUCF_zonal_stats.py run.
#   - cn now has v106_20261006. Set STATE_NODE_SHEET = cn.sheet to use it.
STATE_NODE_SHEET = "v105_20260601"
# Keep PRIMARY_FOREST_ZARR_PATH, PRIMARY_FOREST_RUN_DATE and STATE_NODE_SHEET the same as in the flux script.

# The vegetation zarr was written before the cn vegetation patterns got their veg_ prefix:
#   cn:   veg_gross_emissions__all_C_pools__all_gases__MgCO2e, veg_land_state_node, ...
#   zarr: gross_emissions__all_C_pools__all_gases__MgCO2e,     land_state_node, ...
# True:  removes veg_ from the start of cn names before looking them up in the zarrs.
# False: uses the cn names exactly. Set to False once the zarr names match cn. Keep the same as in the flux script.
DROP_VEG_PREFIX = True

# Contextual layers the class area tables are grouped by (plus adm0 and year).
# Land state nodes are converted to land_state_broad_class and tall_veg_type in the output tables.
CLASS_AREA_CONTEXT_KEYS = ["primary_forest", "land_state", "IPCC_class", "IPCC_change", "IPCC_summary", "IPCC_node"]


# Version with underscores instead of dots, as used in zarr paths and file names (e.g., 1.0.5 -> 1_0_5)
def underscore(version):
    return version.replace(".", "_")


# Local folder for the zonal stats tables: cn.land_use_zonal_stats_table_folder with the IPCC land use version from Run settings
def zonal_stats_folder():
    cn_folder = cn.land_use_zonal_stats_table_folder
    cn_version_part = f"_v{cn.IPCC_LU_version_underscore}_"
    if cn_version_part in cn_folder:
        return cn_folder.replace(cn_version_part, f"_v{underscore(IPCC_LU_VERSION)}_")
    return cn_folder


def read_tile_ids_file(tile_ids_file):
    with open(tile_ids_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


# Returns the land state variable from the vegetation zarr (removes veg_ from the cn name if DROP_VEG_PREFIX is True)
def get_land_state_da(veg_ds):
    name = cn.land_state_pattern
    if DROP_VEG_PREFIX and name.startswith("veg_"):
        name = name[len("veg_"):]
    if name not in veg_ds.data_vars:
        raise KeyError(f"{name} is not in the vegetation zarr (DROP_VEG_PREFIX = {DROP_VEG_PREFIX}). Variables in it: {sorted(veg_ds.data_vars)}")
    return veg_ds[name]


# Patterns and expected codes for each grouping layer (land_state codes are filled in from the state node table)
def get_class_area_layer_config():
    return {
        "primary_forest": {"pattern": cn.starting_composite_primary_forest_pattern, "codes": cn.composite_primary_codes},
        "land_state":     {"pattern": cn.land_state_pattern, "codes": None},
        "IPCC_class":     {"pattern": cn.IPCC_class_pattern, "codes": cn.ipcc_class_codes},
        "IPCC_change":    {"pattern": cn.IPCC_change_pattern, "codes": cn.ipcc_change_codes},
        "IPCC_summary":   {"pattern": cn.IPCC_summary_pattern, "codes": cn.ipcc_change_codes},
        "IPCC_node":      {"pattern": cn.IPCC_node_pattern, "codes": cn.ipcc_node_codes},
    }


# True if a file name is an output of this run: ..._v<version>_<description>_<timestamp>.parquet
def is_run_output(file_name, zonal_stats_description):
    return re.search(rf"_v\d+_\d+_\d+_{re.escape(zonal_stats_description)}_\d{{8}}_\d{{2}}_\d{{2}}_\d{{2}}\.parquet$",
                     file_name) is not None


# Subsets a layer to the tile. If the layer doesn't extend to the tile, returns all 0s.
def subset_layer(da, name, west, south, east, north, zero_template, tile_id, main_logger):
    subset = da.sel(x=slice(west, east), y=slice(north, south))
    if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
        main_logger.info(f"  {name} not in {tile_id}. Creating xarray of all 0s.")
        return xr.zeros_like(zero_template).rename(name)
    return subset.rename(name)


def main(cluster_name, veg_input_date, lu_input_date, model_type="standard", veg_model_path_description="global",
         lu_model_path_description="global", zonal_stats_description="global", chunk_shapefile_uri=False, bounding_box=None,
         first_tiles_to_process=None, log_note=None, tile_ids_file=None):

    ### Step 1: Preparation

    stage = "IPCC_class_area_zonal_statistics"
    # Versions from Run settings, not cn
    veg_model_version = underscore(VEG_MODEL_VERSION)
    lu_model_version = underscore(IPCC_LU_VERSION)

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # If an area smaller than 6x6 deg is given (for testing), the sub_tile_test flag is activated
    if (not chunk_shapefile_uri) and ((abs(bounding_box[0] - bounding_box[2]) < 6) and (abs(bounding_box[1] - bounding_box[3]) < 6)):
        sub_tile_test = True
    else:
        sub_tile_test = False

    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    main_logger.info(f"Stage {stage} started at: {uu.timestr()}")
    main_logger.info(f"Zonal stats description: {zonal_stats_description}")
    main_logger.info(f"Vegetation version: {VEG_MODEL_VERSION}")
    main_logger.info(f"Vegetation model path description: {veg_model_path_description}")
    main_logger.info(f"Vegetation input date: {veg_input_date}")
    main_logger.info(f"IPCC land use model version: {IPCC_LU_VERSION}")
    main_logger.info(f"IPCC land use path description: {lu_model_path_description}")
    main_logger.info(f"IPCC land use input date: {lu_input_date}")
    main_logger.info(f"Running sub-tile test area: {sub_tile_test}")

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
    chunk_size_deg = 1
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, None, fishnet_iso_df, main_logger)

    unique_tile_ids = sorted({uu.xy_to_tile_id(chunk[0], chunk[3]) for chunk in chunk_list})

    if tile_ids_file:
        tile_ids_to_keep = read_tile_ids_file(tile_ids_file)
        original_count = len(unique_tile_ids)
        unique_tile_ids = [tile_id for tile_id in unique_tile_ids if tile_id in tile_ids_to_keep]
        main_logger.info(f"Filtered tile list from {original_count} to {len(unique_tile_ids)} using {tile_ids_file}")

    tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process] if first_tiles_to_process else unique_tile_ids
    main_logger.info(f"tile_ids to calculate class area for: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    source_zarr_chunk_size = cn.chunk_dims  # 4000x4000

    # Same zarr paths as the flux script
    veg_zarr_path = zu.create_zarr_path(cn.veg_outputs_path_zarr.replace("/zarr/", "/mega_zarr/annual_intervals/"),
                                        source_zarr_chunk_size, veg_input_date, main_logger,
                                        model_version=veg_model_version, model_type=model_type,
                                        model_path_description=veg_model_path_description)
    # Built here because the cn path has cn's IPCC land use version in it. The land use folder uses the version with dots.
    lu_zarr_path = zu.create_zarr_path(
        f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/land_use/{IPCC_LU_VERSION}/mega_zarr/annual_intervals/CHUNK_SIZE_pixels/RUN_DATE/land_use_zarr.zarr",
        source_zarr_chunk_size, lu_input_date, main_logger)
    if PRIMARY_FOREST_ZARR_PATH:
        primary_forest_zarr_path = PRIMARY_FOREST_ZARR_PATH
    else:
        primary_forest_zarr_path = zu.create_zarr_path(
            cn.starting_composite_primary_forest_zarr_path.replace(cn.starting_composite_primary_forest_run_date, PRIMARY_FOREST_RUN_DATE),
            source_zarr_chunk_size, PRIMARY_FOREST_RUN_DATE, main_logger,
            model_version=veg_model_version, model_type="standard", model_path_description="global")
    main_logger.info(f"Vegetation zarr path: {veg_zarr_path}")
    main_logger.info(f"IPCC land use zarr path: {lu_zarr_path}")
    main_logger.info(f"Primary forest zarr path: {primary_forest_zarr_path}")

    main_logger.info(f"State node lookup table sheet: {STATE_NODE_SHEET} (cn.sheet: {cn.sheet})")
    state_node_df = zsu.create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, STATE_NODE_SHEET)
    # Includes a 0 code so pixels without a land state are still counted in total area
    node_codes = np.unique(np.concatenate([[0], np.array(list(state_node_df['land_state']), dtype=np.uint32)])).astype(np.uint32)

    layer_config = get_class_area_layer_config()
    layer_config["land_state"]["codes"] = node_codes

    group_counts = {"adm0": len(cn.gadm_adm0_ids), **{key: len(layer_config[key]["codes"]) for key in CLASS_AREA_CONTEXT_KEYS}}
    n_combinations = math.prod(group_counts.values()) * len(cn.veg_outputs_years)
    main_logger.info(f"Class area code counts: {group_counts}; possible combinations: {n_combinations:.2e}")
    if n_combinations > 2**62:
        raise ValueError(f"Too many class area combinations for flox ({n_combinations:.2e}). Remove a layer from CLASS_AREA_CONTEXT_KEYS.")

    local_zonal_stats_folder = Path(zonal_stats_folder())   # Folder name uses IPCC_LU_VERSION from Run settings
    local_class_area_folder = local_zonal_stats_folder / "ipcc_class_area"
    local_class_area_folder.mkdir(parents=True, exist_ok=True)


    ### Step 2: Prepare input zarrs

    prep_start_time = time.time()

    pixel_area_xr = xr.open_zarr(cn.pixel_area_zarr_path, consolidated=False).rename_vars(band_data=cn.pixel_area_zstats_pattern)
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])

    adm0_xr = xr.open_zarr(cn.adm0_zarr_path, consolidated=False).rename_vars(band_data=cn.adm0_pattern)
    adm0_aligned = zsu.safe_crop(zsu.round_coords(adm0_xr), reference)[cn.adm0_pattern]

    veg_ds = xr.open_zarr(veg_zarr_path, consolidated=False)
    lu_ds = zsu.safe_crop(zsu.round_coords(xr.open_zarr(lu_zarr_path, consolidated=False)), reference)

    # Analysis years 2016-2024 (index values of the vegetation zarr's year dimension)
    analysis_years = veg_ds["year"]

    primary_xr = xr.open_zarr(primary_forest_zarr_path, consolidated=False)  # Variable already named starting_composite_primary_forest
    primary_da = zsu.safe_crop(zsu.round_coords(primary_xr), reference)[cn.starting_composite_primary_forest_pattern]
    if "year" in primary_da.dims:
        primary_da = primary_da.isel(year=0, drop=True)

    # IPCC outputs have 10 annual positions (2015-2024); 2016-2024 are used.
    #   IPCC class/node: positions 0-9 are annual outputs
    #   IPCC change: position 0 is empty, positions 1-9 are change interval end years
    #   IPCC summary: position 0 is static 2015_2024 summary, positions 1-9 are empty
    layer_das = {
        "primary_forest": primary_da,
        "land_state": zsu.safe_crop(zsu.round_coords(get_land_state_da(veg_ds)), reference).assign_coords(year=analysis_years.values),
        "IPCC_class": lu_ds[cn.IPCC_class_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years.values),
        "IPCC_change": lu_ds[cn.IPCC_change_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years.values),
        "IPCC_summary": lu_ds[cn.IPCC_summary_pattern].isel(year=0, drop=True).expand_dims(year=analysis_years.values),
        "IPCC_node": lu_ds[cn.IPCC_node_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years.values),
    }
    layer_das = {key: da.reset_coords(drop=True) for key, da in layer_das.items()}
    adm0_aligned = adm0_aligned.reset_coords(drop=True)

    # Pixel area in ha (no year dimension)
    pixel_area_ha_2d = (reference * cn.m2_to_ha).astype("float32").reset_coords(drop=True).rename("area_ha")

    main_logger.info(f"  Finished prep, took {round(time.time() - prep_start_time)} seconds: {uu.timestr()}")


    ### Step 3: Class area tile by tile

    tiles_processed = 0
    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} ({i + 1}/{len(tile_ids_to_process)}): {uu.timestr()}")
        tile_start_time = time.time()

        existing = set(os.listdir(local_class_area_folder))
        if any(tile_id in fname and is_run_output(fname, zonal_stats_description) for fname in existing):
            main_logger.info(f"  Skipping {tile_id}; output already exists")
            continue

        tiles_processed += 1
        if sub_tile_test:
            west, south, east, north = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            main_logger.info("  Running test area")
        else:
            west, south, east, north = uu.get_10x10_tile_bounds(tile_id)

        area_3d = pixel_area_ha_2d.sel(x=slice(west, east), y=slice(north, south)).expand_dims(year=analysis_years.values)
        zero_template = xr.zeros_like(area_3d)

        adm0_da = subset_layer(adm0_aligned, cn.adm0_pattern, west, south, east, north, zero_template, tile_id, main_logger)
        by_das = [subset_layer(layer_das[key], layer_config[key]["pattern"], west, south, east, north, zero_template, tile_id, main_logger)
                  for key in CLASS_AREA_CONTEXT_KEYS]

        aligned = xr.align(area_3d, adm0_da, *by_das, join="override")
        area_3d, adm0_da, by_das = aligned[0], aligned[1], list(aligned[2:])

        main_logger.info(f"  Computing IPCC class area for {tile_id}: {uu.timestr()}")
        results = xarray_reduce(
            area_3d,
            adm0_da,
            *by_das,
            area_3d["year"],
            func="sum",
            expected_groups=(cn.gadm_adm0_ids,)
                            + tuple(layer_config[key]["codes"] for key in CLASS_AREA_CONTEXT_KEYS)
                            + (area_3d.year.values,),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)
        df = zsu.create_ipcc_class_area_df(coord_dict, state_node_df, tile_id, main_logger)
        df_name = f"ipcc_class_area_{tile_id}_v{lu_model_version}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}"
        df.to_parquet(f"{local_class_area_folder}/{df_name}.parquet")
        main_logger.info(f"  IPCC class area table for {tile_id} has {len(df.index)} rows: {uu.timestr()}")

        del results, coord_dict, df, area_3d
        gc.collect()

        main_logger.info(f"  Done with {tile_id}, took {round(time.time() - tile_start_time)} seconds: {uu.timestr()}")

    main_logger.info(f"Finished tiles ({tiles_processed} processed), took {round(time.time() - prep_start_time)} seconds: {uu.timestr()}")

    # Shuts down the cluster because combining is done locally
    try:
        if not run_local and cluster is not None:
            client.close()
            cluster.shutdown()
            main_logger.info(f"Cluster {cluster_name} shut down: {uu.timestr()}")
    except Exception as e:
        main_logger.warning(f"Could not shut down cluster {cluster_name} ({e}). Check the Coiled dashboard.")

    ### Combines tile tables (total area per country per IPCC land use class per year)
    tile_id_regex = r"\d{2}[NS]_\d{3}[EW]"
    class_area_files = sorted(
        str(local_class_area_folder / f)
        for f in os.listdir(local_class_area_folder)
        if f.startswith("ipcc_class_area_") and re.search(tile_id_regex, f) and is_run_output(f, zonal_stats_description)
    )
    if not class_area_files:
        main_logger.info("No IPCC class area tables found.")
        return

    main_logger.info(f"Combining {len(class_area_files)} IPCC class area tables: {uu.timestr()}")
    class_area_long, class_area_wide = zsu.combine_ipcc_class_area_tables([pd.read_parquet(f) for f in class_area_files], main_logger)
    class_area_name = f'ipcc_class_area_by_country_v{lu_model_version}_{zonal_stats_description}_{time.strftime("%Y%m%d_%H_%M_%S")}'
    class_area_long.to_parquet(f"{local_zonal_stats_folder}/{class_area_name}.parquet")
    class_area_long.to_csv(f"{local_zonal_stats_folder}/{class_area_name}.csv", index=False)
    class_area_wide.to_csv(f"{local_zonal_stats_folder}/{class_area_name}_wide.csv", index=False)
    main_logger.info(f"IPCC class area tables written to {local_zonal_stats_folder}/{class_area_name}*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total area of IPCC land use classes by country, land state and IPCC layers.")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")
    parser.add_argument("-vid", "--veg_input_date", default=VEG_INPUT_DATE, help=f"Vegetation model run date (default {VEG_INPUT_DATE})")
    parser.add_argument("-lid", "--lu_input_date", default=LU_INPUT_DATE, help=f"IPCC land use run date (default {LU_INPUT_DATE})")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="1x1 fishnet shapefile (default: CHUNK_SHAPEFILE_URI if no bounding box is given)")
    parser.add_argument("-ft", "--first_tiles_to_process", type=int)
    parser.add_argument("-mt", "--model_type", default="standard")
    parser.add_argument("-vmpd", "--veg_model_path_description", default="global")
    parser.add_argument("-lmpd", "--lu_model_path_description", default="global")
    parser.add_argument("-zd", "--zonal_stats_description", required=True)
    parser.add_argument("-ln", "--log_note")
    parser.add_argument("--tile_ids_file", required=False, help="Path to txt file containing 10x10 tile IDs to process, one per line.")

    args = parser.parse_args()

    if not args.bounding_box and not args.chunk_shapefile_uri:
        args.chunk_shapefile_uri = CHUNK_SHAPEFILE_URI

    for arg_name in ["veg_input_date", "lu_input_date"]:
        if not re.fullmatch(r"\d{8}", str(getattr(args, arg_name))):
            parser.error(f"{arg_name} is '{getattr(args, arg_name)}'. Set it in the Run settings block or pass it on the command line.")

    main(
        cluster_name=args.cluster_name,
        veg_input_date=args.veg_input_date,
        lu_input_date=args.lu_input_date,
        model_type=args.model_type,
        veg_model_path_description=args.veg_model_path_description,
        lu_model_path_description=args.lu_model_path_description,
        zonal_stats_description=args.zonal_stats_description,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        first_tiles_to_process=args.first_tiles_to_process,
        log_note=args.log_note,
        tile_ids_file=args.tile_ids_file,
    )

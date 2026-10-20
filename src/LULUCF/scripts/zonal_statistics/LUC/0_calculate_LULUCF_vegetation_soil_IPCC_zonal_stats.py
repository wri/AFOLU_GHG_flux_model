"""
Purpose: Calculates zonal statistics for LULUCF flux outputs using IPCC land use classifications and other contextual datasets.
The script reads vegetation, LULUCF, mineral soil (SOC), organic soil, and IPCC land use zarrs, converts per-hectare fluxes to per-pixel totals,
and summarizes emissions, removals, net flux, and area by IPCC land use class + transition + summary, land state, administrative unit, and other
contextual layers for each 10×10° tile. Tile-level outputs are written as parquet tables and optionally combined into global zonal statistics tables.

Two kinds of area are output:
    1. area_ha in the flux tables: area of the contextual-layer combination, attached only to rows where that flux layer is non-zero.
       Summing it for one analysis layer gives the area with that flux, NOT the total area of a country/class.
    2. ipcc_class_area tables (separate outputs in the ipcc_class_area/ subfolder): total pixel area of all pixels, whether or not
       they have flux, by country, primary forest, land state (broad class, tall vegetation type) and IPCC class/change/summary/node,
       2016-2024. Use these for total country/class areas.

Output columns (in order; columns of contextual layers that aren't selected are left out, and columns of extra selected
contextual layers such as WDPA go before gas):
    Flux:       analysis_layer, adm0, country_name, region_L1, region_L2_L3, cont_eco, continent, ecozone, continent_ecozone,
                climate_domain, starting_composite_primary_forest, land_state_broad_class, tall_veg_type, IPCC_class,
                IPCC_class_name, IPCC_change, IPCC_change_name, IPCC_summary, IPCC_summary_name, IPCC_node_code,
                IPCC_node_code_name, gas, year, tile_id, value, area_ha
    Class area: adm0, country_name, region_L1, region_L2_L3, starting_composite_primary_forest, land_state_broad_class,
                tall_veg_type, IPCC_class, IPCC_class_name, IPCC_change, IPCC_change_name, IPCC_summary, IPCC_summary_name,
                IPCC_node_code, IPCC_node_code_name, year, tile_id, area_ha
Zonal stats are run on veg_land_state_node; nodes are then converted to land_state_broad_class and tall_veg_type and rows
that become identical are summed, so the tables have fewer rows.

Soil analysis layers are mapped onto the LULUCF analysis years (2016-2024) using cn.mineral_soil_year_map and cn.organic_soil_year_map.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

10x10:
python -m src.utilities.create_cluster -n 10 -m 128 -cn IPCC_land_use_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_land_use_zonal_stats -bb 100 10 110 20 -zd 20N_100W

Central and East Africa (8 10x10 tiles):
python -m src.utilities.create_cluster -n 25 -m 32 -cn IPCC_land_use_zonal_stats__Central_Africa --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_land_use_zonal_stats__Central_Africa -bb 13 -14 44 -3 -zd Central_Africa

Sweden:
python -m src.utilities.create_cluster -n 10 -m 32 -cn IPCC_land_use_zonal_stats__Sweden --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_land_use_zonal_stats__Sweden -bb 10 55 25 70 -zd Sweden -d vegetation mineral_soil organic_soil -g CO2 all_gases

EU27 (30 10x10 tiles):
python -m src.utilities.create_cluster -n 50 -m 32 -cn IPCC_land_use_zonal_stats__EU27 --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_land_use_zonal_stats__EU27 -bb -11 34 35 71 -zd EU27 -d vegetation mineral_soil organic_soil -g CO2 all_gases

Global run:
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_land_use_zonal_stats__global --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_land_use_zonal_stats__global -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

global_south
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_land_use_zonal_stats__global_south --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_LULUCF_zonal_stats_global_south --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_south.txt -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

global_north
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_land_use_zonal_stats__global_north --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_LULUCF_zonal_stats_global_north --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_north.txt -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

Notes:
    - Took 5 minutes to run for 10x10 degree area (00N_110E) with 25 workers (17 credits, $1).
        Analysis layers = LULUCF emissions, LULUCF removals, and LULUCF net flux, vegetation net flux, pixel area
        Contextual layers = IPCC class, node, change and summary, land state node, continent/ecozone, driver, primary/IFL
    - Toook 19 minutes to run for Central Africa (8 tiles) with 50 workers (128 credits, $7)

Selecting analysis and contextual layers (defaults are set in the Run settings block):
    -d  / --datasets            LULUCF, vegetation, mineral_soil, organic_soil     (default: all)
    -g  / --gases               CO2, non_CO2, all_gases                            (default: all)
    -f  / --fluxes              emissions, removals, netflux                       (default: all)
    -cl / --contextual_layers   any of CONTEXTUAL_LAYER_OPTIONS, or all            (default: DEFAULT_CONTEXTUAL_LAYERS)

    Contextual layers multiply: flox indexes every combination of their codes, and the product has to fit in a 64-bit integer.
    The defaults are ~3e13 combinations. "all" (~2e21) is too many for one pass and the script stops before loading anything.
    Leaving out watersheds and cont_eco from "all" brings it under the limit.

    An analysis layer is used if it matches ALL of the selected dataset, gas and flux filters.
    Removals and mineral soil fluxes are CO2 only, so they count as both CO2 and all_gases.
    Organic soil only has emissions. Mineral soil has loss (emissions), gain (removals) and net (netflux).

    Examples:
        # Only CO2 from vegetation and LULUCF, by country and IPCC class only
        ... -d LULUCF vegetation -g CO2 -cl adm0 IPCC_class -zd adm0_vegetation_LULUCF_CO2
        # Only soils, all gases, with default contextual layers
        ... -d mineral_soil organic_soil -zd soils
        # Protected areas, Indigenous lands and KBAs with country and IPCC class
        ... -cl adm0 IPCC_class WDPA landmark KBA -zd protected_areas

TODO:
- Paths, version, etc had to be modified in this script because cn is more up-to-date than our previous outputs. Update
  constants and utilities in this script once everything matches in cn.
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
# from src.utilities import terminate_cluster   # Not used: see cluster shutdown at the end of main()

########################################################################################################################
# Run settings: defaults used when the corresponding command line argument isn't given.
########################################################################################################################
# 1x1 deg fishnet of chunks to process (used when no bounding box is given)
CHUNK_SHAPEFILE_URI = "s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp"

VEG_MODEL_VERSION = "1.0.5"       # Vegetation model version of the VEG_INPUT_DATE run (overrides cn.veg_model_version for the vegetation zarr)
PRIMARY_FOREST_MODEL_VERSION = cn.veg_model_version # Model version folder of the starting composite primary forest zarr (fills version_MODEL_VERSION__TYPE__DESCRIPTION in its cn path)
# Starting composite primary forest zarr (version and run date differ from cn.starting_composite_primary_forest_zarr_path)
PRIMARY_FOREST_ZARR_PATH = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/starting_composite_primary_forest/version_1_0_5__standard__global/2015/presence/zarr/4000_pixels/20260210/starting_composite_primary_forest.zarr"

LULUCF_INPUT_DATE = "20260614"    # Total LULUCF model run date
VEG_INPUT_DATE = "20260130"       # Vegetation model run date
SOC_INPUT_DATE = "20260611"       # Mineral soil (SOC) zarr run date
LU_INPUT_DATE = "20260617"        # IPCC land use run date
#TODO: Add organic soil run date

# Position of each organic soil interval in the organic soil zarr's year dimension (5 positions).
# Only 2016 onward is used: 2016-2020 analysis years use the 2016_2020 position, 2021-2024 use the 2021_2024 position.
ORGANIC_SOIL_INTERVAL_POSITIONS = {'2016_2020': -2, '2021_2024': -1}

# Analysis layer filters (all used by default)
DATASET_OPTIONS = ["LULUCF", "vegetation", "mineral_soil", "organic_soil"]
GAS_OPTIONS = ["CO2", "non_CO2", "all_gases"]
FLUX_OPTIONS = ["emissions", "removals", "netflux"]

# Contextual layers that can be used, and the ones used by default
CONTEXTUAL_LAYER_OPTIONS = [
    "adm0", "land_state", "WDPA", "cont_eco", "landmark", "primary_forest", "KBA", "watersheds", "drivers",
    "forest_age", "BRA_biomes", "managed_land_CAN", "managed_land_USA",
    "IPCC_class", "IPCC_node", "IPCC_change", "IPCC_summary",
]
DEFAULT_CONTEXTUAL_LAYERS = [
    "adm0", "land_state", "cont_eco", "primary_forest", "IPCC_class", "IPCC_node", "IPCC_change", "IPCC_summary",
]

########################################################################################################################
# Utilities
########################################################################################################################
def read_tile_ids_file(tile_ids_file):
    with open(tile_ids_file, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


# Builds a zarr path by filling in the model version/type/description, chunk size and run date placeholders.
# Same replacements as zu.create_zarr_path but with explicit keyword arguments, so argument order can't get mixed up.
def build_zarr_path(zarr_basic_path, chunk_size_pixels, run_date, model_version, model_type, model_path_description):
    zarr_path = zarr_basic_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    zarr_path = zarr_path.replace("RUN_DATE", run_date)
    zarr_path = zarr_path.replace("CHUNK_SIZE", str(chunk_size_pixels))
    return zarr_path


# Finds the name a variable actually has in a dataset (with or without the _ha_yr per-hectare suffix).
# Returns None if the variable isn't in the dataset.
def resolve_var_name(ds, var_name):
    candidates = [var_name, f"{var_name}_ha_yr", var_name.replace("_ha_yr", "")]
    if var_name.startswith("veg_"):
        candidates += [c[len("veg_"):] for c in candidates]
    for candidate in candidates:
        if candidate in ds.data_vars:
            return candidate
    return None


# Maps a soil layer that has its own time steps (e.g., SOC change intervals or organic soil five-year intervals)
# onto the LULUCF analysis years, so it can be stacked into the same flux cube as the vegetation and LULUCF layers.
#   da: soil DataArray with dims (time_dim, y, x) or (y, x)
#   analysis_year_coord: the year coordinate of the LULUCF/vegetation analysis layers (index values)
#   analysis_year_values: the actual reporting years corresponding to analysis_year_coord (e.g., 2016-2024)
#   year_map: dict of reporting year -> soil interval label (e.g., cn.mineral_soil_year_map, cn.organic_soil_year_map)
#   interval_labels: soil interval labels in the order they are stored in the zarr (used if the zarr's time coordinate is an index)
def map_layer_to_analysis_years(da, analysis_year_coord, analysis_year_values, year_map, interval_labels, layer_name, main_logger):

    time_dims = [d for d in da.dims if d not in ("x", "y")]

    # Static layer: same value for every analysis year
    if not time_dims:
        main_logger.info(f"  {layer_name} has no time dimension; using it for all analysis years")
        return da.expand_dims(year=analysis_year_coord.values).assign_coords(year=analysis_year_coord.values)

    time_dim = time_dims[0]
    targets = [year_map[int(y)] for y in analysis_year_values]

    coord_vals = da[time_dim].values if time_dim in da.coords else np.arange(da.sizes[time_dim])
    lookup = {str(v): v for v in coord_vals}

    if da.sizes[time_dim] == 1:
        # Only one time step stored; use it for all years
        mapped = da.isel({time_dim: [0] * len(targets)})
        main_logger.info(f"  {layer_name} has a single time step; using it for all analysis years")
    elif all(str(t) in lookup for t in targets):
        # Time coordinate holds actual interval labels/years; select by label
        mapped = da.sel({time_dim: [lookup[str(t)] for t in targets]})
        main_logger.info(f"  {layer_name} mapped to analysis years by {time_dim} label: {dict(zip(analysis_year_values, targets))}")
    else:
        # Time coordinate is an index; convert interval labels to positions
        n_positions = da.sizes[time_dim]
        if isinstance(interval_labels, dict):
            # Explicit interval -> position mapping (e.g., ORGANIC_SOIL_INTERVAL_POSITIONS); only the listed intervals are used
            positions = [interval_labels[t] % n_positions for t in targets]
        else:
            # Stops rather than guessing: mapping years to the wrong positions gives wrong fluxes without any error
            if n_positions != len(interval_labels):
                raise ValueError(
                    f"{layer_name}: {time_dim} has {n_positions} positions but {len(interval_labels)} interval labels "
                    f"were given ({interval_labels}). Give one label per position.")
            positions = [list(interval_labels).index(t) for t in targets]
        mapped = da.isel({time_dim: positions})
        main_logger.info(f"  {layer_name} mapped to analysis years by {time_dim} position: {dict(zip(analysis_year_values, positions))}")

    if time_dim != "year":
        mapped = mapped.rename({time_dim: "year"})

    return mapped.assign_coords(year=analysis_year_coord.values)


# All analysis layers that can be run, tagged by dataset, flux type and gas. "name" is the variable name in the source zarr (with units).
# Fluxes that only involve CO2 (removals, mineral soil) are tagged as both CO2 and all_gases, because CO2 is all of their gases.
def get_analysis_layer_config():

    def with_units(pattern):
        return zu.add_units_year_to_pattern(pattern, 9999)[0]

    co2_is_all_gases = ["CO2", "all_gases"]

    return [
        # LULUCF totals
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["CO2"],           "name": with_units(cn.gross_emis_all_C_pools_CO2_only_LULUCF_pattern)},
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["non_CO2"],       "name": with_units(cn.gross_emis_all_C_pools_non_CO2_only_LULUCF_pattern)},
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["all_gases"],     "name": with_units(cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern)},
        {"dataset": "LULUCF", "flux": "removals",  "gases": co2_is_all_gases,  "name": with_units(cn.gross_removals_all_C_pools_LULUCF_pattern)},
        {"dataset": "LULUCF", "flux": "netflux",   "gases": ["CO2"],           "name": with_units(cn.net_flux_all_C_pools_CO2_only_LULUCF_pattern)},
        {"dataset": "LULUCF", "flux": "netflux",   "gases": ["all_gases"],     "name": with_units(cn.net_flux_all_C_pools_all_gases_LULUCF_pattern)},

        # Vegetation
        {"dataset": "vegetation", "flux": "emissions", "gases": ["CO2"],       "name": with_units(cn.gross_emis_all_C_pools_CO2_only_pattern)},
        {"dataset": "vegetation", "flux": "emissions", "gases": ["non_CO2"],   "name": with_units(cn.gross_emis_all_C_pools_non_CO2_only_pattern)},
        {"dataset": "vegetation", "flux": "emissions", "gases": ["all_gases"], "name": with_units(cn.gross_emis_all_C_pools_all_gases_pattern)},
        {"dataset": "vegetation", "flux": "removals",  "gases": co2_is_all_gases, "name": with_units(cn.gross_removals_all_C_pools_pattern)},
        {"dataset": "vegetation", "flux": "netflux",   "gases": ["CO2"],       "name": with_units(cn.net_flux_all_C_pools_CO2_only_pattern)},
        {"dataset": "vegetation", "flux": "netflux",   "gases": ["all_gases"], "name": with_units(cn.net_flux_all_C_pools_all_gases_pattern)},

        # Mineral soil (CO2 only)
        # {"dataset": "mineral_soil", "flux": "emissions", "gases": co2_is_all_gases, "name": with_units(cn.SOC_loss_min_soil_extent_pattern)},
        # {"dataset": "mineral_soil", "flux": "removals",  "gases": co2_is_all_gases, "name": with_units(cn.SOC_gain_min_soil_extent_pattern)},
        {"dataset": "mineral_soil", "flux": "netflux",   "gases": co2_is_all_gases, "name": with_units(cn.SOC_net_min_soil_extent_pattern)},
        #TODO: Uncomment min soil gain/loss

        # Organic soil (emissions only). Names already include units.
        # output_name is the analysis_layer name in the output tables (prefixed with organic_soil_); name is the zarr variable.
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["CO2"],       "name": "burned_co2_Mg_CO2_ha_yr",     "output_name": "organic_soil_burned_co2_Mg_CO2_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["CO2"],       "name": "drained_co2_Mg_CO2_ha_yr",    "output_name": "organic_soil_drained_co2_Mg_CO2_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["all_gases"], "name": "burned_total_Mg_CO2e_ha_yr",  "output_name": "organic_soil_burned_total_Mg_CO2e_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["all_gases"], "name": "drained_total_Mg_CO2e_ha_yr", "output_name": "organic_soil_drained_total_Mg_CO2e_ha_yr"},
    ]


# All contextual layers that can be used for grouping. Keys match CONTEXTUAL_LAYER_OPTIONS.
#   pattern: column name in the output tables (and name the layer is given)
#   source:  static (single-band contextual zarr), veg (from vegetation zarr), forest_age (categorized from vegetation zarr), ipcc (from IPCC land use zarr)
#   zarr_path / rename: for static layers; rename=True renames band_data to pattern
#   codes:   expected group values for flox (land_state codes are filled in from the state node table)
def get_contextual_layer_config():
    return {
        "adm0":             {"pattern": cn.adm0_pattern, "source": "static", "zarr_path": cn.adm0_zarr_path, "rename": True, "codes": cn.gadm_adm0_ids},
        "land_state":       {"pattern": cn.land_state_pattern, "source": "veg", "codes": None},
        "WDPA":             {"pattern": cn.WDPA_pattern, "source": "static", "zarr_path": cn.WDPA_zarr_path, "rename": True, "codes": cn.WDPA_codes},
        "cont_eco":         {"pattern": cn.cont_eco_zstats_pattern, "source": "static", "zarr_path": cn.cont_eco_zarr_path, "rename": True, "codes": cn.cont_eco_codes},
        "landmark":         {"pattern": cn.landmark_pattern, "source": "static", "zarr_path": cn.landmark_zarr_path, "rename": True, "codes": cn.landmark_codes},
        "primary_forest":   {"pattern": cn.starting_composite_primary_forest_pattern, "source": "static", "zarr_path": cn.starting_composite_primary_forest_zarr_path, "rename": False, "codes": cn.composite_primary_codes},
        "KBA":              {"pattern": cn.KBA_pattern, "source": "static", "zarr_path": cn.KBA_zarr_path, "rename": True, "codes": cn.KBA_codes},
        "watersheds":       {"pattern": cn.watersheds_pattern, "source": "static", "zarr_path": cn.watersheds_zarr_path, "rename": True, "codes": cn.watershed_codes},
        "drivers":          {"pattern": cn.drivers_of_loss_pattern, "source": "static", "zarr_path": cn.drivers_of_loss_zarr_path, "rename": True, "codes": cn.drivers_codes},
        "forest_age":       {"pattern": cn.forest_age_category_pattern, "source": "forest_age", "codes": cn.forest_age_category_codes},
        "BRA_biomes":       {"pattern": cn.BRA_biomes_pattern, "source": "static", "zarr_path": cn.BRA_biomes_zarr_path, "rename": True, "codes": cn.BRA_biomes_codes},
        "managed_land_CAN": {"pattern": cn.managed_land_CAN_pattern, "source": "static", "zarr_path": cn.managed_land_CAN_zarr_path, "rename": True, "codes": cn.managed_land_codes},
        "managed_land_USA": {"pattern": cn.managed_land_USA_pattern, "source": "static", "zarr_path": cn.managed_land_USA_zarr_path, "rename": True, "codes": cn.managed_land_codes},
        "IPCC_class":       {"pattern": cn.IPCC_class_pattern, "source": "ipcc", "codes": cn.ipcc_class_codes},
        "IPCC_node":        {"pattern": cn.IPCC_node_pattern, "source": "ipcc", "codes": cn.ipcc_node_codes},
        "IPCC_change":      {"pattern": cn.IPCC_change_pattern, "source": "ipcc", "codes": cn.ipcc_change_codes},
        "IPCC_summary":     {"pattern": cn.IPCC_summary_pattern, "source": "ipcc", "codes": cn.ipcc_change_codes},
    }


# Contextual layers the IPCC class area tables are grouped by (plus adm0 and year), whatever layers are selected for the flux tables.
# Land state nodes are converted to land_state_broad_class and tall_veg_type in the output tables.
CLASS_AREA_CONTEXT_KEYS = ["primary_forest", "land_state", "IPCC_class", "IPCC_change", "IPCC_summary", "IPCC_node"]


# True if a file name is an output of this zonal stats run: ..._v<version>_<description>_<timestamp>.parquet.
# Exact match, so -zd Sweden doesn't pick up files from runs like Sweden_class_area.
def is_run_output(file_name, zonal_stats_description):
    return re.search(rf"_v\d+_\d+_\d+_{re.escape(zonal_stats_description)}_\d{{8}}_\d{{2}}_\d{{2}}_\d{{2}}\.parquet$",
                     file_name) is not None


# Subsets a contextual layer to the tile. If the layer doesn't extend to the tile, returns all 0s.
def subset_context_layer(da, name, west, south, east, north, zero_template, tile_id, main_logger):
    subset = da.sel(x=slice(west, east), y=slice(north, south))
    if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
        main_logger.info(f"  {name} not in {tile_id}. Creating xarray of all 0s.")
        return xr.zeros_like(zero_template).rename(name)
    return subset.rename(name)


def main(cluster_name, lulucf_input_date, veg_input_date, lu_input_date, soc_input_date, model_type="standard", lulucf_model_path_description="global",
         veg_model_path_description="global", lu_model_path_description="global", soc_model_path_description="global", zonal_stats_description="global",
         chunk_shapefile_uri=False, bounding_box=None, first_variables_to_process=None, first_tiles_to_process=None,
         model_chunk_stats_table_name=None, log_note=None, tile_ids_file=None,
         datasets=None, gases=None, fluxes=None, contextual_layers=None):

    datasets = datasets or DATASET_OPTIONS
    gases = gases or GAS_OPTIONS
    fluxes = fluxes or FLUX_OPTIONS
    contextual_layers = contextual_layers or DEFAULT_CONTEXTUAL_LAYERS

    ### Step 1: Preparation

    # Model stage being run
    stage = "IPCC_land_use_zonal_statistics"
    veg_model_version = VEG_MODEL_VERSION.replace(".", "_")
    lulucf_model_version = cn.LULUCF_model_version_underscore
    lu_model_version = cn.IPCC_LU_version_underscore

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # If an area smaller than 6x6 deg is given (for testing), the sub_tile_test flag is activated and the analysis extent changes further down
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
    main_logger.info(f"Vegetation version: {VEG_MODEL_VERSION}")
    main_logger.info(f"Vegetation model path description: {veg_model_path_description}")
    main_logger.info(f"Vegetation input date: {veg_input_date}")
    main_logger.info(f"LULUCF version: {cn.LULUCF_model_version}")
    main_logger.info(f"LULUCF model path description: {lulucf_model_path_description}")
    main_logger.info(f"LULUCF input date: {lulucf_input_date}")
    main_logger.info(f"IPCC land use model version: {cn.IPCC_LU_version}")
    main_logger.info(f"IPCC land use path description: {lu_model_path_description}")
    main_logger.info(f"IPCC land use input date: {lu_input_date}")
    main_logger.info(f"Mineral soil version: {cn.SOC_model_version}")
    main_logger.info(f"Mineral soil model path description: {soc_model_path_description}")
    main_logger.info(f"Mineral soil input date: {soc_input_date}")
    main_logger.info(f"Organic soil version: {cn.organic_soil_model_version}")
    main_logger.info(f"Datasets: {datasets}")
    main_logger.info(f"Gases: {gases}")
    main_logger.info(f"Fluxes: {fluxes}")
    main_logger.info(f"Contextual layers: {contextual_layers}")
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

    # Analysis layers to run zonal stats on: those matching the selected datasets, gases and fluxes
    all_analysis_layers = get_analysis_layer_config()
    selected_analysis_layers = [
        layer for layer in all_analysis_layers
        if layer["dataset"] in datasets
        and layer["flux"] in fluxes
        and set(layer["gases"]) & set(gases)
    ]
    if not selected_analysis_layers:
        raise ValueError(f"No analysis layers match datasets {datasets}, gases {gases} and fluxes {fluxes}")

    # Limits the processed variables to the supplied number (for testing)
    n_selected = len(selected_analysis_layers)
    if first_variables_to_process:
        selected_analysis_layers = selected_analysis_layers[0:first_variables_to_process]
    main_logger.info(f"Analysis layers to run zonal stats on ({len(selected_analysis_layers)} out of {n_selected} selected):")
    for layer in selected_analysis_layers:
        output_name_note = f" (output: {layer['output_name']})" if "output_name" in layer else ""
        main_logger.info(f"  {layer['dataset']} | {layer['flux']} | {'/'.join(layer['gases'])} | {layer['name']}{output_name_note}")

    # Contextual layers to group by, in a fixed order (order of CONTEXTUAL_LAYER_OPTIONS)
    contextual_layer_config = get_contextual_layer_config()
    selected_contextual_layers = [key for key in CONTEXTUAL_LAYER_OPTIONS if key in contextual_layers]
    main_logger.info(f"Contextual layers to group by: {selected_contextual_layers}")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to perform zonal stats on: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  # 4000x4000

    # Vegetation zarr
    # Vegetation outputs are in a mega_zarr folder, not the zarr folder in cn.veg_outputs_path_zarr
    # Vegetation outputs are in mega_zarr/annual_intervals/, not the zarr/ folder in cn.veg_outputs_path_zarr
    veg_zarr_path = build_zarr_path(cn.veg_outputs_path_zarr.replace("/zarr/", "/mega_zarr/annual_intervals/"),
                                    source_zarr_chunk_size, veg_input_date, veg_model_version, model_type, veg_model_path_description)
    #TODO: zarr vs mega_zarr mismatch in constants and names for vegetation

    # Mineral soil zarr
    SOC_zarr_path = build_zarr_path(cn.SOC_path_zarr, source_zarr_chunk_size, soc_input_date,
                                    cn.SOC_model_version_underscore, model_type, soc_model_path_description)
    # Organic soil zarr
    organic_soil_zarr_path = cn.organic_soil_zarr_path
    main_logger.info(f"Vegetation zarr path: {veg_zarr_path}")
    main_logger.info(f"Mineral soil zarr path: {SOC_zarr_path}")
    main_logger.info(f"Organic soil zarr path: {organic_soil_zarr_path}")


    # LULUCF zarr
    lulucf_zarr_path = (
        f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs_LULUCF_totals/"
        f"LULUCF_version_{lulucf_model_version}_standard__{lulucf_model_path_description}"
        f"__veg_v{veg_model_version}__org_soil_v{cn.organic_soil_model_version_underscore}"
        f"__min_soil_v{cn.SOC_model_version_underscore}/zarr/annual_intervals/"
        f"{source_zarr_chunk_size}_pixels/{lulucf_input_date}/LULUCF_annual.zarr"
    )

    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {lulucf_zarr_path}")

    # Land use outputs are in mega_zarr/annual_intervals/
    lu_zarr_path = build_zarr_path( cn.IPCC_outputs_path_mega_zarr.replace("/mega_zarr/", "/mega_zarr/annual_intervals/"),
                                    source_zarr_chunk_size, lu_input_date, lu_model_version, model_type, lu_model_path_description)
    main_logger.info(f"IPCC land use zarr path: {lu_zarr_path}")
    #TODO: annual_intervals doesn't match naming convention

    # Creates dataframe of state_node codes and meanings
    state_node_df = zsu.create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, cn.sheet)
    # Includes 0 (no land state) so those pixels aren't dropped from the zonal stats.
    # Note: [0] + np.array(...) adds 0 to every element rather than prepending 0, so concatenate is used.
    node_codes = np.unique(np.concatenate([[0], np.array(list(state_node_df['land_state']), dtype=np.uint32)])).astype(np.uint32)

    # flox indexes every possible combination of contextual layer codes (x years), so the product of the number of codes
    # in each layer has to fit in a 64-bit integer. Checks that before any zarrs are opened.
    contextual_layer_config["land_state"]["codes"] = node_codes
    group_counts = {key: len(contextual_layer_config[key]["codes"]) for key in selected_contextual_layers}
    n_group_combinations = math.prod(group_counts.values()) * len(cn.veg_outputs_years)
    main_logger.info(f"Contextual layer code counts: {group_counts}; possible combinations: {n_group_combinations:.2e}")
    if n_group_combinations > 2**62:
        raise ValueError(
            f"Too many contextual layer combinations for flox ({n_group_combinations:.2e}; limit ~{2**62:.1e}). "
            f"Codes per layer: {group_counts}. Drop some layers with many codes (e.g., watersheds, cont_eco, adm0, IPCC_node, "
            f"IPCC_change, IPCC_summary), or run them in separate passes with different -zd descriptions."
        )
    # Same check for the IPCC class area tables (adm0 x CLASS_AREA_CONTEXT_KEYS x year)
    class_area_group_counts = {"adm0": len(cn.gadm_adm0_ids),
                               **{key: len(contextual_layer_config[key]["codes"]) for key in CLASS_AREA_CONTEXT_KEYS}}
    n_class_area_combinations = math.prod(class_area_group_counts.values()) * len(cn.veg_outputs_years)
    main_logger.info(f"IPCC class area code counts: {class_area_group_counts}; possible combinations: {n_class_area_combinations:.2e}")
    if n_class_area_combinations > 2**62:
        raise ValueError(f"Too many IPCC class area combinations for flox ({n_class_area_combinations:.2e}). "
                         f"Remove a layer from CLASS_AREA_CONTEXT_KEYS.")

    # main_logger.info(f"State nodes are from: {cn.state_node_lookup_table_local} or {cn.state_node_lookup_table_s3}, sheet {cn.sheet}")
    # main_logger.info(f"State nodes are: {node_codes}")

    local_zonal_stats_folder = Path(cn.land_use_zonal_stats_table_folder)
    local_zonal_stats_folder.mkdir(parents=True, exist_ok=True)

    # Separate folder for total area of IPCC land use classes per country (not restricted to pixels with flux)
    local_class_area_folder = local_zonal_stats_folder / "ipcc_class_area"
    local_class_area_folder.mkdir(parents=True, exist_ok=True)

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

    # Pixel area is the reference grid that everything else is cropped and aligned to
    pixel_area_xr = xr.open_zarr(cn.pixel_area_zarr_path, consolidated=False).rename_vars(band_data=cn.pixel_area_zstats_pattern)
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])
    x_coords = reference.coords['x']
    y_coords = reference.coords['y']

    # Country boundaries are always opened because the IPCC class area tables are by country,
    # even if adm0 isn't one of the selected contextual layers
    adm0_xr = xr.open_zarr(cn.adm0_zarr_path, consolidated=False).rename_vars(band_data=cn.adm0_pattern)
    adm0_aligned = zsu.safe_crop(zsu.round_coords(adm0_xr), reference)[cn.adm0_pattern].reset_coords(drop=True)

    # Vegetation and land use zarrs are always opened (analysis years, land state, IPCC class area tables)
    veg_ds = zsu.round_coords(xr.open_zarr(veg_zarr_path, consolidated=False))
    lu_ds = xr.open_zarr(lu_zarr_path, consolidated=False)
    lu_ds = zsu.safe_crop(zsu.round_coords(lu_ds), reference)

    # LULUCF analysis years (index values) and the reporting years they correspond to (2016-2024)
    analysis_years = veg_ds["year"]
    analysis_year_values = [int(cn.veg_outputs_years[0] + int(i)) for i in analysis_years.values]   # Same conversion as zsu.create_df
    main_logger.info(f"Analysis years: {analysis_year_values}")

    # Opens only the zarrs needed for the selected datasets
    selected_dataset_names = {str(layer["dataset"]) for layer in selected_analysis_layers}
    dataset_sources = {}
    if "LULUCF" in selected_dataset_names:
        dataset_sources["LULUCF"] = xr.open_zarr(lulucf_zarr_path, consolidated=False)
    if "vegetation" in selected_dataset_names:
        dataset_sources["vegetation"] = veg_ds
    if "mineral_soil" in selected_dataset_names:
        dataset_sources["mineral_soil"] = xr.open_zarr(SOC_zarr_path, consolidated=False)
    if "organic_soil" in selected_dataset_names:
        dataset_sources["organic_soil"] = xr.open_zarr(organic_soil_zarr_path, consolidated=False)
    for name, ds in dataset_sources.items():
        main_logger.info(f"{name} zarr variables: {list(ds.data_vars)}; dims: {dict(ds.sizes)}")

    # Year mapping for soil datasets, which have their own time steps
    soil_year_mapping = {
        "mineral_soil": (cn.mineral_soil_year_map, cn.SOC_density_intervals),
        "organic_soil": (cn.organic_soil_year_map, ORGANIC_SOIL_INTERVAL_POSITIONS),
    }

    # Pixel area in ha, expanded to the analysis years
    main_logger.info(f"Calculating pixel area: {uu.timestr()}")
    pixel_area_layer = (reference.expand_dims(year=analysis_years.values) * cn.m2_to_ha).astype("float32")

    # Multiplies each analysis layer by pixel area to get per-pixel values
    main_logger.info(f"Calculating per-pixel values for analysis layers: {uu.timestr()}")
    flux_layers = []
    analysis_layer_names = []

    for layer in selected_analysis_layers:
        ds = dataset_sources[layer["dataset"]]
        ds_var_name = resolve_var_name(ds, layer["name"])
        if ds_var_name is None:
            main_logger.warning(f"{layer['name']} not found in {layer['dataset']} zarr. Skipping it.")
            continue

        main_logger.info(f"Preparing {layer['dataset']} layer {layer['name']}: {uu.timestr()}")
        da = zsu.safe_crop(zsu.round_coords(ds[ds_var_name]), reference)

        if layer["dataset"] in soil_year_mapping:
            year_map, interval_labels = soil_year_mapping[layer["dataset"]]
            da = map_layer_to_analysis_years(da, analysis_years, analysis_year_values,
                                             year_map, interval_labels, layer["name"], main_logger)
        else:
            if da.sizes.get("year") != analysis_years.size:
                raise ValueError(f"{layer['name']} has {da.sizes.get('year')} years; expected {analysis_years.size}")
            da = da.assign_coords(year=analysis_years.values)

        da = da.assign_coords(x=x_coords, y=y_coords).transpose("year", "y", "x")
        flux_layers.append((da * pixel_area_layer).astype("float32"))
        analysis_layer_names.append(layer.get("output_name", layer["name"]))   # Name in the output tables

    if not flux_layers:
        raise ValueError("None of the selected analysis layers were found in their zarrs")

    # Total IPCC-context area
    flux_layers.append(pixel_area_layer)
    analysis_layer_names.append("pixel_area_ha")

    # Drops non-dimension coordinates (e.g., spatial_ref, band) so layers from different zarrs can be stacked
    flux_layers = [layer.reset_coords(drop=True) for layer in flux_layers]

    # Stack into one flux cube: shape (analysis_layer, year, y, x)
    main_logger.info(f"Stacking analysis layers into flux cube: {uu.timestr()}")
    flux_cube = xr.concat(flux_layers, dim="analysis_layer")
    flux_cube = flux_cube.assign_coords(analysis_layer=("analysis_layer", analysis_layer_names))
    flux_cube = zsu.round_coords(flux_cube)
    main_logger.info(f"flux_cube: {flux_cube}")

    ### Contextual layers, cropped to the reference grid. Annual layers are aligned to the analysis years.
    # IPCC outputs have 10 annual positions (2015-2024); LULUCF flux outputs have 9 (2016-2024)
    #   IPCC class/node: positions 0-9 are annual outputs
    #   IPCC change: position 0 is empty, positions 1-9 are change interval end years
    #   IPCC summary: position 0 is static 2015_2024 summary, positions 1-9 are empty
    main_logger.info(f"Preparing contextual layers: {uu.timestr()}")

    def prepare_context_layer(key):
        cfg = contextual_layer_config[key]
        pattern = cfg["pattern"]

        if cfg["source"] == "static":
            if key == "adm0":
                da = adm0_aligned
            else:
                # Fills any version/type/description and chunk size placeholders (the primary forest path has them)
                # Primary forest path is set in Run settings; other static layers use their cn paths
                ctx_zarr_path = PRIMARY_FOREST_ZARR_PATH if key == "primary_forest" else cfg["zarr_path"]
                main_logger.info(f"  {key} zarr path: {ctx_zarr_path}")
                ctx_ds = xr.open_zarr(ctx_zarr_path, consolidated=False)
                if cfg["rename"]:
                    ctx_ds = ctx_ds.rename_vars(band_data=pattern)
                da = zsu.safe_crop(zsu.round_coords(ctx_ds), reference)[pattern]
            # Some static layers (e.g., composite primary forest) are stored with a length-1 year dimension
            if "year" in da.dims:
                da = da.isel(year=0, drop=True)


        elif cfg["source"] == "veg":

            veg_var = resolve_var_name(veg_ds, pattern)

            if veg_var is None:
                raise KeyError(f"{pattern} not found in vegetation zarr. Variables: {list(veg_ds.data_vars)}")

            da = zsu.safe_crop(veg_ds[veg_var], reference).assign_coords(year=analysis_years.values)


        elif cfg["source"] == "forest_age":

            age_var = resolve_var_name(veg_ds, cn.forest_age_output_pattern)

            if age_var is None:
                raise KeyError(
                    f"{cn.forest_age_output_pattern} not found in vegetation zarr. Variables: {list(veg_ds.data_vars)}")

            forest_age = zsu.safe_crop(veg_ds[age_var], reference)
            da = zsu.categorize_age(forest_age).astype(np.uint8).assign_coords(year=analysis_years.values)

        elif cfg["source"] == "ipcc":
            if key == "IPCC_summary":
                da = lu_ds[pattern].isel(year=0, drop=True).expand_dims(year=analysis_years.values)
            else:
                da = lu_ds[pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years.values)

        else:
            raise ValueError(f"Unknown contextual layer source {cfg['source']} for {key}")

        da = da.reset_coords(drop=True).rename(pattern)
        main_logger.info(f"  {key} ({pattern}): dims {dict(da.sizes)}")
        return da

    # Contextual layers for the flux tables
    context_das = {key: prepare_context_layer(key) for key in selected_contextual_layers}

    # Contextual layers for the IPCC class area tables (reuses the flux table layers where they're the same)
    main_logger.info(f"Preparing IPCC class area layers: {uu.timestr()}")
    class_area_das = {key: context_das[key] if key in context_das else prepare_context_layer(key)
                      for key in CLASS_AREA_CONTEXT_KEYS}

    # Pixel area in ha (no year dimension) for the class area tables
    pixel_area_ha_2d = (reference * cn.m2_to_ha).astype("float32").reset_coords(drop=True).rename("area_ha")

    prep_end_time = time.time()
    main_logger.info(f"  Finished zonal stats prep, took {round(prep_end_time - prep_start_time)} seconds: {uu.timestr()}")


    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    tiles_processed = 0  # The number of tiles actually processed (since some are skipped)

    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} ({i + 1}/{len(tile_ids_to_process)}): {uu.timestr()}")
        tile_start_time = time.time()

        # Skips if an output for this tile_id and zonal stats description already exists (to not repeat that tile if restarting the zonal stats).
        # Matches on the description so that runs with different layer selections don't skip each other's tiles.
        existing = set(os.listdir(local_zonal_stats_folder))
        if any(tile_id in fname and is_run_output(fname, zonal_stats_description) for fname in existing):
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

        pixel_area_ha_2d_subset = pixel_area_ha_2d.sel(x=slice(west, east), y=slice(north, south))

        # Contextual layers for this tile. Layers that don't extend to the tile become all 0s.
        zero_template = flux_cube_subset.isel(analysis_layer=0, drop=True).reset_coords(drop=True)
        adm0_da = subset_context_layer(adm0_aligned, cn.adm0_pattern, west, south, east, north, zero_template, tile_id, main_logger)
        context_tile_das = [
            subset_context_layer(context_das[key], contextual_layer_config[key]["pattern"],
                                 west, south, east, north, zero_template, tile_id, main_logger)
            for key in selected_contextual_layers
        ]

        # Final alignment
        main_logger.info(f"  Aligning {tile_id}: {uu.timestr()}")
        aligned = xr.align(flux_cube_subset, adm0_da, *context_tile_das, join="override")
        flux_cube_subset, adm0_da, context_tile_das = aligned[0], aligned[1], list(aligned[2:])

        ### Total area by country x primary forest x land state x IPCC class/change/summary/node x year
        # (all pixels, not just pixels with flux)
        main_logger.info(f"  Computing IPCC class area for {tile_id}: {uu.timestr()}")
        area_3d = pixel_area_ha_2d_subset.expand_dims(year=analysis_years.values)
        area_zero_template = xr.zeros_like(area_3d)
        class_area_by_das = [
            subset_context_layer(class_area_das[key], contextual_layer_config[key]["pattern"],
                                 west, south, east, north, area_zero_template, tile_id, main_logger)
            for key in CLASS_AREA_CONTEXT_KEYS
        ]

        aligned_area = xr.align(area_3d, adm0_da.reset_coords(drop=True), *class_area_by_das, join="override")
        area_3d, adm0_for_area_da, class_area_by_das = aligned_area[0], aligned_area[1], list(aligned_area[2:])

        class_area_results = xarray_reduce(
            area_3d,
            adm0_for_area_da,
            *class_area_by_das,
            area_3d["year"],
            func="sum",
            expected_groups=(cn.gadm_adm0_ids,)
                            + tuple(contextual_layer_config[key]["codes"] for key in CLASS_AREA_CONTEXT_KEYS)
                            + (area_3d.year.values,),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        class_area_coord_dict = zsu.convert_to_coord_dict(class_area_results, tile_id, main_logger)
        class_area_df = zsu.create_ipcc_class_area_df(class_area_coord_dict, state_node_df, tile_id, main_logger)
        class_area_df_name = f"ipcc_class_area_{tile_id}_v{lu_model_version}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}"
        class_area_df.to_parquet(f"{local_class_area_folder}/{class_area_df_name}.parquet")
        main_logger.info(f"  IPCC class area table for {tile_id} has {len(class_area_df.index)} rows: {uu.timestr()}")

        del class_area_results, class_area_coord_dict, class_area_df, area_3d
        gc.collect()

        main_logger.info(f"  Computing {tile_id}: {uu.timestr()}")
        results = xarray_reduce(
            flux_cube_subset,
            *context_tile_das,
            flux_cube_subset["year"],
            func='sum',
            expected_groups=tuple(contextual_layer_config[key]["codes"] for key in selected_contextual_layers)
                            + (flux_cube_subset.year.values,),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        # Contextual layers to use to merge pixel_area against other analysis layers (to calculate flux/ha)
        contextual_layer_names = [contextual_layer_config[key]["pattern"] for key in selected_contextual_layers] + ['year']

        main_logger.info(f"  Done computing {tile_id}: {uu.timestr()}")
        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)

        del results
        gc.collect()

        df = zsu.create_df(coord_dict, state_node_df, contextual_layer_names, tile_id, 'vegetation', main_logger)

        del coord_dict
        gc.collect()

        # Land state nodes -> land_state_broad_class and tall_veg_type; rows that become identical are summed
        df = zsu.collapse_flux_df(df, tile_id, main_logger)

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

    # Matches tile_ids like 00N_110E in file names
    tile_id_regex = r"\d{2}[NS]_\d{3}[EW]"

    # Shuts down the cluster because all further processing is done locally.
    # Uses the cluster object this run is connected to. terminate_cluster.terminate_cluster() calls coiled.Cluster(name=...),
    # which creates a new cluster if the old one has already stopped. A failed shutdown doesn't stop the combining below.
    try:
        if not run_local and cluster is not None:
            client.close()
            cluster.shutdown()
            main_logger.info(f"Cluster {cluster_name} shut down: {uu.timestr()}")
    except Exception as e:
        main_logger.warning(f"Could not shut down cluster {cluster_name} ({e}). Check the Coiled dashboard.")

    ### Combines IPCC class area tables (total area per country per IPCC land use class per year)
    class_area_files = sorted(
        str(local_class_area_folder / f)
        for f in os.listdir(local_class_area_folder)
        if f.startswith("ipcc_class_area_") and re.search(tile_id_regex, f) and is_run_output(f, zonal_stats_description)
    )
    if class_area_files:
        main_logger.info(f"Combining {len(class_area_files)} IPCC class area tables: {uu.timestr()}")
        class_area_long, class_area_wide = zsu.combine_ipcc_class_area_tables(
            [pd.read_parquet(f) for f in class_area_files], main_logger)
        class_area_name = f'ipcc_class_area_by_country_v{lu_model_version}_{zonal_stats_description}_{time.strftime("%Y%m%d_%H_%M_%S")}'
        class_area_long.to_parquet(f"{local_zonal_stats_folder}/{class_area_name}.parquet")
        class_area_long.to_csv(f"{local_zonal_stats_folder}/{class_area_name}.csv", index=False)
        class_area_wide.to_csv(f"{local_zonal_stats_folder}/{class_area_name}_wide.csv", index=False)
        main_logger.info(f"IPCC class area tables written to {local_zonal_stats_folder}/{class_area_name}*")
    else:
        main_logger.info("No IPCC class area tables found.")

    # Collects all tile parquet files.
    # Only tile-level files (with a tile_id in the name), so combined outputs from earlier runs aren't counted twice.
    parquet_files = sorted(
        str(local_zonal_stats_folder / f)
        for f in os.listdir(local_zonal_stats_folder)
        if f.startswith("ipcc_lulucf_zonal_stats_") and re.search(tile_id_regex, f) and is_run_output(f, zonal_stats_description)
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

        # Combines all the tile-level dfs in the list into a single df
        main_logger.info(f"Combining dataframes: {uu.timestr()}")
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)

        main_logger.info(f"Rows in combined dataframe: {len(combined_df.index)}")
        if len(combined_df.index) != total_rows:
            main_logger.warning("Sum of row count from individual tables and row count in combined table do not match!")
        main_logger.info(combined_df.head())

        combined_df_name = f'ipcc_lulucf_zonal_stats_v{lu_model_version}_{zonal_stats_description}_{time.strftime("%Y%m%d_%H_%M_%S")}'
        combined_df.to_parquet(f"{local_zonal_stats_folder}/{combined_df_name}.parquet")
        if len(combined_df.index) < 900_000:  # Only writes combined file to Excel if it's not giant
            combined_df.to_csv(f"{local_zonal_stats_folder}/{combined_df_name}.csv", index=False)

        # Converts combined df from long to wide
        combined_wide_df = zsu.create_wide_df(combined_df, main_logger)

        combined_wide_df_name = f'ipcc_lulucf_zonal_stats_v{lu_model_version}_{zonal_stats_description}_wide_{time.strftime("%Y%m%d_%H_%M_%S")}'
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

    parser.add_argument("-vid", "--veg_input_date", default=VEG_INPUT_DATE, help=f"Vegetation model run date (default {VEG_INPUT_DATE})")
    parser.add_argument("-tid", "--lulucf_input_date", default=LULUCF_INPUT_DATE, help=f"Total LULUCF model run date (default {LULUCF_INPUT_DATE})")
    parser.add_argument("-lid", "--lu_input_date", default=LU_INPUT_DATE, help=f"IPCC land use run date (default {LU_INPUT_DATE})")
    parser.add_argument("-sid", "--soc_input_date", default=SOC_INPUT_DATE, help=f"Mineral soil (SOC) zarr run date (default {SOC_INPUT_DATE})")

    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W S E N")
    parser.add_argument('-fv', '--first_variables_to_process', type=int, help='Number of variables to process from raw mega-zarr (for testing)')
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="1x1 fishnet shapefile (default: CHUNK_SHAPEFILE_URI if no bounding box is given)")
    parser.add_argument("-ft", "--first_tiles_to_process", type=int)
    parser.add_argument("-mt", "--model_type", default="standard")

    parser.add_argument("-vmpd", "--veg_model_path_description", default="global")
    parser.add_argument("-tmpd", "--lulucf_model_path_description", default="global")
    parser.add_argument("-lmpd", "--lu_model_path_description", default="global")
    parser.add_argument("-smpd", "--soc_model_path_description", default="global")
    parser.add_argument("-zd", "--zonal_stats_description", required=True)

    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=False, help='local path for model chunk stats to check if tile had any pixels in it, and skip if empty')
    parser.add_argument("-ln", "--log_note")

    # Analysis and contextual layer selection
    parser.add_argument("-d", "--datasets", nargs="+", choices=DATASET_OPTIONS, default=DATASET_OPTIONS,
                        help=f"Dataset(s) to run zonal stats on. Options: {', '.join(DATASET_OPTIONS)}. Default: all")
    parser.add_argument("-g", "--gases", nargs="+", choices=GAS_OPTIONS, default=GAS_OPTIONS,
                        help=f"Gas(es) to include. Options: {', '.join(GAS_OPTIONS)}. Default: all. "
                             f"Removals and mineral soil are CO2 only and are included for CO2 or all_gases.")
    parser.add_argument("-f", "--fluxes", nargs="+", choices=FLUX_OPTIONS, default=FLUX_OPTIONS,
                        help=f"Flux type(s) to include. Options: {', '.join(FLUX_OPTIONS)}. Default: all")
    parser.add_argument("-cl", "--contextual_layers", nargs="+", choices=CONTEXTUAL_LAYER_OPTIONS + ["all"], default=DEFAULT_CONTEXTUAL_LAYERS,
                        help=f"Contextual layer(s) to group by, or 'all'. Options: {', '.join(CONTEXTUAL_LAYER_OPTIONS)}. "
                             f"Default: {', '.join(DEFAULT_CONTEXTUAL_LAYERS)}")
    parser.add_argument( "--tile_ids_file", required=False, help="Path to txt file containing 10x10 tile IDs to process, one per line.")

    args = parser.parse_args()

    if "all" in args.contextual_layers:
        args.contextual_layers = CONTEXTUAL_LAYER_OPTIONS

    # Uses the fishnet shapefile set in the script unless a bounding box or a different shapefile is given.
    # Bounding box runs keep using the bounding box (needed for sub-tile tests).
    if not args.bounding_box and not args.chunk_shapefile_uri:
        args.chunk_shapefile_uri = CHUNK_SHAPEFILE_URI

    # Stops before connecting to the cluster if a run date is still a placeholder
    for arg_name in ["veg_input_date", "lulucf_input_date", "lu_input_date", "soc_input_date"]:
        if not re.fullmatch(r"\d{8}", str(getattr(args, arg_name))):
            parser.error(f"{arg_name} is '{getattr(args, arg_name)}'. Set it in the Run settings block or pass it on the command line.")

    main(
        cluster_name=args.cluster_name,
        lulucf_input_date=args.lulucf_input_date,
        veg_input_date=args.veg_input_date,
        lu_input_date=args.lu_input_date,
        soc_input_date=args.soc_input_date,
        model_type=args.model_type,
        lulucf_model_path_description=args.lulucf_model_path_description,
        veg_model_path_description=args.veg_model_path_description,
        lu_model_path_description=args.lu_model_path_description,
        soc_model_path_description=args.soc_model_path_description,
        zonal_stats_description=args.zonal_stats_description,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        first_variables_to_process=args.first_variables_to_process,
        first_tiles_to_process=args.first_tiles_to_process,
        model_chunk_stats_table_name=args.model_chunk_stats_table_name,
        log_note=args.log_note,
        tile_ids_file=args.tile_ids_file,
        datasets=args.datasets,
        gases=args.gases,
        fluxes=args.fluxes,
        contextual_layers=args.contextual_layers,
    )

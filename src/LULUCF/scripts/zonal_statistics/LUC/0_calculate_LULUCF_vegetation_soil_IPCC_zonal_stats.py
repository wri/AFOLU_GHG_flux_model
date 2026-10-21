"""
Purpose: Calculates zonal statistics for LULUCF flux outputs using IPCC land use classifications and other contextual datasets.
The script reads vegetation, LULUCF, mineral soil (SOC), organic soil, and IPCC land use zarrs, converts per-hectare fluxes to per-pixel totals,
and summarizes emissions, removals, net flux, and area by IPCC land use class + transition + summary, land state, administrative unit, and other
contextual layers for each 10x10 deg tile. Tile-level outputs are written as parquet tables and optionally combined into global zonal statistics tables.

Contextual layers are also filters: flox drops any pixel whose value in a grouping layer isn't in that layer's expected codes.
So totals only match the old script when the default contextual layers are used. Runs with other -cl selections can have
different totals for the same analysis layer.

Total area by IPCC class, country, etc. (all pixels whether or not they have flux) is calculated by a separate script:
    0_calculate_IPCC_class_area_zonal_stats.py

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

10x10:
python -m src.utilities.create_cluster -n 10 -m 128 -cn IPCC_flux_zonal_stats --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats -bb 100 10 110 20 -zd 20N_100W

Central and East Africa (8 10x10 tiles):
python -m src.utilities.create_cluster -n 25 -m 32 -cn IPCC_flux_zonal_stats__Central_Africa --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__Central_Africa -bb 13 -14 44 -3 -zd Central_Africa

Sweden:
python -m src.utilities.create_cluster -n 15 -m 32 -cn IPCC_flux_zonal_stats__Sweden --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__Sweden -bb 10 55 25 70 -zd Sweden -d vegetation mineral_soil organic_soil -g CO2 all_gases

EU27 (30 10x10 tiles):
python -m src.utilities.create_cluster -n 50 -m 32 -cn IPCC_flux_zonal_stats__EU27 --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__EU27 -bb -11 34 35 71 -zd EU27 -d vegetation mineral_soil organic_soil -g CO2 all_gases

Global run:
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_flux_zonal_stats__global --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__global -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

global_south
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_flux_zonal_stats__global_south --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__global_south --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_south.txt -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

global_north
python -m src.utilities.create_cluster -n 25 -m 64 -cn IPCC_flux_zonal_stats__global_north --zonal_stats
python -m src.LULUCF.scripts.zonal_statistics.LUC.0_calculate_LULUCF_vegetation_soil_IPCC_zonal_stats -cn IPCC_flux_zonal_stats__global_north --tile_ids_file /mnt/c/GIS/AFOLU_flux_model/land_use/global_north.txt -zd global --log_note "Zonal stats for IPCC land use model v1.0.0."

Notes:
    - Took 5 minutes to run for 10x10 degree area (00N_110E) with 25 workers (17 credits, $1).
        Analysis layers = LULUCF emissions, LULUCF removals, and LULUCF net flux, vegetation net flux, pixel area
        Contextual layers = IPCC class, node, change and summary, land state node, continent/ecozone, primary/IFL
    - Toook 19 minutes to run for Central Africa (8 tiles) with 50 workers (128 credits, $7)

Selecting analysis and contextual layers (defaults are set in the Run settings block):
    -d  / --datasets            LULUCF, vegetation, mineral_soil, organic_soil     (default: all)
    -g  / --gases               CO2, non_CO2, all_gases                            (default: all)
    -f  / --fluxes              emissions, removals, netflux                       (default: all)
    -cl / --contextual_layers   any of CONTEXTUAL_LAYER_OPTIONS, or all            (default: DEFAULT_CONTEXTUAL_LAYERS)

    Contextual layers multiply: flox indexes every combination of their codes, and the product has to fit in a 64-bit integer.
    "all" is too many for one pass and the script stops before loading anything.

    An analysis layer is used if it matches ALL of the selected dataset, gas and flux filters.
    Removals and mineral soil fluxes are CO2 only, so they count as both CO2 and all_gases.
    Organic soil only has emissions.

    Examples:
        # Only CO2 from vegetation and LULUCF
        ... -d LULUCF vegetation -g CO2 -zd vegetation_LULUCF_CO2
        # Only soils, all gases, with default contextual layers
        ... -d mineral_soil organic_soil -zd soils
        # Protected areas, Indigenous lands and KBAs with country and IPCC class (totals can differ from default runs; see above)
        ... -cl adm0 IPCC_class WDPA landmark KBA -zd protected_areas
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
# constants_and_names, so updating cn doesn't change which zarrs are read. Same versions as the validated IPCC_LULUCF_zonal_stats.py run.
VEG_MODEL_VERSION = "1.0.5"
LULUCF_MODEL_VERSION = "1.0.0"
IPCC_LU_VERSION = "1.0.0"
SOC_MODEL_VERSION = "1.0.1"
ORGANIC_SOIL_MODEL_VERSION = "1.0.1"

LULUCF_INPUT_DATE = "20260614"    # Total LULUCF model run date
VEG_INPUT_DATE = "20260130"       # Vegetation model run date
SOC_INPUT_DATE = "20260611"       # Mineral soil (SOC) zarr run date
LU_INPUT_DATE = "20260617"        # IPCC land use run date
#TODO: Add organic soil run date

# Starting composite primary forest zarr. Primary forest is also a filter (pixels with values other than 0 or 1 are left out),
# so it changes totals, not only the primary/non-primary split.
#   - PRIMARY_FOREST_ZARR_PATH: read this zarr. The default is the one the validated IPCC_LULUCF_zonal_stats.py run read
#     (Sweden -38.3 Mt CO2e/yr). It's in a different folder from the path cn builds now.
#   - Set PRIMARY_FOREST_ZARR_PATH = None to build the path from cn with PRIMARY_FOREST_RUN_DATE instead.
PRIMARY_FOREST_ZARR_PATH = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/starting_composite_primary_forest/version_1_0_5__standard__global/2015/presence/zarr/4000_pixels/20260210/starting_composite_primary_forest.zarr"
# TODO: Not in sync with cn
PRIMARY_FOREST_RUN_DATE = "20260210"

# Sheet of the state node lookup table. Its land state codes are the expected groups for flox, so pixels with land states that
# aren't in the sheet are left out of the zonal stats. It has to match the vegetation zarr being summarized.
#   - Default: v105_20260601, the sheet cn had for the validated IPCC_LULUCF_zonal_stats.py run.
#   - cn now has v106_20261006. Set STATE_NODE_SHEET = cn.sheet to use it.
STATE_NODE_SHEET = "v105_20260601"

# The vegetation zarr was written before the cn vegetation patterns got their veg_ prefix:
#   cn:   veg_gross_emissions__all_C_pools__all_gases__MgCO2e, veg_land_state_node, ...
#   zarr: gross_emissions__all_C_pools__all_gases__MgCO2e,     land_state_node, ...
# True:  removes veg_ from the start of cn names before looking them up in the zarrs.
# False: uses the cn names exactly. Set to False once the zarr names match cn.
DROP_VEG_PREFIX = True

# Position of each organic soil interval in the organic soil zarr's year dimension (5 positions).
# Only 2016 onward is used: 2016-2020 analysis years use the 2016_2020 position, 2021-2024 use the 2021_2024 position.
#TODO: Update with strings instead of positions
ORGANIC_SOIL_INTERVAL_POSITIONS = {'2016_2020': -2, '2021_2024': -1}

# Analysis layer filters (all used by default)
DATASET_OPTIONS = ["LULUCF", "vegetation", "mineral_soil", "organic_soil"]
GAS_OPTIONS = ["CO2", "non_CO2", "all_gases"]
FLUX_OPTIONS = ["emissions", "removals", "netflux"]

# Contextual layers that can be used, in the order they are grouped by.
CONTEXTUAL_LAYER_OPTIONS = [
    "adm0", "land_state", "WDPA", "cont_eco", "landmark", "primary_forest", "KBA", "watersheds", "drivers",
    "forest_age", "BRA_biomes", "managed_land_CAN", "managed_land_USA",
    "IPCC_class", "IPCC_node", "IPCC_change", "IPCC_summary",
]

# Contextual layers are also filters (see docstring), so these have to stay the same for totals to match the old script.
DEFAULT_CONTEXTUAL_LAYERS = [
    "adm0", "land_state", "cont_eco", "primary_forest", #"drivers",
    "IPCC_class", "IPCC_node", "IPCC_change", "IPCC_summary",
]

########################################################################################################################
# Utilities
########################################################################################################################
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
        return {
            line.strip()
            for line in f
            if line.strip()
        }


# Name of a cn variable in the zarrs: removes a leading veg_ if DROP_VEG_PREFIX is True (see Run settings)
def zarr_var_name(name):
    if DROP_VEG_PREFIX and name.startswith("veg_"):
        return name[len("veg_"):]
    return name


# Returns the zarr name of the first name in names that is a variable in ds, or None. Exact matches only.
def find_var_name(ds, names):
    for name in names:
        if zarr_var_name(name) in ds.data_vars:
            return zarr_var_name(name)
    return None


# Returns the land state variable from the vegetation zarr
def get_land_state_da(veg_ds):
    land_state_var = find_var_name(veg_ds, [cn.land_state_pattern])
    if land_state_var is None:
        raise KeyError(f"{zarr_var_name(cn.land_state_pattern)} is not in the vegetation zarr (DROP_VEG_PREFIX = {DROP_VEG_PREFIX}). "
                       f"Variables in it: {sorted(veg_ds.data_vars)}")
    return veg_ds[land_state_var]


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


# All analysis layers that can be run, tagged by dataset, flux type and gas.
#   names: cn variable names (with units) to look for, in order. Exact matches only, after DROP_VEG_PREFIX is applied.
#   output_name: analysis_layer name in the output tables, if different from the zarr variable
# Fluxes that only involve CO2 (removals, mineral soil) are tagged as both CO2 and all_gases, because CO2 is all of their gases.
def get_analysis_layer_config():

    def with_units(pattern):
        return zu.add_units_year_to_pattern(pattern, 9999)[0]

    # The old script used "LULUCF_" + the vegetation pattern. Its cn patterns didn't have the veg_ prefix, so that
    # gave e.g. LULUCF_gross_emissions__..., which is the current cn LULUCF pattern. "LULUCF_" + the current veg_ pattern is
    # kept as a second option in case a LULUCF zarr uses it.
    def lulucf_names(veg_pattern, lulucf_pattern_attr):
        names = []
        lulucf_pattern = getattr(cn, lulucf_pattern_attr, None)
        if lulucf_pattern:
            names.append(with_units(lulucf_pattern))
        old_style_name = with_units(f"LULUCF_{veg_pattern}")
        if old_style_name not in names:
            names.append(old_style_name)
        return names

    co2_is_all_gases = ["CO2", "all_gases"]

    return [
        # LULUCF totals
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["CO2"],           "names": lulucf_names(cn.gross_emis_all_C_pools_CO2_only_pattern, "gross_emis_all_C_pools_CO2_only_LULUCF_pattern")},
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["non_CO2"],       "names": lulucf_names(cn.gross_emis_all_C_pools_non_CO2_only_pattern, "gross_emis_all_C_pools_non_CO2_only_LULUCF_pattern")},
        {"dataset": "LULUCF", "flux": "emissions", "gases": ["all_gases"],     "names": lulucf_names(cn.gross_emis_all_C_pools_all_gases_pattern, "gross_emis_all_C_pools_all_gases_LULUCF_pattern")},
        {"dataset": "LULUCF", "flux": "removals",  "gases": co2_is_all_gases,  "names": lulucf_names(cn.gross_removals_all_C_pools_pattern, "gross_removals_all_C_pools_LULUCF_pattern")},
        {"dataset": "LULUCF", "flux": "netflux",   "gases": ["CO2"],           "names": lulucf_names(cn.net_flux_all_C_pools_CO2_only_pattern, "net_flux_all_C_pools_CO2_only_LULUCF_pattern")},
        {"dataset": "LULUCF", "flux": "netflux",   "gases": ["all_gases"],     "names": lulucf_names(cn.net_flux_all_C_pools_all_gases_pattern, "net_flux_all_C_pools_all_gases_LULUCF_pattern")},

        # Vegetation
        {"dataset": "vegetation", "flux": "emissions", "gases": ["CO2"],          "names": [with_units(cn.gross_emis_all_C_pools_CO2_only_pattern)]},
        {"dataset": "vegetation", "flux": "emissions", "gases": ["non_CO2"],      "names": [with_units(cn.gross_emis_all_C_pools_non_CO2_only_pattern)]},
        {"dataset": "vegetation", "flux": "emissions", "gases": ["all_gases"],    "names": [with_units(cn.gross_emis_all_C_pools_all_gases_pattern)]},
        {"dataset": "vegetation", "flux": "removals",  "gases": co2_is_all_gases, "names": [with_units(cn.gross_removals_all_C_pools_pattern)]},
        {"dataset": "vegetation", "flux": "netflux",   "gases": ["CO2"],          "names": [with_units(cn.net_flux_all_C_pools_CO2_only_pattern)]},
        {"dataset": "vegetation", "flux": "netflux",   "gases": ["all_gases"],    "names": [with_units(cn.net_flux_all_C_pools_all_gases_pattern)]},

        # Mineral soil (CO2 only)
        # {"dataset": "mineral_soil", "flux": "emissions", "gases": co2_is_all_gases, "names": [with_units(cn.SOC_loss_min_soil_extent_pattern)]},
        # {"dataset": "mineral_soil", "flux": "removals",  "gases": co2_is_all_gases, "names": [with_units(cn.SOC_gain_min_soil_extent_pattern)]},
        {"dataset": "mineral_soil", "flux": "netflux",   "gases": co2_is_all_gases, "names": [with_units(cn.SOC_net_min_soil_extent_pattern)]},
        #TODO: Uncomment min soil gain/loss

        # Organic soil (emissions only). Names already include units.
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["CO2"],       "names": ["burned_co2_Mg_CO2_ha_yr"],     "output_name": "organic_soil_burned_co2_Mg_CO2_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["CO2"],       "names": ["drained_co2_Mg_CO2_ha_yr"],    "output_name": "organic_soil_drained_co2_Mg_CO2_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["all_gases"], "names": ["burned_total_Mg_CO2e_ha_yr"],  "output_name": "organic_soil_burned_total_Mg_CO2e_ha_yr"},
        {"dataset": "organic_soil", "flux": "emissions", "gases": ["all_gases"], "names": ["drained_total_Mg_CO2e_ha_yr"], "output_name": "organic_soil_drained_total_Mg_CO2e_ha_yr"},
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
        # No rename because it's created by a different process where the variable is named starting_composite_primary_forest.
        # Opened from build_zarr_paths() in main() because the cn path has version and chunk size placeholders.
        "primary_forest":   {"pattern": cn.starting_composite_primary_forest_pattern, "source": "static", "zarr_path": None, "rename": False, "codes": cn.composite_primary_codes},
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


# True if a file name is an output of this zonal stats run: ..._v<version>_<description>_<timestamp>.parquet.
# Exact match, so -zd Sweden doesn't pick up files from runs like Sweden_class_area.
def is_run_output(file_name, zonal_stats_description):
    return re.search(rf"_v\d+_\d+_\d+_{re.escape(zonal_stats_description)}_\d{{8}}_\d{{2}}_\d{{2}}_\d{{2}}\.parquet$",
                     file_name) is not None



# NEW: Builds every zarr path the script reads, so the diagnostic script reads exactly the same zarrs.
# The old script built these with cn.veg_outputs_path_mega_zarr and an older zu.create_zarr_path that took an interval argument ("annual").
# Neither exists in the current utilities: the current zu.create_zarr_path only fills placeholders, so annual_intervals is added here.
def build_zarr_paths(model_type, veg_input_date, lulucf_input_date, lu_input_date, soc_input_date,
                     veg_model_path_description, lulucf_model_path_description, lu_model_path_description,
                     soc_model_path_description, main_logger):

    source_zarr_chunk_size = cn.chunk_dims  # 4000x4000
    # Versions from Run settings, not cn
    veg_model_version = underscore(VEG_MODEL_VERSION)
    lulucf_model_version = underscore(LULUCF_MODEL_VERSION)
    soc_model_version = underscore(SOC_MODEL_VERSION)
    organic_soil_model_version = underscore(ORGANIC_SOIL_MODEL_VERSION)

    paths = {}

    # Vegetation zarr (mega_zarr/annual_intervals/)
    paths["vegetation"] = zu.create_zarr_path(cn.veg_outputs_path_zarr.replace("/zarr/", "/mega_zarr/annual_intervals/"),
                                              source_zarr_chunk_size, veg_input_date, main_logger,
                                              model_version=veg_model_version, model_type=model_type,
                                              model_path_description=veg_model_path_description)

    # LULUCF zarr (same as the old script)
    paths["LULUCF"] = (
        f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/"
        f"outputs_LULUCF_totals/"
        f"LULUCF_version_{lulucf_model_version}_standard__{lulucf_model_path_description}"
        f"__veg_v{veg_model_version}"
        f"__org_soil_v{organic_soil_model_version}"
        f"__min_soil_v{soc_model_version}"
        f"/zarr/annual_intervals/{source_zarr_chunk_size}_pixels/{lulucf_input_date}/LULUCF_annual.zarr"
    )

    # IPCC land use zarr (mega_zarr/annual_intervals/). Built here because the cn path has cn's IPCC land use version in it.
    # The land use folder uses the version with dots (land_use/1.0.0/).
    paths["land_use"] = zu.create_zarr_path(
        f"{cn.full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/land_use/{IPCC_LU_VERSION}/mega_zarr/annual_intervals/CHUNK_SIZE_pixels/RUN_DATE/land_use_zarr.zarr",
        source_zarr_chunk_size, lu_input_date, main_logger)

    # Starting composite primary forest zarr: PRIMARY_FOREST_ZARR_PATH if set, otherwise built from cn
    # (the cn path has version and chunk size placeholders)
    if PRIMARY_FOREST_ZARR_PATH:
        paths["primary_forest"] = PRIMARY_FOREST_ZARR_PATH
    else:
        paths["primary_forest"] = zu.create_zarr_path(
            cn.starting_composite_primary_forest_zarr_path.replace(cn.starting_composite_primary_forest_run_date, PRIMARY_FOREST_RUN_DATE),
            source_zarr_chunk_size, PRIMARY_FOREST_RUN_DATE, main_logger,
            model_version=veg_model_version, model_type="standard", model_path_description="global")

    # Soil zarrs (not in the old script)
    paths["mineral_soil"] = zu.create_zarr_path(cn.SOC_path_zarr, source_zarr_chunk_size, soc_input_date, main_logger,
                                                model_version=soc_model_version, model_type=model_type,
                                                model_path_description=soc_model_path_description)
    # The cn organic soil path has cn's organic soil version in it; it's replaced with ORGANIC_SOIL_MODEL_VERSION
    cn_organic_version_part = f"/version_{cn.organic_soil_model_version_underscore}/"
    if cn_organic_version_part not in cn.organic_soil_zarr_path:
        raise ValueError(f"Can't find {cn_organic_version_part} in cn.organic_soil_zarr_path ({cn.organic_soil_zarr_path}) to set the organic soil version")
    paths["organic_soil"] = cn.organic_soil_zarr_path.replace(cn_organic_version_part, f"/version_{organic_soil_model_version}/")

    return paths


# The body of main() is the body of main() in IPCC_LULUCF_zonal_stats.py, in the same order and with the same variable names.
# Lines that differ are marked:
#   # NEW:     not in the old script (dataset/gas/flux/contextual layer selection, soil layers, extra checks and logging)
#   # CHANGED: changed to work with the current utilities or the current zarr variable names, without changing values
def main(cluster_name, lulucf_input_date, veg_input_date, lu_input_date, soc_input_date, model_type="standard", lulucf_model_path_description="global",
         veg_model_path_description="global", lu_model_path_description="global", soc_model_path_description="global", zonal_stats_description="global",
         chunk_shapefile_uri=False, bounding_box=None, first_variables_to_process=None, first_tiles_to_process=None,
         model_chunk_stats_table_name=None, log_note=None, tile_ids_file=None,
         datasets=None, gases=None, fluxes=None, contextual_layers=None):

    # NEW: layer selection (defaults: all datasets, gases and fluxes; the old script's contextual layers)
    datasets = datasets or DATASET_OPTIONS
    gases = gases or GAS_OPTIONS
    fluxes = fluxes or FLUX_OPTIONS
    contextual_layers = contextual_layers or DEFAULT_CONTEXTUAL_LAYERS

    ### Step 1: Preparation

    # Model stage being run
    stage = "IPCC_land_use_zonal_statistics"
    veg_model_version = underscore(VEG_MODEL_VERSION)
    lulucf_model_version = underscore(LULUCF_MODEL_VERSION)
    lu_model_version = underscore(IPCC_LU_VERSION)

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
    main_logger.info(f"Vegetation version: {VEG_MODEL_VERSION}")
    main_logger.info(f"Vegetation model path description: {veg_model_path_description}")
    main_logger.info(f"Vegetation input date: {veg_input_date}")
    main_logger.info(f"LULUCF version: {LULUCF_MODEL_VERSION}")
    main_logger.info(f"LULUCF model path description: {lulucf_model_path_description}")
    main_logger.info(f"LULUCF input date: {lulucf_input_date}")
    main_logger.info(f"IPCC land use model version: {IPCC_LU_VERSION}")
    main_logger.info(f"IPCC land use path description: {lu_model_path_description}")
    main_logger.info(f"IPCC land use input date: {lu_input_date}")
    main_logger.info(f"Running sub-tile test area: {sub_tile_test}")
    main_logger.info(f"Mineral soil version: {SOC_MODEL_VERSION}, path description: {soc_model_path_description}, input date: {soc_input_date}")
    main_logger.info(f"Organic soil version: {ORGANIC_SOIL_MODEL_VERSION}")
    main_logger.info(f"Datasets: {datasets}; gases: {gases}; fluxes: {fluxes}")
    main_logger.info(f"Contextual layers: {contextual_layers}")
    main_logger.info(f"DROP_VEG_PREFIX: {DROP_VEG_PREFIX}")
    main_logger.info(f"State node lookup table sheet: {STATE_NODE_SHEET} (cn.sheet: {cn.sheet})")   # NEW

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
    # CHANGED: chosen with -d/-g/-f instead of a fixed list. With -d LULUCF vegetation -g all_gases -f emissions removals netflux,
    # the old script's four layers (LULUCF emissions, removals and net flux; vegetation net flux) are all included.
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
    main_logger.info(f"Variables to run zonal stats on ({len(selected_analysis_layers)} out of {n_selected} selected):")
    for layer in selected_analysis_layers:
        main_logger.info(f"  {layer['dataset']} | {layer['flux']} | {'/'.join(layer['gases'])} | {[zarr_var_name(n) for n in layer['names']]}")

    # NEW: contextual layers to group by, in a fixed order (the old script's order for its layers)
    contextual_layer_config = get_contextual_layer_config()
    selected_contextual_layers = [key for key in CONTEXTUAL_LAYER_OPTIONS if key in contextual_layers]
    main_logger.info(f"Contextual layers to group by: {selected_contextual_layers}")
    if selected_contextual_layers != DEFAULT_CONTEXTUAL_LAYERS:
        main_logger.warning(
            f"Contextual layers differ from the defaults ({DEFAULT_CONTEXTUAL_LAYERS}). Contextual layers also filter pixels "
            f"(pixels with values outside a layer's expected codes are dropped), so totals from this run won't necessarily "
            f"match runs with the default layers or IPCC_LULUCF_zonal_stats.py.")

    if first_tiles_to_process:
        tile_ids_to_process = unique_tile_ids[0:first_tiles_to_process]
    else:
        tile_ids_to_process = unique_tile_ids
    main_logger.info(f"tile_ids to perform zonal stats on: {tile_ids_to_process} ({len(tile_ids_to_process)} out of {len(unique_tile_ids)})")

    # lat-long chunk size for source zarr
    source_zarr_chunk_size = cn.chunk_dims  # 4000x4000

    # CHANGED: zarr paths are built in build_zarr_paths() (see the note there)
    zarr_paths = build_zarr_paths(model_type, veg_input_date, lulucf_input_date, lu_input_date, soc_input_date,
                                  veg_model_path_description, lulucf_model_path_description, lu_model_path_description,
                                  soc_model_path_description, main_logger)
    veg_zarr_path = zarr_paths["vegetation"]
    lulucf_zarr_path = zarr_paths["LULUCF"]
    lu_zarr_path = zarr_paths["land_use"]

    main_logger.info(f"Vegetation zarr path: {veg_zarr_path}")
    main_logger.info(f"Zonal stats from zarr ({source_zarr_chunk_size} pixel chunks): {lulucf_zarr_path}")
    main_logger.info(f"IPCC land use zarr path: {lu_zarr_path}")                                # NEW
    main_logger.info(f"Primary forest zarr path: {zarr_paths['primary_forest']}")              # NEW
    main_logger.info(f"Mineral soil zarr path: {zarr_paths['mineral_soil']}")                  # NEW
    main_logger.info(f"Organic soil zarr path: {zarr_paths['organic_soil']}")                  # NEW

    # Creates dataframe of state_node codes and meanings
    # CHANGED: sheet from STATE_NODE_SHEET (Run settings), which decides which land states are counted
    state_node_df = zsu.create_state_node_df(cn.state_node_lookup_table_local, cn.state_node_lookup_table_s3, STATE_NODE_SHEET)
    # Same as the old script. NOTE: [0] + np.array(...) adds 0 to every code; it does NOT add a 0 code.
    # So pixels with land state 0 aren't in the expected groups and are left out, as in the old script.
    node_codes = [0] + np.array(list(state_node_df['land_state']), dtype=np.uint32)
    contextual_layer_config["land_state"]["codes"] = node_codes   # NEW

    # NEW: flox indexes every combination of contextual layer codes (x years); the product has to fit in a 64-bit integer
    group_counts = {key: len(contextual_layer_config[key]["codes"]) for key in selected_contextual_layers}
    n_group_combinations = math.prod(group_counts.values()) * len(cn.veg_outputs_years)
    main_logger.info(f"Contextual layer code counts: {group_counts}; possible combinations: {n_group_combinations:.2e}")
    if n_group_combinations > 2**62:
        raise ValueError(
            f"Too many contextual layer combinations for flox ({n_group_combinations:.2e}; limit ~{2**62:.1e}). "
            f"Codes per layer: {group_counts}. Drop some layers with many codes, or run them in separate passes with different -zd descriptions."
        )

    local_zonal_stats_folder = Path(zonal_stats_folder())   # CHANGED: folder name uses IPCC_LU_VERSION from Run settings
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
    cont_eco_xr = xr.open_zarr(cn.cont_eco_zarr_path, consolidated=False).rename_vars(band_data=cn.cont_eco_zstats_pattern)
    # CHANGED: primary forest path from build_zarr_paths (the cn path has placeholders). No rename because it's created by a
    # different process where the variable is named starting_composite_primary_forest
    composite_primary_xr = xr.open_zarr(zarr_paths["primary_forest"], consolidated=False)
    drivers_xr = xr.open_zarr(cn.drivers_of_loss_zarr_path, consolidated=False).rename_vars(band_data=cn.drivers_of_loss_pattern)

    # Open LULUCF, vegetation, and land use zarrs
    # CHANGED: the LULUCF zarr is only opened if a LULUCF layer is selected
    selected_dataset_names = {layer["dataset"] for layer in selected_analysis_layers}
    lulucf_ds = xr.open_zarr(lulucf_zarr_path, consolidated=False) if "LULUCF" in selected_dataset_names else xr.Dataset()
    veg_ds = xr.open_zarr(veg_zarr_path, consolidated=False)
    lu_ds = xr.open_zarr(lu_zarr_path, consolidated=False)

    # CHANGED: variables are found by name (after DROP_VEG_PREFIX), for the LULUCF and vegetation layers that were selected.
    # Same as the old script's  [v for v in vars_to_process if v in ds.data_vars]  otherwise.
    lulucf_vars_to_process = []
    veg_vars_to_process = []
    soil_layers_to_process = []   # NEW: (dataset, zarr variable, output name)
    soil_sources = {}             # NEW
    for layer in selected_analysis_layers:
        if layer["dataset"] == "LULUCF":
            ds = lulucf_ds
        elif layer["dataset"] == "vegetation":
            ds = veg_ds
        else:
            if layer["dataset"] not in soil_sources:
                soil_sources[layer["dataset"]] = xr.open_zarr(zarr_paths[layer["dataset"]], consolidated=False)
            ds = soil_sources[layer["dataset"]]
        var_name = find_var_name(ds, layer["names"])
        if var_name is None:
            main_logger.warning(f"None of {[zarr_var_name(n) for n in layer['names']]} are in the {layer['dataset']} zarr "
                                f"(DROP_VEG_PREFIX = {DROP_VEG_PREFIX}). Skipping it. Variables in that zarr: {sorted(ds.data_vars)}")
            continue
        if layer["dataset"] == "LULUCF":
            lulucf_vars_to_process.append(var_name)
        elif layer["dataset"] == "vegetation":
            veg_vars_to_process.append(var_name)
        else:
            soil_layers_to_process.append((layer["dataset"], var_name, layer.get("output_name", var_name)))

    main_logger.info(f"LULUCF variables to process: {lulucf_vars_to_process}")
    main_logger.info(f"Vegetation variables to process: {veg_vars_to_process}")
    main_logger.info(f"Soil variables to process: {soil_layers_to_process}")   # NEW
    if not (lulucf_vars_to_process or veg_vars_to_process or soil_layers_to_process):
        raise ValueError("None of the selected analysis layers were found in their zarrs")   # NEW

    lulucf_selected = lulucf_ds[lulucf_vars_to_process]
    veg_selected = veg_ds[veg_vars_to_process]

    ds_selected_analysis_vars = xr.merge([lulucf_selected, veg_selected], compat="override")

    # Round and align
    main_logger.info(f"Rounding coordinates: {uu.timestr()}")

    adm0_xr = zsu.round_coords(adm0_xr)
    reference = zsu.round_coords(pixel_area_xr[cn.pixel_area_zstats_pattern])
    cont_eco_xr = zsu.round_coords(cont_eco_xr)
    composite_primary_xr = zsu.round_coords(composite_primary_xr)
    drivers_xr = zsu.round_coords(drivers_xr)
    land_state_node = zsu.round_coords(get_land_state_da(veg_ds))   # CHANGED: land state found by name (after DROP_VEG_PREFIX)
    lu_ds = zsu.round_coords(lu_ds)
    if veg_vars_to_process or lulucf_vars_to_process:   # NEW: an empty dataset (soil layers only) has no coordinates to round
        ds_selected_analysis_vars = zsu.round_coords(ds_selected_analysis_vars)


    main_logger.info(f"Cropping: {uu.timestr()}")
    adm0_aligned = zsu.safe_crop(adm0_xr, reference)
    pixel_area_aligned = reference
    cont_eco_aligned = zsu.safe_crop(cont_eco_xr, reference)
    composite_primary_aligned = zsu.safe_crop(composite_primary_xr, reference)
    drivers_aligned = zsu.safe_crop(drivers_xr, reference)
    land_state_node_aligned = zsu.safe_crop(land_state_node, reference)
    lu_ds = zsu.safe_crop(lu_ds, reference)
    if veg_vars_to_process or lulucf_vars_to_process:   # NEW
        ds_selected_analysis_vars_aligned = zsu.safe_crop(ds_selected_analysis_vars, reference)
    else:
        ds_selected_analysis_vars_aligned = ds_selected_analysis_vars

    # List of selected variable names (already aligned and cropped)
    main_logger.info(f"Selecting datasets: {uu.timestr()}")
    selected_datasets = list(ds_selected_analysis_vars_aligned.data_vars)

    # NEW: analysis years come from the vegetation zarr if only soil layers are selected (no LULUCF or vegetation layers)
    analysis_year_coord = ds_selected_analysis_vars_aligned.year if selected_datasets else veg_ds.year

    # Expand pixel_area to match shape of flux variables
    pixel_area_expanded = pixel_area_aligned.expand_dims(year=analysis_year_coord)

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

    # NEW: soil layers (not in the old script), mapped onto the analysis years, then multiplied by pixel area the same way
    if soil_layers_to_process:
        analysis_year_values = [int(cn.veg_outputs_years[0] + int(i)) for i in analysis_year_coord.values]   # Same conversion as zsu.create_df
        main_logger.info(f"Analysis years for soil layers: {analysis_year_values}")
        soil_year_mapping = {
            "mineral_soil": (cn.mineral_soil_year_map, cn.SOC_density_intervals),
            "organic_soil": (cn.organic_soil_year_map, ORGANIC_SOIL_INTERVAL_POSITIONS),
        }
        for dataset, var_name, output_name in soil_layers_to_process:
            main_logger.info(f"Preparing {dataset} layer {var_name}: {uu.timestr()}")
            da = zsu.safe_crop(zsu.round_coords(soil_sources[dataset][var_name]), reference)
            year_map, interval_labels = soil_year_mapping[dataset]
            da = map_layer_to_analysis_years(da, analysis_year_coord, analysis_year_values, year_map, interval_labels, var_name, main_logger)
            da = da.assign_coords(x=x_coords, y=y_coords)
            flux_scaled = (da * pixel_area_layer).astype("float32").transpose(*pixel_area_layer.dims)
            flux_layers.append(flux_scaled)
            analysis_layer_names.append(output_name)

    # Total IPCC-context area
    flux_layers.append(pixel_area_layer)
    analysis_layer_names.append("pixel_area_ha")

    # NEW: layers from the soil zarrs can have non-dimension coordinates (e.g., spatial_ref, band) that stop them from being stacked.
    # Those coordinates are only dropped when soil layers are used. Values aren't changed either way.
    if soil_layers_to_process:
        flux_layers = [layer.reset_coords(drop=True) for layer in flux_layers]

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
    analysis_years = analysis_year_coord   # CHANGED: same as ds_selected_analysis_vars_aligned.year when LULUCF or vegetation layers are selected

    # IPCC contextual layers aligned to LULUCF flux years, 2016-2024
    ipcc_class = lu_ds[cn.IPCC_class_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_node = lu_ds[cn.IPCC_node_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_change = lu_ds[cn.IPCC_change_pattern].isel(year=slice(1, 10)).assign_coords(year=analysis_years)
    ipcc_summary = (lu_ds[cn.IPCC_summary_pattern].isel(year=0, drop=True).expand_dims(year=analysis_years).assign_coords(year=analysis_years))

    # NEW: optional contextual layers (WDPA, landmark, KBA, watersheds, forest age, BRA biomes, managed land), opened only if selected
    # with -cl. Prepared the same way as the static layers above (round, crop).
    # The layers opened above for every run are skipped here.
    layers_opened_above = ["adm0", "land_state", "cont_eco", "primary_forest", "drivers",
                           "IPCC_class", "IPCC_node", "IPCC_change", "IPCC_summary"]
    extra_context = {}
    for key in selected_contextual_layers:
        cfg = contextual_layer_config[key]
        if key in layers_opened_above:
            continue
        if cfg["source"] == "static":
            ctx_xr = xr.open_zarr(cfg["zarr_path"], consolidated=False)
            if cfg["rename"]:
                ctx_xr = ctx_xr.rename_vars(band_data=cfg["pattern"])
            extra_context[key] = zsu.safe_crop(zsu.round_coords(ctx_xr), reference)[cfg["pattern"]]
        elif cfg["source"] == "forest_age":
            age_var = find_var_name(veg_ds, [cn.forest_age_output_pattern])
            if age_var is None:
                raise KeyError(f"{cn.forest_age_output_pattern} not in the vegetation zarr. Variables in it: {sorted(veg_ds.data_vars)}")
            forest_age_aligned = zsu.safe_crop(zsu.round_coords(veg_ds[age_var]), reference)
            extra_context[key] = zsu.categorize_age(forest_age_aligned).astype(np.uint8).rename(cfg["pattern"])
        else:
            raise ValueError(f"Unknown contextual layer source {cfg['source']} for {key}")


    # Part 3: Do zonal stats tile by tile

    main_logger.info(f"Starting zonal stats: {uu.timestr()}")
    tiles_processed = 0  # The number of tiles actually processed (since some are skipped)
    tiles_skipped = []   # NEW

    for i, tile_id in enumerate(tile_ids_to_process):
        main_logger.info(f"Processing {tile_id} ({i + 1}/{len(tile_ids_to_process)}): {uu.timestr()}")
        tile_start_time = time.time()

        # Skips if an output for this tile_id and zonal stats description already exists (to not repeat that tile if restarting the zonal stats).
        # CHANGED: only files from this -zd description count (the old script skipped a tile if any file in the folder had its tile_id).
        # NEW: logged as a warning, because the skipped tile's numbers come from the earlier run's file, whatever script version wrote it.
        existing = set(os.listdir(local_zonal_stats_folder))
        existing_for_tile = sorted(f for f in existing if tile_id in f and is_run_output(f, zonal_stats_description))
        if existing_for_tile:
            main_logger.warning(f"  Skipping {tile_id}; output already exists from an earlier run with -zd {zonal_stats_description}: "
                                f"{existing_for_tile}. Its values are NOT recalculated. Delete those files or use a new -zd to recalculate.")
            tiles_skipped.append(tile_id)
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
        cont_eco_aligned_subset = cont_eco_aligned.sel(x=slice(west, east), y=slice(north, south))
        composite_primary_aligned_subset = composite_primary_aligned.sel(x=slice(west, east), y=slice(north, south))
        drivers_aligned_subset = drivers_aligned.sel(x=slice(west, east), y=slice(north, south))
        land_state_node_aligned_subset = land_state_node_aligned.sel(x=slice(west, east), y=slice(north, south))
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

        if cont_eco_aligned_subset[cn.cont_eco_zstats_pattern].sizes.get("x", 0) == 0 or cont_eco_aligned_subset[
            cn.cont_eco_zstats_pattern].sizes.get("y", 0) == 0:
            cont_eco_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.cont_eco_zstats_pattern)
            main_logger.info(f"  {cn.cont_eco_zstats_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            cont_eco_da = cont_eco_aligned_subset[cn.cont_eco_zstats_pattern]

        if drivers_aligned_subset[cn.drivers_of_loss_pattern].sizes.get("x", 0) == 0 or drivers_aligned_subset[
            cn.drivers_of_loss_pattern].sizes.get("y", 0) == 0:
            drivers_da = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(
                cn.drivers_of_loss_pattern)
            main_logger.info(f"  {cn.drivers_of_loss_pattern} not in {tile_id}. Creating xarray of all 0s.")
        else:
            drivers_da = drivers_aligned_subset[cn.drivers_of_loss_pattern]

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

        # CHANGED: land state is renamed to cn.land_state_pattern. The old script didn't need to because the zarr variable had the
        # same name as the cn pattern then; with DROP_VEG_PREFIX the zarr name differs, and create_df looks for the cn name.
        land_state_node_aligned_subset = land_state_node_aligned_subset.rename(cn.land_state_pattern)

        # NEW: optional contextual layers (only if selected with -cl). 0s if the layer doesn't reach the tile.
        extra_context_das = {}
        for key, da in extra_context.items():
            pattern = contextual_layer_config[key]["pattern"]
            subset = da.sel(x=slice(west, east), y=slice(north, south))
            if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
                extra_context_das[key] = xr.zeros_like(flux_cube_subset.isel(analysis_layer=0, drop=True)).rename(pattern)
                main_logger.info(f"  {pattern} not in {tile_id}. Creating xarray of all 0s.")
            else:
                extra_context_das[key] = subset.rename(pattern)

        # NEW: the contextual layers to group by, in CONTEXTUAL_LAYER_OPTIONS order. With the default layers this is exactly the
        # old script's list (adm0, land state, cont_eco, primary forest, drivers, IPCC class, node, change, summary), in its order.
        tile_context_das = {
            "adm0": adm0_da,
            "land_state": land_state_node_aligned_subset,
            "cont_eco": cont_eco_da,
            "primary_forest": composite_primary_da,
            "drivers": drivers_da,
            "IPCC_class": ipcc_class_da,
            "IPCC_node": ipcc_node_da,
            "IPCC_change": ipcc_change_da,
            "IPCC_summary": ipcc_summary_da,
            **extra_context_das,
        }
        context_by_das = [tile_context_das[key] for key in selected_contextual_layers]

        # Final alignment
        main_logger.info(f"  Aligning {tile_id}: {uu.timestr()}")
        aligned = xr.align(
            flux_cube_subset,
            pixel_area_expanded_subset,
            *context_by_das,      # CHANGED: the selected contextual layers (the old script's list with the defaults)
            join="override"
        )
        flux_cube_subset, pixel_area_expanded_subset, context_by_das = aligned[0], aligned[1], list(aligned[2:])

        main_logger.info(f"  Computing {tile_id}: {uu.timestr()}")
        results = xarray_reduce(
            flux_cube_subset,
            *(
                *context_by_das,  # CHANGED: the selected contextual layers (the old script's list with the defaults)
                flux_cube_subset["year"]
            ),
            func='sum',
            expected_groups=(
                *[contextual_layer_config[key]["codes"] for key in selected_contextual_layers],  # CHANGED: the old script's codes with the defaults
                flux_cube_subset.year.values,
            ),
            group_dims=["year"],
            reindex=ReindexStrategy(blockwise=False, array_type=ReindexArrayType.SPARSE_COO),
            fill_value=0
        ).compute()

        # Contextual layers to use to merge pixel_area against other analysis layers (to calculate flux/ha)
        # CHANGED: the selected contextual layers (the old script's list with the defaults)
        contextual_layers_names = [contextual_layer_config[key]["pattern"] for key in selected_contextual_layers] + ['year']

        main_logger.info(f"  Done computing {tile_id}: {uu.timestr()}")
        coord_dict = zsu.convert_to_coord_dict(results, tile_id, main_logger)

        del results
        gc.collect()

        df = zsu.create_df(coord_dict, state_node_df, contextual_layers_names, tile_id, 'vegetation', main_logger)

        del coord_dict
        gc.collect()

        main_logger.info(f"  Rows in {tile_id} dataframe: {len(df.index)}: {uu.timestr()}")
        main_logger.info(f"  Saving {tile_id} output table: {uu.timestr()}")
        tile_df_name = (f"ipcc_lulucf_zonal_stats_{tile_id}_v{lu_model_version}_{zonal_stats_description}_{time.strftime('%Y%m%d_%H_%M_%S')}")
        df.to_parquet(f"{local_zonal_stats_folder}/{tile_df_name}.parquet")

        # Clean up at end of tile
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
    # NEW: reminder that skipped tiles weren't recalculated
    if tiles_skipped:
        main_logger.warning(f"{len(tiles_skipped)} tile(s) were skipped and NOT recalculated because outputs with -zd {zonal_stats_description} "
                            f"already existed: {tiles_skipped}. The combined tables below use those earlier outputs.")

    # Matches tile_ids like 00N_110E in file names
    tile_id_regex = r"\d{2}[NS]_\d{3}[EW]"

    # CHANGED: shuts down the cluster this run is connected to. terminate_cluster.terminate_cluster() calls coiled.Cluster(name=...),
    # which creates a new cluster if the old one has already stopped. A failed shutdown doesn't stop the combining below.
    try:
        if not run_local and cluster is not None:
            client.close()
            cluster.shutdown()
            main_logger.info(f"Cluster {cluster_name} shut down: {uu.timestr()}")
    except Exception as e:
        main_logger.warning(f"Could not shut down cluster {cluster_name} ({e}). Check the Coiled dashboard.")

    # Collects all tile parquet files
    # CHANGED: only tile-level files from this -zd description. The old script combined every parquet file in the folder with
    # "zonal_stats_" in its name, including combined tables and other runs' tiles, which can count the same tile more than once.
    parquet_files = sorted(
        str(local_zonal_stats_folder / f)
        for f in os.listdir(local_zonal_stats_folder)
        if f.startswith("ipcc_lulucf_zonal_stats_") and re.search(tile_id_regex, f) and is_run_output(f, zonal_stats_description)
    )

    if not parquet_files:
        main_logger.info("No tile parquet files found.")
        return

    # NEW: warns if a tile has more than one file for this description (its values would be counted more than once)
    files_per_tile = pd.Series([re.search(tile_id_regex, os.path.basename(f)).group(0) for f in parquet_files]).value_counts()
    duplicated_tiles = files_per_tile[files_per_tile > 1]
    if not duplicated_tiles.empty:
        main_logger.warning(f"Some tiles have more than one output file for -zd {zonal_stats_description}, so they'd be counted more than "
                            f"once in the combined tables: {duplicated_tiles.to_dict()}. Delete the extra files and rerun.")

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
            # CHANGED: the old script added each table's rows to total_rows again here, so the row-count check below always warned

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


    end_time = time.time()
    main_logger.info(f"Finished zonal stats, took {round(end_time - prep_start_time)} seconds: {uu.timestr()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run combined LULUCF flux zonal stats by IPCC land use zarr context.")
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
    parser.add_argument("--tile_ids_file", required=False, help="Path to txt file containing 10x10 tile IDs to process, one per line.")

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

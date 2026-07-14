"""
Purpose: This script assigns each pixel to an IPCC land use (LU) category using the entire GLCLU observation timeseries and other contxtual information to disentangle land use from land cover (i.e. mangrove, tree crop, rangelands, temporarily unstocked forest, etc). 
LU classifcations are assigned annualy (2015 - 2024), LU change is determined for each interval (2015_2016 - 2023_2024), and all LUs during the timseries are summarized per pixel(i.e if starts as Forest and ends as Cropland during timeseries, LU summary is 32)

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
Indonesia
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -bb 119 -6 120 -5 -cs 1 --run_local --create_zarr --run_date 20268888

Coiled small tests (0.25x0.25 deg chunk):
python -m src.utilities.create_cluster -n 1 -m 8 -cn IPCC_land_use
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use -bb 119.5 -5.75 119.75 -5.5 -cs 0.25 --run_date 20268888

Coiled small tests (1x1 deg chunk):
python -m src.utilities.create_cluster -n 1 -m 8 -cn IPCC_land_use_1x1
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_1x1 -bb 119 -6 120 -5 -cs 1 --no_upload --no_stats --run_date 20268888
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_1x1 -bb 119 -6 120 -5 -cs 1 --create_zarr --run_date 20268888

Coiled test (10x10 deg chunk):
python -m src.utilities.create_cluster -n 50 -m 8 -cn IPCC_land_use_10x10
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_10x10 -bb 110 -10 120 0 -cs 10 --create_zarr --run_date 20268888

Full run:
python -m src.utilities.create_cluster -n 200 -m 8 -cn IPCC_land_use
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -cs 10 --create_zarr --run_date 20260617 --log_note "This is a global run for IPCC land use model v1.0.0 (2015-2024)"
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use --chunk_ids_to_skip /mnt/c/GIS/AFOLU_flux_model/land_use/processed_1x1.txt -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -cs 10 --create_zarr --run_date 20260617 --log_note "This is a global run for IPCC land use model v1.0.0 (2015-2024). Part 2"


Notes:
    - Took 2 minutes to run for 0.25 degree chunk locally
    - Took 3 minutes to run for 1 degree chunk in coiled with no stats, no zarr, and no upload
    - Took 7 minutes to run for 1 degree chunk locally
    - Took 35 minutes to run for 10 degree area in 1 degree chunks in coiled using 100 workers (30 credits)
    - Took 19.5 hours to run globally (1x1 chunk step) + 3.5 hours to make 10 x 10 outputs (4480 credits, $375)

TODO:
Potentially switch from regex to numba for faster performance
Remove skip existing 1x1 logic? Or change to 10x10 deg tile creation only?
Low resource usage during 10x10 tile creation step. Resize cluster or run separately than with 1x1 chunk run.
Add IFL/primary rule: If Built, Crop, or TCL + antrhopgenic driver then conversion, else Forest remaining Forest.
If any vegetation fluxes with water assume wetland.
"""

import argparse
import concurrent.futures
import gc
import os
import psutil
import time
import pandas as pd
import numpy as np
import re
import rasterio

from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster
from src.utilities import terminate_cluster

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"

print_lu_transition = True

def read_chunk_ids_file(path):
    with open(path, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }

# Returns boolean values for whether a pixel is planted forest or tree crop
def get_sdpt_status(sdpt_type):
    sdpt_planted_forest = not np.isnan(sdpt_type) and int(sdpt_type) == 1
    sdpt_tree_crop      = not np.isnan(sdpt_type) and int(sdpt_type) == 2
    return sdpt_planted_forest, sdpt_tree_crop

# Checks if an oil palm planting year transition happens in the timeseries.
def has_planting_transition(lu_dict):
    planting_year = lu_dict.get("planting_year", 0)
    return (planting_year > min(cn.years_annual) and planting_year <= max(cn.years_annual))

# Gets oil palm planting year index in timeseries.
def planting_idx(planting_year):
    if planting_year <= min(cn.years_annual):
        return 0
    if planting_year > max(cn.years_annual):
        return None
    return int(planting_year - min(cn.years_annual))

# Checks if there was TCL up to 5 years before oil palm planting year. If so, considered F->C transition.
def tcl_prior_to_planting(tcl_year, planting_year):
    n_years = 5     # number of years between TCL and oil palm planting year allowed to be considered F -> C conversion
    return (tcl_year > 0 and planting_year > 0 and planting_year - n_years <= tcl_year < planting_year)

# Skips stable pixels that don't have a LU transition exception
def has_lu_exception(driver, tcl_year, pre_2000_plantation, planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, gpw_cultiv_grass):
    return (driver != 0 or tcl_year != 0 or pre_2000_plantation == 1 or planting_year > 0 or sdpt_oil_palm == 1 or sdpt_tree_crop or sdpt_planted_forest or gmw_mangrove or gpw_cultiv_grass)

# Checks that there is only one land use transition during the entire timeseries. If not, prints pixel information.
def check_single_lu_transition(lu_dict, lu_ts):
    transition_count = sum(lu_ts[i] != lu_ts[i - 1] for i in range(1, len(lu_ts)))

    if transition_count > 1:
        debug_info = {
            k: v for k, v in lu_dict.items()
            if k not in {"node_codes"}
        }

        print(
            f"\nWARNING: More than one LU transition detected.\n"
            f"transition_count: {transition_count}\n"
            f"lu_ts: {lu_ts}\n"
            f"debug_info: {debug_info}\n"
        )

# Gets bounds from 1x1 filename in s3
def bounds_from_1x1_filename(raster_path):
    filename = os.path.basename(raster_path).replace(".tif", "")
    parts = filename.split("__")

    if len(parts) < 3:
        raise ValueError(f"Unexpected 1x1 raster filename format: {filename}")

    bounds_str = parts[1]
    bounds = [float(x) for x in bounds_str.split("_")]

    return bounds  # W, S, E, N

# Mosaics uploaded 1x1 IPCC rasters into a 10x10 array. Keeps zeros where 1x1 chunks are missing.
def mosaic_ipcc_1x1_rasters(key, tile_rasters, tile_bounds, logger_worker):
    if not tile_rasters:
        raise ValueError("No tile rasters supplied for mosaic")

    # Get dtype from first raster
    with rasterio.open(tile_rasters[0]) as src:
        dtype = src.dtypes[0]

    mosaic_array = np.zeros((cn.full_raster_dims, cn.full_raster_dims), dtype=dtype)
    tile_w, tile_s, tile_e, tile_n = tile_bounds

    for raster_path in tile_rasters:
        chunk_bounds = bounds_from_1x1_filename(raster_path)
        chunk_w, chunk_s, chunk_e, chunk_n = chunk_bounds

        row0 = int(round((tile_n - chunk_n) / cn.resolution))
        col0 = int(round((chunk_w - tile_w) / cn.resolution))

        with rasterio.open(raster_path) as src:
            arr = src.read(1)

        row1 = row0 + arr.shape[0]
        col1 = col0 + arr.shape[1]

        mosaic_array[row0:row1, col0:col1] = arr

    lu.print_and_log(f"Mosaicked {key}: {len(tile_rasters)} rasters into array {mosaic_array.shape}: {uu.timestr()}",False, logger_worker)

    return mosaic_array

def expected_1x1_filename_for_output_dir(chunk, output_dir_1x1):
    bounds_str = uu.boundstr(chunk)
    tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])

    output_dir = output_dir_1x1.rstrip("/")
    folder_name = output_dir.split("/")[-4] if len(output_dir.split("/")) >= 4 else output_dir

    year_or_interval = output_dir.rstrip("/").split("/")[-3]
    parent = output_dir.rstrip("/").split("/")[-4]

    if year_or_interval == "2015_2024":
        key = cn.IPCC_summary_pattern
    elif "_" in year_or_interval:
        key = f"{cn.IPCC_change_pattern}_{year_or_interval}"
    elif cn.IPCC_class_path in output_dir or cn.IPCC_class_pattern in output_dir:
        key = f"{cn.IPCC_class_pattern}_{year_or_interval}"
    elif cn.IPCC_node_path in output_dir or cn.IPCC_node_pattern in output_dir:
        key = f"{cn.IPCC_node_pattern}_{year_or_interval}"
    else:
        raise ValueError(f"Could not infer expected output key from {output_dir_1x1}")

    return f"{tile_id}__{bounds_str}__{key}.tif"

#  Lists each 1x1 output folder once and stores existing basenames.
def build_existing_1x1_output_index(output_dir_list_1x1, main_logger):
    existing_by_output_dir = {}

    for output_dir_1x1 in output_dir_list_1x1:
        raster_paths, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_dir_1x1)
        existing_by_output_dir[output_dir_1x1] = {os.path.basename(path) for path in raster_paths}
        main_logger.info(f"Found {file_count} existing rasters in {output_dir_1x1}: {uu.timestr()}")

    return existing_by_output_dir

def chunk_has_all_1x1_outputs_fast(chunk, output_dir_list_1x1, existing_by_output_dir):
    for output_dir_1x1 in output_dir_list_1x1:
        filename = expected_1x1_filename_for_output_dir(chunk, output_dir_1x1)

        if filename not in existing_by_output_dir.get(output_dir_1x1, set()):
            return False

    return True

def filter_chunks_missing_1x1_outputs_fast(chunk_list, output_dir_list_1x1, main_logger):
    existing_by_output_dir = build_existing_1x1_output_index(output_dir_list_1x1, main_logger)

    chunks_to_run = []
    skipped_chunks = []

    for chunk in chunk_list:
        if chunk_has_all_1x1_outputs_fast( chunk, output_dir_list_1x1, existing_by_output_dir):
            skipped_chunks.append(chunk)
        else:
            chunks_to_run.append(chunk)
    main_logger.info(f"Existing 1x1 output check: skipping {len(skipped_chunks)} chunks; running {len(chunks_to_run)} chunks.")

    return chunks_to_run, skipped_chunks

# Lists each 1x1 output folder once and stores full raster paths.
def build_raster_paths_by_output_dir(output_dir_list_1x1, main_logger):
    raster_paths_by_output_dir = {}

    for output_dir_1x1 in output_dir_list_1x1:
        raster_paths, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_dir_1x1)
        raster_paths_by_output_dir[output_dir_1x1] = raster_paths
        main_logger.info(f"Found {file_count} rasters for mosaicking in {output_dir_1x1}: {uu.timestr()}")

    return raster_paths_by_output_dir

def make_10x10_tile_list(chunk_list):
    tile_ids = sorted(set(uu.xy_to_tile_id(chunk[0], chunk[3]) for chunk in chunk_list))
    return tile_ids


# Move general utilities from here up to UU
#######################################################################################################################
""" Regex-based land-use reclassification rules
These rules replace the default land use classes.
1. Convert annual GLAD LC values to LU tokens.
2. Use regex to identify token patterns for exceptions.
3. Reclassify token arrays and assign a matching node_code array.

Node codes used here:
1) Settlements and Infrastructure:
    10: "Built from GLAD data"
    11: "Built following tall vegetation loss before built LC"
    12: "Built after first built LC"

2) Cropland:
    20: "Crop from GLAD data"
    21: "Crop from oil palm extent or planting year"
    22: "Crop from SDPT tree crop extent (not oil palm)"
    23: "Crop following tall vegetation loss before crop LC"
    24: "Crop from TCL + permanent agriculture driver"

3) Forest:
    30 : "Forest from GLAD tall vegetation"
    31 : "Forest from SDPT planted forest extent"
    32 : "Forest from GMW mangrove extent"
    333: "Forest from shifting cultivation driver"
    334: "Forest from logging driver"
    335: "Forest from wildfire driver"
    337: "Forest from natural disturbance driver"
    34 : "Unstocked forest after TCL and before oil palm planting"
    353: "Forest from mixed tall/short vegetation rule"
    357: "Forest from vegetation/water transition rule"
    358: "Forest from mixed snow/ice rule"
    39 : "Forest from majority years rule"

4) Grassland:
    40 : "Grass from GLAD short vegetation"
    41 : "Grass from TCL + permanent agriculture driver + GPW cultivated grassland extent"
    430: "Grass from TCL + unknown driver"
    432: "Grass from TCL + hard commodities driver"
    436: "Grass from TCL + settlements/infrastructure driver"
    44 : "Grass prior to oil palm establishment"
    453: "Grass from mixed tall/short vegetation rule"
    457: "Grass from vegetation/water transition rule"
    458: "Grass from mixed snow/ice rule"
    49 : "Grass from majority years rule"

5) Wetland:
    50: "Wetland from GLAD data"
    51: "Wetland from water/wetland/built transition rule"
    52: "Wetland from vegetation/water transition rule"
    53: "Wetland from bare/ice to water/wetland transition rule"
    59: "Wetland from majority years in mixed water rule"

6) Other Land:
    60: "Bare from GLAD data"
    61: "Bare from mixed bare + tall/short vegetation rule"
    62: "Bare from mixed snow/ice rule"
    69: "Bare from majority years rule"

    70: "Water from GLAD data"
    79: "Water from majority years in mixed water rule"

    80: "Snow/ice from GLAD data"
    89: "Snow/ice from majority years rule"
"""

# IPCC Land use hierarchy: Settlements > Cropland > Forest Land > Grassland > Wetlands > Other
# Default GLAD LC numeric values
settlement_lc   = {250}                                         # Built up
cropland_lc     = {244}                                         # Cropland
forest_lc       = set(range(27, 49)) | set(range(127, 149))     # Tall vegetation
grass_lc        = set(range(5, 27)) | set(range(105, 127))      # Short veg
wetland_lc      = set(range(200, 205))                          # Wetland
bare_lc         = set(range(0, 5)) | set(range(100, 105))       # Bare
water_lc        = set(range(205, 208)) | {254}                  # Open water
ice_lc          = {241}                                         # Snow/ice

# Lookup table to go from GLAD LC code -> default LU token
lc_token_map = {
    **{v: "S" for v in settlement_lc},
    **{v: "C" for v in cropland_lc},
    **{v: "F" for v in forest_lc},
    **{v: "G" for v in grass_lc},
    **{v: "W" for v in wetland_lc},
    **{v: "B" for v in bare_lc},
    **{v: "O" for v in water_lc},
    **{v: "I" for v in ice_lc},
}

# Function to get land use token per GLCLU numeric value (tokens used for regex exception rules)
# Returns U for any unknown numeric values (i.e. 223 and 255)
def token_for_lc(v):
    # if v not in lc_token_map:
    #     print(f"Unknown GLCLU code: {v}")
    return lc_token_map.get(v, "U")

# Node code values based on what exception was applied
node_code_map = {
    "built_glad": 10,
    "built_tall_veg_loss": 11,
    "built_post_s": 12,

    "crop_glad": 20,
    "crop_oil_palm": 21,
    "crop_sdpt_tree_crop": 22,
    "crop_post_c": 23,
    "crop_perm_ag_driver": 24,
    #"crop_glad_majority_years": 29,

    "forest_glad": 30,
    #"forest_glad_perm_ag_driver": 301,
    "forest_sdpt_planted_forest": 31,
    "forest_gmw_mangrove": 32,
    "forest_shift_cult_driver": 333,
    "forest_logging_driver": 334,
    "forest_wildfire_driver": 335,
    "forest_nat_dist_driver": 337,
    "forest_unstocked_pre_oil_palm": 34,
    # "forest_built_mix": 351,
    # "forest_crop_mix": 352,
    "forest_veg_mix": 353,
    "forest_water_mix": 357,
    "forest_ice_mix": 358,
    "forest_glad_majority_years": 39,

    "grass_glad": 40,
    "grass_gpw": 41,
    "grass_unknown_driver": 430,
    "grass_hard_commod_driver": 432,
    "grass_settlement_driver": 436,
    "grass_unstocked_pre_oil_palm": 44,
    # "grass_built_mix": 451,
    # "grass_crop_mix": 452,
    "grass_veg_mix": 453,
    "grass_water_mix": 457,
    "grass_ice_mix": 458,
    "grass_glad_majority_years": 49,

    "wetland_glad": 50,
    "wetland_water_built_mix": 51,
    "wetland_veg_water_mix": 52,
    "wetland_bare_ice_water_mix": 53,
    "wetland_glad_majority_years": 59,

    "bare_glad": 60,
    "bare_tall_short_mix": 61,
    "bare_ice_mix": 62,
    "bare_glad_majority_years": 69,

    "water_glad": 70,
    # "water_veg_water_mix": 71,
    "water_glad_majority_years": 79,

    "ice_glad": 80,
    # "ice_bare_mix": 81,
    "ice_glad_majority_years": 89,
}

# Default node codes before rules are applied
def default_node_code(token):
    if token == "S":
        return node_code_map["built_glad"]
    if token == "C":
        return node_code_map["crop_glad"]
    if token == "F":
        return node_code_map["forest_glad"]
    if token == "G":
        return node_code_map["grass_glad"]
    if token == "W":
        return node_code_map["wetland_glad"]
    if token == "B":
        return node_code_map["bare_glad"]
    if token == "O":
        return node_code_map["water_glad"]
    if token == "I":
        return node_code_map["ice_glad"]
    return None

# Function to override default values based on regex rules
def set_tokens(tokens, node_codes, indices, new_token, node_code, initial_tokens=None):
    for i in indices:
        old_token = initial_tokens[i] if initial_tokens is not None else tokens[i]
        tokens[i] = new_token

        # Only overwrite node code if LU token changed from original LC token
        if old_token != new_token:
            node_codes[i] = node_code

def apply_tokens(lu_dict, indices, new_token, node_code):
    set_tokens(lu_dict["tokens"], lu_dict["node_codes"], indices, new_token, node_code, lu_dict["initial_tokens"])

# # Select pre-transition token by count. Ties go to the earlier token in priority_order.
# def majority_token(tokens, candidates, priority_order):
#     present = [t for t in candidates if t in tokens]
#     return max(present, key=lambda t: (tokens.count(t), -priority_order.index(t)))

# Converts char tokens to final int values in LU map
lu_token_map = {
    "S": 1,
    "C": 2,
    "F": 3,
    "G": 4,
    "W": 5,
    "B": 6,
    "O": 7,
    "I": 8,
}

def apply_extent_rules(lu_dict):
    tokens = lu_dict["tokens"]

    crop_reclass_idx = [i for i, token in enumerate(tokens) if token in {"F", "G", "W", "B", "O", "I"}]
    forest_reclass_idx = [i for i, token in enumerate(tokens) if token in {"G", "W", "B", "O", "I"}]
    # May want to consider not including wetland? water? ice?

    # Get oil palm planting year
    crop_extent = lu_dict["sdpt_tree_crop"] or lu_dict["sdpt_oil_palm"]
    planting_year = lu_dict.get("planting_year", 0)
    planting_later = planting_year > min(cn.years_annual)

    # Crop is highest priority and extents are applied in this order: pre-2000 oil palm plantation -> Descals oil palm -> SDPT tree crop
    if lu_dict["pre_2000_plantation"]:
        apply_tokens(lu_dict, crop_reclass_idx, "C", node_code_map["crop_oil_palm"])
        return True
    if crop_extent and planting_later:
        return False  # Don't apply oil palm exception in SDPT extent if planting year hasn't happened yet
    if crop_extent:
        node_code = node_code_map["crop_oil_palm"] if lu_dict["sdpt_oil_palm"] else node_code_map["crop_sdpt_tree_crop"]
        apply_tokens(lu_dict, crop_reclass_idx, "C", node_code)
        return True

    # If no crop extent applies, forest extents are applied by this order: GMW mangrove -> SDPT planted forest
    if lu_dict["gmw_mangrove"]:
        apply_tokens(lu_dict, forest_reclass_idx, "F", node_code_map["forest_gmw_mangrove"])
        return True
    if lu_dict["sdpt_planted_forest"]:
        apply_tokens(lu_dict, forest_reclass_idx, "F", node_code_map["forest_sdpt_planted_forest"])
        return True
    return False


# Mix of LC classes -> built
def apply_built_transition(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    # Confusion between S/O/W in coastal areas.
    # Allow O/W -> S or S -> O/W only if both groups have >= 3 consecutive years and there is exactly one transition.
    if re.fullmatch(r"[OSW]+", token_seq):
        # O/W -> S: Water/Wetland to Settlement
        transition_match = re.fullmatch(r"([OW]{3,})(S{3,})", token_seq)
        if transition_match:
            transition_idx = transition_match.start(2)

            water_tokens = tokens[:transition_idx]
            if all(t == "O" for t in water_tokens):
                pre_token = "O"
                pre_node = node_code_map["water_glad_majority_years"]
            else:
                pre_token = "W"
                pre_node = node_code_map["wetland_water_built_mix"]

            apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node)
            apply_tokens(lu_dict, range(transition_idx, len(tokens)), "S", node_code_map["built_post_s"])
            return

        # S -> O/W: Settlement to Water/Wetland
        transition_match = re.fullmatch(r"(S{3,})([OW]{3,})", token_seq)
        if transition_match:
            transition_idx = transition_match.start(2)

            water_tokens = tokens[transition_idx:]
            if all(t == "O" for t in water_tokens):
                final_token = "O"
                final_node = node_code_map["water_glad_majority_years"]
            else:
                final_token = "W"
                final_node = node_code_map["wetland_water_built_mix"]

            apply_tokens(lu_dict, range(0, transition_idx), "S", node_code_map["built_glad"])
            apply_tokens(lu_dict, range(transition_idx, len(tokens)), final_token, final_node)
            return

    first_s_idx = tokens.index("S")
    apply_tokens(lu_dict, range(first_s_idx, len(tokens)),"S", node_code_map["built_post_s"])

    # Require at least 2 non-S LC classes
    non_s_classes = set(tokens) - {"S"}
    if len(non_s_classes) < 2:
        return

    pre_node_map = {
        "F": node_code_map["forest_glad_majority_years"],
        "G": node_code_map["grass_glad_majority_years"],
        "W": node_code_map["wetland_glad_majority_years"],
        "B": node_code_map["bare_glad_majority_years"],
        "O": node_code_map["water_glad_majority_years"],
        "I": node_code_map["ice_glad_majority_years"],
    }
    #Not including C because it often shows up post veg loss and prior to built. Assume misclassified as C in between if a mix.

    first_s_idx = token_seq.find("S")

    # Go down hierarchy.
    for candidate in ["F", "G", "W", "B", "O", "I"]:
        if candidate not in token_seq:
            continue

        pre_token = candidate

        # F uses first F loss; everything else uses first S.
        if candidate == "F":
            transition_match = re.search(r"F+[SCGWBOI]", token_seq)
            if not transition_match:
                transition_idx = first_s_idx
            else:
                transition_idx = transition_match.end() - 1
        else:
            transition_idx = first_s_idx

        apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node_map[pre_token])
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), "S", node_code_map["built_tall_veg_loss"])
        return
#TODO: Use TCL up to 5 years prior for F->S exception? Allow for S->O/W transitions but not O/W->S? Require 5 years of O/W for it to be considered permanently flooded land?


# Mix of 2 or more LC classes -> crop
def apply_crop_transition(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)

    pre_node_map = {
        "F": node_code_map["forest_glad_majority_years"],
        "G": node_code_map["grass_glad_majority_years"],
        "W": node_code_map["wetland_glad_majority_years"],
        "B": node_code_map["bare_glad_majority_years"],
        "O": node_code_map["water_glad_majority_years"],
        "I": node_code_map["ice_glad_majority_years"],
    }

    first_c_idx = token_seq.find("C")

    # Go down hierarchy. F uses first F loss; everything else uses first C.
    # If there is a mix of 2 or more non-built classes, it chooses the highest-priority class from this order:
    for candidate in ["F", "G", "W", "B", "O", "I"]:
        if candidate not in token_seq:
            continue

        pre_token = candidate

        if candidate == "F":
            transition_match = re.search(r"F+[CGWBOI]", token_seq)
            if transition_match:
                transition_idx = transition_match.end() - 1
            else:
                transition_idx = first_c_idx
        else:
            transition_idx = first_c_idx

        apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node_map[pre_token])
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), "C", node_code_map["crop_post_c"])

        return
#TODO: Use TCL up to 5 years prior for F->C exception? Allow for permanently flooded land like in the built transition rule?

# #Tall vegetation all years
# def apply_all_tall_veg(lu_dict):
#     tcl_prior = lu_dict["tcl_prior"]
#     driver = lu_dict["driver"]
#
#     tokens = lu_dict["tokens"]
#     all_idx = range(len(tokens))
#
#     # If TCL has occurred by the start of timeseries and the driver is permanent ag, assume tall veg is tree crops
#     if tcl_prior and driver == 1:
#         apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_glad_perm_ag_driver"])
#
#     # # If oil palm planting year in interval, allows for F -> C transitions assuming establishment of tree crops
#     # if has_planting_transition(lu_dict):
#     #     idx = planting_idx(lu_dict["planting_year"])
#     #     apply_tokens(lu_dict, range(idx, len(tokens)), "C", node_code_map["crop_oil_palm"])
#     #     return
#TODO: Do we want to allow for all F to be called anything other than forest?
# If all F and TCL prior to 2015 from permanent ag, assume this will be captured by SDPT + Descals?
# If all F and oil palm planting year during timeseries, allow F->C transition?

# Short vegetation all years
def apply_all_short_veg(lu_dict):
    tcl_prior = lu_dict["tcl_prior"]
    driver = lu_dict["driver"]
    planting_year = lu_dict["planting_year"]

    tokens = lu_dict["tokens"]
    all_idx = range(len(lu_dict["tokens"]))

    driver_to_forest_node = {
        3: node_code_map["forest_shift_cult_driver"],
        4: node_code_map["forest_logging_driver"],
        5: node_code_map["forest_wildfire_driver"],
        7: node_code_map["forest_nat_dist_driver"],
    }

    # If oil palm planting year occurs during interval: If TCL up to 5 years prior, assume F -> C transition. Otherwise, assume G -> C transition.
    if has_planting_transition(lu_dict):
        idx = planting_idx(planting_year)
        if tcl_prior_to_planting(lu_dict["tcl_year"], planting_year):
            pre_token = "F"
            pre_node = node_code_map["forest_unstocked_pre_oil_palm"]
            apply_tokens(lu_dict, range(0, idx), pre_token, pre_node)

        apply_tokens(lu_dict, range(idx, len(tokens)), "C", node_code_map["crop_oil_palm"])
        return

    # If TCL has occurred by the start of the timeseries and the driver is permanent ag and not in cultivated grass extent, assume crop the entire timeseries
    elif tcl_prior and driver == 1:
        if not lu_dict["gpw_cultiv_grass"]:
            apply_tokens(lu_dict, all_idx, "C", node_code_map["crop_perm_ag_driver"])
        else:
            apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_gpw"])

    # If TCL has occurred by the start of the timeseries and the driver is shifting cultivation, logging, wildfire, or other natural disturbances, assume unstocked forest the entire timeseries
    elif tcl_prior and driver in driver_to_forest_node:
        apply_tokens(lu_dict, all_idx, "F", driver_to_forest_node[driver])


# Mix of short veg, tall veg, and bare
def apply_tall_short_bare(lu_dict):
    tcl_prior = lu_dict["tcl_prior"]
    tcl_year = lu_dict["tcl_year"]
    tcl_any = tcl_year > 0
    driver = lu_dict["driver"]
    planting_year = lu_dict["planting_year"]

    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    driver_to_forest_node = {
        3: node_code_map["forest_shift_cult_driver"],
        4: node_code_map["forest_logging_driver"],
        5: node_code_map["forest_wildfire_driver"],
        7: node_code_map["forest_nat_dist_driver"],
    }

    driver_to_grass_node = {
        2: node_code_map["grass_hard_commod_driver"],
        6: node_code_map["grass_settlement_driver"],
    }

    # 1) Check if oil palm planting year occurs during interval (regardless of driver + TCL)
    if has_planting_transition(lu_dict):
        # If oil palm planting year in interval, use the first F -> G transition. Else, use oil palm planting year.
        transition_match = re.search(r"F+[GB]", token_seq)
        if transition_match:
            idx = transition_match.end() - 1
        else:
            idx = planting_idx(planting_year)
        pre_plant_tokens = tokens[:idx]

        # If F present before transition or TCL within 5 years before planting, consider it F -> C
        forest_before_planting = ("F" in pre_plant_tokens or tcl_prior_to_planting(tcl_year, planting_year))
        if forest_before_planting:
            apply_tokens(lu_dict, range(0, idx), "F", node_code_map["forest_unstocked_pre_oil_palm"])
        elif "G" in pre_plant_tokens:
            apply_tokens(lu_dict, range(0, idx), "G", node_code_map["grass_unstocked_pre_oil_palm"])
        else:
            apply_tokens(lu_dict, range(0, idx), "B", node_code_map["bare_tall_short_mix"])
        apply_tokens(lu_dict, range(idx, len(tokens)), "C", node_code_map["crop_oil_palm"])
        return

    # 2) If TCL occurred before the timeseries, use permanent agriculture driver to determine LU for all years.
        # If the driver is permanent ag and not in cultivated grass extent, assume crop. Else, assume grass.
    if tcl_prior and driver == 1:
        if not lu_dict["gpw_cultiv_grass"]:
            apply_tokens(lu_dict, all_idx, "C", node_code_map["crop_perm_ag_driver"])
        else:
            apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_gpw"])
        return

    # 3) If TCL during the timeseries, use permanent agriculture driver and first F -> G transition to determine LU transitions:
        # If the driver is permanent ag and not in cultivated grass extent, assume F -> C transition. Else, assume F -> G transition.
    if tcl_any and not tcl_prior and driver == 1:
        match = re.search(r"F+[GB]", token_seq)
        if match:
            transition_idx = match.end() - 1
            apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_veg_mix"])
            if lu_dict["gpw_cultiv_grass"]:
                apply_tokens(lu_dict, range(transition_idx, len(tokens)), "G", node_code_map["grass_gpw"])
            else:
                apply_tokens(lu_dict, range(transition_idx, len(tokens)), "C", node_code_map["crop_perm_ag_driver"])
            return

    # 4) If TCL in any year and driver is temporary, assume forest all years.
        # Temporary drivers are: shifting cultivation, logging, wildfire, and other natural disturbances
    if tcl_any and driver in driver_to_forest_node:
        apply_tokens(lu_dict, all_idx, "F", driver_to_forest_node[driver])
        return

    # 5) If TCL during timeseries, use hard commodities, settlements/ infrastructure, and unknown driver and an F->G/B transition.
    # There must be at least 3 consecutive Fs, and at least 3 consecutive G/Bs until the end to determine LU transitions:
    if tcl_any and not tcl_prior and driver not in {1, 3, 4, 5, 7}:
        terminal_match = re.search(r"F{3,}[GB]{3,}$", token_seq)

        if terminal_match:
            transition_match = re.search(r"F+[GB]", token_seq)     # Get the first F->G/B transition

            if transition_match:
                transition_idx = transition_match.end() - 1
                apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_veg_mix"])

                final_idx = range(transition_idx, len(tokens))
                final_tokens = tokens[transition_idx:]
                if "G" in final_tokens:
                    if driver in driver_to_grass_node:
                        final_node = driver_to_grass_node[driver]
                    else:
                        final_node = node_code_map["grass_unknown_driver"]
                    apply_tokens(lu_dict, final_idx, "G", final_node)
                else:
                    apply_tokens(lu_dict, final_idx, "B", node_code_map["bare_tall_short_mix"])
                return
    #TODO: Revisit this subrule.

    # 6) Otherwise use regex fallback if no oil palm and no TCL + driver.

    # If there is not at least 3 years F or 3 years G/B, not enough evidence for a true F -> G/B transition. Use majority land use instead.
    f_count = tokens.count("F")
    g_count = tokens.count("G")
    b_count = tokens.count("B")
    if g_count + b_count < 3:
        apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_glad_majority_years"])
        return
    elif f_count < 3:
        if g_count >= b_count:
            apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_glad_majority_years"])
        else:
            apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_glad_majority_years"])
        return
        #TODO: Make G if any G present?

    # Otherwise F -> G/B transition needs a terminal G/B phase that starts with 3 consecutive G/Bs, allows at most one F, and ends on G/B.
    # If any Gs in terminal phase, assume grass; otherwise assume bare.
    else:
        terminal_match = re.search(r"(?P<gb>[GB]{3,}(?:F?[GB]*)?)$", token_seq)

        # If no valid terminal G/B phase, set all years to F
        if not terminal_match:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_veg_mix"])
            return

        # Option to make number of G/Bs in terminal phase > 3
        terminal_gb_start_idx = terminal_match.start("gb")
        terminal_tokens = tokens[terminal_gb_start_idx:]
        terminal_gb_count = sum(t in {"G", "B"} for t in terminal_tokens)
        if terminal_gb_count < 3:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_veg_mix"])
            return

        # If there is a valid terminal G/B phase, look for the first F->G/B transition and sets that as the transition year.
        # If any Gs in terminal phase assume grass, otherwise assume bare.
        transition_match = re.search(r"F+[GB]", token_seq)

        if not transition_match:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_veg_mix"])
            return

        transition_idx = transition_match.end() - 1
        apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_veg_mix"])

        final_tokens = tokens[transition_idx:]
        if "G" in final_tokens:
            apply_tokens(lu_dict, range(transition_idx, len(tokens)), "G", node_code_map["grass_veg_mix"])
        else:
            apply_tokens( lu_dict, range(transition_idx, len(tokens)), "B", node_code_map["bare_glad_majority_years"])

# Mix of short veg and bare
def apply_short_bare(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    # Only considered a LU transition if initial landcover >= 3 consecutive years and final land cover >= 3 consecutive years and there is only 1 transition (i.e. GGGBBBBBBB OR BBBBGGGGGG)
    if re.fullmatch(r"(G{3,}B{3,}|B{3,}G{3,})", token_seq):
        return

    # Otherwise collapse to majority class across all years
    g_count = tokens.count("G")
    b_count = tokens.count("B")

    # If G and B have the same number of years, assume G
    if g_count >= b_count:
        apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_glad_majority_years"])
    else:
        apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_glad_majority_years"])

# Mix of vegetation/bare and water/wetland
def apply_veg_bare_water(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    veg_tokens = {"F", "G", "B"}
    water_tokens = {"W", "O"}

    veg_count = sum(t in veg_tokens for t in tokens)
    water_count = sum(t in water_tokens for t in tokens)

    f_count = tokens.count("F")
    g_count = tokens.count("G")
    b_count = tokens.count("B")

    # If there are <3 vegetation/bare years, collapse to water only if all water/wetland years are O. Otherwise assume wetland.
    if veg_count < 3:
        water_wetland_tokens = [t for t in tokens if t in water_tokens]
        if water_wetland_tokens and all(t == "O" for t in water_wetland_tokens):
            apply_tokens(lu_dict, all_idx, "O", node_code_map["water_glad_majority_years"])
        else:
            apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_veg_water_mix"])

        return

    # If there are <3 water/wetland years, collapse to majority vegetation/bare class. Tie goes to forest.
    if water_count < 3:
        if f_count >= g_count and f_count >= b_count:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_glad_majority_years"])
        elif g_count >= b_count:
            apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_glad_majority_years"])
        else:
            apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_glad_majority_years"])
        return
    #TODO: May want to consider F even when its not majority?

    # Vegetation -> water/wetland transition:
    # 3+ consecutive vegetation/bare years followed by 3+ consecutive water/wetland years until the end.
    transition_match = re.search(r"(?P<veg>[FGB]{3,})(?P<water>[WO]{3,})$", token_seq)

    if transition_match:
        transition_idx = transition_match.start("water")
        pre_tokens = tokens[:transition_idx]
        final_tokens = tokens[transition_idx:]

        # Prominent vegetation/bare class: if F > 2 forest, elif G > 2 grass, else bare.
        if pre_tokens.count("F") > 2:
            pre_token = "F"
            pre_node = node_code_map["forest_water_mix"]
        elif pre_tokens.count("G") > 2:
            pre_token = "G"
            pre_node = node_code_map["grass_water_mix"]
        else:
            pre_token = "B"
            pre_node = node_code_map["bare_glad_majority_years"]

        # If all water/wetland years are O, assume water. Otherwise, assume wetland.
        if all(t == "O" for t in final_tokens):
            final_token = "O"
            final_node = node_code_map["water_glad_majority_years"]
        else:
            final_token = "W"
            final_node = node_code_map["wetland_veg_water_mix"]

        apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node)
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), final_token, final_node)
        return

   # If enough evidence of both groups (both groups >=3) but no valid transition, consider it wetland all years.
    apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
    return

# Mix of wetland and water only
def apply_wetland_water(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    # Only considered a LU transition if initial landcover >= 3 consecutive years and final land cover >= 3 consecutive years and there is only 1 transition (i.e. WWWOOOOOOO OR OOOOWWWWWW)
    if re.fullmatch(r"(W{3,}O{3,}|O{3,}W{3,})", token_seq):
        return

    # Otherwise collapse to wetland if there are >=2 wetland years. Otherwise, assume water.
    w_count = tokens.count("W")

    if w_count >= 2:
        apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
    else:
        apply_tokens(lu_dict, all_idx, "O", node_code_map["water_glad_majority_years"])


# Mix of ice and bare only
def apply_ice_bare(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    # Only considered a LU transition if initial landcover >= 3 consecutive years and final land cover >= 3 consecutive years and there is only 1 transition
    if re.fullmatch(r"(I{3,}B{3,}|B{3,}I{3,})", token_seq):
        return

    # Otherwise collapse to majority class.
    i_count = tokens.count("I")
    b_count = tokens.count("B")

    if b_count >= i_count:
        apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_glad_majority_years"])
    else:
        apply_tokens(lu_dict, all_idx, "I", node_code_map["ice_glad_majority_years"])


# Mix of ice and all other LC classes
def apply_ice_other(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    land_tokens = {"I", "F", "G", "B"}
    water_tokens = {"W", "O"}

    land_count = sum(t in land_tokens for t in tokens)
    water_count = sum(t in water_tokens for t in tokens)

    i_count = tokens.count("I")
    f_count = tokens.count("F")
    g_count = tokens.count("G")
    b_count = tokens.count("B")
    w_count = tokens.count("W")
    o_count = tokens.count("O")

    # If there are <3 land years and none are F, collapse to majority water/wetland class. Tie goes to wetland.
    if land_count < 3 and f_count == 0:
        if w_count >= o_count:
            apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
        else:
            apply_tokens(lu_dict, all_idx, "O", node_code_map["water_glad_majority_years"])
        return

    # If there are <3 water/wetland years, use LC with greatest hierarchy where n_years >= 2.
    if water_count < 3:
        if f_count >= 2:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_ice_mix"])
        elif g_count >= 2:
            apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_ice_mix"])
        elif b_count >= 2:
            apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_ice_mix"])
        else:
            apply_tokens(lu_dict, all_idx, "I", node_code_map["ice_glad_majority_years"])
        return

    # Land -> water/wetland transition:
    # 3+ consecutive ice/bare years followed by 3+ consecutive water/wetland years until the end.
    transition_match = re.search(r"(?P<land>[IFGB]{3,})(?P<water>[WO]{3,})$", token_seq)

    if transition_match:
        transition_idx = transition_match.start("water")
        pre_tokens = tokens[:transition_idx]
        final_tokens = tokens[transition_idx:]

        # Prominent LC class: if LC > 2 years, else ice.
        if pre_tokens.count("F") > 2:
            pre_token = "F"
            pre_node = node_code_map["forest_ice_mix"]
        elif pre_tokens.count("G") > 2:
            pre_token = "G"
            pre_node = node_code_map["grass_ice_mix"]
        elif pre_tokens.count("B") > 2:
            pre_token = "B"
            pre_node = node_code_map["bare_ice_mix"]
        else:
            pre_token = "I"
            pre_node = node_code_map["ice_glad_majority_years"]

        # If all water/wetland years are O, assume water. Otherwise, assume wetland.
        if all(t == "O" for t in final_tokens):
            final_token = "O"
            final_node = node_code_map["water_glad_majority_years"]
        else:
            final_token = "W"
            final_node = node_code_map["wetland_bare_ice_water_mix"]

        apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node)
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), final_token, final_node)

        return

    # If enough evidence of both groups but no valid transition, collapse to  LC with greatest hierarchy where n_years >= 2.
    if f_count >= 2:
        apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_ice_mix"])
    elif g_count >= 2:
        apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_ice_mix"])
    elif b_count >= 2:
        apply_tokens(lu_dict, all_idx, "B", node_code_map["bare_ice_mix"])
    else:
        apply_tokens(lu_dict, all_idx, "I", node_code_map["ice_glad_majority_years"])

    return


def apply_regex_rules(tokens, node_codes, driver, tcl_year, pre_2000_plantation, planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, gpw_cultiv_grass):

    lu_dict = {
        "initial_tokens": tokens.copy(),
        "tokens": tokens,
        "node_codes": node_codes,
        "driver": driver,
        "tcl_year": tcl_year,
        "tcl_prior": (tcl_year != 0 and tcl_year <= min(cn.years_annual)),  # convert to bool
        "pre_2000_plantation": (pre_2000_plantation == 1),
        "planting_year": planting_year,
        "sdpt_oil_palm": (sdpt_oil_palm == 1),
        "sdpt_tree_crop": sdpt_tree_crop,
        "sdpt_planted_forest": sdpt_planted_forest,
        "gmw_mangrove": gmw_mangrove,
        "gpw_cultiv_grass": gpw_cultiv_grass,
    }

    # Built and cropland rules apply to both special cases and regex LC-based rules
    token_seq = "".join(lu_dict["tokens"])  # Creates a concat string
    if "S" in token_seq and not re.fullmatch(r"S+", token_seq):
        apply_built_transition(lu_dict)
    elif "C" in token_seq and not re.fullmatch(r"C+", token_seq):
        apply_crop_transition(lu_dict)

    # Check if oil palm, tree crop or forest based on special cases
    extent_rule_applied = apply_extent_rules(lu_dict)

    if not extent_rule_applied:
        # if re.fullmatch(r"F+", token_seq):
        #     apply_all_tall_veg(lu_dict)
        if re.fullmatch(r"G+", token_seq):
            apply_all_short_veg(lu_dict)
        elif re.fullmatch(r"[FGB]+", token_seq) and "F" in token_seq:
            apply_tall_short_bare(lu_dict)
        elif re.fullmatch(r"[GB]+", token_seq):
            apply_short_bare(lu_dict)
        elif re.fullmatch(r"[FGBWO]+", token_seq) and re.search(r"[FGB]", token_seq) and re.search(r"[WO]", token_seq):
            apply_veg_bare_water(lu_dict)
        elif re.fullmatch(r"[WO]+", token_seq):
            apply_wetland_water(lu_dict)
        elif re.fullmatch(r"[IB]+", token_seq):
            apply_ice_bare(lu_dict)
        elif re.fullmatch(r"[IFGBWO]+", token_seq) and "I" in token_seq:
            apply_ice_other(lu_dict)

    # Final token and node code timeseries
    final_tokens = lu_dict["tokens"]
    node_code_ts = lu_dict["node_codes"]

    # Convert final tokens to numeric LU codes
    lu_ts = [lu_token_map[token] for token in final_tokens]

    # Check that there is only one land use transition during the timeseries
    if print_lu_transition:
        check_single_lu_transition(lu_dict, lu_ts)

    # Create transition timeseries: 2015_2016 through 2023_2024
    transition_ts = [int(f"{lu_ts[i]}{lu_ts[i + 1]}") for i in range(len(lu_ts) - 1)]

    # Create sequential unique LU summary
    summary = []
    for lu in lu_ts:
        if not summary or lu != summary[-1]:
            summary.append(lu)

    # Stable LU gets duplicated, e.g. 4 -> 44
    if len(summary) == 1:
        summary.append(summary[0])

    return lu_ts, node_code_ts, transition_ts, summary



def IPCC_land_use(in_dict):

    # Dictionary for output arrays: IPCC land use class, land use node code, land use change, and land use summary
    out_dict = {}

    # Input data
    LC_2015_block = in_dict[f"{cn.land_cover_pattern}_2015"]
    LC_2016_block = in_dict[f"{cn.land_cover_pattern}_2016"]
    LC_2017_block = in_dict[f"{cn.land_cover_pattern}_2017"]
    LC_2018_block = in_dict[f"{cn.land_cover_pattern}_2018"]
    LC_2019_block = in_dict[f"{cn.land_cover_pattern}_2019"]
    LC_2020_block = in_dict[f"{cn.land_cover_pattern}_2020"]
    LC_2021_block = in_dict[f"{cn.land_cover_pattern}_2021"]
    LC_2022_block = in_dict[f"{cn.land_cover_pattern}_2022"]
    LC_2023_block = in_dict[f"{cn.land_cover_pattern}_2023"]
    LC_2024_block = in_dict[f"{cn.land_cover_pattern}_2024"]

    tcl_block = in_dict[cn.tree_cover_loss_pattern]
    drivers_block = in_dict[cn.drivers_pattern]

    oil_palm_2000_extent_block = in_dict[cn.oil_palm_2000_extent_pattern]  # IDN/ MYS pre-2000 plantation
    oil_palm_first_year_block = in_dict[cn.oil_palm_first_year_pattern]    # Descals oil palm planting year

    planted_forest_tree_crop_block = in_dict[cn.planted_forest_tree_crop_pattern]
    planted_forest_type_block = in_dict[cn.planted_forest_type_pattern]


    # Mangrove extent
    mangrove_extent_2015_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2015"]
    mangrove_extent_2016_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2016"]
    mangrove_extent_2017_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2017"]
    mangrove_extent_2018_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2018"]
    mangrove_extent_2019_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2019"]
    mangrove_extent_2020_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2020"]

    # GPW grassland extent
    gpw_extent_2015_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2015"]
    gpw_extent_2016_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2016"]
    gpw_extent_2017_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2017"]
    gpw_extent_2018_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2018"]
    gpw_extent_2019_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2019"]
    gpw_extent_2020_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2020"]
    gpw_extent_2021_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2021"]
    gpw_extent_2022_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2022"]
    gpw_extent_2023_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2023"]
    gpw_extent_2024_block = in_dict[f"{cn.GPW_extent_processed_pattern}_2024"]

    # Creat empty arrays for output datasets
    LU_2015_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2016_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2017_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2018_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2019_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2020_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2021_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2022_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2023_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_2024_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)

    node_code_2015_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2016_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2017_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2018_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2019_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2020_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2021_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2022_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2023_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    node_code_2024_block = np.zeros(LC_2015_block.shape, dtype=np.uint16)

    LU_change_2015_2016_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2016_2017_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2017_2018_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2018_2019_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2019_2020_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2020_2021_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2021_2022_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2022_2023_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)
    LU_change_2023_2024_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)

    LU_summary_block = np.zeros(LC_2015_block.shape, dtype=np.uint8)

    # Iterates through all pixels in the chunk
    for row in range(LC_2015_block.shape[0]):
        for col in range(LC_2015_block.shape[1]):

            ### Reads input pixel values
            LC_2015 = LC_2015_block[row, col]
            LC_2016 = LC_2016_block[row, col]
            LC_2017 = LC_2017_block[row, col]
            LC_2018 = LC_2018_block[row, col]
            LC_2019 = LC_2019_block[row, col]
            LC_2020 = LC_2020_block[row, col]
            LC_2021 = LC_2021_block[row, col]
            LC_2022 = LC_2022_block[row, col]
            LC_2023 = LC_2023_block[row, col]
            LC_2024 = LC_2024_block[row, col]

            tcl_year = np.int16(tcl_block[row, col])
            if tcl_year != 0:
                tcl_year += 2000
            driver = drivers_block[row, col]

            planted_forest_tree_crop = planted_forest_tree_crop_block[row, col]     # simpleName
            sdpt_planted_forest, sdpt_tree_crop = get_sdpt_status(planted_forest_tree_crop)

            pre_2000_plantation = oil_palm_2000_extent_block[row, col]
            sdpt_oil_palm = planted_forest_type_block[row, col]        # simpleType
            descals_planting_year = oil_palm_first_year_block[row, col]

            # Mangrove extent years (1 = mangrove, 0 = no mangrove)
            mang_2015 = mangrove_extent_2015_block[row, col]
            mang_2016 = mangrove_extent_2016_block[row, col]
            mang_2017 = mangrove_extent_2017_block[row, col]
            mang_2018 = mangrove_extent_2018_block[row, col]
            mang_2019 = mangrove_extent_2019_block[row, col]
            mang_2020 = mangrove_extent_2020_block[row, col]
            gmw_mangrove = (mang_2015 == 1 or mang_2016 == 1 or mang_2017 == 1 or mang_2018 == 1 or mang_2019 == 1 or mang_2020 == 1)

            # GPW grasslands (1 = cultivated grassland, 2 = natural / seminatural grassland)
            gpw_2015 = gpw_extent_2015_block[row, col]
            gpw_2016 = gpw_extent_2016_block[row, col]
            gpw_2017 = gpw_extent_2017_block[row, col]
            gpw_2018 = gpw_extent_2018_block[row, col]
            gpw_2019 = gpw_extent_2019_block[row, col]
            gpw_2020 = gpw_extent_2020_block[row, col]
            gpw_2021 = gpw_extent_2021_block[row, col]
            gpw_2022 = gpw_extent_2022_block[row, col]
            gpw_2023 = gpw_extent_2023_block[row, col]
            gpw_2024 = gpw_extent_2024_block[row, col]
            gpw_cultiv_grass = (gpw_2015 == 1 or gpw_2016 == 1 or gpw_2017 == 1 or gpw_2018 == 1 or gpw_2019 == 1 or gpw_2020 == 1 or gpw_2021 == 1 or gpw_2022 == 1 or gpw_2023 == 1 or gpw_2024 == 1)

            tokens = [
                token_for_lc(LC_2015),
                token_for_lc(LC_2016),
                token_for_lc(LC_2017),
                token_for_lc(LC_2018),
                token_for_lc(LC_2019),
                token_for_lc(LC_2020),
                token_for_lc(LC_2021),
                token_for_lc(LC_2022),
                token_for_lc(LC_2023),
                token_for_lc(LC_2024),
            ]

            # If any years are unknown, keep nodata = 0
            if any(t == "U" for t in tokens):
                continue

            default_lu = [lu_token_map[token] for token in tokens]
            node_codes = [default_node_code(token) for token in tokens]

            # Skip stable pixels that don't have an exception
            stable_lu = all(lu == default_lu[0] for lu in default_lu)

            if stable_lu and not has_lu_exception(driver, tcl_year, pre_2000_plantation, descals_planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, gpw_cultiv_grass):
                lu_code = default_lu[0]
                node_code = node_codes[0]
                change_code = lu_code * 10 + lu_code

                LU_2015_block[row, col] = lu_code
                LU_2016_block[row, col] = lu_code
                LU_2017_block[row, col] = lu_code
                LU_2018_block[row, col] = lu_code
                LU_2019_block[row, col] = lu_code
                LU_2020_block[row, col] = lu_code
                LU_2021_block[row, col] = lu_code
                LU_2022_block[row, col] = lu_code
                LU_2023_block[row, col] = lu_code
                LU_2024_block[row, col] = lu_code

                node_code_2015_block[row, col] = node_code
                node_code_2016_block[row, col] = node_code
                node_code_2017_block[row, col] = node_code
                node_code_2018_block[row, col] = node_code
                node_code_2019_block[row, col] = node_code
                node_code_2020_block[row, col] = node_code
                node_code_2021_block[row, col] = node_code
                node_code_2022_block[row, col] = node_code
                node_code_2023_block[row, col] = node_code
                node_code_2024_block[row, col] = node_code

                LU_change_2015_2016_block[row, col] = change_code
                LU_change_2016_2017_block[row, col] = change_code
                LU_change_2017_2018_block[row, col] = change_code
                LU_change_2018_2019_block[row, col] = change_code
                LU_change_2019_2020_block[row, col] = change_code
                LU_change_2020_2021_block[row, col] = change_code
                LU_change_2021_2022_block[row, col] = change_code
                LU_change_2022_2023_block[row, col] = change_code
                LU_change_2023_2024_block[row, col] = change_code

                LU_summary_block[row, col] = change_code
                continue

            # If not stable pixel or exceptions apply, pass default tokens/node codes to regex rules
            LU_timeseries, node_code_timeseries, LU_change_timeseries, summary = (
                apply_regex_rules( tokens, node_codes, driver, tcl_year, pre_2000_plantation, descals_planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, gpw_cultiv_grass))

            # Write out results
            LU_2015_block[row, col] = LU_timeseries[0]
            LU_2016_block[row, col] = LU_timeseries[1]
            LU_2017_block[row, col] = LU_timeseries[2]
            LU_2018_block[row, col] = LU_timeseries[3]
            LU_2019_block[row, col] = LU_timeseries[4]
            LU_2020_block[row, col] = LU_timeseries[5]
            LU_2021_block[row, col] = LU_timeseries[6]
            LU_2022_block[row, col] = LU_timeseries[7]
            LU_2023_block[row, col] = LU_timeseries[8]
            LU_2024_block[row, col] = LU_timeseries[9]

            node_code_2015_block[row, col] = node_code_timeseries[0]
            node_code_2016_block[row, col] = node_code_timeseries[1]
            node_code_2017_block[row, col] = node_code_timeseries[2]
            node_code_2018_block[row, col] = node_code_timeseries[3]
            node_code_2019_block[row, col] = node_code_timeseries[4]
            node_code_2020_block[row, col] = node_code_timeseries[5]
            node_code_2021_block[row, col] = node_code_timeseries[6]
            node_code_2022_block[row, col] = node_code_timeseries[7]
            node_code_2023_block[row, col] = node_code_timeseries[8]
            node_code_2024_block[row, col] = node_code_timeseries[9]

            LU_change_2015_2016_block[row, col] = LU_change_timeseries[0]
            LU_change_2016_2017_block[row, col] = LU_change_timeseries[1]
            LU_change_2017_2018_block[row, col] = LU_change_timeseries[2]
            LU_change_2018_2019_block[row, col] = LU_change_timeseries[3]
            LU_change_2019_2020_block[row, col] = LU_change_timeseries[4]
            LU_change_2020_2021_block[row, col] = LU_change_timeseries[5]
            LU_change_2021_2022_block[row, col] = LU_change_timeseries[6]
            LU_change_2022_2023_block[row, col] = LU_change_timeseries[7]
            LU_change_2023_2024_block[row, col] = LU_change_timeseries[8]

            # Convert array into single value
            summary_code = int("".join(str(x) for x in summary))
            LU_summary_block[row, col] = summary_code

    # Write final blocks to out_dict
    out_dict[f"{cn.IPCC_class_pattern}_2015"] = LU_2015_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2016"] = LU_2016_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2017"] = LU_2017_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2018"] = LU_2018_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2019"] = LU_2019_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2020"] = LU_2020_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2021"] = LU_2021_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2022"] = LU_2022_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2023"] = LU_2023_block.copy()
    out_dict[f"{cn.IPCC_class_pattern}_2024"] = LU_2024_block.copy()

    out_dict[f"{cn.IPCC_node_pattern}_2015"] = node_code_2015_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2016"] = node_code_2016_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2017"] = node_code_2017_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2018"] = node_code_2018_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2019"] = node_code_2019_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2020"] = node_code_2020_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2021"] = node_code_2021_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2022"] = node_code_2022_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2023"] = node_code_2023_block.copy()
    out_dict[f"{cn.IPCC_node_pattern}_2024"] = node_code_2024_block.copy()

    out_dict[f"{cn.IPCC_change_pattern}_2015_2016"] = LU_change_2015_2016_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2016_2017"] = LU_change_2016_2017_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2017_2018"] = LU_change_2017_2018_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2018_2019"] = LU_change_2018_2019_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2019_2020"] = LU_change_2019_2020_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2020_2021"] = LU_change_2020_2021_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2021_2022"] = LU_change_2021_2022_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2022_2023"] = LU_change_2022_2023_block.copy()
    out_dict[f"{cn.IPCC_change_pattern}_2023_2024"] = LU_change_2023_2024_block.copy()

    out_dict[f"{cn.IPCC_summary_pattern}_2015_2024"] = LU_summary_block.copy()

    return out_dict



def calculate_and_upload_IPCC_land_use(bounds, download_dict_with_data_types, is_large_run, no_upload, output_folders, stage, no_stats=False,  create_zarr=False, mega_zarr_path=None):

    chunk_stats = []
    process = psutil.Process(os.getpid())
    logger_worker = lu.setup_logging_worker()
    chunk_start_time = time.time()
    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels



    ### Part 1: Downloads all inputs for chunk.
    # Replaces the placeholder tile_id in the download data dictionary with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_large_run, logger_worker, False)
    #print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        data, status = future.result()
        if 'success' not in status:  # Prints and logs any inputs that couldn't be accessed (downloaded as all 0s) or had to be padded
            lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
        layers[layer] = data

    # Frees up a little memory
    del updated_download_dict
    del futures
    gc.collect()



    ### Part 2: Calculates min, mode, and max for each input chunk.
    # Calculates stats for the input layers
    # for key, array in layers.items():
    #     chunk_stats.append(uu.calculate_ipcc_stats(array, key, bounds_str, tile_id, 'input_layer'))
    # print(chunk_stats)



    ### Part 3: IPCC land use assignment
    lu.print_and_log(f"Assigning IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}",False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)
    ipcc_start = time.time()

    out_dict = IPCC_land_use(layers)
    # Save pixel area for chujn stats before clearing input layers
    if not no_stats:
        pixel_area = layers[cn.pixel_area_pattern]
    #print("out_dict:", out_dict)

    ipcc_end = time.time()
    lu.print_and_log(f"Done assigning IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after IPCC stage completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    lu.print_and_log(f"Assigned IPCC land use in {bounds_str} in {tile_id} in {round(ipcc_end - ipcc_start)} seconds: {uu.timestr()}",False, logger_worker)

    # Deletes all unnecessary input dictionaries before moving on
    in_dicts = [layers]
    [in_dict.clear() for in_dict in in_dicts]

    # ### Part 4: Populate zarr
    if create_zarr:
        zu.populate_ipcc_zarr(bounds, bounds_str, create_zarr, is_large_run, logger_worker, mega_zarr_path, out_dict, stage, tile_id)


    ### Part 5: Calculates chunk stats
    if not no_stats:
        lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

        for key, array in out_dict.items():
            chunk_stats.append(uu.calculate_ipcc_stats(array, key, bounds_str, tile_id, "output_layer", pixel_area=pixel_area))
        lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    if not no_stats:
        del pixel_area
        pixel_area = None
        gc.collect()



    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if no_upload == False:
        out_no_data_val = 0
        #print("output_folders:", output_folders)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict.items():
            data_type = value.dtype.name
            # print("key:", key)
            # print("data_type:", data_type)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("year_range:", year_range)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output
            matched_output_s3_folders = [item for item in output_folders if out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            if out_pattern_without_pixel_meaning == cn.IPCC_summary_pattern:
                matched_output_s3_folder_list = matched_output_s3_folders
            else:
                matched_output_s3_folder_list = [
                    item for item in matched_output_s3_folders
                    if year_range in item
                ]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict[key] = [value, data_type, out_pattern, year_range, s3_path_without_bucket]

        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict, is_large_run, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} using {cn.IPCC_outputs_path}: {uu.timestr()}", is_large_run, logger_worker)

        del upload_tasks
        gc.collect()

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    out_dict.clear()
    del out_dict
    gc.collect()

    return return_message, chunk_stats  # Return both the success message and the statistics

def combine_ipcc_output_to_10x10(tile_id, output_dir_1x1, output_dir_10x10, raster_paths, no_upload, stage,):
    logger_worker = lu.setup_logging_worker()
    tile_bounds = uu.get_10x10_tile_bounds(tile_id)
    tile_bounds_str = tile_id

    tile_rasters = sorted([p for p in raster_paths if f"{tile_id}__" in p])

    if not tile_rasters:
        return f"No 1x1 rasters found for {tile_id} in {output_dir_1x1}"

    first_name = os.path.basename(tile_rasters[0])
    key = first_name.replace(".tif", "").split("__")[-1]
    out_pattern, year_range = uu.strip_and_extract_years(key)
    key = f"{out_pattern}_{year_range}"

    lu.print_and_log( f"Mosaicking {key} for 10x10 tile {tile_id} from {len(tile_rasters)} 1x1 rasters: {uu.timestr()}", False, logger_worker)

    mosaic_array = mosaic_ipcc_1x1_rasters(key, tile_rasters, tile_bounds, logger_worker)

    if not no_upload:
        data_type = mosaic_array.dtype.name
        s3_path_without_bucket = output_dir_10x10[cn.full_bucket_prefix_length:]
        out_dict = {key: [mosaic_array, data_type, out_pattern, year_range, s3_path_without_bucket]}
        upload_tasks = uu.save_and_upload_raster_10x10( tile_bounds, cn.full_raster_dims, tile_id, tile_bounds_str, out_dict, True, logger_worker, no_data_val=0)

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

    if not no_upload:
        del upload_tasks
        out_dict.clear()
        del out_dict

    del mosaic_array
    gc.collect()

    return f"Success for 10x10 {tile_id} {key}: {uu.timestr()}"


def main(cluster_name, run_date, run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None, log_note=None, skip_existing_1x1=None, chunk_ids_to_skip=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'IPCC_land_use'
    model_type = 'standard_model'

    # Determines if arguments for start and end year are valid
    start_year = cn.first_model_year_annual
    end_year = cn.last_model_year_annual

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers= lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Calculates the interval type, difference between start and end years of intervals, and the model output years for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(start_year, end_year, main_logger)

    # Returns a dataframe of chunk_ids and iso code from the GADM4.1 1x1 deg fishnet used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    requested_chunk_size_deg = chunk_size_deg
    make_10x10_outputs = requested_chunk_size_deg == 10
    processing_chunk_size_deg = 1 if make_10x10_outputs else chunk_size_deg

    chunk_list, chunk_size_pixels = uu.create_chunk_list( bounding_box, chunk_shapefile_uri, processing_chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)

    # Preserve original list before any filtering
    global_chunk_list = chunk_list.copy()

    # Filter to only 1x1 degree chunks that have not been processed yet
    if chunk_ids_to_skip:
        skip_set = read_chunk_ids_file(chunk_ids_to_skip)
        original_count = len(chunk_list)

        skipped_list = [uu.boundstr(chunk) for chunk in chunk_list if uu.boundstr(chunk) in skip_set]
        chunk_list = [chunk for chunk in chunk_list if uu.boundstr(chunk) not in skip_set]

        main_logger.info(f"Skipped {len(skipped_list)} chunks from {chunk_ids_to_skip}")
        main_logger.info(f"First skipped chunks: {skipped_list[:10]}")

    tile_ids_10x10 = make_10x10_tile_list(chunk_list) if make_10x10_outputs else []
    #tile_ids_10x10 = make_10x10_tile_list(global_chunk_list) if make_10x10_outputs else []  #If you give a list of chunks to skip, use global chunk for 10x10 tile creation

    main_logger.info(f"Chunks to process: {len(chunk_list)}")
    if make_10x10_outputs:
        main_logger.info(f"10x10 tiles to combine after 1x1 processing: {tile_ids_10x10}")

    # Runs chunks in batches of specified size.
    try:
        n_workers_int = int(n_workers)
    except (TypeError, ValueError):
        n_workers_int = 1
    batch_size = min(len(chunk_list), n_workers_int * 5)

    start_time = uu.timestr()  # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    # Placeholder tile_id to obtain the datatype of each input tile set. Overwritten when chunks are assigned and analyzed.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to LU assignment).
    download_dict = {
        cn.pixel_area_pattern: f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{sample_tile_id}.tif",
        cn.tree_cover_loss_pattern: f"{cn.tree_cover_loss_dir}{cn.tree_cover_loss_pattern}_{sample_tile_id}.tif",
        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_dir}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_dir}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_dir}{sample_tile_id}.tif",
        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_dir}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
    }

    # GLCLU timeseries
    for year in cn.years_annual:
        download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"

    # GPW grassland extent timeseries
    for year in cn.years_annual:
        download_dict[f"{cn.GPW_extent_processed_pattern}_{year}"] = f"{cn.GPW_extent_processed_dir}{year}/{sample_tile_id}_{cn.GPW_extent_processed_pattern}_{year}.tif"

    # GMW mangrove extent timeseries
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        download_dict[f"{cn.mangrove_extent_processed_pattern}_{year}"] = f"{cn.mangrove_extent_processed_dir}{year}/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_{year}.tif"

    # print("Download dictionary:")
    # for key, item in download_dict.items():
    #     print(f"{key}: {item}")

    # Returns the first tile in each input so that the datatype can be determined per dataset
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)
    main_logger.info(f"download_dict_with_data_types for {stage}:")
    for key, value in download_dict_with_data_types.items():
        main_logger.info(f"  {key}: {value}")

    # Creates a list of output directories for all outputs
    class_node_output_dirs = uu.create_output_dir_name_list( [cn.IPCC_class_dir, cn.IPCC_node_dir], interval_type,
                                                             start_year, chunk_size_pixels, model_type, cn.IPCC_LU_version,
                                                             stage, cn.years_annual, interval_year_diff_list, run_date, False)

    change_years = [f"{a}_{b}" for a, b in zip(cn.years_annual[:-1], cn.years_annual[1:])]
    change_output_dirs = uu.create_output_dir_name_list( [cn.IPCC_change_dir], interval_type, start_year, chunk_size_pixels,
                                                         model_type, cn.IPCC_LU_version, stage,
                                                         change_years, interval_year_diff_list, run_date, False)

    summary_dir = (cn.IPCC_summary_dir .replace("RUN_DATE", run_date) .replace("CHUNK_SIZE", str(chunk_size_pixels)))

    # 1x1 output dirs used by the main classification tasks
    output_dir_list = sorted(class_node_output_dirs + change_output_dirs + [summary_dir])
    output_dir_list_1x1 = output_dir_list

    main_logger.info(f"1x1 output_dir_list for {stage}:")
    for item in output_dir_list_1x1:
        main_logger.info(f"  {item}")

    # 10x10 output dirs used only when user requested -cs 10
    output_dir_list_10x10 = None

    if make_10x10_outputs:
        class_node_output_dirs_10x10 = uu.create_output_dir_name_list( [cn.IPCC_class_dir, cn.IPCC_node_dir], interval_type, start_year, cn.full_raster_dims,
                                            model_type, cn.IPCC_LU_version, stage, cn.years_annual, interval_year_diff_list, run_date, False)
        change_output_dirs_10x10 = uu.create_output_dir_name_list([cn.IPCC_change_dir], interval_type, start_year, cn.full_raster_dims,
                                            model_type, cn.IPCC_LU_version, stage, change_years, interval_year_diff_list, run_date, False)
        summary_dir_10x10 = (cn.IPCC_summary_dir .replace("RUN_DATE", run_date) .replace("CHUNK_SIZE", str(cn.full_raster_dims)))

        output_dir_list_10x10 = sorted(class_node_output_dirs_10x10 + change_output_dirs_10x10 + [summary_dir_10x10])

        main_logger.info(f"10x10 output_dir_list for {stage}:")
        for item in output_dir_list_10x10:
            main_logger.info(f"  {item}")

    ### Step 2: Create empty (metadata-only), global zarr in s3.
    outputs_to_zarr = [cn.IPCC_class_pattern, cn.IPCC_node_pattern, cn.IPCC_change_pattern, cn.IPCC_summary_pattern]
    raw_mega_zarr_path = None
    if create_zarr:
        raw_mega_zarr_path = zu.create_zarr_path(cn.IPCC_outputs_path_mega_zarr, cn.chunk_dims, interval_type, model_type,
                        cn.IPCC_LU_version.replace(".", "_"), "global", run_date, main_logger)

        zu.initialize_ipcc_global_zarr(raw_mega_zarr_path, (1, cn.chunk_dims, cn.chunk_dims), main_logger, fill_value=0)

    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log" + "\n")

    chunks_to_run = chunk_list
    if skip_existing_1x1 and not no_upload:
        chunks_to_run, skipped_1x1_chunks = filter_chunks_missing_1x1_outputs_fast(chunk_list, output_dir_list_1x1, main_logger)

    chunk_batches = [chunks_to_run[i:i + batch_size] for i in range(0, len(chunks_to_run), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    all_results = []
    all_1x1_stats = []
    success_count = 0  # Count of successful chunks

    # Iterates through the batches
    if not chunks_to_run:
        main_logger.info("All expected 1x1 outputs already exist in S3. Skipping 1x1 land use processing and moving to 10x10 mosaicking.")
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        if run_local:
            batch_results = [calculate_and_upload_IPCC_land_use(chunk, download_dict_with_data_types, True, no_upload, output_dir_list_1x1, stage, no_stats, create_zarr, raw_mega_zarr_path)
                             for chunk in chunk_batch]
            all_results.extend(batch_results)

        else:
            futures = [client.submit(calculate_and_upload_IPCC_land_use, chunk, download_dict_with_data_types, True, no_upload, output_dir_list_1x1, stage, no_stats, create_zarr, raw_mega_zarr_path, retries=2)
                       for chunk in chunk_batch]
            batch_results = client.gather(futures)
            all_results.extend(batch_results)

        batch_success_count, batch_stats = uu.count_successful_chunks(chunk_batch, True, main_logger, batch_results)
        success_count += batch_success_count
        all_1x1_stats.extend(batch_stats)

        # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
        # That way there are some basic chunk stats (not sorted or anything) to fall back on.
        if len(chunk_batches) > 1:
            main_logger.info(f"Writing batch stats to spreadsheet: {uu.timestr()}")
            df_batch_stats = pd.DataFrame(batch_stats)
            out_spreadsheet = f'TEMP_BATCH_{stage}__batch_{i}_{uu.timestr()}.xlsx'
            local_spreadsheet = f"{cn.local_chunk_stats_path}{out_spreadsheet}"
            with pd.ExcelWriter(local_spreadsheet) as writer:
                df_batch_stats.to_excel(writer, sheet_name=f'stats__batch_{i}', index=False)

        del batch_results
        del batch_stats
        if not run_local and client is not None:
            del futures
            client.run(gc.collect)

        gc.collect()

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)

    ### Step 4: Gather worker logs preliminarily
    worker_log_local_path_prelim = None
    worker_log_local_path = None
    model_chunk_stats_path = None

    if not run_local:
        worker_log_local_path_prelim = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with preliminary worker log compilation", main_logger)

    ### Step 5: Consolidate chunk stats and export

    if (not no_stats) and (success_count > 0) and all_1x1_stats:
        model_chunk_stats_path = uu.compile_ipcc_1x1_chunk_stats(all_1x1_stats, chunk_shapefile_uri, stage, no_upload, main_logger)

        main_logger.info(f"Final IPCC 1x1 chunk stats table: {model_chunk_stats_path}")

        uu.stage_duration(start_time, uu.timestr(), f"{stage} with 1x1 chunk stats", main_logger)

    ### Step 6b: Combine 1x1 outputs into 10x10 outputs

    if make_10x10_outputs and no_upload:
        main_logger.warning("Skipping 10x10 combine because --no_upload is enabled. The combine step reads uploaded 1x1 rasters from S3.")

    if make_10x10_outputs and not no_upload:
        main_logger.info(f"Starting 1x1 -> 10x10 IPCC combine for {len(tile_ids_10x10)} tiles and {len(output_dir_list_1x1)} outputs: {uu.timestr()}")

        raster_paths_by_output_dir = build_raster_paths_by_output_dir(output_dir_list_1x1, main_logger)

        combine_tasks = []
        for tile_id in tile_ids_10x10:
            for output_dir_1x1 in output_dir_list_1x1:
                output_dir_10x10_matches = [
                    d for d in output_dir_list_10x10
                    if d.replace(f"/{cn.full_raster_dims}_pixels/", f"/{cn.chunk_dims}_pixels/") == output_dir_1x1
                ]

                if not output_dir_10x10_matches:
                    main_logger.warning(f"No matching 10x10 output folder for {output_dir_1x1}")
                    continue

                combine_tasks.append((tile_id, output_dir_1x1, output_dir_10x10_matches[0], raster_paths_by_output_dir[output_dir_1x1]))

        main_logger.info(f"Created {len(combine_tasks)} 10x10 mosaic tasks")

        combine_batch_size = min(len(combine_tasks), max(1, n_workers_int))
        combine_batches = [
            combine_tasks[i:i + combine_batch_size]
            for i in range(0, len(combine_tasks), combine_batch_size)
        ]

        for i, combine_batch in enumerate(combine_batches):
            main_logger.info(f"Processing 10x10 combine batch {i + 1}/{len(combine_batches)} ({len(combine_batch)} output tasks): {uu.timestr()}")

            if run_local:
                combine_batch_results = [
                    combine_ipcc_output_to_10x10(tile_id, output_dir_1x1, output_dir_10x10, raster_paths, no_upload, stage)
                    for tile_id, output_dir_1x1, output_dir_10x10, raster_paths in combine_batch
                ]
            else:
                futures = [
                    client.submit(combine_ipcc_output_to_10x10, tile_id, output_dir_1x1, output_dir_10x10, raster_paths, no_upload, stage, retries=2)
                    for tile_id, output_dir_1x1, output_dir_10x10, raster_paths in combine_batch
                ]

                combine_batch_results = client.gather(futures)

            for result in combine_batch_results:
                main_logger.info(result)

            del combine_batch_results
            if not run_local and client is not None:
                del futures
                client.run(gc.collect)

            gc.collect()

            uu.stage_duration(start_time, uu.timestr(), f"{stage}, 10x10 combine batch {i}", main_logger)

        raster_paths_by_output_dir.clear()
        del raster_paths_by_output_dir
        del combine_tasks
        del combine_batches
        gc.collect()

        uu.stage_duration(start_time, uu.timestr(), f"{stage} with 10x10 combine", main_logger)

    ### Step 7: Gather worker logs
    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

    ### Step 8: Resize cluster down to 1 worker
    if not run_local:
        workers = client.scheduler_info()["workers"]
        current_n_workers = len(workers)

        if current_n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")
            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    ### Step 9: Count output geotifs in s3

    main_logger.info(f"Counting IPCC 1x1 geotifs. Expecting {len(chunk_list)} in each 1x1 folder: {uu.timestr()}")
    #main_logger.info(f"Counting IPCC 1x1 geotifs. Expecting {len(global_chunk_list)} in each 1x1 folder: {uu.timestr()}")

    if not no_upload:
        for output_folder in output_dir_list_1x1:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"1x1 output rasters in {output_folder}: {file_count}")

            if file_count != len(chunk_list):    #Use global_chunk_list if using chunk_ids_to_skip
                main_logger.warning(f"WARNING: 1x1 output file count in {output_folder} does not match expected {len(chunk_list)}!")
            # if file_count != len(global_chunk_list):
            #     main_logger.warning(f"WARNING: 1x1 output file count in {output_folder} does not match expected {len(global_chunk_list)}!")

        if make_10x10_outputs and output_dir_list_10x10:
            main_logger.info(f"Counting IPCC 10x10 geotifs. Expecting {len(tile_ids_10x10)} in each 10x10 folder: {uu.timestr()}")

            for output_folder in output_dir_list_10x10:
                geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
                main_logger.info(f"10x10 output rasters in {output_folder}: {file_count}")

                if file_count != len(tile_ids_10x10):
                    main_logger.warning(f"WARNING: 10x10 output file count in {output_folder} does not match expected {len(tile_ids_10x10)}!")

    ### Step 10: Merge compiled worker log and main log
    if not run_local:
        if worker_log_local_path is None:
            worker_log_local_path = worker_log_local_path_prelim

        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    if not run_local:
        client.close()
        terminate_cluster.terminate_cluster(cluster_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create IPCC land use classes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('--create_zarr', action='store_true', help='Create and populate global mega-zarr with model outputs')
    parser.add_argument("--skip_existing_1x1", action="store_true", help="If all expected 1x1 output rasters already exist in S3, skip land use processing and go straight to 10x10 mosaicking.")
    parser.add_argument( "--chunk_ids_to_skip", help="Text file containing chunk bounds strings to skip, one per line")

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size_deg = args.chunk_size_deg
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    create_zarr = args.create_zarr
    skip_existing_1x1 = args.skip_existing_1x1
    chunk_ids_to_skip = args.chunk_ids_to_skip

    # Create the cluster with command line arguments
    main(cluster_name, run_date, run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size_deg=chunk_size_deg, first_chunks=first_chunks, log_note=log_note,
         skip_existing_1x1=skip_existing_1x1, chunk_ids_to_skip=chunk_ids_to_skip)





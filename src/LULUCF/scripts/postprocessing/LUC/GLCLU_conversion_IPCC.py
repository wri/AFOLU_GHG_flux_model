"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -bb 119.5 -5.75 119.75 -5.5 -cs 0.25 --run_local --run_date 20268888

Coiled small tests (0.25x0.25 deg chunk):
python -m src.utilities.create_cluster -n 1 -m 16 -cn IPCC_land_use
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use -bb 119.5 -5.75 119.75 -5.5 -cs 0.25 --run_date 20268888

Coiled small tests (1x1 deg chunk):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn IPCC_land_use_change
python -m src.LULUCF.scripts.postprocessing.LUC.GLCLU_conversion_IPCC -cn IPCC_land_use_change -bb -64 -22 -63 -21 -cs 1 --create_zarr --run_date YYYYMMDD

Coiled test (10x10 deg chunk):

Full run:

"""

import argparse
import concurrent.futures
import gc
import os
import psutil
import time
import sys
import pandas as pd
import numpy as np
import re

import fsspec
import xarray as xr

from concurrent.futures import ThreadPoolExecutor

from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"



# Returns boolean values for whether a pixel is planted forest or tree crop
def get_sdpt_status(sdpt_type):
    sdpt_planted_forest = not np.isnan(sdpt_type) and int(sdpt_type) == 1
    sdpt_tree_crop      = not np.isnan(sdpt_type) and int(sdpt_type) == 2

    return sdpt_planted_forest, sdpt_tree_crop

# Checks if an oil palm planting year related transition happens in the timeseries.
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

# Move general utilities from here up to UU
#######################################################################################################################
""" Regex-based land-use reclassification rules
These rules replace the default land use classes.
1. Convert annual GLAD LC values to LU tokens.
2. Use regex to identify token patterns for exceptions.
3. Reclassify token arrays and assign a matching node_code array.

Node codes used here:
1) Settlements and Infrastructure:
    10 = Built from GLAD data
    11 = Built following tall veg loss before built LC


2) Cropland:
    20 = Crop from GLAD data
    21 = Crop from oil palm extent or planting year
    22 = Crop from SDPT tree crop extent
    23 = Crop from permanent agriculture driver
    24 = Crop following tall veg loss before crop LC
    25 = Crop from majority years in mixed LC prior to built LC


3) Forest:
    30  = Forest from GLAD tall vegetation
    31  = Forest from SDPT planted forest extent
    32  = Forest from GMW mangrove extent
    33X = Unstocked forest from TCL + drivers rules
        333 = Forest from shifting cultivation driver
        334 = Forest from logging driver
        335 = Forest from wildfire driver
        337 = Forest from natural disturbance driver
    34 = Unstocked forest after TCL and before oil palm planting
    35 = Forest from vegetation/bare to built transition rule
    36 = Forest from vegetation/bare to crop transition rule
    37 = Forest from mixed tall/short vegetation rule
    38 = Forest from mixed vegetation/water rule
    39 = Forest from majority years in mixed class rule


4) Grassland:
    40 = Grass from GLAD short vegetation
    41 = Grass from TCL + permanent agriculture driver + GPW cultivated grassland extent (assume rangeland)
    42x = Grass from TCL + driver rule
        420 = Grass from unknown driver
        422 = Grass from hard commodities driver
        426 = Grass from settlements/infrastructure driver
    43 = Grass from vegetation/bare to built transition rule
    44 = Grass from vegetation/bare to crop transition rule
    45 = Grass from mixed tall/short vegetation rule
    46 = Grass from mixed vegetation/water rule
    47 = Grass from majority years in mixed class rule


5) Wetland:
    50 = Wetland from GLAD data
    51 = Wetland from vegetation/water transition rule
    52 = Wetland from majority years in mixed water rule


6) Other
    60 = Bare from GLAD data
    61 = Bare from majority years in mixed bare/grass rule

    70 = Water from GLAD data
    71 = Water from vegetation/water transition rule
    72 = Water from majority years in mixed water rule

    80 = Snow/ice from GLAD data
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

# Function to get land use token per land cover numeric value (tokens used for regex exception rules)
def token_for_lc(v):
    if v not in lc_token_map:
        raise ValueError(f"Unknown GLCLU code: {v}")
    return lc_token_map[v]

# Node code values based on what exception was applied
node_code_map = {
    "built_glad": 10,
    "built_tall_veg_loss": 11,

    "crop_glad": 20,
    "crop_oil_palm": 21,
    "crop_sdpt_tree_crop": 22,
    "crop_perm_ag_driver": 23,
    "crop_tall_veg_loss": 24,
    "crop_glad_majority_years": 25,

    "forest_glad": 30,
    "forest_sdpt_planted_forest": 31,
    "forest_gmw_mangrove": 32,
    "forest_shift_cult_driver": 333,
    "forest_logging_driver": 334,
    "forest_wildfire_driver": 335,
    "forest_nat_dist_driver": 337,
    "forest_unstocked_pre_oil_palm": 34,
    "forest_veg_bare_built_mix": 35,
    "forest_veg_bare_crop_mix": 36,
    "forest_tall_short_mix": 37,
    "forest_veg_water_mix": 38,
    "forest_glad_majority_years": 39,

    "grass_glad": 40,
    "grass_gpw": 41,
    "grass_hard_commod_driver": 422,
    "grass_settlement_driver": 426,
    "grass_unknown_driver": 420,
    "grass_veg_bare_built_mix": 43,
    "grass_veg_bare_crop_mix": 44,
    "grass_tall_short_mix": 45,
    "grass_veg_water_mix": 46,
    "grass_glad_majority_years": 47,


    "wetland_glad": 50,
    "wetland_veg_water_mix": 51,
    "wetland_glad_majority_years": 52,

    "bare_glad": 60,
    "bare_glad_majority_years": 61,

    "water_glad": 70,
    "water_veg_water_mix": 71,
    "water_glad_majority_years": 72,

    "ice_glad": 80,
    "ice_glad_majority_years": 81,
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

# Select pre-transition token by count. Ties go to the earlier token in priority_order.
def majority_token(tokens, candidates, priority_order):
    present = [t for t in candidates if t in tokens]
    return max(present, key=lambda t: (tokens.count(t), -priority_order.index(t)))

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
    # TODO: May want to consider not including wetland? water? ice?

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

# Mix of 2 or more LC classes -> built
def apply_built_transition(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)

    if "S" not in token_seq:
        return

    # Require at least 2 non-S LC classes
    non_s_classes = set(tokens) - {"S"}
    if len(non_s_classes) < 2:
        return

    pre_node_map = {
        "C": node_code_map["crop_glad_majority_years"],
        "F": node_code_map["forest_glad_majority_years"],
        "G": node_code_map["grass_glad_majority_years"],
        "W": node_code_map["wetland_glad_majority_years"],
        "B": node_code_map["bare_glad_majority_years"],
        "O": node_code_map["water_glad_majority_years"],
        "I": node_code_map["ice_glad_majority_years"],
    }

    first_s_idx = token_seq.find("S")

    # Go down hierarchy. C is handled as tie-breaker when present.
    for candidate in ["F", "G", "W", "B", "O", "I"]:
        if candidate not in token_seq:
            continue

        pre_token = candidate

        # Count majority pre-token. Tie goes to C.
        if "C" in token_seq:
            c_count = tokens.count("C")
            candidate_count = tokens.count(candidate)
            if c_count >= candidate_count:
                pre_token = "C"

        # F uses first F loss; everything else uses first S.
        if candidate == "F" and pre_token == "F":
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

#TODO: Use TCL up to 5 years prior for F->S exception?

# Mix of 2 or more LC classes -> crop
def apply_crop_transition(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)

    if "C" not in token_seq:
        return

    # Require at least 2 non-C LC classes
    non_c_classes = set(tokens) - {"C"}
    if len(non_c_classes) < 2:
        return

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
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), "C", node_code_map["crop_tall_veg_loss"])

        return
#TODO: Use TCL up to 5 years prior for F->C exception?

# Tall vegetation all years
def apply_all_tall_veg(lu_dict):
    tcl_prior = lu_dict["tcl_prior"]
    driver = lu_dict["driver"]

    tokens = lu_dict["tokens"]
    all_idx = range(len(tokens))

    # If TCL has occurred by the start of timeseries and the driver is permanent ag, assume tall veg is tree crops
    if tcl_prior and driver == 1:
        apply_tokens(lu_dict, all_idx, "C", node_code_map["crop_perm_ag_driver"])

    # # If oil palm planting year in interval, allows for F -> C transitions assuming establishment of tree crops
    # if has_planting_transition(lu_dict):
    #     idx = planting_idx(lu_dict["planting_year"])
    #     apply_tokens(lu_dict, range(idx, len(tokens)), "C", node_code_map["crop_oil_palm"])
    #     return
    #TODO: Do we want to allow for F->C transition based on oil palm planting year?

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


# Mix of short veg and tall veg
def apply_tall_short(lu_dict):
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
        transition_match = re.search(r"F+G", token_seq)
        if transition_match:
            idx = transition_match.end() - 1
        else:
            idx = planting_idx(planting_year)
        pre_plant_tokens = tokens[:idx]

        # If F present before transition or TCL within 5 years before planting, consider it F -> C
        forest_before_planting = ("F" in pre_plant_tokens or tcl_prior_to_planting(tcl_year, planting_year))
        if forest_before_planting:
            pre_token = "F"
            pre_node = node_code_map["forest_unstocked_pre_oil_palm"]
            apply_tokens(lu_dict, range(0, idx), pre_token, pre_node)

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
        match = re.search(r"F+G", token_seq)
        if match:
            transition_idx = match.end() - 1
            # apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_glad"])
            # Note: if not using first F->G transition switch node code to forest_tall_short_mix
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

    # 5) If TCL during timeseries, use hard commodities, settlements/ infrastructure, and unknown driver and an F->G transition where it stays G until the end. There must be at least 3 Fs, and at least 3 consecutive Gs until the end to determine LU transitions:
    if tcl_any and not tcl_prior and driver not in {1, 3, 4, 5, 7}:
        terminal_match = re.search(r"F{3,}G{3,}$", token_seq)

        if terminal_match:
            transition_match = re.search(r"F+G", token_seq)  # Get the first F->G transition

            if transition_match:
                transition_idx = transition_match.end() - 1
                apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_tall_short_mix"])

                if driver in driver_to_grass_node:
                    grass_node = driver_to_grass_node[driver]
                else:
                    grass_node = node_code_map["grass_unknown_driver"]
                apply_tokens(lu_dict, range(transition_idx, len(tokens)), "G", grass_node)

                return

    # 6) Otherwise use regex fallback (if no oil palm or TCL + driver, LU can only be forest or grass)
    # If there is not at least 3 years F or 3 years G, not enough evidence for a true F/G transition. Use majority land use instead.
    f_count = tokens.count("F")
    g_count = tokens.count("G")
    if g_count < 3:
        apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_glad_majority_years"])
        return
    elif f_count < 3:
        apply_tokens(lu_dict, all_idx, "G", node_code_map["grass_glad_majority_years"])
        return

    # Otherwise F-> G transition needs a terminal G phase that start with 3 consecutive Gs, allow at most one F, end on G.
    # Option to set total number of G years in terminal phase to >= #.
    # TODO: GGGFGG and GGFGGG allowed but not GGFGG?
    else:
        terminal_match = re.search(r"(?P<g>G{3,}(?:F?G*)?)$", token_seq)

        # If no valid terminal G phase, set all years to F
        if not terminal_match:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_tall_short_mix"])
            return

        # Option to make number of Gs in terminal G phase > 3
        terminal_g_start_idx = terminal_match.start("g")
        terminal_g_count = tokens[terminal_g_start_idx:].count("G")
        if terminal_g_count < 3:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_tall_short_mix"])
            return

        # If there is a valid terminal G phase, look for the first F->G transition and sets that as the transition year since that is when the majority of emissions will occur in the vegetation model.
        transition_match = re.search(r"F+G", token_seq)

        if not transition_match:
            apply_tokens(lu_dict, all_idx, "F", node_code_map["forest_tall_short_mix"])
            return

        transition_idx = transition_match.end() - 1
        # apply_tokens(lu_dict, range(0, transition_idx), "F", node_code_map["forest_glad"])
        # Note: if not using first F->G transition switch node code to forest_tall_short_mix
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), "G", node_code_map["grass_tall_short_mix"])

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
    w_count = tokens.count("W")
    o_count = tokens.count("O")

    # If there are <3 vegetation/bare years, collapse to majority water/wetland. Tie goes to wetland.
    if veg_count < 3:
        if w_count >= o_count:
            apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
        else:
            apply_tokens(lu_dict, all_idx, "O", node_code_map["water_glad_majority_years"])
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
            pre_node = node_code_map["forest_glad_majority_years"]
        elif pre_tokens.count("G") > 2:
            pre_token = "G"
            pre_node = node_code_map["grass_glad_majority_years"]
        else:
            pre_token = "B"
            pre_node = node_code_map["bare_glad_majority_years"]

        # Majority water class: if W > 2, wetland; otherwise water.
        if final_tokens.count("W") > 2:
            final_token = "W"
            final_node = node_code_map["wetland_glad_majority_years"]
        else:
            final_token = "O"
            final_node = node_code_map["water_glad_majority_years"]

        apply_tokens(lu_dict, range(0, transition_idx), pre_token, pre_node)
        apply_tokens(lu_dict, range(transition_idx, len(tokens)), final_token, final_node)
        return

   # If enough evidence of both groups (both groups >=3) but no valid transition, consider it wetland all years.
    apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
    return
#TODO: change node_codes to veg_water_mix?

# Mix of wetland and water only
def apply_wetland_water(lu_dict):
    tokens = lu_dict["tokens"]
    token_seq = "".join(tokens)
    all_idx = range(len(tokens))

    # Only considered a LU transition if initial landcover >= 3 consecutive years and final land cover >= 3 consecutive years and there is only 1 transition (i.e. WWWOOOOOOO OR OOOOWWWWWW)
    if re.fullmatch(r"(W{3,}O{3,}|O{3,}W{3,})", token_seq):
        return

    # Otherwise collapse to majority class across all years
    w_count = tokens.count("W")
    o_count = tokens.count("O")

    # If W and O have the same number of years, assume W
    if  w_count >= o_count:
        apply_tokens(lu_dict, all_idx, "W", node_code_map["wetland_glad_majority_years"])
    else:
        apply_tokens(lu_dict, all_idx, "O", node_code_map["water_glad_majority_years"])


def apply_regex_rules(lc_timeseries, driver, tcl_year, pre_2000_plantation, planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, gpw_cultiv_grass):

    # Create default token array and default node code array from LC timeseries
    tokens = [token_for_lc(v) for v in lc_timeseries]               #char array representing land use timeseries
    node_codes = [default_node_code(token) for token in tokens]     #int array representing class definition rules applied throughout the timeseries

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

    # Check if oil palm, tree crop or forest based on special cases
    extent_rule_applied = apply_extent_rules(lu_dict)

    if not extent_rule_applied:
        token_seq = "".join(lu_dict["tokens"]) #Creates a concat string
        if "S" in token_seq and not re.fullmatch(r"S+", token_seq):
            apply_built_transition(lu_dict)
        elif "C" in token_seq and not re.fullmatch(r"C+", token_seq):
            apply_crop_transition(lu_dict)
        elif re.fullmatch(r"F+", token_seq):
            apply_all_tall_veg(lu_dict)
        elif re.fullmatch(r"G+", token_seq):
            apply_all_short_veg(lu_dict)
        elif re.fullmatch(r"[FG]+", token_seq):
            apply_tall_short(lu_dict)
        elif re.fullmatch(r"[GB]+", token_seq):
            apply_short_bare(lu_dict)
        elif re.fullmatch(r"[FGBWO]+", token_seq) and re.search(r"[FGB]", token_seq) and re.search(r"[WO]", token_seq):
            apply_veg_bare_water(lu_dict)
        elif re.fullmatch(r"[WO]+", token_seq):
            apply_wetland_water(lu_dict)

    # Final token and node code timeseries
    final_tokens = lu_dict["tokens"]
    node_code_ts = lu_dict["node_codes"]

    # Convert final tokens to numeric LU codes
    lu_ts = [lu_token_map[token] for token in final_tokens]

    # Check that there is only one land use transition during the timeseries
    check_single_lu_transition(lu_dict, lu_ts)

    # Create transition timeseries: 2015_2016 through 2023_2024
    transition_ts = [int(f"{lu_ts[i]}{lu_ts[i + 1]}") for i in range(len(lu_ts) - 1)]

    # Create sequential unique LU summary ([3, 3, 3, 4, 4, 4, 2, 2, 2] -> [3, 4, 2] -> Forest to Grass to Crop)
    summary = []
    for lu in lu_ts:
        if not summary or lu != summary[-1]:
            summary.append(lu)

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
    # TODO: Read in as a union so only 1 tile set needed
    mangrove_extent_1996_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_1996"]
    mangrove_extent_2007_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2007"]
    mangrove_extent_2008_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2008"]
    mangrove_extent_2009_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2009"]
    mangrove_extent_2010_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2010"]
    mangrove_extent_2015_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2015"]
    mangrove_extent_2016_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2016"]
    mangrove_extent_2017_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2017"]
    mangrove_extent_2018_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2018"]
    mangrove_extent_2019_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2019"]
    mangrove_extent_2020_block = in_dict[f"{cn.mangrove_extent_processed_pattern}_2020"]

    # GPW cultivated grassland extent
    # TODO: Add gpw_cultiv_grass here
    # TODO: Read in as a union so only 1 tile set needed

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

    LU_summary_block = np.zeros(LC_2015_block.shape, dtype=np.uint64) #TODO: Switch back to 32

    # for year in cn.years_annual:
    #     out_dict[f"{cn.IPCC_class_pattern}_{year}"] =  np.zeros(LC_2015_block.shape, dtype=np.uint8)
    #     out_dict[f"{cn.IPCC_node_pattern}_{year}"] = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    # for year in cn.years_annual[:-1]:
    #     out_dict[f"{cn.IPCC_change_pattern}_{year}_{year+1}"] = np.zeros(LC_2015_block.shape, dtype=np.uint16)
    # out_dict[f"{cn.IPCC_summary_pattern}"] = np.zeros(LC_2016_block.shape, dtype=np.uint32)

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
            LC_timeseries = np.array([LC_2015, LC_2016, LC_2017, LC_2018, LC_2019, LC_2020, LC_2021, LC_2022, LC_2023, LC_2024]).astype('uint8')

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
            mang_1996 = mangrove_extent_1996_block[row, col]
            mang_2007 = mangrove_extent_2007_block[row, col]
            mang_2008 = mangrove_extent_2008_block[row, col]
            mang_2009 = mangrove_extent_2009_block[row, col]
            mang_2010 = mangrove_extent_2010_block[row, col]
            mang_2015 = mangrove_extent_2015_block[row, col]
            mang_2016 = mangrove_extent_2016_block[row, col]
            mang_2017 = mangrove_extent_2017_block[row, col]
            mang_2018 = mangrove_extent_2018_block[row, col]
            mang_2019 = mangrove_extent_2019_block[row, col]
            mang_2020 = mangrove_extent_2020_block[row, col]
            mang_timeseries = np.array([mang_1996, mang_2007, mang_2008, mang_2009, mang_2010, mang_2015, mang_2016, mang_2017, mang_2018, mang_2019, mang_2020]).astype('uint8')
            gmw_mangrove = bool(np.any(mang_timeseries == 1))

            #TODO: Add gpw_cultiv_grass here

            # Pass in values for regex rules
            LU_timeseries, node_code_timeseries, LU_change_timeseries, summary = (
                apply_regex_rules(LC_timeseries, driver, tcl_year, pre_2000_plantation, descals_planting_year, sdpt_oil_palm, sdpt_tree_crop, sdpt_planted_forest, gmw_mangrove, False))
            #TODO: Add gpw_cultiv_grass (currently set to False)

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

    out_dict[f"{cn.IPCC_summary_pattern}"] = LU_summary_block.copy()

    return out_dict



def calculate_and_upload_IPCC_land_use(bounds, download_dict_with_data_types, is_large_run, no_upload, output_folders, stage):

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
    # TODO: What stats do we want to know for input chunks?

    

    ### Part 3: IPCC land use assignment
    lu.print_and_log(f"Assigning IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}",False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)
    ipcc_start = time.time()

    out_dict = IPCC_land_use(layers)
    #print("out_dict:", out_dict)

    ipcc_end = time.time()
    lu.print_and_log(f"Done assigning IPCC land use in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"Memory usage after IPCC stage completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    lu.print_and_log(f"Assigned IPCC land use in {bounds_str} in {tile_id} in {round(ipcc_end - ipcc_start)} seconds: {uu.timestr()}",False, logger_worker)

    # Deletes all unnecessary input dictionaries before moving on
    in_dicts = [layers]
    [in_dict.clear() for in_dict in in_dicts]


    #TODO: Add Zarr step here


    ### Part 5: Calculates chunk stats
    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # # The relevant pixel area (m^2) file in s3
    # pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
    #
    # # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    # pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, 'Float32')
    # pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers
    for key, array in out_dict.items():
        chunk_stats.append(uu.calculate_ipcc_stats(array, key, bounds_str, tile_id, 'output_layer'))

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
    # TODO: updated to pixel counts per class or total pixel area per class. Update with LU_change and LU_summary



    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if no_upload == False:
        out_no_data_val = 0
        print("output_folders:", output_folders)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict.items():
            data_type = value.dtype.name
            print("key", key)
            print("data_type:", data_type)

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

    chunk_end_time = time.time()
    lu.print_and_log(f"{bounds_str} took {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    return return_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, run_date, run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'IPCC_land_use'
    model_type = 'standard_model'

    # Runs chunks in batches of specified size.
    # batch_size = 3200   # 6 batches to cover all chunks
    batch_size = 5      # For testing batch processing

    # Determines if arguments for start and end year are valid
    start_year = cn.first_model_year_annual
    end_year = cn.last_model_year_annual

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr()  # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")

    # Calculates the interval type, difference between start and end years of intervals, and the model output years for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(start_year, end_year, main_logger)

    # Returns a dataframe of chunk_ids and iso code from the GADM4.1 1x1 deg fishnet used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Placeholder tile_id to obtain the datatype of each input tile set. Overwritten when chunks are assigned and analyzed.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to LU assignment).
    download_dict = {
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
        #TODO: Add global pasture watch data

    # GMW mangrove extent timeseries
    for year in cn.mangrove_extent_years:
        download_dict[f"{cn.mangrove_extent_processed_pattern}_{year}"] = f"{cn.mangrove_extent_processed_dir}{year}/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_{year}.tif"


    # Replaces the placeholder parts of the input paths with relevant values
    # download_dict = {key: value.replace("CHUNK_SIZE", '40000') for key, value in download_dict.items()}
    # download_dict = {key: value.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning) for key, value in download_dict.items()}
    #TODO: Delete?

    print("Download dictionary::")
    for key, item in download_dict.items():
        print(f"{key}: {item}")

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


    summary_dir = ( cn.IPCC_summary_dir .replace("RUN_DATE", run_date) .replace("CHUNK_SIZE", str(chunk_size_pixels)))

    output_dir_list = sorted(class_node_output_dirs + change_output_dirs + [summary_dir])

    main_logger.info(f"output_dir_list for {stage}:")
    for item in output_dir_list:
        main_logger.info(f"  {item}")

    # TODO: Add zarr step here

    ### Step 2: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log" + "\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    all_results = []
    all_1x1_stats = []
    success_count = 0  # Count of successful chunks

    # TODO: Run locally or in coiled

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        if run_local:
            batch_results = [calculate_and_upload_IPCC_land_use(chunk, download_dict_with_data_types, True, no_upload, output_dir_list, stage)
                             for chunk in chunk_batch]
            all_results.extend(batch_results)

        else:
            futures = [client.submit(calculate_and_upload_IPCC_land_use, chunk, download_dict_with_data_types, True, no_upload, output_dir_list, stage)
                       for chunk in chunk_batch]
            batch_results = client.gather(futures)
            all_results.extend(batch_results)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, True, main_logger, batch_results)
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
        if client is not None:
            del futures
            client.run(gc.collect)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)

    #TODO: Add from stage 4 on




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

    # Create the cluster with command line arguments
    main(cluster_name, run_date, run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size_deg=chunk_size_deg, first_chunks=first_chunks, log_note=log_note)





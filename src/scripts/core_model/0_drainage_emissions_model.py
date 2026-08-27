"""
drainage_emissions_model.py
Organic‑soils drainage and fire emissions model

* full decision‑tree logic (drainage & burned‑area) kept intact
* parallel execution via Coiled / Dask (mirrors LULUCF model)
* outputs saved to S3 using universal_utilities.save_and_upload_small_raster_set
* optionally target specific 10x10 degree tiles via ``--tile_ids``
* or process the entire list of tiles via ``--full_model``
"""

from __future__ import annotations
import argparse
import concurrent.futures
import sys
from datetime import datetime
from typing import Optional

import dask.bag
import numpy as np
import xarray as xr
from numba import jit, types
from numba.typed import Dict

# project utilities
from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu
from src.scripts.utilities import numba_utilities as nu
from src.scripts.utilities import drainage_emission_factors as defac
from src.scripts.utilities import burned_area_emission_factors as baf
from src.scripts.core_model import state_codes as sc
from src.scripts.utilities import drainage_zarr_utilities as zu

# ----------------------------------------------------------------------
# constants pulled into locals for Numba speed
# ----------------------------------------------------------------------
c_to_co2 = np.float32(cn.c_to_co2)
n2o_n_to_n2o = np.float32(cn.n2o_n_to_n2o)
gwp_ch4 = np.float32(cn.gwp_ch4)
gwp_n2o = np.float32(cn.gwp_n2o)

# Determine the width of finalized (padded) state codes robustly, then derive a
# divisor that extracts the two-digit "root" (peat/non-peat and peat class)
# from any padded code. This makes root extraction independent of whether the
# states are 6-digit, 8-digit, etc.
STATE_PAD_DIGITS = max(len(str(code)) for code in sc.ALL_DRAINED_STATE_CODES)
# Default root divisor for the configured pad width (e.g., 10**6 for 8-digit states)
ROOT_DIVISOR = 10 ** (STATE_PAD_DIGITS - 2)

VALID_DRAINED_STATE_CODES = np.array(
    [np.uint32(int(code)) for code in sc.ALL_DRAINED_STATE_CODES], dtype=np.uint32
)
VALID_BURNED_STATE_CODES = np.array(
    [np.uint32(int(code)) for code in sc.ALL_BURNED_STATE_CODES], dtype=np.uint32
)

# Default peat probability threshold for the OGH dataset. This matches the
# preprocessing threshold previously applied during tiling.
DEFAULT_OGH_THRESHOLD = 10.0
DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M = 500.0
# Engert is a road-density surface, not a distance raster, so it cannot use the
# distance threshold directly. The processed model input is in km of road per km2
# (the raw m/km2 product is divided by 1000 during preprocessing; verified ratio).
# To make Engert respond to the same drainage-distance sweep we convert the
# distance threshold T (m) into an analogous density cutoff via a parallel-road
# approximation: for density rho (km/km2) the mean road spacing is s = 1000 / rho
# (m) and the mean distance to the nearest road is s / 4 = 250 / rho (m). A cell is
# road-drained when that implied distance is within T, i.e. when
#     rho >= 250 / T = ENGERT_DENSITY_DISTANCE_CONSTANT_KM_KM2 / T.
# Approximate (1 km granularity; assumes processed Engert units are km road per
# km2). Revisit the constant if the Engert units ever change.
ENGERT_DENSITY_DISTANCE_CONSTANT_KM_KM2 = 250.0
# Engert stays in the regional set so --exclude_regional_linear_features can still
# drop it. Dadap is no longer here: it is now a distance-to-canal surface that
# responds to --drainage_distance_threshold_m directly.
REGIONAL_LINEAR_FEATURE_LAYERS = ("engert",)
PEAT_THRESHOLD_SCENARIO_ALIASES = {
    "baseline": "baseline",
    "base": "baseline",
    "operational": "baseline",
    "low": "low_area",
    "low_area": "low_area",
    "low-area": "low_area",
    "high": "high_area",
    "high_area": "high_area",
    "high-area": "high_area",
}


def exclude_regional_linear_feature_layers(download_dict: dict) -> dict:
    """Return *download_dict* without fixed regional linear-feature inputs."""

    return {
        key: value
        for key, value in download_dict.items()
        if key not in REGIONAL_LINEAR_FEATURE_LAYERS
    }


def validate_drainage_distance_threshold_m(value: float) -> float:
    """Validate the distance threshold for Dadap/OSM/GRIP drainage rasters."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--drainage_distance_threshold_m must be a numeric distance in meters."
        ) from exc
    if not np.isfinite(threshold):
        raise ValueError("--drainage_distance_threshold_m must be finite.")
    if threshold <= 0:
        raise ValueError("--drainage_distance_threshold_m must be positive.")
    if threshold >= 1000:
        raise ValueError(
            "--drainage_distance_threshold_m must be less than 1000 m because "
            "the current distance rasters are clipped at 1000 m. Update the "
            "distance-raster preprocessing before using a larger threshold."
        )
    return threshold

forest_code = cn.ipcc_codes["forest"]
cropland_code = cn.ipcc_codes["cropland"]
settlement_code = cn.ipcc_codes["settlement"]
wetland_code = cn.ipcc_codes["wetland"]
grassland_code = cn.ipcc_codes["grassland"]
otherland_code = cn.ipcc_codes["otherland"]

boreal_code = cn.ecozone_codes["boreal"]
temperate_code = cn.ecozone_codes["temperate"]
tropical_code = cn.ecozone_codes["tropical"]

poor_nutrient_code = cn.nutrient_status_codes["poor"]
rich_nutrient_code = cn.nutrient_status_codes["rich"]
unknown_nutrient_code = cn.nutrient_status_codes["unknown"]


long_rotation_code = cn.plantation_type_codes["long_rotation"]
short_rotation_code = cn.plantation_type_codes["short_rotation"]
oil_palm_code = cn.plantation_type_codes["oil_palm"]

# helpers for numba dict lookups
@jit(nopython=True)
def lookup_efs(key, table, coastal_code=0):
    """Return (values, is_missing) from drainage factor table.
    Note: allow coastal keys to resolve to their factors (no short-circuit)."""
    if key in table:
        return table[key], False
    return defac.ZERO_ARRAY, True


@jit(nopython=True)
def lookup_befs(key, table):
    """Return (values, is_missing) from burned-area factor table."""
    if key in table:
        return table[key], False
    return baf.ZERO_ARRAY, True

# ----------------------------------------------------------------------
# full decision‑tree function (unchanged logic, ASCII comments only)
# ----------------------------------------------------------------------
@jit(nopython=True)
def _calculate_drainage_and_emissions_numba(
    in_dict_uint8,
    in_dict_int16,
    in_dict_float32,
    drainage_table,
    burned_table,
    count_burned_years,
    drainage_distance_threshold_m,
):
    """
    1) Drainage classification & state
    2) Drainage‑based emissions (CO2, N2O, CH4, off‑site CO2, total CO2e)
    3) Burned‑area emissions (CO2, CO, CH4, total CO2e) if burned layer present.
       When ``count_burned_years`` is ``True``, burned emissions are multiplied
       by the number of burned years for each pixel within the interval.
    Returns two numba typed dicts:
    - uint32 layers: categorical classification (drained_soil, drained_state,
      burned_state, burned_years_count)
    - float32 layers: emissions totals
    """

    out_dict_uint32 = Dict.empty(types.unicode_type, types.uint32[:, :])
    out_dict_float32 = Dict.empty(types.unicode_type, types.float32[:, :])

    # The maximum allowed digits for classification nodes.
    max_digits_state = STATE_PAD_DIGITS

    # required inputs --------------------------------------------------
    peat_block = in_dict_uint8["peat"]
    land_cover_block = in_dict_uint8["land_cover"]
    planted_forest_type_block = in_dict_uint8["planted_forest_type"]
    dadap_block = in_dict_float32["dadap"]  # distance-to-canal in metres (0 = nodata)
    osm_roads_block = in_dict_float32["osm_roads"]
    osm_canals_block = in_dict_float32["osm_canals"]
    engert_block = in_dict_float32["engert"]  # road density in km of road per km2
    grip_block = in_dict_float32["grip"]
    extraction_block = in_dict_uint8["extraction"]
    ecozone_block = in_dict_int16["climate_domain"]
    descals_type_block = in_dict_int16["descals_type"]
    mangrove_block = in_dict_uint8["mangrove_extent"]
    tidal_marsh_block = in_dict_uint8["tidal_marsh"]

    # optional burned‑area mask
    burned_block = None
    for k in in_dict_uint8.keys():
        if k.startswith("burned_area_combined_"):
            burned_block = in_dict_uint8[k]
            break

    rows, cols = peat_block.shape

    # output arrays ----------------------------------------------------
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_block = np.zeros((rows, cols), dtype=np.uint32)

    drained_co2_out = np.zeros((rows, cols), dtype=np.float32)
    drained_n2o_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_land_out = np.zeros((rows, cols), dtype=np.float32)
    drained_ch4_ditch_out = np.zeros((rows, cols), dtype=np.float32)
    drained_co2_offsite_out = np.zeros((rows, cols), dtype=np.float32)
    drained_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)

    burned_state_out = np.zeros((rows, cols), dtype=np.uint32)
    burned_co2_out = np.zeros((rows, cols), dtype=np.float32)
    burned_co_out = np.zeros((rows, cols), dtype=np.float32)
    burned_ch4_out = np.zeros((rows, cols), dtype=np.float32)
    burned_total_co2e_out = np.zeros((rows, cols), dtype=np.float32)
    burned_years_count_out = np.zeros((rows, cols), dtype=np.uint32)

    # Engert road-density cutoff analogous to the distance threshold (see
    # ENGERT_DENSITY_DISTANCE_CONSTANT_KM_KM2): density >= 250/T (km/km2) means the
    # implied mean distance to the nearest road is within the distance threshold.
    engert_density_threshold = (
        ENGERT_DENSITY_DISTANCE_CONSTANT_KM_KM2 / drainage_distance_threshold_m
    )

    # main pixel loop --------------------------------------------------
    for row in range(rows):
        for col in range(cols):

            # pixel values
            peat = peat_block[row, col]
            land_cover = land_cover_block[row, col]
            valid_land_cover = land_cover in (
                forest_code,
                cropland_code,
                settlement_code,
                wetland_code,
                grassland_code,
                otherland_code,
            )
            planted_forest_type = planted_forest_type_block[row, col]
            dadap = dadap_block[row, col]
            osm_roads = osm_roads_block[row, col]
            osm_canals = osm_canals_block[row, col]
            engert = engert_block[row, col]
            grip = grip_block[row, col]
            extraction = extraction_block[row, col]
            ecozone = ecozone_block[row, col]
            descals_type = descals_type_block[row, col]
            mangrove = mangrove_block[row, col]
            tidal_marsh = tidal_marsh_block[row, col]

            # Descals oil palm overrides SDPT plantation subtype. Any
            # Descals-positive pixel is treated as oil palm for plantation routing.
            effective_plantation_type = (
                oil_palm_code if descals_type > 0 else planted_forest_type
            )
            has_effective_plantation = effective_plantation_type > 0

            # default nutrient status by ecozone
            if ecozone == boreal_code:
                nutrient = poor_nutrient_code
            elif ecozone == temperate_code:
                nutrient = rich_nutrient_code
            else:
                nutrient = unknown_nutrient_code

            if mangrove > 0:
                coastal_code = 1
            elif tidal_marsh > 0:
                coastal_code = 2
            else:
                coastal_code = 0

            # initialize
            ef_co2 = np.float32(0.0)
            ef_n2o = np.float32(0.0)
            ef_ch4_land = np.float32(0.0)
            ef_ch4_ditch = np.float32(0.0)
            ef_co2_offsite = np.float32(0.0)
            frac_ditch = np.float32(0.0)
            node = 0
            emission_node = 0
            drained = False

            # A) Drainage classification ----------------------------------
            if peat > 0 and valid_land_cover:
                node = nu.accrete_node(node, 1)
                if (
                    dadap > 0 and dadap <= drainage_distance_threshold_m
                ) or (
                    osm_canals > 0 and osm_canals <= drainage_distance_threshold_m
                ):
                    node = nu.accrete_node(node, 1)
                    drained = True
                elif (
                    (engert >= engert_density_threshold)
                    or (grip > 0 and grip <= drainage_distance_threshold_m)
                    or (osm_roads > 0 and osm_roads <= drainage_distance_threshold_m)
                ):
                    node = nu.accrete_node(node, 2)
                    drained = True
                elif extraction > 0:
                    node = nu.accrete_node(node, 5)
                    drained = True
                elif land_cover in (cropland_code, settlement_code):
                    node = nu.accrete_node(node, 3)
                    drained = True
                elif has_effective_plantation:
                    node = nu.accrete_node(node, 4)
                    drained = True
                else:
                    node = nu.accrete_node(node, 6)

                soil_block[row, col] = 2 if drained else 1
            else:
                node = 0  # non-peat root uses explicit zero code
                soil_block[row, col] = 0  # not peat

            if peat > 0 and coastal_code:
                node = nu.accrete_node(node, 9)
                node = nu.accrete_node(node, coastal_code)


            # B) Drainage emission factors --------------------------------
            if soil_block[row, col] == 2:  # only if drained
                emission_node = nu.accrete_node(emission_node, 1)
                key = ""
                vals = defac.ZERO_ARRAY
                missing = False

                # BOREAL ---------------------------------------------------
                if ecozone == boreal_code:
                    emission_node = nu.accrete_node(emission_node, 1)
                    category_node = emission_node
                    if coastal_code:
                        emission_node = nu.accrete_node(category_node, 9)
                        emission_node = nu.accrete_node(emission_node, coastal_code)
                        if coastal_code == 1:
                            key = "coastal_mangrove"
                        else:
                            key = "coastal_tidal_marsh"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(category_node, 1)
                        key = "boreal_extraction"
                    elif land_cover == forest_code:
                        forest_node = nu.accrete_node(category_node, 3)
                        emission_node = forest_node
                        if nutrient == poor_nutrient_code:
                            emission_node = nu.accrete_node(forest_node, 1)
                            key = "boreal_forest_poor"
                        elif nutrient == rich_nutrient_code:
                            emission_node = nu.accrete_node(forest_node, 2)
                            key = "boreal_forest_rich"
                    elif land_cover == grassland_code:
                        emission_node = nu.accrete_node(category_node, 4)
                        key = "boreal_grassland"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(category_node, 5)
                        key = "boreal_cropland"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(category_node, 6)
                        key = "boreal_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(category_node, 7)
                        key = "boreal_wetland"
                    elif land_cover == otherland_code:
                        emission_node = nu.accrete_node(category_node, 8)
                        key = "boreal_otherland"
                    else:
                        missing = True

                    vals, missing = lookup_efs(key, drainage_table, coastal_code)
                    if missing:
                        raise ValueError(
                            "Missing drainage emission factor for boreal route"
                        )

                # TEMPERATE -----------------------------------------------
                elif ecozone == temperate_code:
                    emission_node = nu.accrete_node(emission_node, 2)
                    category_node = emission_node
                    if coastal_code:
                        emission_node = nu.accrete_node(category_node, 9)
                        emission_node = nu.accrete_node(emission_node, coastal_code)
                        if coastal_code == 1:
                            key = "coastal_mangrove"
                        else:
                            key = "coastal_tidal_marsh"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(category_node, 1)
                        key = "temperate_extraction"
                    elif land_cover == forest_code:
                        emission_node = nu.accrete_node(category_node, 3)
                        key = "temperate_forest"
                    elif land_cover == grassland_code:
                        grassland_node = nu.accrete_node(category_node, 4)
                        emission_node = grassland_node
                        if nutrient == poor_nutrient_code:
                            emission_node = nu.accrete_node(grassland_node, 1)
                            key = "temperate_grassland_poor"
                        elif nutrient == rich_nutrient_code:
                            emission_node = nu.accrete_node(grassland_node, 2)
                            key = "temperate_grassland_rich"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(category_node, 5)
                        key = "temperate_cropland"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(category_node, 6)
                        key = "temperate_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(category_node, 7)
                        key = "temperate_wetland"
                    elif land_cover == otherland_code:
                        emission_node = nu.accrete_node(category_node, 8)
                        key = "temperate_otherland"
                    else:
                        missing = True

                    vals, missing = lookup_efs(key, drainage_table, coastal_code)
                    if missing:
                        raise ValueError(
                            "Missing drainage emission factor for temperate route"
                        )

                # TROPICAL -------------------------------------------------
                elif ecozone == tropical_code:
                    emission_node = nu.accrete_node(emission_node, 3)
                    category_node = emission_node
                    if coastal_code:
                        emission_node = nu.accrete_node(category_node, 9)
                        emission_node = nu.accrete_node(emission_node, coastal_code)
                        if coastal_code == 1:
                            key = "coastal_mangrove"
                        else:
                            key = "coastal_tidal_marsh"
                    elif extraction > 0:
                        emission_node = nu.accrete_node(category_node, 1)
                        key = "tropical_extraction"
                    elif has_effective_plantation:
                        plantation_node = nu.accrete_node(category_node, 2)
                        emission_node = plantation_node
                        if effective_plantation_type == long_rotation_code:
                            emission_node = nu.accrete_node(plantation_node, 1)
                            key = "tropical_long_rotation"
                        elif effective_plantation_type == short_rotation_code:
                            emission_node = nu.accrete_node(plantation_node, 2)
                            key = "tropical_short_rotation"
                        elif effective_plantation_type == oil_palm_code:
                            emission_node = nu.accrete_node(plantation_node, 3)
                            key = "tropical_oil_palm"
                    elif land_cover == forest_code:
                        emission_node = nu.accrete_node(category_node, 3)
                        key = "tropical_forest"
                    elif land_cover == grassland_code:
                        emission_node = nu.accrete_node(category_node, 4)
                        key = "tropical_grassland"
                    elif land_cover == cropland_code:
                        emission_node = nu.accrete_node(category_node, 5)
                        key = "tropical_cropland"
                    elif land_cover == settlement_code:
                        emission_node = nu.accrete_node(category_node, 6)
                        key = "tropical_settlement"
                    elif land_cover == wetland_code:
                        emission_node = nu.accrete_node(category_node, 7)
                        key = "tropical_wetland"
                    elif land_cover == otherland_code:
                        emission_node = nu.accrete_node(category_node, 8)
                        key = "tropical_otherland"
                    else:
                        missing = True

                    vals, missing = lookup_efs(key, drainage_table, coastal_code)
                    if missing:
                        raise ValueError(
                            "Missing drainage emission factor for tropical route"
                        )

                else:
                    if coastal_code:
                        emission_node = nu.accrete_node(emission_node, 9)
                        emission_node = nu.accrete_node(emission_node, coastal_code)
                        if coastal_code == 1:
                            key = "coastal_mangrove"
                        else:
                            key = "coastal_tidal_marsh"
                        vals, missing = lookup_efs(key, drainage_table, coastal_code)
                    else:
                        missing = True
                if not missing:
                    ef_co2 = vals[0]
                    ef_n2o = vals[1]
                    ef_ch4_land = vals[2]
                    ef_ch4_ditch = vals[3]
                    ef_co2_offsite = vals[4]
                    frac_ditch = vals[5]

                    # calculate drainage emissions ---------------------------
                    (
                        co2_em,
                        n2o_co2e,
                        ch4_land_co2e,
                        ch4_ditch_co2e,
                        co2_off,
                        total_co2e,
                    ) = nu.calculate_drainage_emissions_co2e(
                        ef_co2,
                        ef_n2o,
                        ef_ch4_land,
                        ef_ch4_ditch,
                        ef_co2_offsite,
                        frac_ditch,
                        c_to_co2,
                        n2o_n_to_n2o,
                        gwp_n2o,
                        gwp_ch4,
                    )

                    drained_co2_out[row, col] = co2_em
                    drained_n2o_out[row, col] = n2o_co2e
                    drained_ch4_land_out[row, col] = ch4_land_co2e
                    drained_ch4_ditch_out[row, col] = ch4_ditch_co2e
                    drained_co2_offsite_out[row, col] = co2_off
                    drained_total_co2e_out[row, col] = total_co2e

            elif soil_block[row, col] == 1:
                # Undrained peat: append ecozone digit (1=boreal, 2=temperate,
                # 3=tropical, 4=other_domain). No EF lookup — undrained EF = 0.
                if ecozone == boreal_code:
                    emission_node = nu.accrete_node(emission_node, 1)
                elif ecozone == temperate_code:
                    emission_node = nu.accrete_node(emission_node, 2)
                elif ecozone == tropical_code:
                    emission_node = nu.accrete_node(emission_node, 3)
                else:
                    emission_node = nu.accrete_node(emission_node, 4)

            if emission_node > 0:
                node = nu.accrete_node(node, emission_node)

            if node > (10 ** max_digits_state) - 1:
                raise ValueError("Maximum state digits exceeded")

            state_block[row, col] = nu.pad_to_6_digits(node, max_digits_state)

            # C) Burned‑area emissions -------------------------------------
            burned_node = 0
            burned_emission_node = 0
            if burned_block is not None:
                burned_val = burned_block[row, col]
                burned_years_count = (
                    burned_val if count_burned_years else (1 if burned_val > 0 else 0)
                )
                burned_years_count_out[row, col] = burned_years_count
                if burned_val > 0 and soil_block[row, col] in (1, 2):

                    if ecozone == boreal_code:
                        burned_node = nu.accrete_node(burned_node, 1)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        if soil_block[row, col] == 2:
                            bkey = "boreal_drained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        else:
                            bkey = "boreal_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 2)

                    elif ecozone == temperate_code:
                        burned_node = nu.accrete_node(burned_node, 2)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 2)
                        if soil_block[row, col] == 2:
                            bkey = "temperate_drained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                        else:
                            bkey = "temperate_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 2)

                    elif ecozone == tropical_code:
                        burned_node = nu.accrete_node(burned_node, 3)
                        burned_emission_node = nu.accrete_node(burned_emission_node, 3)
                        if soil_block[row, col] == 2:
                            if (
                                land_cover == cropland_code
                                or has_effective_plantation
                            ):
                                bkey = "tropical_drained_crop_or_plantation"
                                burned_emission_node = nu.accrete_node(burned_emission_node, 1)
                            else:
                                bkey = "tropical_drained_other"
                                burned_emission_node = nu.accrete_node(burned_emission_node, 2)
                        else:
                            bkey = "tropical_undrained"
                            burned_emission_node = nu.accrete_node(burned_emission_node, 3)
                    else:
                        burned_node = nu.accrete_node(burned_node, 4)
                        bkey = "other"
                        burned_emission_node = nu.accrete_node(burned_emission_node, 4)

                    bvals, bmissing = lookup_befs(bkey, burned_table)
                    if bmissing:
                        raise ValueError("Missing burned-area emission factor")
                    gef_co2 = bvals[0]
                    gef_co = bvals[1]
                    gef_ch4 = bvals[2]
                    mass_burnt = bvals[3]

                    multiplier = burned_years_count
                    (
                        burn_co2,
                        burn_co,
                        burn_ch4,
                        burn_total_co2e,
                    ) = nu.calculate_burned_area_emissions(
                        np.float32(mass_burnt) * np.float32(multiplier),
                        np.float32(gef_co2),
                        np.float32(gef_co),
                        np.float32(gef_ch4),
                        gwp_ch4,
                    )

                    burned_co2_out[row, col] = burn_co2
                    burned_co_out[row, col] = burn_co
                    burned_ch4_out[row, col] = burn_ch4
                    burned_total_co2e_out[row, col] = burn_total_co2e

            if burned_emission_node > 0:
                burned_node = nu.accrete_node(burned_node, burned_emission_node)

            if burned_node > (10 ** max_digits_state) - 1:
                raise ValueError("Maximum burned state digits exceeded")

            burned_state_out[row, col] = nu.pad_to_6_digits(burned_node, max_digits_state)

    # pack outputs ----------------------------------------------------------
    out_dict_uint32["drained_soil"] = soil_block
    out_dict_uint32["drained_state"] = state_block
    out_dict_uint32["burned_state"] = burned_state_out
    out_dict_uint32["burned_years_count"] = burned_years_count_out

    out_dict_float32["drained_co2_Mg_CO2_ha_yr"] = drained_co2_out
    out_dict_float32["drained_n2o_Mg_CO2e_ha_yr"] = drained_n2o_out
    out_dict_float32["drained_ch4_land_Mg_CO2e_ha_yr"] = drained_ch4_land_out
    out_dict_float32["drained_ch4_ditch_Mg_CO2e_ha_yr"] = drained_ch4_ditch_out
    out_dict_float32["drained_co2_offsite_Mg_CO2_ha_yr"] = drained_co2_offsite_out
    out_dict_float32["drained_total_Mg_CO2e_ha_yr"] = drained_total_co2e_out
    out_dict_float32["burned_co2_Mg_CO2_ha"] = burned_co2_out
    out_dict_float32["burned_co_Mg_CO_ha"] = burned_co_out
    out_dict_float32["burned_ch4_Mg_CO2e_ha"] = burned_ch4_out
    out_dict_float32["burned_total_Mg_CO2e_ha"] = burned_total_co2e_out
    return out_dict_uint32, out_dict_float32


def organic_soil_from_drained_soil(drained_soil: np.ndarray) -> np.ndarray:
    """Return a 0/1 thresholded organic-soil mask from the drainage classifier."""

    return (np.asarray(drained_soil) > 0).astype(np.uint8)


def calculate_drainage_and_emissions(
    in_dict_uint8,
    in_dict_int16,
    in_dict_float32,
    drainage_table,
    burned_table,
    count_burned_years,
    drainage_distance_threshold_m,
):
    drainage_distance_threshold_m = validate_drainage_distance_threshold_m(
        drainage_distance_threshold_m
    )
    return _calculate_drainage_and_emissions_numba(
        in_dict_uint8,
        in_dict_int16,
        in_dict_float32,
        drainage_table,
        burned_table,
        count_burned_years,
        drainage_distance_threshold_m,
    )


# ----------------------------------------------------------------------
# helper: combine burned layers across interval
# ----------------------------------------------------------------------
def combine_burned_area(
    layers: dict,
    iv_start: int,
    iv_end: int,
    count_burned_years: bool = True,
):
    """Combine annual burned area layers over an interval.

    Parameters
    ----------
    layers : dict
        Dictionary of layer arrays.
    iv_start, iv_end : int
        Interval start and end years.
    count_burned_years : bool, optional
        If ``True``, return the number of burned years per pixel. Otherwise a
        binary mask is returned.
    """
    shape = None
    combined = None
    for yr in range(iv_start, iv_end + 1):
        key = f"{cn.burned_area_final_pattern}_{yr}"
        if key in layers:
            arr = layers[key]
            if shape is None:
                shape = arr.shape
                combined = np.zeros(shape, dtype=np.uint8)
            combined += (arr > 0).astype(np.uint8)
    if combined is not None:
        if not count_burned_years:
            combined[combined > 0] = 1
        layers[f"burned_area_combined_{iv_start}_{iv_end}"] = combined


def binarize_extraction_layer(layers: dict):
    """Ensure extraction is consumed as a uint8 0/1 presence layer."""

    if "extraction" in layers:
        layers["extraction"] = (layers["extraction"] > 0).astype(np.uint8)


def validate_land_cover_on_organic_soil(
    layers: dict,
    *,
    tile_id: str,
    bounds_string: str,
    iv_start: int,
    iv_end: int,
) -> None:
    """Reject organic-soil pixels without an explicit IPCC land-use class."""

    peat = layers["peat"]
    land_cover = layers["land_cover"]
    valid_codes = np.asarray(sorted(cn.ipcc_codes.values()), dtype=np.uint8)
    invalid_on_peat = (peat > 0) & ~np.isin(land_cover, valid_codes)
    invalid_count = int(np.count_nonzero(invalid_on_peat))
    if not invalid_count:
        return
    invalid_values = np.unique(land_cover[invalid_on_peat]).tolist()
    raise uu.RequiredInputRasterError(
        "Land cover is missing or invalid on "
        f"{invalid_count} organic-soil pixels in {tile_id} {bounds_string} "
        f"for {iv_start}-{iv_end}; values={invalid_values}. Refusing to route "
        "unknown class values to Otherland."
    )


def validate_climate_domain_source_codes(
    layers: dict,
    *,
    tile_id: str,
    bounds_string: str,
) -> None:
    """Reject source climate-domain codes outside the configured legend."""

    climate = layers["climate_domain"]
    land_cover = layers["land_cover"]
    valid_codes = np.asarray(sorted(cn.climate_domain_remap), dtype=np.int16)
    valid_land_cover_codes = np.asarray(
        sorted(cn.ipcc_codes.values()),
        dtype=np.uint8,
    )
    modeled_pixels = np.isin(land_cover, valid_land_cover_codes)
    invalid = modeled_pixels & ~np.isin(climate, valid_codes)
    invalid_count = int(np.count_nonzero(invalid))
    if not invalid_count:
        return
    invalid_values = np.unique(climate[invalid]).tolist()
    raise uu.RequiredInputRasterError(
        "Climate-domain source contains "
        f"{invalid_count} modeled pixels outside its configured legend in {tile_id} "
        f"{bounds_string}; values={invalid_values}."
    )


# ----------------------------------------------------------------------
# per‑chunk wrapper
# ----------------------------------------------------------------------
def calculate_and_upload_drainage(
    bounds,
    typed_dict,
    is_final,
    no_upload,
    iv_start,
    iv_end,
    closing_year,
    peat_dataset="ogh",
    run_name="ogh_standard_model",
    peat_threshold: Optional[float | dict[int, float]] = None,
    count_burned_years: bool = True,
    emission_factor_variant: str = "default",
    mega_zarr_path: Optional[str] = None,
    outputs_to_zarr: Optional[list[str]] = None,
    interval_end_years: Optional[list[int]] = None,
    run_date: Optional[str] = None,
    required_layers: Optional[set[str]] = None,
    drainage_distance_threshold_m: float = DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M,
):
    """Process a single chunk for a given interval.

    Parameters
    ----------
    bounds : list[float]
        Bounding box for the chunk.
    typed_dict : dict
        Dictionary of input raster paths keyed by layer name.
    is_final : bool
        Whether the run covers the full domain.
    no_upload : bool
        If ``True``, skip uploading outputs to S3.
    iv_start, iv_end : int
        Interval start and end years.
    closing_year : int
        Land cover composite year for this interval.
    peat_dataset : str, optional
        Peat mask dataset name.
    run_name : str, optional
        Model run identifier used to label output paths.
    peat_threshold : float, dict[int, float], or None, optional
        Threshold applied to the peat probability layer when using the OGH
        dataset. A scalar applies a single global threshold. A dict maps
        ecozone codes to per-biome thresholds (e.g., ``{1: 0.15, 2: 0.30,
        3: 0.20, 0: 0.23}``). Values greater than or equal to the threshold
        are treated as peat. ``None`` disables thresholding.
    count_burned_years : bool, optional
        If ``True``, count the number of burned years within the interval and
        multiply burned emissions accordingly.
    emission_factor_variant : {"low", "default", "high"}, optional
        Select which emission factor table set to use for drainage and
        burned‑area emissions.
    mega_zarr_path : str, optional
        Path to the global mega-zarr store for chunk output writes.
    outputs_to_zarr : list[str], optional
        Output datasets to populate into the mega-zarr.
    interval_end_years : list[int], optional
        Ordered year index used to locate the interval end year in zarr.
        This can be the full-model year index when populating a global store.
    run_date : str, optional
        Date string (YYYYMMDD) used in raster output paths. When ``None``,
        falls back to ``cn.today_date``.
    required_layers : set[str], optional
        Input layers that must be readable with the exact expected window
        shape. The driver always requires land cover and climate in the model
        domain, and requires peat where preflight found a stored peat tile.
    drainage_distance_threshold_m : float, optional
        Distance threshold, in meters, for Dadap canals, OSM canals, OSM roads,
        and GRIP roads. Must be positive and less than 1000 m for the current
        distance-raster preprocessing.
    """

    drainage_distance_threshold_m = validate_drainage_distance_threshold_m(
        drainage_distance_threshold_m
    )
    logger = lu.setup_logging_worker()
    bstr = uu.boundstr(bounds)
    tid = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_px = uu.calc_chunk_length_pixels(bounds)
    chunk_stats = []
    required_layers = set(
        required_layers
        if required_layers is not None
        else {"land_cover", "climate_domain", "peat"}
    )

    lu.print_and_log(
        f"Processing {bstr} {iv_start}-{iv_end} using land cover {closing_year}",
        is_final,
        logger,
    )

    # replace {tile_id} with actual
    import copy
    import re

    dwn = copy.deepcopy(typed_dict)
    for k, (uri, dt) in list(dwn.items()):
        dwn[k] = (re.sub(cn.tile_id_pattern, tid, uri), dt)

    # Drop burned area years outside the current interval
    for k in list(dwn.keys()):
        if k.startswith(cn.burned_area_final_pattern):
            try:
                yr = int(k.split("_")[-1])
            except ValueError:
                continue
            if yr < iv_start or yr > iv_end:
                dwn.pop(k)
    lu.print_and_log(
        f"Using burned area years {[k for k in dwn if k.startswith(cn.burned_area_final_pattern)]}",
        is_final,
        logger,
    )
    lu.print_and_log(
        f"Required input layers: {sorted(required_layers)}",
        is_final,
        logger,
    )

    if not uu.check_for_tile(dwn, is_final, logger):
        return f"Skipped {bstr} (tile absent)", chunk_stats

    # ensure burned keys exist
    for yr in range(iv_start, iv_end + 1):
        dwn.setdefault(f"{cn.burned_area_final_pattern}_{yr}", (None, "Byte"))

    # download chunk windows in parallel
    futs = uu.queue_chunk_downloads(
        bounds,
        dwn,
        chunk_px,
        logger,
        is_final=is_final,
        required_layers=required_layers,
    )
    layers = {}
    for fut in concurrent.futures.as_completed(futs):
        layers[futs[fut]] = fut.result()

    # fill missing layers with zeros
    uint8 = [
        "peat",
        "planted_forest_type",
        "extraction",
        "land_cover",
        "mangrove_extent",
        "tidal_marsh",
    ]
    uint8 += [
        f"{cn.burned_area_final_pattern}_{yr}"
        for yr in range(iv_start, iv_end + 1)
    ]
    int16 = ["climate_domain", "descals_type"]
    float32 = [
        "dadap",
        "osm_roads",
        "osm_canals",
        "engert",
        "grip",
    ]
    layers = uu.fill_missing_input_layers_with_no_data(
        layers, uint8, int16, [], float32, bstr, tid, is_final, logger
    )

    # Remap FAO ecozone values to simplified climate domain codes.
    # Done before peat thresholding so per-biome thresholds can use
    # the remapped ecozone codes.
    if "climate_domain" in layers:
        validate_climate_domain_source_codes(
            layers,
            tile_id=tid,
            bounds_string=bstr,
        )
        cd = layers["climate_domain"].astype(np.int16, copy=False)
        remapped = np.zeros_like(cd, dtype=np.int16)
        for src_val, dst_val in cn.climate_domain_remap.items():
            remapped[cd == src_val] = np.int16(dst_val)
        layers["climate_domain"] = remapped

    if peat_dataset in {"ogh", "ogh_unthresholded"} and peat_threshold is not None:
        peat_layer = layers.get("peat")
        if peat_layer is not None:
            if isinstance(peat_threshold, dict):
                cd = layers["climate_domain"]
                binary_peat = np.zeros_like(peat_layer, dtype=np.uint8)
                for ecozone_code, thresh in peat_threshold.items():
                    mask = cd == ecozone_code
                    binary_peat[mask] = (peat_layer[mask] >= thresh).astype(np.uint8)
                layers["peat"] = binary_peat
            else:
                layers["peat"] = (peat_layer >= peat_threshold).astype(np.uint8)

    validate_land_cover_on_organic_soil(
        layers,
        tile_id=tid,
        bounds_string=bstr,
        iv_start=iv_start,
        iv_end=iv_end,
    )

    binarize_extraction_layer(layers)

    # stats for inputs
    for k, arr in layers.items():
        chunk_stats.append(
            uu.calculate_stats(arr, k, bstr, tid, "input_layer", iv_start=iv_start, iv_end=iv_end)
        )

    combine_burned_area(layers, iv_start, iv_end, count_burned_years)

    # create typed dicts for numba
    td8, td16, td32, td32f = nu.create_typed_dicts(layers)

    lu.print_and_log(
        f"starting drainage calc {tid} {bstr} {iv_start}-{iv_end}",
        is_final,
        logger,
    )

    try:
        drainage_table = {
            "low": defac.LOW_TABLE,
            "default": defac.DEFAULT_TABLE,
            "high": defac.HIGH_TABLE,
        }[emission_factor_variant]
        burned_table = {
            "low": baf.LOW_TABLE,
            "default": baf.DEFAULT_TABLE,
            "high": baf.HIGH_TABLE,
        }[emission_factor_variant]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported emission factor variant '{emission_factor_variant}'."
            " Expected one of: low, default, high."
        ) from exc

    out_u32, out_f32 = calculate_drainage_and_emissions(
        td8,
        td16,
        td32f,
        drainage_table,
        burned_table,
        count_burned_years,
        drainage_distance_threshold_m,
    )
    outputs = {**out_u32, **out_f32}

    drained_state = outputs.get("drained_state")
    if drained_state is not None:
        # A) Decide keep/drop using the *actual peat mask* (robust across widths)
        drained_soil = outputs.get("drained_soil")
        has_peat = bool(np.any(drained_soil > 0)) if drained_soil is not None else True
        if not has_peat:
            outputs.pop("drained_state")
        else:
            # B) Normalize width to the declared pad (e.g., 8 digits) if narrower
            m = int(drained_state.max())
            observed_width = (0 if m == 0 else int(np.floor(np.log10(m))) + 1)
            if observed_width and observed_width < STATE_PAD_DIGITS:
                scale = 10 ** (STATE_PAD_DIGITS - observed_width)
                drained_state = (drained_state.astype(np.uint64) * np.uint64(scale)).astype(np.uint32)
                outputs["drained_state"] = drained_state
                lu.print_and_log(
                    f"[state] normalized drained_state width {observed_width}→{STATE_PAD_DIGITS} in {tid}",
                    is_final, logger,
                )

            # C) Validation / diagnostics (unknown nodes, width mismatch)
            eff_width = STATE_PAD_DIGITS if observed_width == 0 else max(observed_width, STATE_PAD_DIGITS)
            root_div = 10 ** (eff_width - 2)
            roots = np.unique(drained_state // root_div)
            if observed_width and observed_width != STATE_PAD_DIGITS:
                lu.print_and_log(
                    f"[state] width mismatch: observed={observed_width}, declared={STATE_PAD_DIGITS} (roots={roots[:5]})",
                    is_final, logger,
                )

            unknown_nodes = np.setdiff1d(
                np.unique(drained_state[drained_state > 0]),
                VALID_DRAINED_STATE_CODES,
            )
            if unknown_nodes.size:
                lu.print_and_log(
                    "Drained-state codes not registered in the core state registry: "
                    f"{unknown_nodes[:10]}",
                    is_final, logger,
                )

    burned_state = outputs.get("burned_state")
    if burned_state is not None and not np.any(burned_state):
        outputs.pop("burned_state")
    elif burned_state is not None:
        unknown_burned_nodes = np.setdiff1d(
            np.unique(burned_state[burned_state > 0]),
            VALID_BURNED_STATE_CODES,
        )
        if unknown_burned_nodes.size:
            lu.print_and_log(
                (
                    "Burned-state codes not registered in the core state registry: "
                    f"{unknown_burned_nodes[:10]}"
                ),
                is_final,
                logger,
            )

    # Unified packed state output (preserves drained + burned semantics in one raster)
    drained_state_for_pack = outputs.get("drained_state")
    burned_state_for_pack = outputs.get("burned_state")
    if drained_state_for_pack is not None:
        if burned_state_for_pack is None:
            burned_state_for_pack = np.zeros_like(drained_state_for_pack, dtype=np.uint32)
        outputs["combined_state"] = sc.pack_combined_state(
            drained_state_for_pack.astype(np.uint32, copy=False),
            burned_state_for_pack.astype(np.uint32, copy=False),
        )

    drained_soil = outputs.pop("drained_soil", None)
    if drained_soil is not None:
        outputs["organic_soil"] = organic_soil_from_drained_soil(drained_soil)

    # burned-area emissions are totals for the whole inventory period; convert
    # to annual values based on the number of years in the period
    interval_length = iv_end - iv_start + 1
    burned_layers = {
        "burned_co2_Mg_CO2_ha": "burned_co2_Mg_CO2_ha_yr",
        "burned_co_Mg_CO_ha": "burned_co_Mg_CO_ha_yr",
        "burned_ch4_Mg_CO2e_ha": "burned_ch4_Mg_CO2e_ha_yr",
        "burned_total_Mg_CO2e_ha": "burned_total_Mg_CO2e_ha_yr",
    }
    for old, new in burned_layers.items():
        if old in outputs:
            outputs[new] = outputs.pop(old) / interval_length

    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tid}.tif"
    pixel_area_chunk = uu.get_tile_dataset_rio(
        pixel_area_uri,
        "Float32",
        bounds,
        chunk_px,
        is_final,
        logger,
        required=True,
    )[0]

    if mega_zarr_path and outputs_to_zarr:
        zarr_outputs = {}
        for output_name in outputs_to_zarr:
            if output_name in outputs:
                zarr_outputs[output_name] = outputs[output_name]
                continue
            if output_name.endswith("_pixel_yr"):
                per_ha_name = output_name.replace("_pixel_yr", "_ha_yr")
                if per_ha_name in outputs:
                    zarr_outputs[output_name] = (
                        outputs[per_ha_name] * pixel_area_chunk * cn.m2_to_ha
                    )
            elif output_name.endswith("_pixel"):
                per_ha_name = output_name.replace("_pixel", "_ha")
                if per_ha_name in outputs:
                    zarr_outputs[output_name] = (
                        outputs[per_ha_name] * pixel_area_chunk * cn.m2_to_ha
                    )

        if zarr_outputs:
            interval_years = interval_end_years or [iv_end]
            try:
                year_index = interval_years.index(iv_end)
            except ValueError as exc:
                raise ValueError(
                    f"Interval end year {iv_end} is missing from zarr year "
                    f"index {interval_years}; refusing to write rasters and "
                    "MegaZarr from divergent interval selections."
                ) from exc
            zu.populate_mega_zarr(
                mega_zarr_path,
                zarr_outputs,
                outputs_to_zarr,
                bounds,
                year_index,
            )

    # stats for outputs, with explicit layer categorization
    drainage_classification_layers = ["organic_soil", "drained_state", "combined_state"]
    burned_classification_layers = ["burned_state"]
    numeric_layers = ["burned_years_count"]

    for k, arr in outputs.items():
        if k in drainage_classification_layers or k in burned_classification_layers:
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer",
                    iv_start=iv_start,
                    iv_end=iv_end,
                )
            )
        elif k in numeric_layers:
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer_numeric",
                    iv_start=iv_start,
                    iv_end=iv_end,
                )
            )
        else:
            per_pixel = arr * pixel_area_chunk * cn.m2_to_ha
            chunk_stats.append(
                uu.calculate_stats(
                    arr,
                    k,
                    bstr,
                    tid,
                    "output_layer",
                    per_pixel,
                    iv_start,
                    iv_end,
                )
            )


    # upload rasters
    if not no_upload:
        if iv_start == iv_end:
            year_tag = f"{iv_start}"
        else:
            # Paths reflect the land cover year closing the period.
            end_for_path = iv_end
            year_tag = f"{iv_start}_{end_for_path}"
        interval_tag = (
            cn.intervals_annual
            if iv_start == iv_end
            else cn.intervals_five_year
        )
        for k, arr in outputs.items():
            outputs[k] = [arr, arr.dtype.name, k, year_tag]
        upload_tasks = uu.save_and_upload_small_raster_set(
            bounds,
            chunk_px,
            tid,
            bstr,
            outputs,
            is_final,
            logger,
            interval_type=interval_tag,
            model_type=run_name,
            no_data_val=0,
            date_str=run_date,
        )
        lu.print_and_log(
            f"Upload tasks created for {bstr} in {tid}. Uploading now: {uu.timestr()}",
            False,
            logger,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            # Consume the iterator so worker exceptions propagate and a failed
            # upload cannot be reported as a successful model chunk.
            list(ex.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks))
        lu.print_and_log(
            f"Uploads completed for {bstr} in {tid} using {cn.outputs_path}: {uu.timestr()}",
            is_final,
            logger,
        )

    return f"Success {bstr} {iv_start}-{iv_end}", chunk_stats


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------


def compute_intervals(start_year, end_year, interval_type, all_five_year_periods):
    """Return only inventory intervals backed by complete land-cover inputs."""
    if interval_type not in {cn.intervals_annual, cn.intervals_five_year}:
        raise ValueError(
            f"Unsupported interval_type {interval_type!r}; expected "
            f"{cn.intervals_annual!r} or {cn.intervals_five_year!r}."
        )

    canonical = list(cn.five_year_inventory_periods)
    if all_five_year_periods:
        interval_type = cn.intervals_five_year
        return (
            canonical,
            canonical[0][0],
            canonical[-1][1],
            interval_type,
        )

    if interval_type == cn.intervals_annual:
        if start_year is None and end_year is None:
            start_year = end_year = cn.annual_land_cover_years[-1]
        elif start_year is None:
            start_year = end_year
        elif end_year is None:
            end_year = start_year
        start_year = int(start_year)
        end_year = int(end_year)
        if end_year < start_year:
            raise ValueError("end_year must be greater than or equal to start_year.")
        requested_years = list(range(start_year, end_year + 1))
        unsupported = sorted(set(requested_years) - set(cn.annual_land_cover_years))
        if unsupported:
            raise ValueError(
                "Annual model periods lack complete global land-cover coverage "
                f"for years {unsupported}. Supported annual years are "
                f"{cn.annual_land_cover_years}."
            )
        return (
            [(year, year) for year in requested_years],
            start_year,
            end_year,
            interval_type,
        )

    if start_year is None and end_year is None:
        selected = [canonical[-1]]
    elif start_year is None:
        selected = [period for period in canonical if period[1] == int(end_year)]
    elif end_year is None:
        selected = [period for period in canonical if period[0] == int(start_year)]
    else:
        start_year = int(start_year)
        end_year = int(end_year)
        selected = [
            period
            for period in canonical
            if period[0] >= start_year and period[1] <= end_year
        ]

    if not selected:
        raise ValueError(
            "Five-year runs must select canonical inventory-period boundaries: "
            f"{canonical}."
        )
    normalized_start = selected[0][0]
    normalized_end = selected[-1][1]
    if start_year is not None and int(start_year) != normalized_start:
        raise ValueError(
            "start_year is not a canonical inventory-period boundary; "
            f"expected one of {[period[0] for period in canonical]}."
        )
    if end_year is not None and int(end_year) != normalized_end:
        raise ValueError(
            "end_year is not a canonical inventory-period boundary; "
            f"expected one of {[period[1] for period in canonical]}."
        )
    return selected, normalized_start, normalized_end, interval_type


def normalize_zarr_write_options(
    *,
    no_upload: bool,
    create_zarr: bool,
    update_existing_zarr: bool,
    mega_zarr_path: Optional[str],
) -> bool:
    """Validate write intent and return the normalized update flag."""

    if create_zarr and update_existing_zarr:
        raise ValueError(
            "--create_zarr and --update_existing_zarr are mutually exclusive. "
            "Use --create_zarr only for new global stores, and "
            "--update_existing_zarr for partial updates."
        )
    if mega_zarr_path and create_zarr:
        raise ValueError(
            "--mega_zarr_path cannot be combined with --create_zarr because "
            "creation always targets a new, automatically-derived store."
        )
    normalized_update = bool(update_existing_zarr or mega_zarr_path)
    if no_upload and (create_zarr or normalized_update):
        raise ValueError(
            "--no_upload forbids all remote writes and cannot be combined with "
            "--create_zarr, --update_existing_zarr, or --mega_zarr_path."
        )
    return normalized_update


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def parse_biome_thresholds(
    raw: Optional[str],
    fallback: Optional[float] = None,
    fscore_metric: str = "f1",
    threshold_scenario: str = "baseline",
) -> Optional[dict[int, float]]:
    """Parse per-biome thresholds from a JSON string or CSV file path.

    Returns a dict mapping ecozone codes (int) to threshold values (float),
    or ``None`` if *raw* is ``None``.

    For CSV input, ``fscore_metric`` selects F1, F2, or mixed threshold columns and
    ``threshold_scenario`` selects the threshold scenario:

    - ``baseline`` reads ``best_f1_threshold``/``best_f2_threshold``/
      ``best_mixed_threshold`` when available, otherwise
      ``operational_threshold``.
    - ``low_area``/``low`` reads the higher-threshold, low-area envelope
      column (``low_area_threshold`` or legacy ``lower_bound_threshold``).
    - ``high_area``/``high`` reads the lower-threshold, high-area envelope
      column (``high_area_threshold`` or legacy ``upper_bound_threshold``).

    ``fscore_metric`` and ``threshold_scenario`` are ignored for JSON input,
    where thresholds are given directly as a biome-to-threshold mapping.
    """
    import json

    if raw is None:
        return None

    raw = raw.strip()

    if raw.startswith("{"):
        name_map = json.loads(raw)
    elif raw.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(raw)
        metric_key = fscore_metric.lower()
        if metric_key not in {"f1", "f2", "mixed"}:
            raise ValueError(
                "fscore_metric must be 'f1', 'f2', or 'mixed'; "
                f"got {fscore_metric!r}"
            )
        scenario_key = normalize_peat_threshold_scenario(threshold_scenario)
        if "metric" in df.columns:
            csv_metrics = sorted(
                df["metric"].dropna().astype(str).str.lower().unique()
            )
            metric_mask = df["metric"].astype(str).str.lower() == metric_key
            if not metric_mask.any():
                raise ValueError(
                    f"CSV metric column does not contain {metric_key!r}; "
                    f"found: {csv_metrics}"
                )
            df = df.loc[metric_mask].copy()
        if "biome" not in df.columns:
            raise KeyError(
                f"CSV must contain a 'biome' column. Found: {df.columns.tolist()}"
            )
        threshold_col = select_biome_threshold_column(
            df.columns,
            metric_key=metric_key,
            scenario_key=scenario_key,
        )
        name_map = dict(zip(df["biome"], df[threshold_col]))
    elif raw.endswith(".json"):
        with open(raw) as fh:
            name_map = json.load(fh)
    else:
        raise ValueError(
            f"Cannot parse --peat_threshold_by_biome: expected JSON string, "
            f".csv, or .json file. Got: {raw!r}"
        )

    import math

    code_map: dict[int, float] = {}
    for name, thresh in name_map.items():
        val = float(thresh)
        if math.isnan(val):
            continue
        name_lower = str(name).strip().lower()
        if name_lower in cn.ecozone_codes:
            code_map[cn.ecozone_codes[name_lower]] = val
        else:
            try:
                code_map[int(name)] = val
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unknown biome name '{name}'. Expected one of: "
                    f"{list(cn.ecozone_codes.keys())}"
                )

    # The fscore script and CLI examples express thresholds on a 0-1 scale,
    # but the OGH raster stores probabilities as uint8 0-100.  Rescale so
    # the comparison `peat_layer >= thresh` works on the native raster values.
    if code_map and all(v <= 1.0 for v in code_map.values()):
        code_map = {k: v * 100.0 for k, v in code_map.items()}
        if fallback is not None and fallback <= 1.0:
            fallback = fallback * 100.0

    unknown_code = cn.ecozone_codes["unknown"]
    if unknown_code not in code_map and fallback is not None:
        code_map[unknown_code] = fallback

    return code_map


def normalize_peat_threshold_scenario(value: str) -> str:
    """Normalize user-facing threshold-scenario aliases."""
    key = str(value).strip().lower()
    try:
        return PEAT_THRESHOLD_SCENARIO_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "--peat_threshold_scenario must be one of: "
            f"{sorted(PEAT_THRESHOLD_SCENARIO_ALIASES)}; got {value!r}"
        ) from exc


def select_biome_threshold_column(
    columns,
    *,
    metric_key: str,
    scenario_key: str,
) -> str:
    """Return the CSV threshold column for a metric/scenario pair."""
    if scenario_key == "baseline":
        candidates = [
            f"best_{metric_key}_threshold",
            "operational_threshold",
            "baseline_threshold",
        ]
    elif scenario_key == "low_area":
        candidates = [
            "low_area_threshold",
            "lower_bound_threshold",
        ]
    elif scenario_key == "high_area":
        candidates = [
            "high_area_threshold",
            "upper_bound_threshold",
        ]
    else:
        raise ValueError(f"Unexpected threshold scenario: {scenario_key!r}")

    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(
        "CSV does not contain a threshold column for "
        f"scenario={scenario_key!r}, metric={metric_key!r}. "
        f"Expected one of {candidates}; found: {list(columns)}"
    )


def run_drainage_model(
    cluster_name=None,
    bounding_box=None,
    chunk_size=None,
    chunk_shapefile_uri=None,
    first_chunks=None,
    run_local=False,
    no_stats=False,
    no_log=False,
    no_upload=False,
    start_year=None,
    end_year=None,
    all_five_year_periods=False,
    interval_type="annual",
    tile_ids=None,
    peat_dataset="ogh",
    run_name="ogh_standard_model",
    peat_threshold: Optional[float | dict[int, float]] = DEFAULT_OGH_THRESHOLD,
    count_burned_years: bool = True,
    emission_factor_variant: str = "default",
    mega_zarr_path: Optional[str] = None,
    outputs_to_zarr: Optional[list[str]] = None,
    include_legacy_state_rasters: bool = False,
    create_zarr: bool = False,
    update_existing_zarr: bool = False,
    run_date: Optional[str] = None,
    drainage_distance_threshold_m: float = DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M,
    exclude_regional_linear_features: bool = False,
):

    model_tile_ids = cn.get_tile_id_list()
    drainage_distance_threshold_m = validate_drainage_distance_threshold_m(
        drainage_distance_threshold_m
    )
    update_existing_zarr = normalize_zarr_write_options(
        no_upload=no_upload,
        create_zarr=create_zarr,
        update_existing_zarr=update_existing_zarr,
        mega_zarr_path=mega_zarr_path,
    )
    stage = "drainage_model"
    start_ts = uu.timestr()
    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    shapefile_provided = bool(chunk_shapefile_uri)
    chunk_shapefile_uri = chunk_shapefile_uri or cn.fishnet_1x1deg_uri

    main_logger, _ = lu.populate_main_log_header(
        bounding_box=None if tile_ids else bounding_box,
        use_shapefile=chunk_shapefile_uri if chunk_shapefile_uri else False,
        client=client,
        cluster=cluster,
        log_note="Organic-soils drainage model",
        run_local=run_local,
        model_type="organic_soils",
        stage=stage,
    )

    tile_ids = list(tile_ids) if tile_ids else []

    if tile_ids:
        unique_tile_ids = list(dict.fromkeys(tile_ids))
        if len(unique_tile_ids) != len(tile_ids):
            main_logger.warning(
                "Received %d tile IDs with duplicates; reducing to %d unique tiles.",
                len(tile_ids),
                len(unique_tile_ids),
            )
            tile_ids = unique_tile_ids
        else:
            tile_ids = unique_tile_ids

        if len(tile_ids) == len(model_tile_ids) and set(tile_ids) == set(model_tile_ids):
            main_logger.info(
                "Full-model tile roster resolved to %d tiles from %s",
                len(tile_ids),
                cn.tile_id_list_source,
            )
        else:
            main_logger.info(
                "Processing %d explicitly-specified 10x10 degree tiles.",
                len(tile_ids),
            )
    elif bounding_box:
        main_logger.info(
            "No tile IDs supplied; deriving extent from bounding box %s.",
            bounding_box,
        )
    else:
        main_logger.info(
            "No tile IDs supplied; default roster contains %d tiles from %s.",
            len(model_tile_ids),
            cn.tile_id_list_source,
        )

    if peat_threshold is None:
        threshold_msg = "none"
    elif isinstance(peat_threshold, dict):
        inv = {v: k for k, v in cn.ecozone_codes.items()}
        threshold_msg = "per-biome: " + "; ".join(
            f"{inv.get(k, k)}: >= {v}" for k, v in sorted(peat_threshold.items())
        )
    else:
        threshold_msg = f">= {peat_threshold}"
    main_logger.info(
        "Peat dataset set to %s with threshold %s",
        peat_dataset,
        threshold_msg,
    )
    main_logger.info(
        "Drainage distance threshold for Dadap canals, OSM canals, OSM roads, and GRIP roads: %s m",
        drainage_distance_threshold_m,
    )
    if exclude_regional_linear_features:
        main_logger.info(
            "Excluding fixed regional linear-feature inputs from drainage classification: %s",
            ", ".join(REGIONAL_LINEAR_FEATURE_LAYERS),
        )

    if chunk_shapefile_uri:
        if shapefile_provided:
            main_logger.info(
                f"Using user-supplied chunk shapefile: {chunk_shapefile_uri}"
            )
        else:
            main_logger.info(
                f"Using default chunk shapefile: {chunk_shapefile_uri}"
            )

    chunk_size = chunk_size or 1
    if tile_ids:
        chunks = []
        for tid in tile_ids:
            bds = uu.get_10x10_tile_bounds(tid)
            chunks.extend(uu.get_chunk_bounds(bds, chunk_size))
    elif chunk_shapefile_uri:
        fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
        chunks, _ = uu.create_chunk_list(
            bounding_box,
            chunk_shapefile_uri,
            chunk_size,
            first_chunks,
            fishnet_iso_df,
            main_logger,
        )
    else:
        bounding_box = bounding_box or [110, -10, 120, 0]
        chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    is_final = len(chunks) > 20

    # Default run date for zarr naming
    if not run_date:
        run_date = datetime.utcnow().strftime("%Y%m%d")

    main_logger.info("Create and populate global mega-zarr: %s", create_zarr)
    main_logger.info("Update existing mega-zarr in place: %s", update_existing_zarr)

    # Normalize interval settings and compute interval list
    intervals, start_year, end_year, interval_type = compute_intervals(
        start_year,
        end_year,
        interval_type,
        all_five_year_periods,
    )
    interval_end_years = [iv[1] for iv in intervals]
    zarr_year_index = interval_end_years

    # Land cover is both a required classifier and a drainage trigger. Use the
    # complete annual 2024 footprint to distinguish valid ocean/edge gaps from
    # an accidentally sparse processing prefix, then check every model period
    # before initializing a Zarr or running model tasks.
    requested_tile_ids = sorted(
        {uu.xy_to_tile_id(bounds[0], bounds[3]) for bounds in chunks}
    )
    reference_start, reference_end = cn.land_cover_coverage_reference_period
    reference_inputs = cn.get_dynamic_download_dict(
        cn.sample_tile_id,
        reference_start,
        reference_end,
        peat_dataset=peat_dataset,
    )
    land_cover_reference_template = reference_inputs["land_cover"]
    reference_land_cover_tile_ids = uu.list_existing_s3_tile_ids(
        land_cover_reference_template,
        model_tile_ids,
    )
    reference_land_cover_tile_ids = uu.validate_tile_set_fingerprint(
        reference_land_cover_tile_ids,
        expected_count=cn.land_cover_reference_tile_count,
        expected_sha256=cn.land_cover_reference_tile_ids_sha256,
        layer_name="Land-cover model-domain",
    )
    required_land_cover_tile_ids = set(requested_tile_ids) & set(
        reference_land_cover_tile_ids
    )
    excluded_tile_ids = sorted(
        set(requested_tile_ids) - required_land_cover_tile_ids
    )
    main_logger.info(
        "Reference land-cover model domain contains %d tiles; %d of %d requested "
        "tiles are in-domain: %s",
        len(reference_land_cover_tile_ids),
        len(required_land_cover_tile_ids),
        len(requested_tile_ids),
        land_cover_reference_template,
    )
    if excluded_tile_ids:
        main_logger.info(
            "Excluding %d requested tiles outside the reference land-cover model "
            "domain (examples: %s)",
            len(excluded_tile_ids),
            ", ".join(excluded_tile_ids[:10]),
        )
        chunks = [
            bounds
            for bounds in chunks
            if uu.xy_to_tile_id(bounds[0], bounds[3])
            in required_land_cover_tile_ids
        ]
    if not chunks:
        raise ValueError(
            "No requested chunks intersect the reference land-cover model domain."
        )
    required_tile_ids = sorted(required_land_cover_tile_ids)

    period_input_templates = {}
    for iv_start, iv_end in intervals:
        input_templates = cn.get_dynamic_download_dict(
            cn.sample_tile_id,
            iv_start,
            iv_end,
            peat_dataset=peat_dataset,
        )
        if exclude_regional_linear_features:
            input_templates = exclude_regional_linear_feature_layers(input_templates)
        period_input_templates[(iv_start, iv_end)] = input_templates

    climate_domain_template = reference_inputs["climate_domain"]
    validated_climate_count = uu.validate_required_s3_tile_coverage(
        climate_domain_template,
        required_land_cover_tile_ids,
        layer_name="climate_domain",
    )
    main_logger.info(
        "Validated required climate-domain coverage: %d land-cover-footprint "
        "tiles at %s",
        validated_climate_count,
        climate_domain_template,
    )

    pixel_area_template = (
        f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{{tile_id}}.tif"
    )
    validated_pixel_area_count = uu.validate_required_s3_tile_coverage(
        pixel_area_template,
        required_tile_ids,
        layer_name="pixel_area",
    )
    main_logger.info(
        "Validated required pixel-area coverage: %d requested tiles at %s",
        validated_pixel_area_count,
        pixel_area_template,
    )

    # Sparse inputs legitimately omit tiles with no data. Resolve every exact
    # object footprint once, then require successful reads wherever an object
    # exists. This preserves sparse-layer semantics without allowing transient
    # read failures or corrupt objects to masquerade as all-zero inputs.
    mandatory_layers = {"land_cover", "climate_domain"}
    sparse_templates = {
        template
        for input_templates in period_input_templates.values()
        for layer_name, template in input_templates.items()
        if layer_name not in mandatory_layers
    }
    sparse_footprints = uu.list_existing_s3_tile_ids_for_templates(
        sparse_templates,
        required_tile_ids,
    )
    required_input_tile_ids_by_interval = {}
    for iv_start, iv_end in intervals:
        input_templates = period_input_templates[(iv_start, iv_end)]
        land_cover_template = input_templates["land_cover"]
        validated_count = uu.validate_required_s3_tile_coverage(
            land_cover_template,
            required_land_cover_tile_ids,
            layer_name=f"land_cover for {iv_start}-{iv_end}",
        )
        main_logger.info(
            "Validated required land-cover coverage for %s-%s: "
            "%d reference-footprint tiles at %s",
            iv_start,
            iv_end,
            validated_count,
            land_cover_template,
        )
        interval_footprints = {}
        for layer_name, template in input_templates.items():
            if layer_name in mandatory_layers:
                interval_footprints[layer_name] = set(
                    required_land_cover_tile_ids
                )
            else:
                interval_footprints[layer_name] = sparse_footprints[template]
        required_input_tile_ids_by_interval[(iv_start, iv_end)] = (
            interval_footprints
        )
        main_logger.info(
            "Input object footprints for %s-%s (present objects become "
            "required reads): %s",
            iv_start,
            iv_end,
            ", ".join(
                f"{name}={len(tile_set)}"
                for name, tile_set in sorted(interval_footprints.items())
            ),
        )

    peat_template = reference_inputs["peat"]
    required_peat_tile_ids = sparse_footprints[peat_template]
    main_logger.info(
        "Peat input contains %d of %d requested tiles; absent tiles are "
        "allowed because preprocessing omits all-zero peat rasters: %s",
        len(required_peat_tile_ids),
        len(required_tile_ids),
        peat_template,
    )

    if create_zarr:
        if not chunks:
            raise ValueError("No chunks available to determine zarr chunk size.")
        chunk_size_pixels = uu.calc_chunk_length_pixels(chunks[0])
        # A store must advertise only the intervals this run will populate.
        # This prevents a one-period run from looking like a complete temporal
        # series with zero-filled years.
        zarr_year_index = interval_end_years
        main_logger.info(
            "Zarr year index (%s): %s",
            interval_type,
            zarr_year_index,
        )
        mega_zarr_path = zu.create_mega_zarr_path(
            cn.drainage_outputs_path_mega_zarr,
            chunk_size_pixels,
            interval_type,
            run_name,
            run_date,
            main_logger,
        )
        if outputs_to_zarr is None:
            outputs_to_zarr = list(cn.drainage_outputs_to_zarr)
            if include_legacy_state_rasters:
                outputs_to_zarr.extend(cn.drainage_optional_state_outputs)
        outputs_to_zarr = list(dict.fromkeys(outputs_to_zarr))
        main_logger.info(
            "Mega-zarr output datasets (%d): %s",
            len(outputs_to_zarr),
            outputs_to_zarr,
        )
        zu.initialize_global_mega_zarr(
            mega_zarr_path,
            outputs_to_zarr,
            zarr_year_index,
            chunk_size_pixels,
            interval_type,
            main_logger,
        )
        ds = zu.open_mega_zarr_dataset(mega_zarr_path)
        main_logger.info("mega-zarr coords: %s", ds.coords)
        main_logger.info(
            "y range: %s, %s", ds.y.values.min(), ds.y.values.max()
        )
        main_logger.info(
            "x range: %s, %s", ds.x.values.min(), ds.x.values.max()
        )
        main_logger.info(
            "mega-zarr chunk size (years, y, x): %s", ds.chunksizes
        )
    elif update_existing_zarr:
        if not chunks:
            raise ValueError("No chunks available to determine zarr chunk size.")
        chunk_size_pixels = uu.calc_chunk_length_pixels(chunks[0])
        if mega_zarr_path is None:
            mega_zarr_path = zu.create_mega_zarr_path(
                cn.drainage_outputs_path_mega_zarr,
                chunk_size_pixels,
                interval_type,
                run_name,
                run_date,
                main_logger,
            )
        if outputs_to_zarr is None:
            outputs_to_zarr = list(cn.drainage_outputs_to_zarr)
            if include_legacy_state_rasters:
                outputs_to_zarr.extend(cn.drainage_optional_state_outputs)
        outputs_to_zarr = list(dict.fromkeys(outputs_to_zarr))
        ds = zu.open_mega_zarr_dataset(mega_zarr_path)
        zarr_year_index = [int(year) for year in ds.year.values]
        missing_outputs = [
            output_name
            for output_name in outputs_to_zarr
            if output_name not in ds.data_vars
        ]
        if missing_outputs:
            raise ValueError(
                "Requested outputs are missing from existing mega-zarr "
                f"{mega_zarr_path}: {missing_outputs}"
            )
        main_logger.info("Updating existing mega-zarr path: %s", mega_zarr_path)
        main_logger.info(
            "Existing mega-zarr year index (%s): %s",
            interval_type,
            zarr_year_index,
        )
        main_logger.info(
            "Mega-zarr output datasets (%d): %s",
            len(outputs_to_zarr),
            outputs_to_zarr,
        )

    # build task list & run with dask.bag
    bag_items = [(bds, iv[0], iv[1]) for iv in intervals for bds in chunks]
    bag = dask.bag.from_sequence(bag_items, npartitions=len(bag_items))

    if create_zarr:
        zu.set_mega_zarr_run_status(
            mega_zarr_path,
            "running",
            expected_task_count=len(bag_items),
            expected_chunk_count=len(chunks),
            selected_inventory_periods=[
                f"{iv_start}_{iv_end}" for iv_start, iv_end in intervals
            ],
        )

    typed_dict_cache = {}

    def _wrap(t):
        final_year = cn.five_year_inventory_periods[-1][1]
        closing_year = t[2]
        bstr = uu.boundstr(t[0])
        main_logger.info(
            f"{bstr} interval {t[1]}-{t[2]} uses land cover {closing_year}"
        )

        key = (t[1], closing_year)
        if key not in typed_dict_cache:
            download_dict = cn.get_dynamic_download_dict(
                cn.sample_tile_id,
                t[1],
                closing_year,
                peat_dataset=peat_dataset,
            )
            if exclude_regional_linear_features:
                download_dict = exclude_regional_linear_feature_layers(download_dict)
            first_tiles = uu.first_file_name_in_s3_folder(download_dict)
            typed_dict_cache[key] = uu.add_file_type_to_dict(first_tiles)

        typed_dict = typed_dict_cache[key]
        tile_id = uu.xy_to_tile_id(t[0][0], t[0][3])
        interval_footprints = required_input_tile_ids_by_interval[(t[1], t[2])]
        required_layers = {
            layer_name
            for layer_name, tile_set in interval_footprints.items()
            if tile_id in tile_set
        }
        missing_required_layers = sorted(required_layers - set(typed_dict))
        if missing_required_layers:
            raise uu.RequiredInputRasterError(
                "Required inputs could not be typed for interval "
                f"{t[1]}-{closing_year}: {missing_required_layers}."
            )

        return calculate_and_upload_drainage(
            t[0],
            typed_dict,
            is_final,
            no_upload,
            t[1],
            t[2],
            closing_year,
            peat_dataset,
            run_name,
            peat_threshold,
            count_burned_years,
            emission_factor_variant,
            mega_zarr_path=mega_zarr_path,
            outputs_to_zarr=outputs_to_zarr,
            interval_end_years=zarr_year_index,
            run_date=run_date,
            required_layers=required_layers,
            drainage_distance_threshold_m=drainage_distance_threshold_m,
        )

    try:
        results = bag.map(_wrap).compute()
    except Exception as exc:
        if create_zarr:
            try:
                zu.set_mega_zarr_run_status(
                    mega_zarr_path,
                    "failed",
                    failure_type=type(exc).__name__,
                )
            except Exception:
                main_logger.exception(
                    "Could not mark failed mega-zarr run status: %s",
                    mega_zarr_path,
                )
        raise

    # Summarize chunk results and gather per-chunk statistics
    success_count, all_stats = uu.count_successful_chunks(
        bag_items, is_final, main_logger, results
    )
    if success_count != len(bag_items):
        if create_zarr:
            zu.set_mega_zarr_run_status(
                mega_zarr_path,
                "failed",
                successful_task_count=success_count,
                expected_task_count=len(bag_items),
            )
        raise RuntimeError(
            "Drainage model did not complete every submitted task: "
            f"{success_count} of {len(bag_items)} succeeded."
        )

    # Aggregate per‑chunk statistics and merge with the fishnet shapefile
    if (not no_stats) and (success_count > 0):
        uu.compile_1x1_chunk_stats(
            all_stats,
            chunk_shapefile_uri,
            stage,
            no_upload,
            main_logger,
            run_name=run_name,
            run_date=run_date,
        )

    if create_zarr:
        zu.set_mega_zarr_run_status(
            mega_zarr_path,
            "complete",
            successful_task_count=success_count,
            expected_task_count=len(bag_items),
        )


    uu.stage_duration(start_ts, uu.timestr(), stage)
    if not run_local:
        client.close()
        cluster.close()


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------
def main(argv=None):
    """
    CLI entry point with an IDE‑friendly quick‑test.

    * If you pass arguments, they are honored (same flags as before).
    * If you run the script with no args (for example, click Run in an IDE),
      it spins up a tiny 1 x 1 degree chunk locally so you can debug fast.
    """
    argv = argv if argv is not None else sys.argv[1:]

    # quick test when no CLI args ------------------------------------
    if not argv:
        print("No CLI args: running 1-tile local smoke test")
        run_drainage_model(
            cluster_name=None,
            bounding_box=[112, -2, 113, -1],  # 1 x 1 degree tile
            chunk_size=1,
            chunk_shapefile_uri=None,
            first_chunks=None,
            run_local=True,
            no_stats=True,
            no_log=False,
            no_upload=True,
            start_year=2021,
            end_year=2024,
            interval_type=cn.intervals_five_year,
            all_five_year_periods=False,
            peat_dataset="ogh",
            run_name="ogh_standard_model",
            peat_threshold=DEFAULT_OGH_THRESHOLD,
            count_burned_years=True,
            emission_factor_variant="default",
            drainage_distance_threshold_m=DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M,
        )
        return

    # normal CLI parsing --------------------------------------------
    p = argparse.ArgumentParser("Organic-soils drainage model")
    p.add_argument("--cluster_name")
    p.add_argument(
        "--bounding_box",
        "-bb",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
    )
    p.add_argument(
        "--tile_ids",
        action="append",
        help="Comma separated 10x10 tile IDs (e.g. 00N_110E). Can be used multiple times.",
    )
    p.add_argument(
        "--full_model",
        action="store_true",
        help="Process all available 10x10 degree tiles",
    )
    p.add_argument("--chunk_size", "-cs", type=float)
    p.add_argument(
        "-cshp",
        "--chunk_shapefile_uri",
        help="S3 location for shapefile of 1x1 deg chunk footprints",
    )
    p.add_argument(
        "-f",
        "--first_chunks",
        type=int,
        help="Number of chunks to process from shapefile",
    )
    p.add_argument("--run_local", action="store_true")
    p.add_argument("--no_stats", action="store_true")
    p.add_argument("--no_log", action="store_true")
    p.add_argument("--no_upload", action="store_true")
    p.add_argument("--start_year", type=int)
    p.add_argument("--end_year", type=int)
    p.add_argument(
        "--interval_type",
        choices=[cn.intervals_annual, cn.intervals_five_year],
        default=cn.intervals_annual,
    )
    p.add_argument(
        "--all_five_year_periods",
        action="store_true",
        help=(
            "Process all available five year intervals "
            "(sets interval_type to five_year)"
        ),
    )
    p.add_argument(
        "--peat_dataset",
        default="ogh",
        choices=cn.peat_dataset_choices,
        help="Peat mask dataset to use",
    )
    p.add_argument(
        "--peat_threshold",
        type=parse_optional_float,
        default=DEFAULT_OGH_THRESHOLD,
        help=(
            "Threshold applied to OGH peat probabilities; values greater than "
            "or equal to the threshold are treated as peat. Pass 'none' "
            "to disable thresholding. When --peat_threshold_by_biome is "
            "also set, this value is used as the fallback for unknown ecozones."
        ),
    )
    p.add_argument(
        "--drainage_distance_threshold_m",
        type=float,
        default=DEFAULT_DRAINAGE_DISTANCE_THRESHOLD_M,
        help=(
            "Distance threshold, in meters, for Dadap canals, OSM canals, "
            "OSM roads, and GRIP roads."
        ),
    )
    p.add_argument(
        "--exclude_regional_linear_features",
        action="store_true",
        help=(
            "Exclude fixed presence-only regional linear-feature drainage inputs "
            "(Engert roads) so drainage-distance sensitivity runs depend only on "
            "the distance rasters (Dadap canals, OSM canals/roads, GRIP roads). "
            "Dadap is now distance-based and responds to the sweep, so it is no "
            "longer excluded."
        ),
    )
    p.add_argument(
        "--peat_threshold_by_biome",
        type=str,
        default=None,
        help=(
            "Per-biome peat probability thresholds. Accepts either an inline "
            "JSON string (e.g., '{\"tropical\": 0.15, \"boreal\": 0.30, "
            "\"temperate\": 0.20}') or a path to a CSV file with a 'biome' "
            "column and baseline/scenario threshold columns. Baseline summary "
            "CSVs can use 'best_f1_threshold'/'best_f2_threshold'; scenario "
            "CSVs can use 'operational_threshold', 'low_area_threshold', and "
            "'high_area_threshold'. Use --fscore_metric and "
            "--peat_threshold_scenario to choose which thresholds to read. "
            "When set, --peat_threshold is used as the fallback for unknown "
            "ecozones."
        ),
    )
    p.add_argument(
        "--fscore_metric",
        choices=["f1", "f2", "mixed"],
        default="f1",
        help=(
            "When --peat_threshold_by_biome points to a CSV, select the F1, "
            "F2, or mixed threshold set. For biome-threshold summary CSVs "
            "this reads 'best_f1_threshold', 'best_f2_threshold', or "
            "'best_mixed_threshold'. For custom scenario-bound CSVs, use "
            "'operational_threshold', 'low_area_threshold', and "
            "'high_area_threshold'. Ignored for JSON input. Default: f1."
        ),
    )
    p.add_argument(
        "--peat_threshold_scenario",
        choices=sorted(PEAT_THRESHOLD_SCENARIO_ALIASES),
        default="baseline",
        help=(
            "When --peat_threshold_by_biome points to a CSV, choose which "
            "per-biome threshold scenario to read. 'baseline' reads best-F1/"
            "best-F2 or operational thresholds; 'low_area'/'low' reads the "
            "higher-threshold low-area envelope; 'high_area'/'high' reads the "
            "lower-threshold high-area envelope. Ignored for JSON input. "
            "Default: baseline."
        ),
    )
    p.add_argument(
        "--run_name",
        default="ogh_standard_model",
        help="Run name used to label output directories",
    )
    p.add_argument(
        "--count_burned_years",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Multiply burned emissions by the number of burned years in each interval "
            "(use --no-count_burned_years to disable)."
        ),
    )
    p.add_argument(
        "--emission_factor_variant",
        choices=["default", "low", "high"],
        default="default",
        help="Select drainage and burned-area emission factor set for sensitivity runs",
    )
    p.add_argument(
        "--create_zarr",
        action="store_true",
        help="Create and populate global mega-zarr with model outputs",
    )
    p.add_argument(
        "--update_existing_zarr",
        action="store_true",
        help=(
            "Populate an existing mega-zarr in place without reinitializing it. "
            "Use this for tile-level runs."
        ),
    )
    p.add_argument(
        "--mega_zarr_path",
        help=(
            "Existing mega-zarr path to update. Supplying this implies "
            "--update_existing_zarr and cannot be combined with --create_zarr."
        ),
    )
    p.add_argument(
        "--include_legacy_state_rasters",
        action="store_true",
        help="Also write component drained_state and burned_state rasters (default: omit).",
    )
    p.add_argument(
        "--run_date",
        help="Run date used in mega-zarr path naming (YYYYMMDD)",
    )
    args = p.parse_args(argv)

    tile_ids = []
    if args.full_model:
        tile_ids = cn.get_tile_id_list()
    elif args.tile_ids:
        for item in args.tile_ids:
            tile_ids.extend(t.strip() for t in item.split(",") if t.strip())

    biome_thresholds = parse_biome_thresholds(
        args.peat_threshold_by_biome,
        fallback=args.peat_threshold,
        fscore_metric=args.fscore_metric,
        threshold_scenario=args.peat_threshold_scenario,
    )
    effective_threshold = biome_thresholds if biome_thresholds is not None else args.peat_threshold

    run_drainage_model(
        cluster_name=args.cluster_name,
        bounding_box=args.bounding_box,
        chunk_size=args.chunk_size,
        chunk_shapefile_uri=args.chunk_shapefile_uri,
        first_chunks=args.first_chunks,
        run_local=args.run_local,
        no_stats=args.no_stats,
        no_log=args.no_log,
        no_upload=args.no_upload,
        start_year=args.start_year,
        end_year=args.end_year,
        interval_type=args.interval_type,
        all_five_year_periods=args.all_five_year_periods,
        tile_ids=tile_ids,
        peat_dataset=args.peat_dataset,
        run_name=args.run_name,
        peat_threshold=effective_threshold,
        count_burned_years=args.count_burned_years,
        emission_factor_variant=args.emission_factor_variant,
        create_zarr=args.create_zarr,
        update_existing_zarr=args.update_existing_zarr,
        mega_zarr_path=args.mega_zarr_path,
        include_legacy_state_rasters=args.include_legacy_state_rasters,
        run_date=args.run_date,
        drainage_distance_threshold_m=args.drainage_distance_threshold_m,
        exclude_regional_linear_features=args.exclude_regional_linear_features,
    )


if __name__ == "__main__":
    main()

"""
Computes per-pixel asymmetric uncertainty (U⁻_Δ,j and U⁺_Δ,j, Mg C per 120m pixel) in SOC
stock change for the 2020 interval (avg 2010–2015 → avg 2015–2020) at 120m resolution,
and the global U⁻_global / U⁺_global (Mg C and Tg C).

Method: Uncertainty_Propagation_in_SOC_Stocks__from_Serkan_Isik_20260605.pdf
        (OpenGeoHub Foundation, 2026-06-05)

Per Claude session 'Mineral soil carbon change uncertainty analysis'

A 120m pixel is included if it contains at least one mineral-soil 30m pixel (not majority). This matches the
central estimate, which uses the 30m mineral-soil mask directly — so every 120m pixel that contributes any
mass to the central estimate is also represented in the uncertainty analysis. A majority-mineral threshold would
leave some central-estimate mass with no associated uncertainty.

Per-pixel computation for each 120m mineral-soil pixel j [PDF §3]:
    Per time block t:
        U⁻_t,j = max(SOC_t,j − p16_t,j, 0)      [lower; clamp to avoid negatives]
        U⁺_t,j = max(p84_t,j − SOC_t,j, 0)      [upper; clamp to avoid negatives]

    Stock-change propagation, assuming uncorrelated time blocks [PDF §5]:
        U⁻_Δ,j = sqrt( (U⁻_t2,j)² + (U⁺_t1,j)² )   [lowest change: t2 low, t1 high]
        U⁺_Δ,j = sqrt( (U⁺_t2,j)² + (U⁻_t1,j)² )   [highest change: t2 high, t1 low]

Global aggregation [PDF §6–7]:
    PDF §7 sums country-level U⁻_c over countries: U⁻_global = sqrt( Σ_c (U⁻_c)² ).
    Country-level step skipped per Serkan's email (2026-06-08); summing directly over pixels j
    is algebraically equivalent because root sum of squares (RSS) is associative: sqrt(Σ_c (sqrt(Σ_{j∈c} U²))²) = sqrt(Σ_j U²).
    Net change:   U⁻_global = sqrt( Σ_j (U⁻_Δ,j)² ),  U⁺_global = sqrt( Σ_j (U⁺_Δ,j)² )
    Gross loss:   U⁻_loss   = sqrt( Σ_{j∈loss} (U⁻_Δ,j)² ),  U⁺_loss = sqrt( Σ_{j∈loss} (U⁺_Δ,j)² )
    Gross gain:   U⁻_gain   = sqrt( Σ_{j∈gain} (U⁻_Δ,j)² ),  U⁺_gain = sqrt( Σ_{j∈gain} (U⁺_Δ,j)² )
    where loss/gain classification is based on the sign of delta_mean = mean_t2 − mean_t1.
    Note: pixels near zero where U⁺_Δ > |delta_mean| could plausibly flip sign; this fixed
    classification treats them as pure loss or gain (optimistic; standard first-order practice).

Unit convention [PDF §1]:
    All values are Mg C per 120m pixel (not per ha). The OGH input COGs are in mg/cm³
    volumetric density (raw integer × 0.1 = kg C/m³). Conversion to Mg C per 120m pixel
    uses latitude-dependent pixel area so that the result is geographically correct.
    Conversion (two steps):
        Step 1 — Mg C/m³ → Mg C/ha: raw × 0.1 kg/m³ × 0.3 m depth × 10000 m²/ha / 1000 kg/Mg
        Step 2 — Mg C/ha → Mg C/pixel: × pixel_area_ha  (latitude-dependent)

Inputs per time block [PDF §1]:
    - mean SOC stock map at 120m (Mg C per 120m pixel after conversion)
    - p16 and p84 SOC stock maps at 120m (same units)
    - organic-soil mask at 30m (to exclude organic soils)

Local test with intermediates:
  python -m src.LULUCF.scripts.mineral_soil_organic_carbon.4_SOC_uncertainty_2020_interval  -bb 110 -1 111 0 -cs 1 -mpd test_box --upload_intermediates

Small Coiled run in area with data:
  python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn SOC_uncertainty
  python -m src.LULUCF.scripts.mineral_soil_organic_carbon.4_SOC_uncertainty_2020_interval -cn SOC_uncertainty -bb 110 -1 111 0 -cs 1 -mpd test_box --upload_intermediates

Small Coiled run in area without data:
  python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn SOC_uncertainty
  python -m src.LULUCF.scripts.mineral_soil_organic_carbon.4_SOC_uncertainty_2020_interval -cn SOC_uncertainty -bb 0 77 1 78 -cs 1 -mpd test_box --upload_intermediates

Full run:
  python -m src.utilities.create_cluster -n 200 -t 1 -m 8 -cn SOC_uncertainty
  python -m src.LULUCF.scripts.mineral_soil_organic_carbon.4_SOC_uncertainty_2020_interval -cn SOC_uncertainty -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp
"""

import argparse
import gc
import os
import concurrent.futures
import numpy as np
import pandas as pd
import psutil
import rasterio
import boto3
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print
import sys
import time

from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import resize_cluster


# ─── Input COGs ───────────────────────────────────────────────────────────────
# [PDF §1] Six 120m OGH layers: mean, p16, p84 for t1 (2010-2015) and t2 (2015-2020).
# Source: OpenLandMap_soildb_COGS.csv (commit b1264a8).
#   mean uses the '_m_120m_' identifier (not '_mean_'); p16/p84 confirmed on lines 34, 35, 49, 50.
# Units: raw integer × 0.1 = kg C/m³ volumetric density, depth 0–30 cm.
_OGH_BASE   = 'https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics'
_OGH_VAR    = 'oc_iso.10694.1995.mg.cm3'
_SUFFIX_T1  = 'b0cm..30cm_20100101_20151231_g_epsg.4326_v20250204.tif'
_SUFFIX_T2  = 'b0cm..30cm_20150101_20201231_g_epsg.4326_v20250204.tif'

SOC_UNCERTAINTY_COGS = {
    'mean_t1': f'{_OGH_BASE}/{_OGH_VAR}_m_120m_{_SUFFIX_T1}',
    'p16_t1':  f'{_OGH_BASE}/{_OGH_VAR}_p16_120m_{_SUFFIX_T1}',
    'p84_t1':  f'{_OGH_BASE}/{_OGH_VAR}_p84_120m_{_SUFFIX_T1}',
    'mean_t2': f'{_OGH_BASE}/{_OGH_VAR}_m_120m_{_SUFFIX_T2}',
    'p16_t2':  f'{_OGH_BASE}/{_OGH_VAR}_p16_120m_{_SUFFIX_T2}',
    'p84_t2':  f'{_OGH_BASE}/{_OGH_VAR}_p84_120m_{_SUFFIX_T2}',
}

# ─── Output constants ─────────────────────────────────────────────────────────
# Main outputs: U⁻_Δ and U⁺_Δ (Mg C per 120m pixel), masked to mineral soil
U_MINUS_DELTA_PATTERN = 'SOC_uncertainty_lower__mineral_soil_extent__MgC_per_pixel'
U_PLUS_DELTA_PATTERN  = 'SOC_uncertainty_upper__mineral_soil_extent__MgC_per_pixel'

# Intermediate outputs (uploaded with --upload_intermediates)
U_MINUS_T1_PATTERN = 'SOC_uncertainty_lower_t1__mineral_soil_extent__MgC_per_pixel'
U_PLUS_T1_PATTERN  = 'SOC_uncertainty_upper_t1__mineral_soil_extent__MgC_per_pixel'
U_MINUS_T2_PATTERN = 'SOC_uncertainty_lower_t2__mineral_soil_extent__MgC_per_pixel'
U_PLUS_T2_PATTERN  = 'SOC_uncertainty_upper_t2__mineral_soil_extent__MgC_per_pixel'
DELTA_MEAN_PATTERN = 'SOC_delta_mean__mineral_soil_extent__MgC_per_pixel'
LOSS_MASK_PATTERN  = 'SOC_loss_mask__mineral_soil_extent'
GAIN_MASK_PATTERN  = 'SOC_gain_mask__mineral_soil_extent'


# ─── Physical constants ───────────────────────────────────────────────────────
# [PDF §1] OGH COG encoding: raw integer × 0.1 = kg C/m³ volumetric density.
# Depth: 0–30 cm = 0.3 m.
# Conversion to Mg C per pixel: density × depth × pixel_area_m² / 1000
# pixel_area_m² is latitude-dependent (see _pixel_area_m2 helper).
OGH_SCALE  = np.float64(0.1)    # raw → kg C/m³
DEPTH_M    = np.float64(0.3)    # 0–30 cm layer
KG_TO_MG   = np.float64(1000)  # kg/Mg
M2_PER_HA  = np.float64(10000) # m²/ha

INTERVAL_LABEL = '2020'   # avg 2015-2020 minus avg 2010-2015


# ─── Helpers ──────────────────────────────────────────────────────────────────

def block_nansum_2d(arr_30m, target_rows, target_cols):
    """
    Aggregate a 2D 30m array to (target_rows, target_cols) by summing blocks.

    [PDF §4] The 120m pixel is the uncertainty unit; the 16 underlying 30m pixels
    share one 120m uncertainty estimate. This function aggregates the 30m mineral-soil
    indicator to the 120m grid to determine which 120m pixels contain any mineral soil.

    NaN → 0 before summing. Trims input to the nearest exact multiple of block size.
    """
    src_rows, src_cols = arr_30m.shape
    block_r = max(1, int(round(src_rows / target_rows)))   # 30m pixels per 120m pixel, row direction
    block_c = max(1, int(round(src_cols / target_cols)))   # 30m pixels per 120m pixel, col direction
    rows_trim = target_rows * block_r   # drop any fractional-block edge rows
    cols_trim = target_cols * block_c   # drop any fractional-block edge cols
    arr_trimmed = arr_30m[:rows_trim, :cols_trim].astype(np.float32)
    arr_no_nan = np.where(np.isnan(arr_trimmed), np.float32(0.0), arr_trimmed)   # NaN → 0 so nansum = sum
    # reshape to (target_rows, block_r, target_cols, block_c), then sum the two block axes
    return arr_no_nan.reshape(target_rows, block_r, target_cols, block_c).sum(axis=(1, 3), dtype=np.float32)


def pixel_area_m2(bounds, n_pixels, pixel_size_deg):
    """
    Latitude-dependent pixel area (m²) for each row of the 120m chunk, broadcast to 2D.

    [PDF §1] OGH COGs store volumetric density (kg/m³); converting to Mg C per pixel
    requires the actual pixel area, which varies with latitude. Uses standard equatorial
    approximations: meridional 110574 m/deg, zonal 111320 m/deg × cos(lat). Error < 0.3%
    within ±80° latitude, which is well within any other uncertainty in the analysis.
    """
    lat_top = bounds[3]
    lat_centers = lat_top - pixel_size_deg * (np.arange(n_pixels) + 0.5)
    cos_lat = np.cos(np.radians(lat_centers)).astype(np.float64)
    area_per_row = (pixel_size_deg * 111320.0 * cos_lat) * (pixel_size_deg * 110574.0)
    return (area_per_row[:, np.newaxis] * np.ones(n_pixels, dtype=np.float64)).astype(np.float32)


# ─── Per-chunk worker ─────────────────────────────────────────────────────────

def compute_soc_uncertainty(bounds, is_large_run, stage, no_upload, upload_intermediates,
                             nodata_val_120m, pixel_size_deg_120m, output_s3_dir):
    """
    Process a single 1×1 degree chunk. Returns:
        (return_message,
         sum_U_minus_sq, sum_U_plus_sq,           # net change [all valid pixels]
         sum_U_minus_loss_sq, sum_U_plus_loss_sq, # gross loss [delta_mean < 0]
         sum_U_minus_gain_sq, sum_U_plus_gain_sq, # gross gain [delta_mean > 0]
         chunk_stats_list)

    Follows PDF §8 recommended workflow steps 2–5 (pixel level).
    Country-level aggregation (steps 4–5 of PDF §8) is skipped per project scope;
    pixel values feed directly into global root-sum-of-squares [PDF §7].
    """
    chunk_stats_combined = []
    process = psutil.Process(os.getpid())
    logger_worker = lu.setup_logging_worker()
    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, 'preprocessing_', is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels_30m  = uu.calc_chunk_length_pixels(bounds)
    chunk_deg = bounds[2] - bounds[0]
    chunk_length_pixels_120m = max(1, int(round(chunk_deg / pixel_size_deg_120m)))

    lu.print_and_log(f"Processing {bounds_str} ({tile_id}): 30m={chunk_length_pixels_30m}px  120m={chunk_length_pixels_120m}px",False, logger_worker)


    ### Part 1: Download 120m COGs (6 files, concurrent)
    # [PDF §1, §8 step 2] Mean, p16, p84 for t1 (2010-2015) and t2 (2015-2020).
    # Inaccessible COG → NaN array so the failure appears as missing data rather than
    # zero uncertainty, which would be silently wrong.

    raw_arrays = {}
    expected_shape = (chunk_length_pixels_120m, chunk_length_pixels_120m)

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_key = {
            executor.submit(uu.get_tile_dataset_rio, url, bounds, chunk_length_pixels_120m, logger_worker, 'float32'): key
            for key, url in SOC_UNCERTAINTY_COGS.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            data, status = future.result()
            if 'success' not in status and 'padded' not in status:
                lu.print_and_log(f"WARNING: {key} inaccessible for {bounds_str} — filling with NaN. Status: {status}",False, logger_worker)
                data = np.full(expected_shape, np.nan, dtype=np.float32)
            raw_arrays[key] = data
    print("raw_arrays:", raw_arrays)

    ### Part 2: Download 30m organic soil mask
    # [PDF §1, §6] Organic-soil pixels are excluded from uncertainty outputs.
    # On access failure, all pixels are treated as mineral soil (over-inclusive); log a warning.

    organic_soil_uri = f"{cn.organic_soil_extent_dir}{tile_id}__{cn.organic_soil_extent_pattern}.tif"
    organic_soil_mask, organic_status = uu.get_tile_dataset_rio(organic_soil_uri, bounds, chunk_length_pixels_30m, logger_worker, 'uint8')
    if 'success' not in organic_status and 'padded' not in organic_status:
        lu.print_and_log(f"WARNING: Organic soil mask inaccessible for {bounds_str} — all pixels treated as mineral soil. Status: {organic_status}",False, logger_worker)
    # print("organic_soil_mask:", organic_soil_mask)


    ### Part 3: Build mineral-soil mask at 120m
    # [PDF §4, §6] A 120m pixel is included if it contains at least one mineral-soil
    # 30m pixel (block sum of mineral indicator > 0).

    mineral_indicator_30m = (organic_soil_mask != cn.organic_soil_mask_val).astype(np.float32)
    mineral_sum_120m = block_nansum_2d(mineral_indicator_30m, chunk_length_pixels_120m, chunk_length_pixels_120m)
    has_mineral_soil = mineral_sum_120m > np.float32(0.0)
    # print("has_mineral_soil:", has_mineral_soil)


    ### Part 4: Convert raw OGH integer → Mg C per 120m pixel (two steps)
    # [PDF §1] nodata_val_120m → NaN; NaN arrays from inaccessible COGs propagate through all steps.

    # Step 1: raw integer → Mg C/ha  (volumetric density × depth, unit conversion)
    # raw × 0.1 (kg/m³) × 0.3 m depth × 10000 m²/ha / 1000 (kg/Mg) = Mg C/ha
    density_to_MgC_per_ha = np.float32(OGH_SCALE * DEPTH_M * M2_PER_HA / KG_TO_MG)

    # Step 2: Mg C/ha → Mg C per 120m pixel  (multiply by latitude-dependent pixel area in ha)
    pixel_area_120m_m2 = pixel_area_m2(bounds, chunk_length_pixels_120m, pixel_size_deg_120m)
    pixel_area_120m_ha = (pixel_area_120m_m2 / M2_PER_HA).astype(np.float32)

    def convert(arr):
        nodata_masked = np.where(arr == nodata_val_120m, np.nan, arr.astype(np.float32))
        MgC_per_ha = nodata_masked * density_to_MgC_per_ha   # Mg C/ha
        return (MgC_per_ha * pixel_area_120m_ha).astype(np.float32)  # Mg C/pixel

    mean_t1 = convert(raw_arrays['mean_t1'])
    p16_t1  = convert(raw_arrays['p16_t1'])
    p84_t1  = convert(raw_arrays['p84_t1'])
    mean_t2 = convert(raw_arrays['mean_t2'])
    p16_t2  = convert(raw_arrays['p16_t2'])
    p84_t2  = convert(raw_arrays['p84_t2'])


    ### Part 5: Per-pixel asymmetric uncertainty for each time block
    # [PDF §3, §8 step 2]
    # U⁻_t,j = max(SOC_t,j − p16_t,j, 0)   [lower: how far mean is above p16]
    # U⁺_t,j = max(p84_t,j − SOC_t,j, 0)   [upper: how far p84 is above mean]
    # Clamped to ≥ 0 to handle cases where the mean falls outside the p16–p84 interval.
    # np.maximum propagates NaN, so inaccessible COGs → NaN in U (correctly missing, not zero).

    U_minus_t1 = np.maximum(mean_t1 - p16_t1, np.float32(0.0))
    U_plus_t1  = np.maximum(p84_t1 - mean_t1, np.float32(0.0))
    U_minus_t2 = np.maximum(mean_t2 - p16_t2, np.float32(0.0))
    U_plus_t2  = np.maximum(p84_t2 - mean_t2, np.float32(0.0))


    ### Part 6: Propagate uncertainty through the stock-change computation
    # [PDF §5, §8 step 3] Assuming uncorrelated time blocks (no ρ term):
    # U⁻_Δ,j = sqrt( (U⁻_t2)² + (U⁺_t1)² )  [t2 at its low, t1 at its high → smallest change]
    # U⁺_Δ,j = sqrt( (U⁺_t2)² + (U⁻_t1)² )  [t2 at its high, t1 at its low → largest change]

    U_minus_delta = np.sqrt(U_minus_t2 ** 2 + U_plus_t1  ** 2).astype(np.float32)
    U_plus_delta  = np.sqrt(U_plus_t2  ** 2 + U_minus_t1 ** 2).astype(np.float32)


    ### Part 7: Apply masks and classify loss/gain pixels
    # [PDF §6] Exclude pixels outside mineral-soil mask.
    # has_data: NaN in either time block's U → exclude (data quality; propagated from Parts 4-5).

    has_data = ~(np.isnan(U_minus_delta) | np.isnan(U_plus_delta))
    valid = has_data & has_mineral_soil

    U_minus_delta_masked = np.where(valid, U_minus_delta, np.nan).astype(np.float32)
    U_plus_delta_masked  = np.where(valid, U_plus_delta,  np.nan).astype(np.float32)

    # Per-time-block intermediates: same mineral-soil mask, but per-block data validity
    has_data_t1 = ~(np.isnan(U_minus_t1) | np.isnan(U_plus_t1))
    has_data_t2 = ~(np.isnan(U_minus_t2) | np.isnan(U_plus_t2))
    U_minus_t1_masked = np.where(has_data_t1 & has_mineral_soil, U_minus_t1, np.nan).astype(np.float32)
    U_plus_t1_masked  = np.where(has_data_t1 & has_mineral_soil, U_plus_t1,  np.nan).astype(np.float32)
    U_minus_t2_masked = np.where(has_data_t2 & has_mineral_soil, U_minus_t2, np.nan).astype(np.float32)
    U_plus_t2_masked  = np.where(has_data_t2 & has_mineral_soil, U_plus_t2,  np.nan).astype(np.float32)

    # Central estimate of stock change at 120m; used to classify loss vs gain pixels.
    # NaN where has_data is False (inaccessible COG) so those pixels are excluded from
    # both gross loss and gross gain — they don't silently inflate either category.
    delta_mean_raw = (mean_t2 - mean_t1).astype(np.float32)
    delta_mean_masked = np.where(valid, delta_mean_raw, np.nan).astype(np.float32)

    # Loss/gain classification based on central estimate sign (fixed classification).
    # Pixels where |delta_mean| < U⁺_Δ could plausibly flip sign, but are treated as
    # pure loss or gain — a standard first-order approximation (see module docstring).
    is_loss = valid & (delta_mean_raw < np.float32(0.0))
    is_gain = valid & (delta_mean_raw > np.float32(0.0))

    # Binary masks: 1.0 where classified, NaN elsewhere (float32 for GeoTIFF compatibility)
    loss_mask = np.where(is_loss, np.float32(1.0), np.nan).astype(np.float32)
    gain_mask = np.where(is_gain, np.float32(1.0), np.nan).astype(np.float32)


    ### Part 8: Per-chunk contributions for global aggregation
    # [PDF §7] Accumulated separately for net change, gross loss, and gross gain.
    # Country-level aggregation [PDF §6] is skipped; pixel values used directly.
    # For gross loss: U⁻_Δ = how much deeper the loss could be; U⁺_Δ = how much shallower.
    # For gross gain: U⁺_Δ = how much larger the gain could be; U⁻_Δ = how much smaller.

    sum_U_minus_squared     = float(np.nansum(U_minus_delta_masked ** 2))
    sum_U_plus_squared      = float(np.nansum(U_plus_delta_masked  ** 2))
    sum_U_minus_loss_sq     = float(np.nansum(np.where(is_loss, U_minus_delta ** 2, np.float32(0.0))))
    sum_U_plus_loss_sq      = float(np.nansum(np.where(is_loss, U_plus_delta  ** 2, np.float32(0.0))))
    sum_U_minus_gain_sq     = float(np.nansum(np.where(is_gain, U_minus_delta ** 2, np.float32(0.0))))
    sum_U_plus_gain_sq      = float(np.nansum(np.where(is_gain, U_plus_delta  ** 2, np.float32(0.0))))

    lu.print_and_log(
        f"After computing U⁻/U⁺_Δ for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB",
        False, logger_worker
    )


    ### Part 9: Chunk stats (always; useful for QC even without upload)

    for arr, name in [
        (U_minus_delta_masked, f"{U_MINUS_DELTA_PATTERN}_{INTERVAL_LABEL}"),
        (U_plus_delta_masked,  f"{U_PLUS_DELTA_PATTERN}_{INTERVAL_LABEL}"),
        (U_minus_t1_masked,    f"{U_MINUS_T1_PATTERN}_{INTERVAL_LABEL}"),
        (U_plus_t1_masked,     f"{U_PLUS_T1_PATTERN}_{INTERVAL_LABEL}"),
        (U_minus_t2_masked,    f"{U_MINUS_T2_PATTERN}_{INTERVAL_LABEL}"),
        (U_plus_t2_masked,     f"{U_PLUS_T2_PATTERN}_{INTERVAL_LABEL}"),
        (delta_mean_masked,    f"{DELTA_MEAN_PATTERN}_{INTERVAL_LABEL}"),
        (loss_mask,            f"{LOSS_MASK_PATTERN}_{INTERVAL_LABEL}"),
        (gain_mask,            f"{GAIN_MASK_PATTERN}_{INTERVAL_LABEL}"),
    ]:
        chunk_stats_combined.append(uu.calculate_stats(arr, name, bounds_str, tile_id, 'output_layer'))


    ### Part 10: Save GeoTIFFs and upload

    if not no_upload:
        output_without_bucket = output_s3_dir[cn.full_bucket_prefix_length:]
        intermediates_base = f"{output_without_bucket}intermediates"

        out_dict = {
            f"{U_MINUS_DELTA_PATTERN}_{INTERVAL_LABEL}": [
                U_minus_delta_masked, 'float32', U_MINUS_DELTA_PATTERN, INTERVAL_LABEL, output_without_bucket,
            ],
            f"{U_PLUS_DELTA_PATTERN}_{INTERVAL_LABEL}": [
                U_plus_delta_masked, 'float32', U_PLUS_DELTA_PATTERN, INTERVAL_LABEL, output_without_bucket,
            ],
        }

        if upload_intermediates:
            out_dict.update({
                f"{U_MINUS_T1_PATTERN}_{INTERVAL_LABEL}": [
                    U_minus_t1_masked, 'float32', U_MINUS_T1_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/U_minus_t1/",
                ],
                f"{U_PLUS_T1_PATTERN}_{INTERVAL_LABEL}": [
                    U_plus_t1_masked, 'float32', U_PLUS_T1_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/U_plus_t1/",
                ],
                f"{U_MINUS_T2_PATTERN}_{INTERVAL_LABEL}": [
                    U_minus_t2_masked, 'float32', U_MINUS_T2_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/U_minus_t2/",
                ],
                f"{U_PLUS_T2_PATTERN}_{INTERVAL_LABEL}": [
                    U_plus_t2_masked, 'float32', U_PLUS_T2_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/U_plus_t2/",
                ],
                # Central estimate of stock change and loss/gain classification masks
                f"{DELTA_MEAN_PATTERN}_{INTERVAL_LABEL}": [
                    delta_mean_masked, 'float32', DELTA_MEAN_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/delta_mean/",
                ],
                f"{LOSS_MASK_PATTERN}_{INTERVAL_LABEL}": [
                    loss_mask, 'float32', LOSS_MASK_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/loss_mask/",
                ],
                f"{GAIN_MASK_PATTERN}_{INTERVAL_LABEL}": [
                    gain_mask, 'float32', GAIN_MASK_PATTERN, INTERVAL_LABEL,
                    f"{intermediates_base}/gain_mask/",
                ],
            })

        upload_tasks = uu.save_and_upload_small_raster_set(
            bounds, chunk_length_pixels_120m, tile_id, bounds_str,
            out_dict, is_large_run, logger_worker, np.nan
        )

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        lu.print_and_log(
            f"Uploaded {len(upload_tasks)} GeoTIFF(s) for {bounds_str} ({tile_id}): {uu.timestr()}",
            False, logger_worker
        )

    chunk_end_time = time.time()
    lu.print_and_log(
        f"  Total chunk processing for {bounds_str} in {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}",
        False, logger_worker
    )

    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"
    return (return_message,
            sum_U_minus_squared, sum_U_plus_squared,
            sum_U_minus_loss_sq, sum_U_plus_loss_sq,
            sum_U_minus_gain_sq, sum_U_plus_gain_sq,
            chunk_stats_combined)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cluster_name, run_local=False, no_stats=False, no_log=False, no_upload=False,
         upload_intermediates=False, chunk_shapefile_uri=False, bounding_box=None,
         chunk_size_deg=None, first_chunks=None, run_date=None,
         model_type=None, model_path_description=None, log_note=None):

    stage = 'mineral_soil_change_uncertainty_2020_interval'
    batch_size = 3800

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(
        client, cluster, log_note, run_local, model_type, stage)

    if not run_date:
        run_date = date.today().strftime('%Y%m%d')

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Model path description: {model_path_description}")
    main_logger.info(f"Interval: {INTERVAL_LABEL} (avg 2010-2015 → avg 2015-2020)  [PDF §2]")
    main_logger.info(f"Uncertainty method: asymmetric U⁻/U⁺ per pixel  [PDF §3]")
    main_logger.info(f"Time-block correlation: uncorrelated (no ρ term)  [PDF §5]")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"upload_intermediates: {upload_intermediates}")

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)

    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    is_large_run = len(chunk_list) > 20
    if is_large_run:
        main_logger.info(f"Large-scale run: {is_large_run}")

    output_s3_dir = f"{cn.full_bucket_prefix}/{cn.SOC_uncertainty_output_base}/{run_date}/"
    output_s3_dir = output_s3_dir.replace(cn.model_version_type_description_placeholder, f"version_{cn.SOC_model_version_underscore}__{model_type}__{model_path_description}")
    main_logger.info(f"Main output S3 directory: {output_s3_dir}")
    if upload_intermediates:
        main_logger.info(f"Intermediates S3 directory: {output_s3_dir}intermediates/")


    ### Step 1: Read metadata from 120m COGs
    # [PDF §1] Confirm nodata is set before launching workers; a missing nodata value would
    # cause convert() to silently treat fill pixels as valid data.
    # Also spot-check that all 6 COGs share the same nodata (warn if not; don't block).

    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(SOC_UNCERTAINTY_COGS['p16_t1']) as src:
            nodata_val_120m     = src.nodata
            pixel_size_deg_120m = abs(src.transform.a)

    if nodata_val_120m is None:
        raise RuntimeError(
            f"No nodata value in {SOC_UNCERTAINTY_COGS['p16_t1']}. "
            "Cannot distinguish fill values from valid zero-uncertainty pixels. "
            "Inspect COG metadata and set nodata_val_120m manually if needed."
        )

    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        for key, url in SOC_UNCERTAINTY_COGS.items():
            if key == 'p16_t1':
                continue
            with rasterio.open(url) as src:
                nd = src.nodata
            if nd != nodata_val_120m:
                main_logger.warning(
                    f"WARNING: nodata mismatch — {key} nodata={nd}, expected {nodata_val_120m}. "
                    "Check convert() in worker.")

    main_logger.info(f"120m COG nodata: {nodata_val_120m}")
    main_logger.info(f"120m COG pixel size: {pixel_size_deg_120m:.7f} deg  ({1/pixel_size_deg_120m:.1f} px/deg)")


    ### Step 2: Process chunks in batches

    main_logger.info("Workers' logs to be appended after main function log\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"Batches to process: {len(chunk_batches)}: {uu.timestr()}")

    all_stats = []
    success_count = 0
    total_sum_U_minus_sq      = 0.0   # net change: Σ (U⁻_Δ,j)²  [all valid pixels]
    total_sum_U_plus_sq       = 0.0   # net change: Σ (U⁺_Δ,j)²
    total_sum_U_minus_loss_sq = 0.0   # gross loss: Σ (U⁻_Δ,j)²  [delta_mean < 0 pixels]
    total_sum_U_plus_loss_sq  = 0.0   # gross loss: Σ (U⁺_Δ,j)²
    total_sum_U_minus_gain_sq = 0.0   # gross gain: Σ (U⁻_Δ,j)²  [delta_mean > 0 pixels]
    total_sum_U_plus_gain_sq  = 0.0   # gross gain: Σ (U⁺_Δ,j)²

    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(
            f"Processing batch {i+1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        uu.create_s3_task_files(stage, chunk_batch)

        futures = []
        for chunk in chunk_batch:
            future = client.submit(
                compute_soc_uncertainty,
                chunk, is_large_run, stage, no_upload, upload_intermediates,
                nodata_val_120m, pixel_size_deg_120m, output_s3_dir
            )
            futures.append(future)

        batch_results = client.gather(futures)

        # Extract U² accumulators; reformat to 2-tuples for count_successful_chunks
        # (which expects (return_message, chunk_stats)).
        formatted_results = []
        for result in batch_results:
            if result is not None:
                try:
                    (return_message,
                     sum_U_minus_sq, sum_U_plus_sq,
                     sum_U_minus_loss_sq, sum_U_plus_loss_sq,
                     sum_U_minus_gain_sq, sum_U_plus_gain_sq,
                     chunk_stats) = result
                    total_sum_U_minus_sq      += sum_U_minus_sq
                    total_sum_U_plus_sq       += sum_U_plus_sq
                    total_sum_U_minus_loss_sq += sum_U_minus_loss_sq
                    total_sum_U_plus_loss_sq  += sum_U_plus_loss_sq
                    total_sum_U_minus_gain_sq += sum_U_minus_gain_sq
                    total_sum_U_plus_gain_sq  += sum_U_plus_gain_sq
                    formatted_results.append((return_message, chunk_stats))
                except (TypeError, ValueError):
                    formatted_results.append(result)
            else:
                formatted_results.append(result)

        batch_success_count, batch_stats = uu.count_successful_chunks(
            chunk_batch, is_large_run, main_logger, formatted_results)
        success_count += batch_success_count
        all_stats.extend(batch_stats)

        if len(chunk_batches) > 1:
            df_batch_stats = pd.DataFrame(batch_stats)
            out_file = f"TEMP_BATCH_{stage}__batch_{i}_{uu.timestr()}.xlsx"
            local_path = f"{cn.local_chunk_stats_path}{out_file}"
            with pd.ExcelWriter(local_path) as writer:
                df_batch_stats.to_excel(writer, sheet_name=f"stats__batch_{i}", index=False)

        del futures, batch_results
        client.run(gc.collect)
        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)


    ### Step 3: Preliminary worker log

    if not run_local:
        worker_log_local_path_prelim = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with preliminary worker log", main_logger)


    ### Step 4: Compile chunk stats

    if (not no_stats) and (success_count > 0):
        model_chunk_stats_path = uu.compile_1x1_chunk_stats(
            all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stats", main_logger)


    ### Step 5: Compute and save global uncertainty scalars
    # [PDF §7] Root-sum-of-squares across all valid 120m mineral-soil pixels.
    # Country-level step [PDF §6] skipped; pixels used directly (spatially uncorrelated assumption).
    # Net change uses all valid pixels; gross loss/gain use only pixels classified by delta_mean sign.

    U_minus_global_MgC = float(np.sqrt(total_sum_U_minus_sq))
    U_plus_global_MgC  = float(np.sqrt(total_sum_U_plus_sq))
    U_minus_loss_MgC   = float(np.sqrt(total_sum_U_minus_loss_sq))
    U_plus_loss_MgC    = float(np.sqrt(total_sum_U_plus_loss_sq))
    U_minus_gain_MgC   = float(np.sqrt(total_sum_U_minus_gain_sq))
    U_plus_gain_MgC    = float(np.sqrt(total_sum_U_plus_gain_sq))

    for label, val in [
        ('U_net_lower',       U_minus_global_MgC),
        ('U_net_upper',       U_plus_global_MgC),
        ('U_loss_deeper',     U_minus_loss_MgC),
        ('U_loss_shallower',  U_plus_loss_MgC),
        ('U_gain_smaller',    U_minus_gain_MgC),
        ('U_gain_larger',     U_plus_gain_MgC),
    ]:
        main_logger.info(f"{label} = {val:.4e} Mg C  =  {val/1e6:.4f} Tg C  [PDF §7]")

    if not no_upload:
        df_global = pd.DataFrame([{
            'interval': INTERVAL_LABEL,
            'time_block_t1': '2010-2015',
            'time_block_t2': '2015-2020',
            'temporal_correlation': 'uncorrelated (PDF §5)',
            'spatial_correlation': 'uncorrelated pixels; country-level step skipped (PDF §6-7)',
            # Net change uncertainty (all valid mineral-soil pixels)
            # U_net_lower: net change could be this much more negative
            # U_net_upper: net change could be this much more positive
            'U_net_lower_MgC':  U_minus_global_MgC,
            'U_net_upper_MgC':  U_plus_global_MgC,
            'U_net_lower_TgC':  U_minus_global_MgC / 1e6,
            'U_net_upper_TgC':  U_plus_global_MgC  / 1e6,
            # Gross loss uncertainty (pixels where delta_mean < 0)
            # U_loss_deeper:    gross loss could be this much larger (more negative)
            # U_loss_shallower: gross loss could be this much smaller (less negative)
            'U_loss_deeper_MgC':    U_minus_loss_MgC,
            'U_loss_shallower_MgC': U_plus_loss_MgC,
            'U_loss_deeper_TgC':    U_minus_loss_MgC / 1e6,
            'U_loss_shallower_TgC': U_plus_loss_MgC  / 1e6,
            # Gross gain uncertainty (pixels where delta_mean > 0)
            # U_gain_larger:  gross gain could be this much larger (more positive)
            # U_gain_smaller: gross gain could be this much smaller (less positive)
            'U_gain_larger_MgC':  U_plus_gain_MgC,
            'U_gain_smaller_MgC': U_minus_gain_MgC,
            'U_gain_larger_TgC':  U_plus_gain_MgC  / 1e6,
            'U_gain_smaller_TgC': U_minus_gain_MgC / 1e6,
            'run_date': run_date,
            'model_path_description': model_path_description or '',
        }])

        global_csv_filename = f"SOC_uncertainty_2020_interval_global_{run_date}.csv"
        local_csv_path = f"/tmp/{global_csv_filename}"
        df_global.to_csv(local_csv_path, index=False)

        s3_key = f"{SOC_UNCERTAINTY_OUTPUT_BASE}/{run_date}/{global_csv_filename}"
        boto3.client('s3').upload_file(local_csv_path, cn.short_bucket_prefix, s3_key)
        main_logger.info(f"Global uncertainty CSV: s3://{cn.short_bucket_prefix}/{s3_key}")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with global uncertainty", main_logger)


    ### Step 6: Resize cluster

    if not run_local:
        n_workers_now = len(client.scheduler_info()["workers"])
        if n_workers_now > 10:
            main_logger.info("Resizing cluster to 1 worker")
            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 7: Count output GeoTIFFs

    if not no_upload and is_large_run:
        geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_s3_dir)
        expected = success_count * 2   # U⁻_Δ and U⁺_Δ per chunk
        main_logger.info(f"Output rasters in {output_s3_dir}: {file_count}  (expected {expected})")
        if file_count != expected:
            main_logger.warning("WARNING: File count mismatch — check for failed uploads.")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 8: Final worker log

    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute asymmetric SOC uncertainty (U⁻_Δ, U⁺_Δ) for the 2020 interval. "
                    "Method: Uncertainty_Propagation_in_SOC_Stocks__from_Serkan_Isik_20260605.pdf")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W S E N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size in degrees')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='S3 URI for 1×1 deg fishnet shapefile')
    parser.add_argument('-f', '--first_chunks', type=int, help='Process only the first N chunks from shapefile')
    parser.add_argument('-mpd', '--model_path_description', help='Description for this run')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Skip chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Skip combined worker log')
    parser.add_argument('--no_upload', action='store_true', help='Skip all S3 uploads')
    parser.add_argument('--upload_intermediates', action='store_true', help='Also upload U⁻_t1, U⁺_t1, U⁻_t2, U⁺_t2, delta_mean, loss_mask, gain_mask GeoTIFFs for QC')

    args = parser.parse_args()

    main(
        args.cluster_name,
        args.run_local, args.no_stats, args.no_log, args.no_upload,
        args.upload_intermediates,
        args.chunk_shapefile_uri, args.bounding_box, args.chunk_size_deg, args.first_chunks,
        args.run_date, args.model_type, args.model_path_description, args.log_note,
    )

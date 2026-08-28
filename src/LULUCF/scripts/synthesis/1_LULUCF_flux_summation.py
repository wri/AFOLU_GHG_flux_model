"""
Creates global LULUCF-level 30m outputs by summing vegetation (annual, 2016-2024), mineral SOC (block-based),
and organic soil (block-based) components.
With Claude session 'LULUCF 30-m outputs script'

Temporal block mapping:
  Veg years 2016-2020  →  SOC 2010-2015 block vs. 2015-2020 block (i.e. change interval 2020, zarr idx 3)  +  org soil 2016-2020 block (zarr idx 3)
  Veg years 2021-2024  →  SOC 2010-2015 block vs. 2015-2020 block (i.e. change interval 2020, zarr idx 3)  +  org soil 2021-2024 block (zarr idx 4)
  We are using the mineral soil 2020 change interval for all years because the 2022 change interval had
  much higher gross loss and gain than preceding intervals and we didn't trust it.
  Best to just use the 2020 interval for all years for now.

SOC zarr index notes:
  The SOC zarr has 5 time slices keyed by cn.SOC_density_intervals = [2005, 2010, 2015, 2020, 2022].
  Change variables have fill (NaN) at idx 0 (year 2005 has no change computed ending on it).
  Change ending 2020 (2015-2020 interval) → zarr idx 3.
  Change ending 2022 (2020-2022 interval) → zarr idx 4.

Sign convention (matching vegetation model):
  Positive = net emission to atmosphere; negative = net removal.
  Gross emissions: positive.  Gross removals: negative.

NaN handling:
  Each of the three components (veg, SOC, org soil) contributes 0 where it has no data.
  Output is NaN only when all three components are NaN for a pixel.

NoData = np.nan.

Outputs (all-gases, MgCO2e/ha/yr):
  LULUCF_gross_emissions__all_C_pools__all_gases__MgCO2e  = veg gross emis + SOC loss + org soil total
  LULUCF_gross_removals__all_C_pools__MgCO2               = veg gross removals + SOC gain
  LULUCF_net_flux__all_C_pools__all_gases__MgCO2e         = veg net + SOC net + org soil total

Zarrs:
  • Timeseries zarr: 9 annual years, chunks (9, 4000, 4000)
  • Annual average zarr: 1 year labelled 'avg', chunks (1, 4000, 4000)

Geotif outputs (timeseries and annual avg):
  • 1x1 deg per-ha
  • 10x10 deg per-ha, per-pixel, and 0.04x0.04 deg aggregated

Processing unit: 10x10 deg tiles (each worker processes 100 constituent 1x1 sub-chunks,
then assembles 10x10 geotifs in the same pass).

Because processing is in 10x10 deg tiles, which can take 2-3 hours to run apiece, this script
progressively shuts down workers as they are no longer needed. That way, only a few workers more than are needed
are actually still running, rather than all workers running the whole time but not doing anything once they finish.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test:
python -m src.LULUCF.synthesis.scripts.1_LULUCF_flux_summation --run_local --no_upload -bb 110 -10 120 0 -mt standard -mpd test_box --veg_date 20260130 --veg_mpd global --soc_date 20260611 --soc_mpd global  -create_zarr

Coiled small test:
python -m src.utilities.create_cluster -n 1 -t 1 -m 64 -cn LULUCF_summation
python -m src.LULUCF.synthesis.scripts.1_LULUCF_flux_summation -cn LULUCF_summation --no_upload -bb 110 -10 120 0 -mt standard -mpd test_tile_00N_110E --veg_date 20260130 --veg_mpd global --soc_date 20260611 --soc_mpd global --create_zarr

Full run (150 workers based on discussion with Claude session 'LULUCF 30-m outputs script' about how different numbers of workers will affect runtime):
python -m src.utilities.create_cluster -n 150 -t 1 -m 64 -cn LULUCF_summation
python -m src.LULUCF.synthesis.scripts.1_LULUCF_flux_summation -cn LULUCF_summation -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --veg_date 20260130 --veg_mpd global --soc_date 20260611 --soc_mpd global --create_zarr --log_note "LULUCF v1.0.0 fluxes: veg v1.0.5 + SOC v1.0.1 + org soil v1.0.1, 2016-2024."

#TODO Parallelize 10x10 deg tile uploads in create_10x10_deg_geotif_from_zarr, per Claude session 'LULUCF 30-m outputs script'. Applies to veg, SOC, and LULUCF. Haven't tried at all.
#TODO Parallelize outer loop for var in LULUCF_OUTPUTS_TO_ZARR: with max_workers=2 to speed 10x10 deg uploads (separate from change to create_10x10_deg_geotif_from_zarr, per Claude session 'LULUCF 30-m outputs script'
#TODO Figure out why LULUCF chunk stats emissions and the emissions from the zonal stats-based LULUCF table don't match by 0.024 Gt CO2/yr, as described in https://app.asana.com/1/25496124013636/task/1215782176192854/comment/1215782176192873?focus=true
#TODO Include output for total carbon density (vegetation + soil) (need to use different SOC blocks for different vegetation years)
"""

import argparse
import gc
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import time
import ctypes
import resource
from rasterio.transform import from_origin
import fsspec
import numpy as np
import pandas as pd
import psutil
import zarr
from dask.distributed import print, as_completed
from dask import config

from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SPLIT_YEAR = 2021  # First year of second soil block

# SOC zarr time indices for the two LULUCF-relevant change blocks.
# SOC zarr time axis = cn.SOC_density_intervals = [2005, 2010, 2015, 2020, 2022].
# Change ending 2020 (2015-2020 interval) is at idx 3; change ending 2022 is at idx 4.
SOC_BLOCK1_ZARR_IDX = 3   # 2016-2020 years → SOC 2015-2020 interval (end year 2020)

# Organic soil zarr time indices for the two LULUCF-relevant blocks.
ORG_SOIL_BLOCK1_ZARR_IDX = 3   # 2016-2020 block
ORG_SOIL_BLOCK2_ZARR_IDX = 4   # 2021-2024 block


# Variable names within source zarrs
VEG_EMIS_VAR     = f"{cn.gross_emis_all_C_pools_all_gases_pattern}{cn.flux_density_pixel_meaning}"
VEG_REMOVALS_VAR = f"{cn.gross_removals_all_C_pools_pattern}{cn.flux_density_pixel_meaning}"
VEG_NET_VAR      = f"{cn.net_flux_all_C_pools_all_gases_pattern}{cn.flux_density_pixel_meaning}"
SOC_LOSS_VAR     = f"{cn.SOC_loss_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}"
SOC_GAIN_VAR     = f"{cn.SOC_gain_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}"
SOC_NET_VAR      = f"{cn.SOC_net_min_soil_extent_pattern}{cn.flux_density_pixel_meaning}"
ORG_BURNED_VAR   = f"{cn.organic_soil_burned_pattern}{cn.flux_density_pixel_meaning}"
ORG_DRAINED_VAR  = f"{cn.organic_soil_drained_pattern}{cn.flux_density_pixel_meaning}"

# LULUCF output patterns (base, without unit suffix)
LULUCF_OUTPUTS_TO_ZARR = [
    cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern,
    cn.gross_removals_all_C_pools_LULUCF_pattern,
    cn.net_flux_all_C_pools_all_gases_LULUCF_pattern,
]

# LULUCF output variable names with unit suffix (as stored in zarrs)
LULUCF_EMIS_VAR     = f"{cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern}{cn.flux_density_pixel_meaning}"
LULUCF_REMOVALS_VAR = f"{cn.gross_removals_all_C_pools_LULUCF_pattern}{cn.flux_density_pixel_meaning}"
LULUCF_NET_VAR      = f"{cn.net_flux_all_C_pools_all_gases_LULUCF_pattern}{cn.flux_density_pixel_meaning}"


# ---------------------------------------------------------------------------
# Helper: combine components, treating NaN as absent (not missing)
# ---------------------------------------------------------------------------

def _combine_components(*arrays):
    """
    Sums arrays element-wise, treating NaN as absent.
    A component is considered 'present' (has real data) only when it is non-NaN AND non-zero.
    Output is NaN when no component has real data — this correctly handles:
      • NaN fill values (standard NoData)
      • 0 fill values (e.g. org soil for non-peatland pixels)
      • SOC 0 meaning 'no change' in pixels outside the veg/org soil domain
    Output is 0 only when at least one component is non-NaN/non-zero but the sum happens to be 0.
    """
    has_data = np.zeros(arrays[0].shape, dtype=bool)
    total    = np.zeros(arrays[0].shape, dtype=np.float32)
    for arr in arrays:
        nan_mask  = np.isnan(arr)
        total    += np.where(nan_mask, np.float32(0), arr)
        has_data |= (~nan_mask & (arr != 0))   # present = non-NaN AND non-zero
    return np.where(has_data, total, np.nan).astype(np.float32)


def period_mean(arr_3d):
    """
    Computes annual average over the full 9-year period by always dividing by cn.end_year_count,
    not by the count of non-NaN years. A pixel with data only in 2016 gets value/9, not value/1.
    Pixels that are NaN in all 9 years remain NaN.
    """
    total = np.nansum(arr_3d, axis=0).astype(np.float32)  # NaN treated as 0 in sum
    all_nan = np.all(np.isnan(arr_3d), axis=0)  # True where no data in any year
    return np.where(all_nan, np.nan, total / cn.veg_end_year_count).astype(np.float32)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def calculate_LULUCF_fluxes(tile_id, is_large_run, stage, no_upload, create_zarr,
                              veg_zarr_path, soc_zarr_path,
                              lulucf_zarr_path, lulucf_avg_zarr_path,
                              outputs_1x1_dir_by_year, outputs_1x1_avg_dirs,
                              output_base_10x10, model_type, model_path_description,
                              valid_sub_chunks=None):
    """
    Processes one 10x10 deg tile end-to-end:
      Phase 1: Process each of the (up to) 100 1x1 sub-chunks; write 1x1 per-ha timeseries and average geotifs and populate respective zarrs.
      Phase 2: Create 10x10 timeseries geotifs (per-ha, per-pixel, 0.04 deg) from LULUCF zarr.
      Phase 3: Create 10x10 annual-average geotifs from in-memory tile accumulators.
    """
    process = psutil.Process(os.getpid())
    logger_worker = lu.setup_logging_worker()
    tile_start = time.time()

    lu.print_and_log(f"Starting tile {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    # Open source zarrs once per tile
    fs = fsspec.filesystem("s3", anon=False)
    veg_zarr  = zarr.open_group(fs.get_mapper(veg_zarr_path),      mode="r", use_consolidated=False)
    soc_zarr  = zarr.open_group(fs.get_mapper(soc_zarr_path),      mode="r", use_consolidated=False)
    org_zarr  = zarr.open_group(fs.get_mapper(cn.organic_soil_zarr_path), mode="r", use_consolidated=False)

    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)
    chunk_list_for_tile = uu.get_chunk_bounds_from_bounding_box([min_x, min_y, max_x, max_y], 1)

    # Gets just the chunks in the tile that are in the chunk list shapefile (i.e. skips ocean ones)
    if valid_sub_chunks is not None:
        chunk_list_for_tile = [b for b in chunk_list_for_tile if tuple(b) in valid_sub_chunks]

    lu.print_and_log(f"--- Creating 1x1 outputs for {tile_id}: {len(chunk_list_for_tile)} valid chunks in tile: {uu.timestr()}",False, logger_worker)

    tile_pixels = cn.full_raster_dims  # 40000
    # Tile-level annual-average accumulators (held throughout tile processing)
    avg_emis_tile     = np.full((tile_pixels, tile_pixels), np.nan, dtype=np.float32)
    avg_removals_tile = np.full((tile_pixels, tile_pixels), np.nan, dtype=np.float32)
    avg_net_tile      = np.full((tile_pixels, tile_pixels), np.nan, dtype=np.float32)

    chunk_stats_combined = []

    # -----------------------------------------------------------------------
    # Phase 1: process each 1x1 sub-chunk
    # -----------------------------------------------------------------------

    lu.print_and_log(f"--- Creating 1x1 outputs for {tile_id}: {uu.timestr()}", False, logger_worker)

    for chunk_idx, bounds in enumerate(chunk_list_for_tile):
    # for chunk_idx, bounds in enumerate(chunk_list_for_tile[92:97]): # for testing

        lu.print_and_log(f"Processing chunk {chunk_idx+1} ({bounds}) of {len(chunk_list_for_tile)} in {tile_id}", is_large_run, logger_worker)

        bounds_str       = uu.boundstr(bounds)
        subtile_id       = uu.xy_to_tile_id(bounds[0], bounds[3])
        chunk_len_pixels = uu.calc_chunk_length_pixels(bounds)  # 4000

        # Global zarr spatial slice indices
        lat0, lon0 = zu.latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)
        lat1, lon1 = zu.latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)

        # Position of this sub-chunk within the 40000×40000 tile accumulator
        tile_row0 = int(round((max_y - bounds[3]) / cn.resolution))
        tile_col0 = int(round((bounds[0] - min_x) / cn.resolution))
        tile_row1 = tile_row0 + chunk_len_pixels
        tile_col1 = tile_col0 + chunk_len_pixels

        # --- Read veg (all 9 annual years at once): shape (9, 4000, 4000) ---
        veg_emis_all     = veg_zarr[VEG_EMIS_VAR    ][0:cn.veg_end_year_count, lat0:lat1, lon0:lon1].astype(np.float32)
        veg_removals_all = veg_zarr[VEG_REMOVALS_VAR][0:cn.veg_end_year_count, lat0:lat1, lon0:lon1].astype(np.float32)
        veg_net_all      = veg_zarr[VEG_NET_VAR      ][0:cn.veg_end_year_count, lat0:lat1, lon0:lon1].astype(np.float32)

        # --- Read SOC blocks (direct zarr index: 3 (used for all years for now)) ---
        soc_loss_b1 = soc_zarr[SOC_LOSS_VAR][SOC_BLOCK1_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)
        soc_gain_b1 = soc_zarr[SOC_GAIN_VAR][SOC_BLOCK1_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)
        soc_net_b1  = soc_zarr[SOC_NET_VAR ][SOC_BLOCK1_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)

        # --- Read org soil blocks (indices 3 and 4) ---
        org_burned_b1  = org_zarr[ORG_BURNED_VAR ][ORG_SOIL_BLOCK1_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)
        org_burned_b2  = org_zarr[ORG_BURNED_VAR ][ORG_SOIL_BLOCK2_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)
        org_drained_b1 = org_zarr[ORG_DRAINED_VAR][ORG_SOIL_BLOCK1_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)
        org_drained_b2 = org_zarr[ORG_DRAINED_VAR][ORG_SOIL_BLOCK2_ZARR_IDX, lat0:lat1, lon0:lon1].astype(np.float32)

        # Combine burned+drained for each org soil block (NaN where both absent)
        org_total_b1 = _combine_components(org_burned_b1, org_drained_b1)
        org_total_b2 = _combine_components(org_burned_b2, org_drained_b2)
        del org_burned_b1, org_burned_b2, org_drained_b1, org_drained_b2

        # --- Compute 9 annual LULUCF outputs ---
        ny = lat1 - lat0
        nx = lon1 - lon0
        lulucf_emis     = np.full((cn.veg_end_year_count, ny, nx), np.nan, dtype=np.float32)
        lulucf_removals = np.full((cn.veg_end_year_count, ny, nx), np.nan, dtype=np.float32)
        lulucf_net      = np.full((cn.veg_end_year_count, ny, nx), np.nan, dtype=np.float32)

        for i, year in enumerate(cn.veg_outputs_years):
            # SOC: always use the 2015-2020 block for all years
            soc_loss = soc_loss_b1;  soc_gain = soc_gain_b1;  soc_net = soc_net_b1
            # Org soil: split which block to use at SPLIT_YEAR
            org_total = org_total_b1 if year < SPLIT_YEAR else org_total_b2

            lulucf_emis[i]     = _combine_components(veg_emis_all[i],     soc_loss,  org_total)
            lulucf_removals[i] = _combine_components(veg_removals_all[i], soc_gain)
            lulucf_net[i]      = _combine_components(veg_net_all[i],      soc_net,   org_total)

        del (veg_emis_all, veg_removals_all, veg_net_all,
             soc_loss_b1, soc_gain_b1, soc_net_b1,
             org_total_b1, org_total_b2)

        # --- Annual averages for this sub-chunk across all 9 years---
        lulucf_emis_avg = period_mean(lulucf_emis)
        lulucf_removals_avg = period_mean(lulucf_removals)
        lulucf_net_avg = period_mean(lulucf_net)

        # Place sub-chunk averages into 10x10 tile accumulators
        avg_emis_tile    [tile_row0:tile_row1, tile_col0:tile_col1] = lulucf_emis_avg
        avg_removals_tile[tile_row0:tile_row1, tile_col0:tile_col1] = lulucf_removals_avg
        avg_net_tile     [tile_row0:tile_row1, tile_col0:tile_col1] = lulucf_net_avg

        # --- Populate timeseries zarr ---
        zarr_out_dict = {}
        for i, year in enumerate(cn.veg_outputs_years):
            zarr_out_dict[f"{LULUCF_EMIS_VAR}_{year}"]     = lulucf_emis[i]
            zarr_out_dict[f"{LULUCF_REMOVALS_VAR}_{year}"] = lulucf_removals[i]
            zarr_out_dict[f"{LULUCF_NET_VAR}_{year}"]      = lulucf_net[i]

        lu.print_and_log(f"Writing select outputs to global timeseries zarr for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
        zu.populate_zarr(bounds, bounds_str, create_zarr, cn.veg_outputs_years, False, logger_worker,
                         lulucf_zarr_path, zarr_out_dict, LULUCF_OUTPUTS_TO_ZARR, stage, subtile_id)

        # --- Populate annual average zarr (year label 'avg', stored at index 0) ---
        avg_zarr_dict = {
            f"{LULUCF_EMIS_VAR}_avg":     lulucf_emis_avg,
            f"{LULUCF_REMOVALS_VAR}_avg": lulucf_removals_avg,
            f"{LULUCF_NET_VAR}_avg":      lulucf_net_avg,
        }
        lu.print_and_log(f"Writing select outputs to global average zarr for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
        zu.populate_zarr(bounds, bounds_str, create_zarr, ['avg'], False, logger_worker,
                         lulucf_avg_zarr_path, avg_zarr_dict, LULUCF_OUTPUTS_TO_ZARR, stage, subtile_id)

        # --- Write 1x1 deg per-ha geotifs ---
        if not no_upload:
            lu.print_and_log(f"Saving 1x1 deg outputs in cluster for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
            upload_dict = {}
            for i, year in enumerate(cn.veg_outputs_years):
                emis_dir = outputs_1x1_dir_by_year[(cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern, year)]
                rem_dir  = outputs_1x1_dir_by_year[(cn.gross_removals_all_C_pools_LULUCF_pattern,       year)]
                net_dir  = outputs_1x1_dir_by_year[(cn.net_flux_all_C_pools_all_gases_LULUCF_pattern,   year)]
                upload_dict[f"{LULUCF_EMIS_VAR}_{year}"]     = [lulucf_emis[i],     'float32', cn.flux_density_pixel_meaning, year, emis_dir[cn.full_bucket_prefix_length:]]
                upload_dict[f"{LULUCF_REMOVALS_VAR}_{year}"] = [lulucf_removals[i], 'float32', cn.flux_density_pixel_meaning, year, rem_dir[cn.full_bucket_prefix_length:]]
                upload_dict[f"{LULUCF_NET_VAR}_{year}"]      = [lulucf_net[i],      'float32', cn.flux_density_pixel_meaning, year, net_dir[cn.full_bucket_prefix_length:]]

            upload_dict[f"{LULUCF_EMIS_VAR}_avg_{cn.veg_year_range_str}"]  = [lulucf_emis_avg, 'float32', cn.flux_density_pixel_meaning,
                                                      'avg', outputs_1x1_avg_dirs[cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern][cn.full_bucket_prefix_length:]]
            upload_dict[f"{LULUCF_REMOVALS_VAR}_avg_{cn.veg_year_range_str}"] = [lulucf_removals_avg, 'float32', cn.flux_density_pixel_meaning,
                                                         'avg', outputs_1x1_avg_dirs[cn.gross_removals_all_C_pools_LULUCF_pattern][cn.full_bucket_prefix_length:]]
            upload_dict[f"{LULUCF_NET_VAR}_avg_{cn.veg_year_range_str}"] = [lulucf_net_avg, 'float32', cn.flux_density_pixel_meaning,
                                                    'avg', outputs_1x1_avg_dirs[cn.net_flux_all_C_pools_all_gases_LULUCF_pattern][cn.full_bucket_prefix_length:]]

            upload_tasks = uu.save_and_upload_small_raster_set(
                bounds, chunk_len_pixels, subtile_id, bounds_str,
                upload_dict, is_large_run, logger_worker, np.nan
            )
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        # --- Chunk stats ---
        pixel_area_uri   = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{subtile_id}.tif"
        pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_len_pixels, logger_worker, 'Float32')[0]

        for i, year in enumerate(cn.veg_outputs_years):
            for key, arr in [
                (f"{LULUCF_EMIS_VAR}_{year}",     lulucf_emis[i]),
                (f"{LULUCF_REMOVALS_VAR}_{year}", lulucf_removals[i]),
                (f"{LULUCF_NET_VAR}_{year}",      lulucf_net[i]),
            ]:
                per_pixel = arr * pixel_area_chunk * cn.m2_to_ha
                chunk_stats_combined.append(uu.calculate_stats(arr, key, bounds_str, subtile_id, 'output_layer', per_pixel))

        for key, arr in [
            (f"{LULUCF_EMIS_VAR}_avg_{cn.veg_year_range_str}", lulucf_emis_avg),
            (f"{LULUCF_REMOVALS_VAR}_avg_{cn.veg_year_range_str}", lulucf_removals_avg),
            (f"{LULUCF_NET_VAR}_avg_{cn.veg_year_range_str}", lulucf_net_avg),
        ]:
            per_pixel = arr * pixel_area_chunk * cn.m2_to_ha
            chunk_stats_combined.append(uu.calculate_stats(arr, key, bounds_str, subtile_id, 'output_layer', per_pixel))

        lu.print_and_log(f"  {bounds_str}: {process.memory_info().rss / 1024 ** 2:.1f} MB RAM", False, logger_worker)

        del lulucf_emis, lulucf_removals, lulucf_net
        del lulucf_emis_avg, lulucf_removals_avg, lulucf_net_avg
        del pixel_area_chunk
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # returns free glibc pages to OS
        except Exception:
            pass
        lu.print_and_log(f"Memory before Phase 2: {psutil.Process().memory_info().rss / 1e6:.1f} MB", is_large_run, logger_worker)

    tile_end_1x1 = time.time()
    lu.print_and_log(f"Completed 1x1 processing in {tile_id} in {round(tile_end_1x1 - tile_start)} seconds: {uu.timestr()}", is_large_run, logger_worker)

    # -----------------------------------------------------------------------
    # Phase 2: 10x10 deg timeseries geotifs from the LULUCF zarr
    # Reads one (var, year) at a time from zarr to avoid holding all 9 years
    # (~173 GB) in memory simultaneously.
    # -----------------------------------------------------------------------
    lu.print_and_log(f"--- Creating 10x10 timeseries geotifs for {tile_id}: {uu.timestr()}", False, logger_worker)

    #TODO Parallelize with max_workers=2 to speed 10x10 deg uploads (separate from change to create_10x10_deg_geotif_from_zarr, per Claude session 'LULUCF 30-m outputs script'
    for var in LULUCF_OUTPUTS_TO_ZARR:
        for year_idx in range(cn.veg_end_year_count):
            zu.create_10x10_deg_geotif_from_zarr(
                var, year_idx, tile_id, lulucf_zarr_path, output_base_10x10,
                cn.LULUCF_model_version_underscore, model_type, model_path_description,
                no_upload, False, np.nan
            )

    # -----------------------------------------------------------------------
    # Phase 3: 10x10 deg annual average geotifs from in-memory tile accumulators
    # Produces per-ha, per-pixel, and 0.04 deg outputs directly without re-reading zarr.
    # -----------------------------------------------------------------------
    lu.print_and_log(f"--- Creating 10x10 annual average geotifs for {tile_id}: {uu.timestr()}", False, logger_worker)

    if not no_upload:

        pixel_area_zarr_store = uu.get_pixel_area_store()
        lat_arr_pa = pixel_area_zarr_store["y"][:]
        lon_arr_pa = pixel_area_zarr_store["x"][:]

        y0_pa = len(lat_arr_pa) - np.searchsorted(lat_arr_pa[::-1], min_y, side='left')
        y1_pa = len(lat_arr_pa) - np.searchsorted(lat_arr_pa[::-1], max_y, side='right')
        x0_pa = np.searchsorted(lon_arr_pa, min_x, side='left')
        x1_pa = np.searchsorted(lon_arr_pa, max_x, side='right')
        if y0_pa > y1_pa:
            y0_pa, y1_pa = y1_pa, y0_pa
        pixel_area_tile = pixel_area_zarr_store['band_data'][y0_pa:y1_pa, x0_pa:x1_pa]

        tile_transform   = from_origin(min_x, max_y, cn.resolution,                    cn.resolution)
        coarse_transform = from_origin(min_x, max_y, cn.global_geotif_resolution, cn.global_geotif_resolution)

        LULUCF_run = cn.LULUCF_full_version_underscore.replace("MODEL_TYPE", model_type)
        LULUCF_run = LULUCF_run.replace("MODEL_PATH_DESCRIPTION", model_path_description)

        for pattern, avg_arr in [
            (cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern, avg_emis_tile),
            (cn.gross_removals_all_C_pools_LULUCF_pattern,       avg_removals_tile),
            (cn.net_flux_all_C_pools_all_gases_LULUCF_pattern,   avg_net_tile),
        ]:
            data_per_pixel = avg_arr * pixel_area_tile * cn.m2_to_ha

            # 0.04 deg aggregation (sum of per-pixel values within each coarse cell)
            ny, nx = data_per_pixel.shape
            ny_trim = ny - (ny % cn.global_aggregation_factor)
            nx_trim = nx - (nx % cn.global_aggregation_factor)
            data_fine_trim = data_per_pixel[:ny_trim, :nx_trim]
            reshaped = data_fine_trim.reshape(
                ny_trim // cn.global_aggregation_factor, cn.global_aggregation_factor,
                nx_trim // cn.global_aggregation_factor, cn.global_aggregation_factor
            )
            coarse_agg = np.nansum(reshaped, axis=(1, 3)).astype(np.float32)
            # Cells with no valid fine pixels → NaN
            coarse_agg[np.sum(~np.isnan(reshaped), axis=(1, 3)) == 0] = np.nan

            var_with_unit = f"{pattern}{cn.flux_density_pixel_meaning}"
            base = (
                cn.LULUCF_outputs_path
                .replace(cn.model_version_type_description_placeholder, LULUCF_run)
            )
            run_date_str = outputs_1x1_avg_dirs[pattern].rstrip('/').split('/')[-1]

            s3_ha  = f"{base}{pattern}/annual_intervals/avg_{cn.veg_year_range_str}/{cn.flux_density_pixel_meaning}/{cn.full_raster_dims}_pixels/{run_date_str}/{tile_id}__{var_with_unit}_avg_{cn.veg_year_range_str}.tif"
            s3_px  = f"{base}{pattern}/annual_intervals/avg_{cn.veg_year_range_str}/{cn.flux_per_pixel_pixel_meaning}/{cn.full_raster_dims}_pixels/{run_date_str}/{tile_id}__{var_with_unit.replace(cn.flux_density_pixel_meaning, cn.flux_per_pixel_pixel_meaning)}_avg_{cn.veg_year_range_str}.tif"
            s3_crs = f"{base}{pattern}/annual_intervals/avg_{cn.veg_year_range_str}/{cn.flux_aggreg_pixel_meaning}/{cn.global_aggregation_factor}_pixels/{run_date_str}/{tile_id}__{var_with_unit.replace(cn.flux_density_pixel_meaning, cn.flux_aggreg_pixel_meaning)}_avg_{cn.veg_year_range_str}.tif"

            uu.write_single_geotiff_to_s3(pattern, 'avg', tile_id, avg_arr,        np.nan, tile_transform,   s3_ha,  logger_worker)
            uu.write_single_geotiff_to_s3(pattern, 'avg', tile_id, data_per_pixel, np.nan, tile_transform,   s3_px,  logger_worker)
            uu.write_single_geotiff_to_s3(pattern, 'avg', tile_id, coarse_agg,     np.nan, coarse_transform, s3_crs, logger_worker)

            del data_per_pixel, data_fine_trim, reshaped, coarse_agg

        del pixel_area_tile, avg_emis_tile, avg_removals_tile, avg_net_tile

    tile_end = time.time()
    lu.print_and_log( f"Completed 10x10 processing for {tile_id} in {round(tile_end - tile_end_1x1)} seconds: {uu.timestr()}", is_large_run, logger_worker)
    lu.print_and_log(f"Completed full processing for {tile_id} in {round(tile_end - tile_start)} seconds: {uu.timestr()}", is_large_run, logger_worker)
    lu.print_and_log(f"Peak memory for {tile_id}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GB", False, logger_worker)

    return f"Success for tile {tile_id}: {uu.timestr()}", chunk_stats_combined


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main(cluster_name, model_type,
         veg_date, veg_model_path_description,
         soc_date, soc_model_path_description,
         run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None,
         run_date=None, model_path_description=None, log_note=None):

    # -----------------------------------------------------------------------
    # Step 1: Preparation
    # -----------------------------------------------------------------------
    stage      = 'LULUCF_flux_summation'

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    config.set({"distributed.scheduler.allowed-failures": 2})

    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(
        client, cluster, log_note, run_local, model_type, stage)

    if not run_date:
        run_date = date.today().strftime("%Y%m%d")

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"LULUCF model version: {cn.LULUCF_model_version}")
    main_logger.info(f"Veg: {cn.veg_model_version};  SOC: {cn.SOC_model_version}; Organic soil: {cn.organic_soil_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description};  Run date: {run_date}")
    main_logger.info(f"Veg zarr date: {veg_date};  SOC zarr date: {soc_date}")
    main_logger.info(f"Organic soil zarr: {cn.organic_soil_zarr_path}")
    main_logger.info(f"no_upload: {no_upload};  create_zarr: {create_zarr}")
    main_logger.info(f"SOC block 1 zarr idx: {SOC_BLOCK1_ZARR_IDX} (SOC_density_intervals[3]=2020, i.e. avg(2010-2015) vs. avg(2015-2020)")
    main_logger.info(f"Org soil block 1 idx: {ORG_SOIL_BLOCK1_ZARR_IDX};  block 2 idx: {ORG_SOIL_BLOCK2_ZARR_IDX}")

    # Verify org soil zarr year coordinate
    fs = fsspec.filesystem("s3", anon=False)
    org_zarr_check  = zarr.open_group(fs.get_mapper(cn.organic_soil_zarr_path), mode="r", use_consolidated=False)
    org_soil_years  = org_zarr_check["year"][:]
    main_logger.info(f"Org soil zarr year values: {org_soil_years.tolist()}")
    main_logger.info(f"  Block 1 (idx {ORG_SOIL_BLOCK1_ZARR_IDX}) = {org_soil_years[ORG_SOIL_BLOCK1_ZARR_IDX]}")
    main_logger.info(f"  Block 2 (idx {ORG_SOIL_BLOCK2_ZARR_IDX}) = {org_soil_years[ORG_SOIL_BLOCK2_ZARR_IDX]}")

    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)
    chunk_list, chunk_size_pixels = uu.create_chunk_list(
        bounding_box, chunk_shapefile_uri, chunk_size_deg or 1, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"1x1 deg chunks to cover: {len(chunk_list)}")

    # Build a dict mapping tile_id → frozenset of valid (w, s, e, n) bounds
    valid_chunks_by_tile = {}
    for c in chunk_list:
        tid = uu.xy_to_tile_id(c[0], c[3])
        valid_chunks_by_tile.setdefault(tid, set()).add(tuple(c))

    tile_ids = sorted(valid_chunks_by_tile.keys())

    main_logger.info(f"10x10 deg tiles to process (only chunks from chunklist): {len(tile_ids)}")

    # is_large_run = True  # For testing
    is_large_run = len(tile_ids) > 5
    if is_large_run:
        create_zarr = True
        main_logger.info("Large run: forcing create_zarr=True")

    # -----------------------------------------------------------------------
    # Step 2: Build source zarr paths
    # -----------------------------------------------------------------------
    veg_zarr_path = zu.create_zarr_path(cn.veg_outputs_path_zarr, cn.chunk_dims, veg_date, main_logger,
                                        cn.veg_model_version_underscore, model_type, veg_model_path_description)

    soc_zarr_path = zu.create_zarr_path(cn.SOC_path_zarr, cn.chunk_dims, soc_date, main_logger,
                                         cn.SOC_model_version_underscore, model_type, soc_model_path_description)

    for label, path in [("veg", veg_zarr_path), ("SOC", soc_zarr_path), ("org soil", cn.organic_soil_zarr_path)]:
        exists = fs.exists(path)
        main_logger.info(f"{label} zarr exists ({path}): {exists}")
        if not exists:
            import sys; sys.exit(f"{label} zarr not found: {path}")

    # -----------------------------------------------------------------------
    # Step 3: Build LULUCF output paths
    # -----------------------------------------------------------------------
    LULUCF_run = cn.LULUCF_full_version_underscore.replace("MODEL_TYPE", model_type)
    LULUCF_run = LULUCF_run.replace("MODEL_PATH_DESCRIPTION", model_path_description)

    LULUCF_base_path = (
        cn.LULUCF_outputs_path
        .replace(cn.model_version_type_description_placeholder, LULUCF_run)
    )

    # Keeps underscore, matching the 10x10 outputs
    def s3_dir_1x1(pattern, year_or_avg):
        if year_or_avg == 'avg':
            yr_str = f"{year_or_avg}_{cn.veg_outputs_years[0]}_{cn.veg_outputs_years[-1]}"
        else:
            yr_str = str(year_or_avg)
        return f"{LULUCF_base_path}{pattern}/annual_intervals/{yr_str}/{cn.flux_density_pixel_meaning}/{cn.chunk_dims}_pixels/{run_date}/"

    outputs_1x1_dir_by_year = {
        (pattern, year): s3_dir_1x1(pattern, year)
        for pattern in LULUCF_OUTPUTS_TO_ZARR
        for year in cn.veg_outputs_years
    }
    outputs_1x1_avg_dirs = {p: s3_dir_1x1(p, 'avg') for p in LULUCF_OUTPUTS_TO_ZARR}

    # output_base_10x10 uses placeholder tokens that create_10x10_deg_geotif_from_zarr substitutes internally
    output_base_10x10 = (
        f"{LULUCF_base_path}"
        f"PATTERN/annual_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{run_date}/"
    )

    main_logger.info(f"Sample 1x1 output annual dir (emis 2016): {outputs_1x1_dir_by_year[(cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern, 2016)]}")
    main_logger.info(f"Sample 1x1 output avg dir (emis):  {outputs_1x1_avg_dirs[cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern]}")
    main_logger.info(f"10x10 output base: {output_base_10x10}")


    # -----------------------------------------------------------------------
    # Step 4: Create LULUCF zarrs (separate for timeseries and annual average)
    # -----------------------------------------------------------------------
    LULUCF_annual_zarr_path     = None
    LULUCF_avg_zarr_path = None

    # Two separate zarrs: one for the timeseries and one for annual averages
    if create_zarr:
        zarr_base = (
            cn.LULUCF_outputs_path_zarr
            .replace(cn.model_version_type_description_placeholder, LULUCF_run)
            .replace("MODEL_INTERVAL_TYPE", "annual")
            .replace("CHUNK_SIZE_pixels",   f"{cn.chunk_dims}_pixels")
            .replace("RUN_DATE",            run_date)
        )
        LULUCF_annual_zarr_path     = f"{zarr_base}{cn.LULUCF_annual_zarr_name}"
        LULUCF_avg_zarr_path = f"{zarr_base}{cn.LULUCF_avg_zarr_name}"

        outputs_with_unit = [f"{p}{cn.flux_density_pixel_meaning}" for p in LULUCF_OUTPUTS_TO_ZARR]

        # Timeseries zarr: n_years=9, chunks=(9, 4000, 4000)
        zu.initialize_global_zarr(
            LULUCF_annual_zarr_path, outputs_with_unit, cn.veg_end_year_count,
            (cn.veg_end_year_count, cn.chunk_dims, cn.chunk_dims), main_logger)

        # Annual average zarr: n_years=1, chunks=(1, 4000, 4000)
        zu.initialize_global_zarr(
            LULUCF_avg_zarr_path, outputs_with_unit, 1,
            (1, cn.chunk_dims, cn.chunk_dims), main_logger)

        main_logger.info(f"Timeseries zarr: {LULUCF_annual_zarr_path}")
        main_logger.info(f"Annual avg zarr: {LULUCF_avg_zarr_path}")


    # -----------------------------------------------------------------------
    # Step 5: Process tiles in parallel, retiring surplus workers as the
    #         queue drains so idle workers don't run up costs.
    # -----------------------------------------------------------------------
    main_logger.info("Workers' logs to be appended after main function log\n")

    all_stats = []
    success_count = 0

    n_workers_start = len(client.scheduler_info()["workers"])

    futures = [
        client.submit(
            calculate_LULUCF_fluxes,
            tile_id, is_large_run, stage, no_upload, create_zarr,
            veg_zarr_path, soc_zarr_path,
            LULUCF_annual_zarr_path, LULUCF_avg_zarr_path,
            outputs_1x1_dir_by_year, outputs_1x1_avg_dirs,
            output_base_10x10, model_type, model_path_description,
            valid_chunks_by_tile[tile_id]
        )
        for tile_id in tile_ids
    ]

    n_total = len(futures)
    n_done = 0
    main_logger.info(f"Submitted {n_total} tiles to {n_workers_start} workers: {uu.timestr()}")

    for future in as_completed(futures):
        n_done += 1
        n_remaining = n_total - n_done

        result = future.result()
        if isinstance(result, tuple) and result[0].startswith("Success"):
            success_count += 1
            all_stats.extend(result[1])

        main_logger.info(f"Tile {n_done}/{n_total} done ({n_remaining} remaining): {uu.timestr()}")

        # Progressive retirement: once remaining work is less than the
        # starting worker count, there are guaranteed idle workers.
        # retire_workers() is graceful — Dask will not assign new tasks to
        # flagged workers; each drains its current task before shutting down,
        # ensuring logs are fully streamed to Coiled before the instance exits.
        # It only shuts down workers that are idle (not processing a task).
        # The >= 5 threshold avoids repeated API calls at the very tail end.
        # This could perhaps also be done with Coiled's cluster.adapt, which reduces workers as they're not needed.
        # But I didn't try that.
        if not run_local and n_remaining < n_workers_start:
            workers_info = client.scheduler_info()["workers"]
            current_count = len(workers_info)
            desired_count = max(1, n_remaining)
            surplus = current_count - desired_count
            if surplus >= 5:
                idle_ids = [
                    wid for wid, info in workers_info.items()
                    if not info.get("processing")
                ]
                to_retire = idle_ids[:surplus]
                if to_retire:
                    time.sleep(15)
                    client.retire_workers(to_retire, close_workers=True)
                    main_logger.info(
                        f"Retired {len(to_retire)} idle workers: {current_count} → {current_count - len(to_retire)} "
                        f"({n_remaining} tiles remaining)"
                    )

    del futures
    client.run(gc.collect)
    uu.stage_duration(start_time, uu.timestr(), stage, main_logger)

    # -----------------------------------------------------------------------
    # Steps 6-11: Logs, stats, resize, count, merge (same pattern as SOC script)
    # -----------------------------------------------------------------------
    if not run_local:
        worker_log_local_path_prelim = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with preliminary worker log", main_logger)

    if (not no_stats) and success_count > 0:
        uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with chunk stats", main_logger)

    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)

    if not run_local:
        workers = client.scheduler_info()["workers"]
        if len(workers) > 10:
            main_logger.info("Resizing cluster to 1 worker for counts/log")
            resize_cluster.resize_coiled_cluster(cluster_name, 1)

    # Counts output 1x1 and 10x10 deg geotifs
    if not no_upload and is_large_run:
        for pattern in LULUCF_OUTPUTS_TO_ZARR:
            for year in list(cn.veg_outputs_years) + ['avg']:
                folder_1x1 = outputs_1x1_dir_by_year.get((pattern, year)) or outputs_1x1_avg_dirs.get(pattern)
                if folder_1x1:
                    _, count = uu.list_raster_full_paths_in_s3_folder_and_count(folder_1x1)
                    main_logger.info(f"  1x1 outputs in {folder_1x1}: {count}")

                year_str = f"avg_{cn.veg_year_range_str}" if year == 'avg' else str(year)
                for units, dims in [
                    (cn.flux_density_pixel_meaning,   cn.full_raster_dims),
                    (cn.flux_per_pixel_pixel_meaning, cn.full_raster_dims),
                    (cn.flux_aggreg_pixel_meaning,    cn.global_aggregation_factor),
                ]:
                    folder_10x10 = (output_base_10x10
                                    .replace("PATTERN",          pattern)
                                    .replace("START_END",        year_str)
                                    .replace("PER_HA_OR_PIXEL",  units)
                                    .replace("CHUNK_SIZE_pixels", f"{dims}_pixels"))
                    _, count = uu.list_raster_full_paths_in_s3_folder_and_count(folder_10x10)
                    main_logger.info(f"  10x10 outputs in {folder_10x10}: {count}")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)

    if not run_local:
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create LULUCF-level 30m flux outputs.")

    parser.add_argument('-cn',   '--cluster_name')
    parser.add_argument('-rd',   '--run_date',                    help='Run date YYYYMMDD (default: today)')
    parser.add_argument('-bb',   '--bounding_box',  nargs=4, type=float, help='W S E N')
    parser.add_argument('-cs',   '--chunk_size_deg', type=float)
    parser.add_argument('-cshp', '--chunk_shapefile_uri')
    parser.add_argument('-f',    '--first_chunks',   type=int)
    parser.add_argument('-mt',   '--model_type',     default='standard')
    parser.add_argument('-mpd',  '--model_path_description')
    parser.add_argument('-ln',   '--log_note')

    parser.add_argument('--veg_date', required=True)
    parser.add_argument('--veg_mpd',  required=True)
    parser.add_argument('--soc_date', required=True)
    parser.add_argument('--soc_mpd',  required=True)

    parser.add_argument('--run_local',   action='store_true')
    parser.add_argument('--no_stats',    action='store_true')
    parser.add_argument('--no_log',      action='store_true')
    parser.add_argument('--no_upload',   action='store_true')
    parser.add_argument('--create_zarr', action='store_true')

    args = parser.parse_args()

    main(
        cluster_name               = args.cluster_name,
        model_type                 = args.model_type,
        veg_date                   = args.veg_date,
        veg_model_path_description = args.veg_mpd,
        soc_date                   = args.soc_date,
        soc_model_path_description = args.soc_mpd,
        run_local                  = args.run_local,
        no_stats                   = args.no_stats,
        no_log                     = args.no_log,
        no_upload                  = args.no_upload,
        create_zarr                = args.create_zarr,
        chunk_shapefile_uri        = args.chunk_shapefile_uri,
        bounding_box               = args.bounding_box,
        chunk_size_deg             = args.chunk_size_deg,
        first_chunks               = args.first_chunks,
        run_date                   = args.run_date,
        model_path_description     = args.model_path_description,
        log_note                   = args.log_note,
    )
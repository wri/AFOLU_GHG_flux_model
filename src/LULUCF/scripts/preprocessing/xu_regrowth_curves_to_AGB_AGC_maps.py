"""
Creates AGB density maps and AGC growth rate maps from the Xu et al. 2026 Chapman-Richards
regrowth curve parameter raster.
Rasters are from https://drive.google.com/drive/folders/1ANjxrL4LItXtteHIGFRU70_MfZiFVF9n?usp=sharing,
per email from Yidi on 2026-06-28.

Input:
  4-band GeoTIF at 1 degree resolution. Bands are the four fitted parameters:
    Band 1: AGBmax  (Mg AGB/ha)
    Band 2: b
    Band 3: c
    Band 4: d       (starting biomass; non-zero for degraded-forest disturbances)

  Formula: AGB(age) = AGBmax × (1 - exp(-b × age))^c + d

Outputs:
  AGB density maps (Mg AGB/ha) at ages 0, 5, 10, 15, 20, 40, 60, 80, 100 years.
  AGC rate maps (Mg AGC/ha/yr) over the intervals between those ages, converted from
  AGB using cn.biomass_to_carbon_non_mangrove. Thus, the rate maps are not simply (delta AGB)/(delta t)--
  you need to multiply by AGB:AGC.

Made with Claude session 'Gridded regrowth maps from Chapman-Richards curves'

Runs locally because it's processing a few 1x1 deg geotifs.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.LULUCF.scripts.preprocessing.xu_regrowth_curves_to_AGB_AGC_maps
"""

import os
from datetime import date
import numpy as np
import rasterio

from src.utilities import universal_utilities as uu
from src.utilities import constants_and_names as cn

# --- Paths (patterns defined in cn) ---
INPUT_S3   = f"{cn.Xu_regrowth_raw_dir}{cn.Xu_regrowth_raw_pattern}"
AGB_S3_DIR = cn.Xu_regrowth_AGB_rate_dir
AGC_S3_DIR = cn.Xu_regrowth_AGC_rate_global_all_ages_dir

LOCAL_TMP     = "/tmp/xu_regrowth/"
LOCAL_OUT_DIR = "/mnt/c/GIS/AFOLU_flux_model/LULUCF/Xu_et_al_2026_regrowth_rasters/from_Yidi_Xu_20260628/"
LOCAL_AGB_DIR = f"{LOCAL_OUT_DIR}AGB_density_by_age/"
LOCAL_AGC_DIR = f"{LOCAL_OUT_DIR}AGC_rate/global/"

RUN_DATE = cn.Xu_global_date

# Ages (years) at which to compute AGB density snapshots
AGES = [0, 5, 10, 15, 20, 40, 60, 80, 100]

# Consecutive pairs define the rate intervals
INTERVALS = list(zip(AGES[:-1], AGES[1:]))


def chapman_richards(agbmax, b, c, d, age):
    """AGB(age) = AGBmax × (1 - exp(-b × age))^c + d

    Clamps the base of the power to [0, 1] so that:
      - age=0 always yields d (base = 0)
      - no NaN arises from raising a negative base to a fractional exponent
    """
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        inner = np.clip(1.0 - np.exp(-b * age), 0.0, 1.0)
        return (agbmax * np.power(inner, c) + d).astype(np.float32)


def save_raster(array, path, profile):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)


def save_and_upload(array, fname, local_out_dir, s3_dir, profile):
    # Save to persistent local directory
    local_keep = f"{local_out_dir}{fname}"
    save_raster(array, local_keep, profile)
    print(f"  Saved  locally: {local_keep}")

    # Write a temp copy for S3 upload (upload_s3_file does not delete the file)
    tmp_path = f"{LOCAL_TMP}{fname}"
    save_raster(array, tmp_path, profile)
    print(f"  Uploading to:   {s3_dir}{fname}")
    uu.upload_s3_file(f"{s3_dir}{fname}", tmp_path)
    os.remove(tmp_path)


def main():
    os.makedirs(LOCAL_TMP, exist_ok=True)
    os.makedirs(LOCAL_AGB_DIR, exist_ok=True)
    os.makedirs(LOCAL_AGC_DIR, exist_ok=True)

    # Download parameter raster from S3
    local_input = f"{LOCAL_TMP}growthcurve_otherdeg.tif"
    print(f"Downloading {INPUT_S3} ...")
    uu.download_s3_file(INPUT_S3, local_input)
    print("Download complete.\n")

    # Read the four parameter bands; fill nodata → NaN so masking is uniform
    with rasterio.open(local_input) as src:
        base_profile = src.profile.copy()
        agbmax = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        b_par  = src.read(2, masked=True).filled(np.nan).astype(np.float32)
        c_par  = src.read(3, masked=True).filled(np.nan).astype(np.float32)
        d_par  = src.read(4, masked=True).filled(np.nan).astype(np.float32)

    # Pixels where any parameter is missing
    nodata_mask = np.isnan(agbmax) | np.isnan(b_par) | np.isnan(c_par) | np.isnan(d_par)

    # Single-band float32 output profile
    out_profile = base_profile.copy()
    out_profile.update(
        count=1,
        dtype='float32',
        compress='lzw',
        nodata=np.nan,
    )

    # --- Step 1: AGB density maps ---
    print("Computing and uploading AGB density maps ...")
    agb_maps = {}
    for age in AGES:
        agb = chapman_richards(agbmax, b_par, c_par, d_par, age)
        agb[nodata_mask] = np.nan
        agb_maps[age] = agb

        fname = f"{cn.Xu_regrowth_AGB_rate_pattern}__{age}_years_{RUN_DATE}.tif"
        save_and_upload(agb, fname, LOCAL_AGB_DIR, AGB_S3_DIR, out_profile)
        print(f"    age={age:3d} yr  |  non-NaN pixels: {np.sum(~np.isnan(agb)):,}")

    # --- Step 2: AGC rate maps ---
    print(f"\nComputing and uploading AGC rate maps for intervals {INTERVALS}")
    for (age_min, age_max) in INTERVALS:
        delta_agb   = agb_maps[age_max] - agb_maps[age_min]   # Mg AGB/ha over the interval
        rate_agb    = delta_agb / (age_max - age_min)          # Mg AGB/ha/yr
        rate_agc    = (rate_agb * cn.biomass_to_carbon_non_mangrove).astype(np.float32)
        rate_agc[nodata_mask] = np.nan

        fname = f"{cn.Xu_regrowth_AGC_rate_pattern}__{age_min}_{age_max}_years_{RUN_DATE}.tif"
        save_and_upload(rate_agc, fname, LOCAL_AGC_DIR, AGC_S3_DIR, out_profile)
        print(f"    {age_min:3d}–{age_max:3d} yr  |  mean rate: {np.nanmean(rate_agc):.3f} Mg AGC/ha/yr")

    # --- Diagnostic: print all inputs and outputs for one pixel ---
    row, col = np.argwhere(~nodata_mask)[0]
    print(f"\nSample pixel  row={row}, col={col}:")
    print(f"  Inputs:  AGBmax={agbmax[row,col]:.4f}  b={b_par[row,col]:.6f}  c={c_par[row,col]:.4f}  d={d_par[row,col]:.4f}")
    print(f"  AGB density (Mg AGB/ha):")
    for age in AGES:
        print(f"    age={age:3d} yr  ->  {agb_maps[age][row,col]:.4f}")
    print(f"  AGC rates (Mg AGC/ha/yr):")
    for (age_min, age_max) in INTERVALS:
        delta_agb = agb_maps[age_max][row,col] - agb_maps[age_min][row,col]
        rate_agc = delta_agb / (age_max - age_min) * cn.biomass_to_carbon_non_mangrove
        print(f"    {age_min:3d}–{age_max:3d} yr  ->  {rate_agc:.4f}")

    os.remove(local_input)
    print("\nDone.")


if __name__ == "__main__":
    main()

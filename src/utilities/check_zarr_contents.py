"""
Script to check contents of mega-zarr in a 1x1 deg chunk (to confirm it has data).
Adapted from zu.zarr_1x1_deg_stats

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.check_zarr_contents
"""

import sys
import fsspec
from dask.distributed import print
import numpy as np
import xarray as xr
import zarr
from bisect import bisect_left, bisect_right

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities.constants_and_names import intervals_annual

### Settings-- modify these

# For vegetation model outputs
# bounds = [23, -4, 24, -3]
bounds = [-111, 60, -110, 61]
zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/mega_zarr/annual_intervals/4000_pixels/20260130/vegetation_zarr.zarr'
var_name = 'gross_emissions__all_C_pools__all_gases__MgCO2e_ha_yr'
interval_end_years = cn.interval_end_years_annual

# # For carbon density timeseries
# bounds = [110, -1 ,111, 0]
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/mega_zarr/4000_pixels/20251224/SOC_zarr.zarr'
# var_name = 'SOC_density__full_extent__0-30cm_MgC'
# interval_end_years = cn.SOC_density_intervals

# # For carbon density change timeseries
# bounds = [10, 49, 11, 50]
# # bounds = [23, -4, 24, -3]
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__test_box/mega_zarr/4000_pixels/20260310/SOC_zarr.zarr'
# var_name = 'SOC_change__full_extent__0-30cm_MgC_ha_yr'
# interval_end_years = cn.SOC_change_intervals

# # For starting carbon density
# bounds = [114, -4, 115, -3]
# zarr_path = 's3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/mega_zarr/4000_pixels/20260121/starting_C_densities_zarr.zarr'
# var_name = 'carbon_density__BGC__landcover_masked__MgC'
# interval_end_years = [2015]

# # For starting composite primary forest
# bounds = [9, -1, 10, 0]
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/starting_composite_primary_forest/2015/zarr/4000_pixels/20260210/starting_composite_primary_forest.zarr'
# var_name = 'starting_composite_primary_forest'
# interval_end_years = [2015]

# # For AGC RF (added to vegetation zarr later)
# bounds = [10, 48, 11, 49]
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/mega_zarr/annual_intervals/4000_pixels/20260130/vegetation_zarr.zarr'
# var_name = 'removal_factor__AGC__MgC_ha_yr'
# interval_end_years = [2024]

# # For flox contextual layers
# bounds = [119, -6, 120, -5]  # For continent-ecozone: mix of 0, 4018 and 4020, with 4020 in upper right (00N_110E)
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/FAO_ecozone_continents/20260206_fillValue_removed/FAO_ecozone_continents_20260206.zarr'  # does not work

# bounds = [13, 48, 14, 49]  # For GADM: Three countries meet in Europe (50N_010E)
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/GADM4_1_adm0_global/20251209_fillValue_removed/global_GADM41_adm0_20251209.zarr'  # works

# bounds = [119, -3, 120, -2]  # For IFL/primary forest: Extensive primary forest, should have primary forest in lower-left and upper-right corners (00N_110E)
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/IFL2000_tropical_primary_forest_2001/20251209_fillValue_removed/ifl_primary_forest_merged_20251209.zarr' # works

# bounds = [13, 48, 14, 49]  # For pixel area (50N_010E)
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/pixel_area/20251209_fillValue_removed/global_pixel_area_20251209.zarr'  # works

# bounds = [-58, -16, -57, -15]  # For Brazil biomes: Three biomes meet (10S_060W)
# zarr_path = 's3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/BRA_biomes/20251229_fillValue_removed/BRA_biomes_20251229.zarr'  # works

###################################################

bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W

print(f"Getting stats for {bounds_str}: {uu.timestr()}")
fs = fsspec.filesystem("s3", anon=False)

### For zarrs with year dimension

zarr_stats_raw_all_years = []
# Bounding box to get stats for, reformatted for zarr extraction
target_box = {
    "lat_min": bounds[1],
    "lat_max": bounds[3],
    "lon_min": bounds[0],
    "lon_max": bounds[2]
}

# print(f"Getting indices for {bounds_str}")
lat0, lon0 = zu.latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
lat1, lon1 = zu.latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

print(f"Getting mapper for {bounds_str}")
zarr_mapper = fs.get_mapper(zarr_path)
print(f"Opening zarr for {bounds_str}")
zarr_group = zarr.open(zarr_mapper, mode="r")
print(zarr_group)
# print(f"Getting array for {bounds_str}")
zarr_chunk_array = zarr_group[var_name][:, lat0:lat1, lon0:lon1]
print("zarr_chunk_array:", zarr_chunk_array)

ds = xr.open_zarr(zarr_mapper, consolidated=False)
print(f"mega-zarr coords: {ds.coords}")
print(f"y range: {ds.y.values.min()}, {ds.y.values.max()}")
print(f"x range: {ds.x.values.min()}, {ds.x.values.max()}")
print(f"mega-zarr chunk size (years, y, x): {ds.chunksizes}")

# For [years]x4000x4000 zarr
for year_idx, year in enumerate(interval_end_years):
    zarr_chunk_array_year = zarr_chunk_array[year_idx]
    # print("zarr_chunk_array_year:", zarr_chunk_array_year)

    pattern_with_year = f"{var_name}_{year}"

    # print(f"Calculating stats for {bounds_str}")
    zarr_stats_raw_year = uu.calculate_stats(zarr_chunk_array_year, pattern_with_year, bounds_str, tile_id,'zarr_stats')
    # print(zarr_stats_raw_year)

    zarr_stats_raw_all_years.append(zarr_stats_raw_year)

    for key, value in zarr_stats_raw_year.items():
        print(f"{key}: {value}")
    print("\n")


###################################################

# ### For zarrs with no year dimension (no time component; may not be global, so spatial indexes are different)
# ### From https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6986043f-c8b0-832c-837f-7329873aa948
#
# # For bounds = [lon_min, lat_min, lon_max, lat_max]
# def get_index_range(coords, min_val, max_val, descending=False):
#     if descending:
#         coords = coords[::-1]
#         i0 = bisect_left(coords, max_val)
#         i1 = bisect_right(coords, min_val)
#         return len(coords) - i1, len(coords) - i0
#     else:
#         i0 = bisect_left(coords, min_val)
#         i1 = bisect_right(coords, max_val)
#         return i0, i1
#
# # zarrs without time dimension have the variable name band_data
# var_name = 'band_data'
#
# # print(f"Getting array for {bounds_str}")
# zarr_mapper_band_data = fs.get_mapper(f"{zarr_path}/{var_name}")
# zarr_array = zarr.open_array(zarr_mapper_band_data, mode="r")
#
# # Step 1: Points to the Zarr group
# fs = fsspec.filesystem("s3", anon=False)
# zarr_store = fs.get_mapper(zarr_path)
# zarr_group = zarr.open_group(zarr_store, mode="r")
#
# # Step 2: Lists keys
# print("Zarr keys:", list(zarr_group.array_keys()))
#
# # Step 3: Tries reading coordinate arrays
# if "y" in zarr_group:
#     y_coords = zarr_group["y"][:]
#     print("y shape:", y_coords.shape)
# else:
#     sys.exit("No y coordinate array")
#
# if "x" in zarr_group:
#     x_coords = zarr_group["x"][:]
#     print("x shape:", x_coords.shape)
# else:
#     sys.exit("No x coordinate array")
#
# lat_vals = y_coords  # usually descending
# lon_vals = x_coords  # usually ascending
#
# lat0, lat1 = get_index_range(lat_vals, bounds[3], bounds[1], descending=True)
# lon0, lon1 = get_index_range(lon_vals, bounds[0], bounds[2])
#
# zarr_chunk_array = zarr_array[lat0:lat1, lon0:lon1]
# print("zarr_chunk_array:", zarr_chunk_array)
#
# print("min:", float(np.nanmin(zarr_chunk_array)))
# print("mean:", float(np.nanmean(zarr_chunk_array)))
# print("max:", float(np.nanmax(zarr_chunk_array)))
# print("count:", np.count_nonzero(zarr_chunk_array))





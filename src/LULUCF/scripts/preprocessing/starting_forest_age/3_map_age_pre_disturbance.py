"""
Script to convert Simon Besnard's age pre-disturbance 1deg zarr into a global geotif.
Finished with assistance from ChatGPT: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68669be7-8698-800a-a2e9-f1bd7c83c037
Simon sent the zarr on 2025-07-02/2025-07-03.
It is currently accepted at Nature Ecology and Evolution.
Simon's explanation of this map (email 2025-07-03):
"So stand_replaced_diff is the age difference between 2020 and 2010 for forests disturbed between 2010 and 2020.
Initially, it is negative because I do age 2020 - age 2010. One can have positive value if very young forests in 2010 (<10 years)
where disturbed between 2010 and 2020. It is not directly age-pre disturbance but a very good approximation."

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/src/LULUCF/scripts/preprocessing/starting_forest_age
python _map_age_pre_disturbance.py

Input and output are local. Doesn't use s3.
Inputs and outputs in s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/forest_age/age_pre_disturbance_Besnard_et_al/

This only needs to be run once, unless Simon updates the age pre-disturbance map.

"""
import xarray as xr
import os
import numpy as np

# Access the data directory
data_dir = '/mnt/c/GIS/AFOLU_flux_model/forest_age'

## From Simon
# The zarr file that needs to be access is 'AgeDiff_1deg'
stand_replaced_diff =  xr.open_zarr(os.path.join(data_dir, 'AgeDiff_1deg')).stand_replaced_diff.median(dim = 'members')
stand_replaced_diff = np.abs(stand_replaced_diff.where(stand_replaced_diff < 0))
print(stand_replaced_diff)

## From ChatGPT
# Set CRS and spatial metadata
stand_replaced_diff.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
stand_replaced_diff.rio.write_crs("EPSG:4326", inplace=True)

# Define output path
output_tif = os.path.join(data_dir, "age_pre_disturbance_median_1deg_global__20250703.tif")

# Export to GeoTIFF
stand_replaced_diff.rio.to_raster(output_tif)

print(f"GeoTIFF saved to {output_tif}")

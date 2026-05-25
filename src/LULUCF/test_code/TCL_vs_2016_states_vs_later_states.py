"""
For a given chunk, produces counts of the state nodes and AGC emissions for each year, categorized by whether pre-2015 TCL occurred
and whether Potapov loss in 2016 occurred.
The goal is to see how much loss and emissions occur in pixels that had pre-model TCL and/or first model year loss.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/src/LULUCF/test_code
python TCL_vs_2016_states_vs_later_states.py

From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6973beb7-e4b0-8326-9e37-5408f6871889
"""

import rasterio
import numpy as np
import os


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)


def starts_with_3(arr):
    """Returns a boolean mask where values start with digit '3' (e.g., 3xxxxxxx)."""
    return np.floor_divide(arr, 10_000_000) == 3


def count_values_with_masks(year_data, mask):
    masked_data = year_data[mask]
    values, counts = np.unique(masked_data, return_counts=True)
    return dict(zip(values, counts))


if __name__ == "__main__":

    # Path setup
    base_path = "/mnt/c/GIS/AFOLU_flux_model/test_data/output/v1_0_5/v12_land_state_nodes"
    emissions_path_base = "/mnt/c/GIS/AFOLU_flux_model/test_data/output/v1_0_5/v12_land_state_nodes"
    area_path = "/mnt/c/GIS/AFOLU_flux_model/test_data/00N_020E_inputs/hanson_2013_area_00N_020E_23_-4_24_-3_extent.tif"
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # Read 2016 raster and tree cover loss raster
    print("Loading 2016 and TCL masks...")
    land_2016_path = os.path.join(base_path, "00N_020E__23_-4_24_-3__land_state_node_2016__20260122_23_44.tif")
    land_2016 = read_raster(land_2016_path)
    mask_2016 = starts_with_3(land_2016)

    tree_loss_path = "/mnt/c/GIS/AFOLU_flux_model/test_data/00N_020E_inputs/GFW2024_00N_020E__23_-4_24_-3_extent.tif"
    tree_loss = read_raster(tree_loss_path)
    mask_tcl = tree_loss < 15

    # Load area raster (m²)
    area_m2 = read_raster(area_path)

    # Define 4 combination masks
    print("Creating 2x2 masks...")
    mask_tt = np.logical_and(mask_2016, mask_tcl)         # 2016=3 AND TCL<15
    mask_tf = np.logical_and(mask_2016, ~mask_tcl)        # 2016=3 AND TCL>=15
    mask_ft = np.logical_and(~mask_2016, mask_tcl)        # 2016≠3 AND TCL<15
    mask_ff = np.logical_and(~mask_2016, ~mask_tcl)       # 2016≠3 AND TCL>=15

    masks = {
        "2016 loss & prior TCL": mask_tt,
        "2016 loss & no prior TCL": mask_tf,
        "2016 not loss & prior TCL": mask_ft,
        "2016 not los & no prior TCL": mask_ff
    }

    # Process each year and count pixels by 2x2 mask
    for year in years:
        # print(f"\n--- Year {year} ---")
        land_path = os.path.join(
            base_path, f"00N_020E__23_-4_24_-3__land_state_node_{year}__20260122_23_44.tif"
        )
        year_data = read_raster(land_path)


        # Read emissions raster for this year (Mg CO2 / ha)
        emissions_path = os.path.join(
            emissions_path_base,
            f"00N_020E__23_-4_24_-3__gross_emissions__AGC__MgCO2_ha_yr_{year}.tif"
        )
        emissions_per_ha = read_raster(emissions_path)

        # Convert to Mg CO2 per pixel
        emissions_per_pixel = emissions_per_ha * (area_m2 / 10_000.0)

        for label, mask in masks.items():

            # Apply mask to land state and emissions
            masked_land = year_data[mask]
            masked_emissions = emissions_per_pixel[mask]

            # Unique pixel values and total emissions per pixel value
            unique_values = np.unique(masked_land)

            for val in unique_values:
                val_mask = masked_land == val
                count = np.sum(val_mask)
                emissions_sum = np.sum(masked_emissions[val_mask])
                print(f"  Year: {year}, Mask: {label}, Pixel Value: {val}, Count: {count}, Emissions MgCO2: {emissions_sum:.2f}")

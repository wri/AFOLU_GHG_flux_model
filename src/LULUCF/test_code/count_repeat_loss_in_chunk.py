"""
For a given chunk, this counts how many pixels had their second, third, or fourth loss event in a given year.
The goal is to track how much repeat loss is occurring.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/src/LULUCF/test_code
python count_repeat_loss_in_chunk.py

From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6973beb7-e4b0-8326-9e37-5408f6871889
"""


import rasterio
import numpy as np
import os

def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1)

def starts_with_3(arr):
    """Returns a boolean mask where values start with digit '3'."""
    return np.floor_divide(arr, 10_000_000) == 3

if __name__ == "__main__":

    # Years and paths
    base_path = "/mnt/c/GIS/AFOLU_flux_model/test_data/output/v1_0_5/v12_land_state_nodes"
    years = list(range(2016, 2025))  # 2016 to 2024 inclusive

    # Initialize loss count per pixel
    first_raster_path = os.path.join(
        # base_path, f"00N_020E__23_-4_24_-3__land_state_node_2016__20260122_23_44.tif"  # DRC
        base_path, f"20S_060W__-51_-25_-50_-24__land_state_node_2016__20260124.tif"   # BRA
    )
    raster_shape = read_raster(first_raster_path).shape
    loss_counter = np.zeros(raster_shape, dtype=np.uint8)

    print("Year\t1st Loss\t2nd Loss\t3rd Loss\t4th Loss")
    for year in years:
        raster_path = os.path.join(
            base_path,
            # f"00N_020E__23_-4_24_-3__land_state_node_{year}__20260122_23_44.tif"   # DRC
            f"20S_060W__-51_-25_-50_-24__land_state_node_{year}__20260124.tif"   # BRA
        )
        data = read_raster(raster_path)

        # Identify pixels with loss this year
        loss_this_year = starts_with_3(data)

        # Before updating: get how many times pixel had loss before this year
        previous_losses = loss_counter[loss_this_year]

        # Determine how many are experiencing N-th loss *this year*
        first_loss  = np.sum(previous_losses == 0)
        second_loss = np.sum(previous_losses == 1)
        third_loss  = np.sum(previous_losses == 2)
        fourth_loss = np.sum(previous_losses == 3)

        # Update loss counter
        loss_counter[loss_this_year] += 1

        print(f"{year}\t{first_loss}\t\t{second_loss}\t\t{third_loss}\t\t{fourth_loss}")


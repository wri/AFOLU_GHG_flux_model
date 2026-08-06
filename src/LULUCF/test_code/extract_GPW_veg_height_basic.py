"""
Basic script for downloading a portion of a global SOC COG, mostly for checking outputs.
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model
python src/LULUCF/test_code/extract_GPW_veg_height_basic.py

Does not use Dask or Coiled.
Output is in kg C/m^3, multipled by 10 (to keep geotif in int instead of float)

ChatGPT: https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6877a34b-02cc-800a-88cc-a123cdc9ed1b.
Main issue was getting the bounding box to be exactly 1x1 deg and 4000x4000 pixels; it sometimes had an extra pixel
due to centroid issues.
"""

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# Desired tile extent (1x1 deg)
lon_min = 124
lat_max = -29
pixel_size = 0.00025
tile_px = int(1.0 / pixel_size)  # 4000

for year in range(2015, 2025, 1):

    print(f"Extracting {year}")

    url = f"https://s3.opengeohub.org/gpw/arco/gpw_short.veg.height_egbt_m_30m_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"

    with rasterio.Env(AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(url) as src:
            # Compute col/row in source raster for exact tile corner (upper-left pixel edge)
            col_start = round((lon_min - src.bounds.left) / pixel_size)
            row_start = round((src.bounds.top - lat_max) / pixel_size)

            window = Window(col_start, row_start, tile_px, tile_px)
            data = src.read(1, window=window)
            data_rescale = data/10

            # Force transform to exact corner alignment
            transform = Affine.translation(lon_min, lat_max) * Affine.scale(pixel_size, -pixel_size)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": tile_px,
                "width": tile_px,
                "transform": transform,
                "dtype": 'float32'
            })

            with rasterio.open(f"20S_120E__124_-30_125_-29__GPW_veg_height_m_{year}.tif", "w", **out_meta) as dest:
                dest.write(data_rescale, 1)
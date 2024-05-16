import os
import logging
import rasterio
from rasterio.features import rasterize
from rasterio import features  # Ensure 'features' is imported correctly
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import boto3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_length/'

# Local paths for processed output
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\grip"
grip_raw = r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region1_vector_shp\GRIP4_region1.shp"


def get_tile_metadata(tile_path):
    with rasterio.open(tile_path) as src:
        pixel_size = src.transform[0]  # Extract the pixel size from the transform (size of pixel in coordinate units)
        transform = src.transform  # Use the existing transform directly
        width = src.width
        height = src.height
        return src.crs, transform, width, height, src.bounds


def process_vector_data_to_raster(vector_data_path, tile_bounds, output_raster_path, transform, width, height):
    tile_box = box(*tile_bounds)
    vector_data = gpd.read_file(vector_data_path, bbox=tile_box)

    logging.info(f"Loaded {len(vector_data)} vector features for processing.")

    if vector_data.empty:
        logging.info("No vector data found within the tile bounds. Skipping rasterization.")
        return

    # Optional: Check and log road types if applicable
    # in global version, the field is called RoadType
    if 'GP_RTP' in vector_data.columns:
        road_types = vector_data['GP_RTP'].value_counts()
        logging.info(f"Road types present: {road_types.to_dict()}")

    vector_data['length_km'] = vector_data.geometry.to_crs(epsg=6933).length / 1000  # Convert lengths to kilometers

    # Rasterize the lengths
    shapes = ((geom, value) for geom, value in zip(vector_data.geometry, vector_data['length_km']))
    rasterized_data = features.rasterize(shapes, out_shape=(height, width), fill=0, transform=transform,
                                         all_touched=True)

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform
    }
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(rasterized_data, 1)

    logging.info(f"Processed and saved {output_raster_path}")


os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
tiles_list = ["30N_090W_peat_mask_processed.tif"]  # Adjust as necessary

for tile_key in tiles_list:
    tile_path = f"s3://{s3_bucket_name}/{s3_tiles_prefix}/{tile_key}"  # Full path for reading directly from S3
    local_output_path = os.path.join(output_dir, tile_key.replace('_peat_mask_processed.tif',
                                                                  '_grip_length.tif'))  # New output path
    crs, transform, width, height, bounds = get_tile_metadata(tile_path)
    process_vector_data_to_raster(grip_raw, bounds, local_output_path, transform, width, height)

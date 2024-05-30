"""
This script processes GRIP (Global Roads Inventory Project) roads by tiles using a pre-existing raster template from S3.
It reads the raster tiles from S3, resamples them to a target resolution, creates a fishnet grid, reads corresponding
roads shapefiles for each tile, assigns road lengths to the fishnet cells, converts the lengths to road density, and
saves the results as raster files. This will have to be re-run with the updated organic soils extent

The script uses Dask to parallelize the processing of multiple tiles.

Functions:
- get_raster_bounds: Reads and returns the bounds of a raster file.
- resample_raster: Resamples a raster to a target resolution.
- mask_raster: Masks raster data to highlight specific values.
- create_fishnet_from_raster: Creates a fishnet grid from raster data.
- reproject_gdf: Reprojects a GeoDataFrame to a specified EPSG code.
- read_tiled_roads: Reads and reprojects roads shapefile for a given tile.
- assign_road_segments_to_cells: Assigns road segments to fishnet cells and calculates lengths.
- convert_length_to_density: Converts road lengths to road density.
- fishnet_to_raster: Converts a fishnet GeoDataFrame to a raster and saves it.
- process_tile: Processes a single tile (Dask delayed function).
- process_all_tiles: Processes all tiles using Dask for parallelization.
- main: Main function to orchestrate the processing based on provided arguments.

Usage examples:
- Process a specific tile (00N_110E):
  python script.py --tile_id 00N_110E

- Process all tiles:
  python script.py

Dependencies:
- geopandas
- pandas
- shapely
- numpy
- rasterio
- boto3
- logging
- os
- fiona
- dask
- dask.distributed
- dask.diagnostics

Note: The script relies on the presence of AWS credentials for accessing S3 buckets.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import boto3
import logging
import os
import fiona
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_base_dir = 'climate/AFOLU_flux_model/organic_soils'
roads_tiles_directory = r"C:\GIS\Data\Global\GRIP\roads_by_tile"

# Local paths
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\grip_density"
os.makedirs(output_dir, exist_ok=True)

logging.info("Directories and paths set up")

def get_raster_bounds(raster_path):
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds

def resample_raster(src, target_resolution_m):
    logging.info(f"Resampling raster to {target_resolution_m} meter resolution (1 km by 1 km)")
    # Calculate new width and height based on the target resolution in meters
    transform = src.transform
    target_resolution_deg = target_resolution_m / 111320  # Approximate conversion factor

    # Calculate new width and height
    width = int((src.bounds.right - src.bounds.left) / target_resolution_deg)
    height = int((src.bounds.top - src.bounds.bottom) / target_resolution_deg)

    new_transform = rasterio.transform.from_bounds(
        src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, width, height)

    profile = src.profile
    profile.update(transform=new_transform, width=width, height=height)

    data = src.read(
        out_shape=(src.count, height, width),
        resampling=Resampling.nearest
    )

    return data, profile

def mask_raster(data, profile):
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    profile.update(dtype=rasterio.uint8)
    return mask.astype(rasterio.uint8), profile

def create_fishnet_from_raster(data, transform):
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
                x, y = transform * (col, row)
                polygons.append(box(x, y, x + transform[0], y + transform[4]))

    fishnet_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def reproject_gdf(gdf, epsg):
    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)

def read_tiled_roads(tile_id, roads_tile_dir, epsg):
    logging.info(f"Reading tiled roads shapefile for tile ID: {tile_id}")
    try:
        tile_id = '_'.join(tile_id.split('_')[:2])  # e.g., '00N_110E'
        file_path = os.path.join(roads_tile_dir, f"roads_{tile_id}.shp")
        if os.path.exists(file_path):
            with fiona.open(file_path) as roads:
                roads_gdf = gpd.GeoDataFrame.from_features(roads, crs=roads.crs)
            logging.info(f"Read {len(roads_gdf)} road features for tile {tile_id}")
            roads_gdf = reproject_gdf(roads_gdf, epsg)
            return roads_gdf
        else:
            logging.warning(f"No shapefile found for tile {tile_id}")
            return gpd.GeoDataFrame(columns=['geometry'])
    except fiona.errors.DriverError as e:
        logging.error(f"Error reading roads shapefile: {e}")
        return gpd.GeoDataFrame(columns=['geometry'])

def assign_road_segments_to_cells(fishnet_gdf, roads_gdf):
    logging.info("Assigning road segments to fishnet cells and calculating lengths")
    road_lengths = []

    for idx, cell in fishnet_gdf.iterrows():
        roads_in_cell = gpd.clip(roads_gdf, cell.geometry)
        total_length = roads_in_cell.geometry.length.sum()
        road_lengths.append(total_length)

    fishnet_gdf['length'] = road_lengths
    logging.info(f"Fishnet with road lengths: {fishnet_gdf.head()}")
    return fishnet_gdf

def convert_length_to_density(fishnet_gdf, crs):
    logging.info("Converting length to density (km/km2)")
    if crs.axis_info[0].unit_name == 'metre':
        fishnet_gdf['density'] = fishnet_gdf['length'] / (1 * 1)  # lengths are in meters, cell area in km2
    elif crs.axis_info[0].unit_name == 'kilometre':
        fishnet_gdf['density'] = fishnet_gdf['length']  # lengths are already in km, cell area in km2
    else:
        raise ValueError("Unsupported CRS units")
    return fishnet_gdf

def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    transform = profile['transform']
    out_shape = (profile['height'], profile['width'])
    fishnet_gdf = fishnet_gdf.to_crs(profile['crs'])

    if fishnet_gdf.empty:
        logging.info(f"No valid geometries found for {output_raster_path}. Skipping rasterization.")
        return

    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry, fishnet_gdf['density'])],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.float32
    )

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(rasterized, 1)

    logging.info("Fishnet converted to raster and saved")

@dask.delayed
def process_tile(tile_key):
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"grip_density_{tile_id}.tif")

    if os.path.exists(local_output_path):
        logging.info(f"{local_output_path} already exists. Skipping processing.")
        return

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'

    try:
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_path) as src:
                # Define target resolution (1km in meters)
                target_resolution = 1000  # 1 km resolution in meters

                # Resample the template raster to the same resolution and alignment
                resampled_data, resampled_profile = resample_raster(src, target_resolution)

                # Mask the resampled raster
                masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)

                # Create the fishnet grid from the masked raster
                fishnet_gdf = create_fishnet_from_raster(masked_data, resampled_profile['transform'])

                # Reproject the fishnet to Albers Equal Area Conic (EPSG:5070)
                fishnet_gdf = reproject_gdf(fishnet_gdf, 5070)

                # Read the roads shapefile for the given tile ID and reproject to EPSG:5070
                roads_gdf = read_tiled_roads(tile_id, roads_tiles_directory, 5070)

                # Assign road segments to cells and calculate lengths
                fishnet_with_lengths = assign_road_segments_to_cells(fishnet_gdf, roads_gdf)

                # Convert length to density
                fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

                # Convert the fishnet to a raster and save it locally
                fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

                logging.info(f"Saved {local_output_path}")
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles():
    # Get the list of rasters from S3
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_tiles_prefix)

    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith('_peat_mask_processed.tif'):
                    tile_keys.append(tile_key)

    # Use Dask to parallelize the processing of tiles
    dask_tiles = [process_tile(tile_key) for tile_key in tile_keys]
    with ProgressBar():
        dask.compute(*dask_tiles)

def main(tile_id=None):
    # Initialize Dask client
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        if tile_id:
            tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
            process_tile(tile_key).compute()
        else:
            process_all_tiles()
    finally:
        # Close Dask client
        client.close()

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    # main(tile_id='00N_110E')

    # To process all tiles, comment out the above line and uncomment the line below
    main()

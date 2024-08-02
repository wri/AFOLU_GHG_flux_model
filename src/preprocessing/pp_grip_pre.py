import os
import geopandas as gpd
import logging
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd

import pp_utilities as uu
import constants_and_names as cn

"""
This script processes GRIP (Global Roads Inventory Project) roads by tiles using a pre-existing tile index shapefile.
It reads the tile index shapefile, identifies the regions each tile overlaps with, processes the corresponding
roads shapefiles, clips the roads data to the tile boundaries, and saves the results as shapefiles for each tile.

The script uses Dask to parallelize the processing of multiple tiles.

Functions:
- read_tiles_shapefile: Reads and returns the tile index shapefile from S3.
- download_regional_shapefiles: Downloads regional shapefiles from S3 to a local directory.
- process_tile: Processes a single tile by reading and clipping road data within the tile's bounds.
- process_all_tiles: Processes all tiles using Dask for parallelization.
- upload_to_s3: Compresses and uploads processed tiles to S3.
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
- dask
- dask.distributed
- dask.diagnostics

Note: The script assumes the presence of the required shapefiles and directories.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs(cn.output_dir, exist_ok=True)

def read_tiles_shapefile():
    logging.info("Downloading tiles shapefile from S3 to local directory")
    uu.read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)
    shapefile_path = os.path.join(cn.local_temp_dir, 'Global_Peatlands.shp')
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

def download_regional_shapefiles():
    downloaded_shapefiles = []
    for s3_path in cn.s3_regional_shapefiles:
        local_shapefile_path = os.path.join(cn.local_temp_dir, os.path.splitext(os.path.basename(s3_path))[0])

        # Check if the shapefile already exists locally
        if all(os.path.exists(local_shapefile_path + ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            logging.info(f"Shapefile {local_shapefile_path} already exists locally. Skipping download.")
            downloaded_shapefiles.append(local_shapefile_path + '.shp')
            continue

        logging.info(f"Downloading regional shapefile {s3_path}")
        uu.read_shapefile_from_s3(s3_path, cn.local_temp_dir, cn.s3_bucket_name)
        downloaded_shapefiles.append(local_shapefile_path + '.shp')
        logging.info(f"Downloaded and saved regional shapefile to {local_shapefile_path + '.shp'}")

    return downloaded_shapefiles

@dask.delayed
def process_tile(tile, regional_shapefiles):
    tile_id = tile['tile_id']  # Assuming the tile ID is stored in a column named 'tile_id'
    output_path = os.path.join(cn.output_dir, f"roads_{tile_id}.shp")

    if os.path.exists(output_path):
        logging.info(f"Output file {output_path} already exists. Skipping tile {tile_id}.")
        return

    tile_bounds = tile.geometry.bounds
    logging.info(f"Processing tile {tile_id} with bounds {tile_bounds}")

    combined_roads = []

    try:
        for region_shapefile in regional_shapefiles:
            logging.info(f"Reading roads shapefile {region_shapefile} within bounds {tile_bounds}")
            roads_gdf = gpd.read_file(region_shapefile, bbox=tile_bounds)
            logging.info(f"Number of roads read within bounds from {region_shapefile} for tile {tile_id}: {len(roads_gdf)}")

            if not roads_gdf.empty:
                roads_in_tile = gpd.clip(roads_gdf, tile.geometry)
                logging.info(f"Number of roads after clipping from {region_shapefile} for tile {tile_id}: {len(roads_in_tile)}")
                combined_roads.append(roads_in_tile)

        if not combined_roads:
            logging.info(f"No roads found for tile {tile_id}. Skipping.")
            return

        combined_roads_gdf = pd.concat(combined_roads, ignore_index=True)
        combined_roads_gdf.to_file(output_path)
        logging.info(f"Saved combined roads for tile {tile_id} to {output_path}")
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(tiles_gdf):
    regional_shapefiles = download_regional_shapefiles()
    tasks = [process_tile(tile, regional_shapefiles) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        dask.compute(*tasks)

def upload_to_s3():
    logging.info(f"Starting upload of processed tiles to S3: {cn.s3_output_prefix}")
    uu.compress_and_upload_directory_to_s3(cn.output_dir, cn.s3_bucket_name, cn.s3_output_prefix)
    logging.info(f"Finished uploading processed tiles to S3: {cn.s3_output_prefix}")

def main(tile_id=None):
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        regional_shapefiles = download_regional_shapefiles()
        process_tile(tile, regional_shapefiles).compute()
    else:
        process_all_tiles(tiles_gdf)
    upload_to_s3()

if __name__ == '__main__':
    # Initialize Dask client
    cluster = LocalCluster()
    client = Client(cluster)
    logging.info("Dask client initialized")

    try:
        # Example usage
        # Replace 'test_tile_id' with the actual tile ID you want to test
        main(tile_id='00N_110E')

        # To process all tiles, comment out the above line and uncomment the line below
        # main()
    finally:
        # Close Dask client
        client.close()
        logging.info("Dask client closed")

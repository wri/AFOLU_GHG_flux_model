import os
import geopandas as gpd
import logging
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd
import argparse
import sys

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
- upload_final_output_to_s3: Uploads the final output file to S3.
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

os.makedirs(cn.datasets['grip']['roads']['local_processed'], exist_ok=True)

def read_tiles_shapefile():
    """
    Reads the tile index shapefile from S3 and returns it as a GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the tiles.
    """
    logging.info("Downloading tiles shapefile from S3 to local directory")
    uu.read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)
    shapefile_path = os.path.join(cn.local_temp_dir, 'Global_Peatlands.shp')
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

def download_regional_shapefiles():
    """
    Downloads regional shapefiles from S3 to the local directory if they do not already exist.

    Returns:
        list: List of local paths to the downloaded shapefiles.
    """
    downloaded_shapefiles = []
    for s3_path in cn.grip_regional_shapefiles:
        local_shapefile_path = os.path.join(cn.local_temp_dir, os.path.splitext(os.path.basename(s3_path))[0])

        # Check if the shapefile already exists locally
        if all(os.path.exists(local_shapefile_path + ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            logging.info(f"Shapefile {local_shapefile_path} already exists locally. Skipping download.")
            downloaded_shapefiles.append(local_shapefile_path + '.shp')
            continue

        logging.info(f"Downloading regional shapefile {s3_path}")
        s3_file_path = f"/vsis3/{cn.s3_bucket_name}/{s3_path}"
        try:
            gdf = gpd.read_file(s3_file_path)
            gdf.to_file(f"{local_shapefile_path}.shp")
            downloaded_shapefiles.append(f"{local_shapefile_path}.shp")
            logging.info(f"Downloaded and saved regional shapefile to {local_shapefile_path}.shp")
        except Exception as e:
            logging.error(f"Error downloading shapefile from S3: {e}")
            continue

    return downloaded_shapefiles

@dask.delayed
def process_tile(tile, regional_shapefiles):
    """
    Processes a single tile by reading and clipping road data within the tile's bounds.

    Args:
        tile (gpd.GeoSeries): GeoSeries containing the tile data.
        regional_shapefiles (list): List of local paths to the regional shapefiles.

    Returns:
        str: Path to the processed shapefile, or None if no processing was done.
    """
    tile_id = tile['tile_id']  # Assuming the tile ID is stored in a column named 'tile_id'
    output_path = os.path.join(cn.datasets['grip']['roads']['local_processed'], f"roads_{tile_id}.shp")

    if os.path.exists(output_path):
        logging.info(f"Output file {output_path} already exists. Skipping tile {tile_id}.")
        return None

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
            return None

        combined_roads_gdf = pd.concat(combined_roads, ignore_index=True)
        combined_roads_gdf.to_file(output_path)
        logging.info(f"Saved combined roads for tile {tile_id} to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")
        return None

def process_all_tiles(tiles_gdf):
    """
    Processes all tiles using Dask for parallelization.

    Args:
        tiles_gdf (gpd.GeoDataFrame): GeoDataFrame containing the tiles.

    Returns:
        list: List of paths to the processed shapefiles.
    """
    regional_shapefiles = download_regional_shapefiles()
    tasks = [process_tile(tile, regional_shapefiles) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        results = dask.compute(*tasks)
    return results

def upload_final_output_to_s3(output_path):
    """
    Uploads the final output shapefile to S3.

    Args:
        output_path (str): Path to the local shapefile to upload.
    """
    if output_path is None:
        return

    local_file_path = output_path
    tile_id = os.path.basename(local_file_path).replace("roads_", "").replace(".shp", "")
    s3_file_path = os.path.join(cn.datasets['grip']['roads']['s3_processed'], f"roads_{tile_id}.shp")

    if os.path.exists(local_file_path):
        uu.upload_file_to_s3(local_file_path, cn.s3_bucket_name, s3_file_path)
        logging.info(f"Uploaded {local_file_path} to s3://{cn.s3_bucket_name}/{s3_file_path}")
        os.remove(local_file_path)  # Remove local file after upload
        logging.info(f"Removed local file {local_file_path}")
    else:
        logging.warning(f"Local file {local_file_path} does not exist. Skipping upload.")

def main(tile_id=None):
    """
    Main function to orchestrate the processing based on the provided arguments.

    Args:
        tile_id (str, optional): ID of the tile to process. If None, all tiles will be processed.
    """
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        regional_shapefiles = download_regional_shapefiles()
        output_path = process_tile(tile, regional_shapefiles).compute()
        upload_final_output_to_s3(output_path)
    else:
        results = process_all_tiles(tiles_gdf)
        for output_path in results:
            upload_final_output_to_s3(output_path)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process GRIP roads by tiles.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local', help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    if not any(sys.argv[1:]):  # Check if there are no command-line arguments
        # Direct execution examples for PyCharm
        # Example usage for processing a specific tile with the local Dask client
        tile_id = '00N_110E'  # Replace with your desired tile ID
        client_type = 'local'  # Replace with 'coiled' if you want to use the Coiled cluster

        # Initialize Dask client based on the direct execution setup
        if client_type == 'coiled':
            client, cluster = uu.setup_coiled_cluster()
        else:
            cluster = LocalCluster()
            client = Client(cluster)

        logging.info(f"Dask client initialized with {client_type} cluster")

        try:
            main(tile_id=tile_id)
        finally:
            client.close()
            logging.info("Dask client closed")
            if client_type == 'coiled':
                cluster.close()
                logging.info("Coiled cluster closed")

        # Example usage for processing all tiles with the local Dask client
        client_type = 'local'  # Replace with 'coiled' if you want to use the Coiled cluster

        # Initialize Dask client based on the direct execution setup
        if client_type == 'coiled':
            client, cluster = uu.setup_coiled_cluster()
        else:
            cluster = LocalCluster()
            client = Client(cluster)

        logging.info(f"Dask client initialized with {client_type} cluster")

        try:
            main()
        finally:
            client.close()
            logging.info("Dask client closed")
            if client_type == 'coiled':
                cluster.close()
                logging.info("Coiled cluster closed")
    else:
        # Initialize Dask client based on the argument
        if args.client == 'coiled':
            client, cluster = uu.setup_coiled_cluster()
        else:
            cluster = LocalCluster()
            client = Client(cluster)

        logging.info(f"Dask client initialized with {args.client} cluster")

        try:
            main(tile_id=args.tile_id)
        finally:
            client.close()
            logging.info("Dask client closed")
            if args.client == 'coiled':
                cluster.close()
                logging.info("Coiled cluster closed")

import os
import geopandas as gpd
import logging
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd

from pp_utilities import get_existing_s3_files, compress_and_upload_directory_to_s3

"""
This script processes GRIP (Global Roads Inventory Project) roads by tiles using a pre-existing tile index shapefile.
It reads the tile index shapefile, identifies the regions each tile overlaps with, processes the corresponding
roads shapefiles, clips the roads data to the tile boundaries, and saves the results as shapefiles for each tile.

The script uses Dask to parallelize the processing of multiple tiles.

Functions:
- read_tiles_shapefile: Reads and returns the tile index shapefile.
- process_tile: Processes a single tile by reading and clipping road data within the tile's bounds.
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
- dask
- dask.distributed
- dask.diagnostics

Note: The script assumes the presence of the required shapefiles and directories.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
regional_shapefiles = [
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region1_vector_shp\GRIP4_region1.shp",
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region2_vector_shp\GRIP4_region2.shp",
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region3_vector_shp\GRIP4_region3.shp",
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region4_vector_shp\GRIP4_region4.shp",
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region5_vector_shp\GRIP4_region5.shp",
    r"C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region6_vector_shp\GRIP4_region6.shp"
]
tiles_shapefile_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
output_dir = r"C:\GIS\Data\Global\GRIP\roads_by_tile"

os.makedirs(output_dir, exist_ok=True)

def read_tiles_shapefile():
    """
    Reads the tiles shapefile from the local directory.
    Returns:
        GeoDataFrame: A GeoDataFrame containing the tiles.
    """
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(tiles_shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

@dask.delayed
def process_tile(tile):
    """
    Processes a single tile by reading and clipping road data within the tile's bounds.

    Args:
        tile (GeoSeries): The tile to process.
    """
    tile_id = tile['tile_id']  # Assuming the tile ID is stored in a column named 'tile_id'
    output_path = os.path.join(output_dir, f"roads_{tile_id}.shp")

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
    """
    Processes all tiles using Dask for parallelization.

    Args:
        tiles_gdf (GeoDataFrame): GeoDataFrame containing all tiles.
    """
    tasks = [process_tile(tile) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        dask.compute(*tasks)

def main(tile_id=None):
    """
    Main function to orchestrate the processing based on provided arguments.

    Args:
        tile_id (str, optional): ID of the tile to process. Defaults to None.
    """
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        # Process a single tile
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        process_tile(tile).compute()
    else:
        # Process all tiles
        process_all_tiles(tiles_gdf)

if __name__ == '__main__':
    # Initialize Dask client
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        # Example usage
        # Replace 'test_tile_id' with the actual tile ID you want to test
        # main(tile_id='00N_110E')

        # To process all tiles, comment out the above line and uncomment the line below
        main()
    finally:
        # Close Dask client
        client.close()

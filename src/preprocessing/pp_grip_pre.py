import geopandas as gpd
import os
import logging
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd

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
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(tiles_shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

@dask.delayed
def process_tile(tile):
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
    # Use Dask to parallelize the processing of tiles
    tasks = [process_tile(tile) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        dask.compute(*tasks)

def main(tile_id=None):
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

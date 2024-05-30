"""
This script processes OSM (OpenStreetMap) data for roads and canals by tile. It reads a tile index shapefile,
identifies the regions each tile overlaps with, and processes the corresponding OSM PBF files. The processed
data is then saved as shapefiles for each tile.

Unlike the previous script, this script checks tile and region intersections on the fly instead of using a
preindexed shapefile with region information. The script is currently not being used in production.

Functions:
- read_tiles_shapefile: Reads the tile index shapefile.
- run_ogr2ogr_local: Runs ogr2ogr to extract data from OSM PBF files within specified bounds.
- filter_linestrings: Filters only LineString geometries from a GeoDataFrame.
- process_tile: Processes a single tile for roads and/or canals.
- process_all_tiles: Processes all tiles for roads and/or canals.
- main: Main function to orchestrate the processing based on provided arguments.
"""

import geopandas as gpd
import os
import logging
import subprocess
import shapely.geometry
from datetime import datetime
import pandas as pd
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# Paths
filtered_canals_path = r"C:\GIS\Data\Global\OSM\filtered_canals"
filtered_highways_path = r"C:\GIS\Data\Global\OSM\filtered_highways"
output_dir_roads = r"C:\GIS\Data\Global\OSM\roads_by_tile"
output_dir_canals = r"C:\GIS\Data\Global\OSM\canals_by_tile"
os.makedirs(output_dir_roads, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)

tiles_shapefile_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"

# Hardcoded bounds dictionary using filenames
bounds_dict = {
    "north-america-latest.osm.pbf": (-180.0, 5.57228, 180.0, 85.04177),
    "africa-latest.osm.pbf": (-27.262032, -60.3167, 66.722766, 37.77817),
    "antarctica-latest.osm.pbf": (-180.0, -90.0, 180.0, -60.0),
    "asia-latest.osm.pbf": (-180.0, -13.01165, 180.0, 84.52666),
    "australia-oceania-latest.osm.pbf": (-179.999999, -57.16482, 180.0, 26.27781),
    "central-america-latest.osm.pbf": (-99.82733, 3.283755, -44.93667, 28.05483),
    "europe-latest.osm.pbf": (-34.49296, 29.635548, 46.75348, 81.47299),
    "south-america-latest.osm.pbf": (-82.64999, -56.20053, -34.1092, 12.4575)
}

def read_tiles_shapefile():
    """
    Reads the tile index shapefile.
    Returns:
        GeoDataFrame: A GeoDataFrame containing the tiles.
    """
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(tiles_shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

def run_ogr2ogr_local(pbf_file, tile_bounds, tile_id):
    """
    Runs ogr2ogr to extract data from an OSM PBF file within specified bounds.

    Args:
        pbf_file (str): Path to the OSM PBF file.
        tile_bounds (tuple): Bounding box (minx, miny, maxx, maxy) of the tile.
        tile_id (str): Tile ID for temporary file naming.

    Returns:
        GeoDataFrame: A GeoDataFrame with the extracted data.
    """
    temp_geojson = f'temp_{tile_id}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.geojson'
    cmd = [
        'ogr2ogr', '-f', 'GeoJSON', temp_geojson, pbf_file,
        '-spat', str(tile_bounds[0]), str(tile_bounds[1]), str(tile_bounds[2]), str(tile_bounds[3]), 'lines'
    ]
    try:
        subprocess.check_call(cmd)
        gdf = gpd.read_file(temp_geojson, memory_map=True)
        os.remove(temp_geojson)
        return gdf
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ogr2ogr: {e}")
        if os.path.exists(temp_geojson):
            os.remove(temp_geojson)
        return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Error reading GeoJSON file: {e}")
        if os.path.exists(temp_geojson):
            os.remove(temp_geojson)
        return gpd.GeoDataFrame()

def filter_linestrings(gdf):
    """
    Filters only LineString geometries from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.

    Returns:
        GeoDataFrame: A GeoDataFrame with only LineString geometries.
    """
    if not gdf.empty:
        linestring_gdf = gdf[gdf.geometry.type == 'LineString']
        return linestring_gdf
    return gpd.GeoDataFrame()

def process_tile(tile, filtered_canals_path, filtered_highways_path):
    """
    Processes a single tile for roads and/or canals.

    Args:
        tile (GeoSeries): The tile to process.
        filtered_canals_path (str): Path to the filtered canals directory.
        filtered_highways_path (str): Path to the filtered highways directory.
    """
    tile_id = tile['tile_id']
    roads_output_path = os.path.join(output_dir_roads, f"roads_{tile_id}.shp")
    canals_output_path = os.path.join(output_dir_canals, f"canals_{tile_id}.shp")

    roads_needed = not os.path.exists(roads_output_path)
    canals_needed = not os.path.exists(canals_output_path)

    if not roads_needed and not canals_needed:
        logging.info(f"Output files for {tile_id} already exist. Skipping tile {tile_id}.")
        return

    tile_bounds = tile.geometry.bounds
    tile_geom = shapely.geometry.box(*tile_bounds)
    logging.info(f"Processing tile {tile_id} with bounds {tile_bounds}")

    combined_roads = []
    combined_canals = []

    try:
        if roads_needed:
            for pbf_file in os.listdir(filtered_highways_path):
                pbf_file_path = os.path.join(filtered_highways_path, pbf_file)
                file_name = os.path.basename(pbf_file_path).replace('highways_', '').replace('canals_', '')
                logging.debug(f"Checking intersection for file: {file_name}")
                if file_name in bounds_dict and tile_geom.intersects(shapely.geometry.box(*bounds_dict[file_name])):
                    logging.info(f"Reading roads from PBF file {pbf_file_path} within bounds {tile_bounds}")
                    gdf = run_ogr2ogr_local(pbf_file_path, tile_bounds, tile_id)
                    gdf = filter_linestrings(gdf)

                    if not gdf.empty:
                        gdf = gdf[['geometry', 'highway']]
                        roads_in_tile = gpd.clip(gdf, tile.geometry)
                        combined_roads.append(roads_in_tile)

        if canals_needed:
            for pbf_file in os.listdir(filtered_canals_path):
                pbf_file_path = os.path.join(filtered_canals_path, pbf_file)
                file_name = os.path.basename(pbf_file_path).replace('highways_', '').replace('canals_', '')
                logging.debug(f"Checking intersection for file: {file_name}")
                if file_name in bounds_dict and tile_geom.intersects(shapely.geometry.box(*bounds_dict[file_name])):
                    logging.info(f"Reading canals from PBF file {pbf_file_path} within bounds {tile_bounds}")
                    gdf = run_ogr2ogr_local(pbf_file_path, tile_bounds, tile_id)
                    gdf = filter_linestrings(gdf)

                    if not gdf.empty:
                        gdf = gdf[['geometry', 'waterway']]
                        canals_in_tile = gpd.clip(gdf, tile.geometry)
                        combined_canals.append(canals_in_tile)

        if combined_roads:
            combined_roads_gdf = gpd.GeoDataFrame(pd.concat(combined_roads, ignore_index=True))
            combined_roads_gdf.to_file(roads_output_path)
            logging.info(f"Saved combined roads for tile {tile_id} to {roads_output_path}")
            combined_roads_gdf = None
            combined_roads = []

        if combined_canals:
            combined_canals_gdf = gpd.GeoDataFrame(pd.concat(combined_canals, ignore_index=True))
            combined_canals_gdf.to_file(canals_output_path)
            logging.info(f"Saved combined canals for tile {tile_id} to {canals_output_path}")
            combined_canals_gdf = None
            combined_canals = []

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")
    finally:
        # Clean up memory
        combined_roads = []
        combined_canals = []

def process_all_tiles(tiles_gdf, filtered_canals_path, filtered_highways_path):
    """
    Processes all tiles for roads and/or canals.

    Args:
        tiles_gdf (GeoDataFrame): GeoDataFrame containing all tiles.
        filtered_canals_path (str): Path to the filtered canals directory.
        filtered_highways_path (str): Path to the filtered highways directory.
    """
    for idx, tile in tiles_gdf.iterrows():
        process_tile(tile, filtered_canals_path, filtered_highways_path)

def main(tile_id=None):
    """
    Main function to orchestrate the processing based on provided arguments.

    Args:
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
    """
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        process_tile(tile, filtered_canals_path, filtered_highways_path)
    else:
        process_all_tiles(tiles_gdf, filtered_canals_path, filtered_highways_path)

if __name__ == '__main__':
    tile_id = None
    main(tile_id=tile_id)

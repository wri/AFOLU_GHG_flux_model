"""
This script processes OSM (OpenStreetMap) data for roads and canals by tile. It reads a tile index shapefile,
identifies the regions each tile overlaps with, and processes the corresponding OSM PBF files. The processed
data is then saved as shapefiles for each tile.

This script processes one tile at a time, which makes it slower but more reliable. It is being kept for reference
and is not currently used in production.

Functions:
- read_tiles_shapefile: Reads the tile index shapefile.
- run_ogr2ogr_local: Runs ogr2ogr to extract data from OSM PBF files within specified bounds.
- kill_processes_by_name: Kills all processes with a given name.
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
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# Paths
regional_pbf_files = [
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\antarctica-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf"
]

# Hardcoded bounds dictionary
bounds_dict = {
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf": (
        -180.0, 5.57228, 180.0, 85.04177),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf": (
        -27.262032, -60.3167, 66.722766, 37.77817),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\antarctica-latest.osm.pbf": (-180.0, -90.0, 180.0, -60.0),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf": (-180.0, -13.01165, 180.0, 84.52666),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf": (
        -179.999999, -57.16482, 180.0, 26.27781),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf": (
        -99.82733, 3.283755, -44.93667, 28.05483),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf": (
        -34.49296, 29.635548, 46.75348, 81.47299)
}

tiles_shapefile_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
output_dir_roads = r"C:\GIS\Data\Global\OSM\roads_by_tile"
output_dir_canals = r"C:\GIS\Data\Global\OSM\canals_by_tile"
temp_output_dir = r"C:\GIS\Data\Global\OSM\temp"
os.makedirs(output_dir_roads, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)
os.makedirs(temp_output_dir, exist_ok=True)

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
    temp_geojson = os.path.join(temp_output_dir, f'temp_{tile_id}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.geojson')
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

def kill_processes_by_name(process_name):
    """
    Kills all processes with the given name.

    Args:
        process_name (str): Name of the process to kill.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in proc.info['cmdline']:
                logging.info(f"Killing process {proc.info['pid']} - {proc.info['name']} - {' '.join(proc.info['cmdline'])}")
                proc.terminate()  # or proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def process_tile(tile, pbf_files, bounds_dict):
    """
    Processes a single tile for roads and/or canals.

    Args:
        tile (GeoSeries): The tile to process.
        pbf_files (list): List of paths to the OSM PBF files.
        bounds_dict (dict): Dictionary with bounds for each PBF file.
    """
    tile_id = tile['tile_id']
    roads_output_path = os.path.join(output_dir_roads, f"roads_{tile_id}.shp")
    canals_output_path = os.path.join(output_dir_canals, f"canals_{tile_id}.shp")

    if os.path.exists(roads_output_path) and os.path.exists(canals_output_path):
        logging.info(f"Output files for {tile_id} already exist. Skipping tile {tile_id}.")
        return

    tile_bounds = tile.geometry.bounds
    logging.info(f"Processing tile {tile_id} with bounds {tile_bounds}")

    combined_roads = []
    combined_canals = []

    try:
        for pbf_file in pbf_files:
            if tile.geometry.intersects(shapely.geometry.box(*bounds_dict[pbf_file])):
                logging.info(f"Reading roads from PBF file {pbf_file} within bounds {tile_bounds}")
                gdf = run_ogr2ogr_local(pbf_file, tile_bounds, tile_id)

                if not gdf.empty:
                    roads = gdf[gdf['highway'].notnull()]
                    canals = gdf[(gdf['waterway'].notnull()) & (gdf['waterway'].isin(['ditch', 'canal', 'drain']))]

                    roads_in_tile = gpd.clip(roads, tile.geometry)
                    canals_in_tile = gpd.clip(canals, tile.geometry)

                    combined_roads.append(roads_in_tile)
                    combined_canals.append(canals_in_tile)

        if combined_roads:
            all_roads_gdf = pd.concat(combined_roads, ignore_index=True)
            all_roads_gdf.to_file(roads_output_path)
            logging.info(f"Saved roads shapefile for tile {tile_id}")
        else:
            logging.info(f"No roads data found for tile {tile_id}")

        if combined_canals:
            all_canals_gdf = pd.concat(combined_canals, ignore_index=True)
            all_canals_gdf.to_file(canals_output_path)
            logging.info(f"Saved canals shapefile for tile {tile_id}")
        else:
            logging.info(f"No canals data found for tile {tile_id}")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(tiles_gdf, pbf_files, bounds_dict):
    """
    Processes all tiles for roads and/or canals.

    Args:
        tiles_gdf (GeoDataFrame): GeoDataFrame containing the tiles.
        pbf_files (list): List of paths to the OSM PBF files.
        bounds_dict (dict): Dictionary with bounds for each PBF file.
    """
    for _, tile in tiles_gdf.iterrows():
        process_tile(tile, pbf_files, bounds_dict)

def main():
    """
    Main function to orchestrate the processing based on provided arguments.
    """
    tiles_gdf = read_tiles_shapefile()
    process_all_tiles(tiles_gdf, regional_pbf_files, bounds_dict)
    kill_processes_by_name('ogr2ogr')

if __name__ == "__main__":
    main()

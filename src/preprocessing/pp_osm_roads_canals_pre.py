"""
This script processes OSM (OpenStreetMap) data for roads and canals by tile. It reads a tile index shapefile,
identifies the regions each tile overlaps with, and processes the corresponding OSM PBF files. The processed data
is then saved as shapefiles for each tile. This script relies on a pre-indexed tile shapefile.

The script can process:
- Only roads
- Only canals
- Both roads and canals

Command-line arguments can be used to specify:
- The ID of the tile to process
- Whether to process roads
- Whether to process canals

Usage examples:
- Process both roads and canals for all tiles:
  python script.py --process_roads --process_canals

- Process only canals for a specific tile:
  python script.py --tile_id 50N_110E --process_canals

Functions:
- read_tiles_shapefile: Reads the updated tile index shapefile with regions.
- run_ogr2ogr_local: Runs ogr2ogr to extract data from OSM PBF files within specified bounds.
- filter_linestrings: Filters only LineString geometries from a GeoDataFrame.
- filter_only_linestrings: Filters out non-LineString geometries.
- process_tile: Processes a single tile for roads and/or canals.
- process_all_tiles: Processes all tiles for roads and/or canals.
- main: Main function to orchestrate the processing based on provided arguments.
"""

import geopandas as gpd
import os
import logging
import subprocess
from datetime import datetime
import pandas as pd

import pp_utilities as uu

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths TODO add logic to automatically create these paths and integrate with other osm preprocessing
filtered_canals_path = r"C:\GIS\Data\Global\OSM\filtered_canals"
filtered_highways_path = r"C:\GIS\Data\Global\OSM\filtered_highways"
output_dir_roads = r"C:\GIS\Data\Global\OSM\roads_by_tile"
output_dir_canals = r"C:\GIS\Data\Global\OSM\canals_by_tile"
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
s3_bucket_name = 'gfw2-data'
index_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/index/Global_Peatlands'

os.makedirs(output_dir_roads, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)

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
    Reads the updated tiles shapefile with regions.
    Returns:
        GeoDataFrame: A GeoDataFrame containing the tiles.
    """
    logging.info("Downloading tiles shapefile from S3 to local directory")
    uu.read_shapefile_from_s3(index_shapefile_prefix, local_temp_dir, s3_bucket_name)
    shapefile_path = os.path.join(local_temp_dir, 'Global_Peatlands.shp')
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(shapefile_path)
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

def filter_only_linestrings(gdf):
    """
    Filters out non-LineString geometries from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.

    Returns:
        GeoDataFrame: A GeoDataFrame with only LineString geometries.
    """
    return gdf[gdf.geometry.type == 'LineString']

def process_tile(tile, filtered_canals_path, filtered_highways_path, process_roads=True, process_canals=True):
    """
    Processes a single tile for roads and/or canals.

    Args:
        tile (GeoSeries): The tile to process.
        filtered_canals_path (str): Path to the filtered canals directory.
        filtered_highways_path (str): Path to the filtered highways directory.
        process_roads (bool): Whether to process roads.
        process_canals (bool): Whether to process canals.
    """
    tile_id = tile['tile_id']
    roads_output_path = os.path.join(output_dir_roads, f"roads_{tile_id}.shp")
    canals_output_path = os.path.join(output_dir_canals, f"canals_{tile_id}.shp")

    roads_needed = process_roads and not os.path.exists(roads_output_path)
    canals_needed = process_canals and not os.path.exists(canals_output_path)

    if not roads_needed and not canals_needed:
        logging.info(f"Output files for {tile_id} already exist. Skipping tile {tile_id}.")
        return

    tile_bounds = tile.geometry.bounds
    logging.info(f"Processing tile {tile_id} with bounds {tile_bounds}")

    combined_roads = []
    combined_canals = []

    try:
        if 'regions' in tile:
            regions = tile['regions'].split(',')
            for region in regions:
                pbf_file = region
                if pbf_file in bounds_dict:
                    if roads_needed:
                        pbf_file_path = os.path.join(filtered_highways_path, f"highways_{pbf_file}")
                        logging.info(f"Reading roads from PBF file {pbf_file_path} within bounds {tile_bounds}")
                        gdf = run_ogr2ogr_local(pbf_file_path, tile_bounds, tile_id)
                        gdf = filter_only_linestrings(gdf)

                        if not gdf.empty:
                            gdf = gdf[['geometry', 'highway']]
                            roads_in_tile = gpd.clip(gdf, tile.geometry)
                            combined_roads.append(roads_in_tile)
                        else:
                            logging.info(f"No roads data found in {pbf_file_path} for tile {tile_id}")

                    if canals_needed:
                        pbf_file_path = os.path.join(filtered_canals_path, f"canals_{pbf_file}")
                        logging.info(f"Reading canals from PBF file {pbf_file_path} within bounds {tile_bounds}")
                        gdf = run_ogr2ogr_local(pbf_file_path, tile_bounds, tile_id)
                        gdf = filter_only_linestrings(gdf)

                        if not gdf.empty:
                            gdf = gdf[['geometry', 'waterway']]
                            canals_in_tile = gpd.clip(gdf, tile.geometry)
                            combined_canals.append(canals_in_tile)
                        else:
                            logging.info(f"No canals data found in {pbf_file_path} for tile {tile_id}")

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

def process_all_tiles(tiles_gdf, filtered_canals_path, filtered_highways_path, process_roads=True, process_canals=True):
    """
    Processes all tiles for roads and/or canals.

    Args:
        tiles_gdf (GeoDataFrame): GeoDataFrame containing all tiles.
        filtered_canals_path (str): Path to the filtered canals directory.
        filtered_highways_path (str): Path to the filtered highways directory.
        process_roads (bool): Whether to process roads.
        process_canals (bool): Whether to process canals.
    """
    for idx, tile in tiles_gdf.iterrows():
        process_tile(tile, filtered_canals_path, filtered_highways_path, process_roads, process_canals)

def main(tile_id=None, process_roads=True, process_canals=True):
    """
    Main function to orchestrate the processing based on provided arguments.

    Args:
        tile_id (str, optional): ID of the tile to process. Defaults to None.
        process_roads (bool, optional): Whether to process roads. Defaults to True.
        process_canals (bool, optional): Whether to process canals. Defaults to True.
    """
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        process_tile(tile, filtered_canals_path, filtered_highways_path, process_roads, process_canals)
    else:
        process_all_tiles(tiles_gdf, filtered_canals_path, filtered_highways_path, process_roads, process_canals)

if __name__ == '__main__':
    # Directly call main with the desired parameters to process tile 50N_110E canals only
    # main(tile_id="50N_110E", process_roads=False, process_canals=True)
    # Directly call full run
    main(process_roads=True, process_canals=True)

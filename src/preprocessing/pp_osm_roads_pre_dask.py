import geopandas as gpd
import os
import logging
import subprocess
import shapely.geometry
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

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
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf": (-180.0, 5.57228, 180.0, 85.04177),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf": (-27.262032, -60.3167, 66.722766, 37.77817),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\antarctica-latest.osm.pbf": (-180.0, -90.0, 180.0, -60.0),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf": (-180.0, -13.01165, 180.0, 84.52666),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf": (-179.999999, -57.16482, 180.0, 26.27781),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf": (-99.82733, 3.283755, -44.93667, 28.05483),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf": (-34.49296, 29.635548, 46.75348, 81.47299)
}

tiles_shapefile_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
output_dir_roads = r"C:\GIS\Data\Global\OSM\roads_by_tile"
output_dir_canals = r"C:\GIS\Data\Global\OSM\canals_by_tile"
os.makedirs(output_dir_roads, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)

def read_tiles_shapefile():
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(tiles_shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

def run_ogr2ogr_local(pbf_file, tile_bounds, tile_id):
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

@dask.delayed
def process_tile(tile, pbf_files, bounds_dict):
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
            combined_roads_gdf = gpd.GeoDataFrame(pd.concat(combined_roads, ignore_index=True))
            combined_roads_gdf.to_file(roads_output_path)
            logging.info(f"Saved combined roads for tile {tile_id} to {roads_output_path}")

        if combined_canals:
            combined_canals_gdf = gpd.GeoDataFrame(pd.concat(combined_canals, ignore_index=True))
            combined_canals_gdf.to_file(canals_output_path)
            logging.info(f"Saved combined canals for tile {tile_id} to {canals_output_path}")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(tiles_gdf, pbf_files, bounds_dict, max_concurrent_tasks=4):
    tasks = [process_tile(tile, pbf_files, bounds_dict) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        for i in range(0, len(tasks), max_concurrent_tasks):
            dask.compute(*tasks[i:i+max_concurrent_tasks])

def main(tile_id=None):
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        process_tile(tile, regional_pbf_files, bounds_dict).compute()
    else:
        process_all_tiles(tiles_gdf, regional_pbf_files, bounds_dict)

if __name__ == '__main__':
    tile_id = None

    dask.config.set({"distributed.scheduler.work-stealing": True})
    dask.config.set({"distributed.scheduler.worker-ttl": "60s"})
    dask.config.set({"distributed.comm.timeouts.connect": "60s"})
    dask.config.set({"distributed.comm.timeouts.tcp": "120s"})

    local_cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='8GB')
    local_client = Client(local_cluster)

    try:
        main(tile_id=tile_id)
    finally:
        local_client.close()

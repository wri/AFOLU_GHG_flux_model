import geopandas as gpd
import os
import logging
import subprocess
import shapely.geometry
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)  # Set root logger to debug

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

def run_ogr2ogr_memory(pbf_file, tile_bounds):
    # Use ogr2ogr to convert the PBF to GeoJSON in memory
    cmd = [
        'ogr2ogr', '-f', 'GeoJSON', '/vsimem/temp.geojson', pbf_file,
        '-spat', str(tile_bounds[0]), str(tile_bounds[1]), str(tile_bounds[2]), str(tile_bounds[3]), 'lines'
    ]
    try:
        subprocess.check_call(cmd)
        gdf = gpd.read_file('/vsimem/temp.geojson')
        subprocess.check_call(['gdalmanage', 'delete', '/vsimem/temp.geojson'])
        return gdf
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ogr2ogr: {e}")
        return gpd.GeoDataFrame()

@dask.delayed
def process_tile(tile, pbf_files):
    tile_id = tile['tile_id']  # Assuming the tile ID is stored in a column named 'tile_id'
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
            logging.info(f"Reading roads from PBF file {pbf_file} within bounds {tile_bounds}")
            gdf = run_ogr2ogr_memory(pbf_file, tile_bounds)

            if not gdf.empty:
                roads = gdf[gdf['highway'].notnull()]
                canals = gdf[(gdf['waterway'].notnull()) & (gdf['waterway'].isin(['ditch', 'canal', 'drain']))]

                roads_in_tile = gpd.clip(roads, tile.geometry)
                canals_in_tile = gpd.clip(canals, tile.geometry)

                combined_roads.append(roads_in_tile)
                combined_canals.append(canals_in_tile)

        if combined_roads:
            combined_roads_gdf = pd.concat(combined_roads, ignore_index=True)
            combined_roads_gdf.to_file(roads_output_path)
            logging.info(f"Saved combined roads for tile {tile_id} to {roads_output_path}")

        if combined_canals:
            combined_canals_gdf = pd.concat(combined_canals, ignore_index=True)
            combined_canals_gdf.to_file(canals_output_path)
            logging.info(f"Saved combined canals for tile {tile_id} to {canals_output_path}")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(tiles_gdf, pbf_files):
    # Use Dask to parallelize the processing of tiles
    tasks = [process_tile(tile, pbf_files) for idx, tile in tiles_gdf.iterrows()]
    with ProgressBar():
        dask.compute(*tasks)

def main(tile_id=None):
    tiles_gdf = read_tiles_shapefile()
    if tile_id:
        # Process a single tile
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        process_tile(tile, regional_pbf_files).compute()
    else:
        # Process all tiles
        process_all_tiles(tiles_gdf, regional_pbf_files)

if __name__ == '__main__':
    # Specify the tile_id to process a single tile or set it to None to process all tiles
    tile_id = None  # or a specific tile ID like "00N_110E"

    # Initialize Dask client
    cluster = LocalCluster()
    client = Client(cluster)

    try:
        # Run main function with or without tile_id argument
        main(tile_id=tile_id)
    finally:
        # Close Dask client
        client.close()

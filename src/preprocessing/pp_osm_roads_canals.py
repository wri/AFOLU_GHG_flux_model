import os
import subprocess
import logging
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError
from shapely.geometry import box, mapping
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import Pool
import psutil

"""
The purpose of this script is to split the data into smaller PBFs for easier processing.
It includes checks to ensure there is data in the tile before exporting.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# S3 Paths
s3_bucket = "gfw2-data"
s3_output_dir = "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/osm_roads/pbf_tiles"

# Paths to the original regional PBF files locally
regional_pbf_files = [
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf"
]

# Hardcoded bounds dictionary
bounds_dict = {
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf": (-180.0, 5.57228, 180.0, 85.04177),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf": (-27.262032, -60.3167, 66.722766, 37.77817),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf": (-180.0, -13.01165, 180.0, 84.52666),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf": (-179.999999, -57.16482, 180.0, 26.27781),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf": (-99.82733, 3.283755, -44.93667, 28.05483),
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf": (-34.49296, 29.635548, 46.75348, 81.47299)
}

tiles_shapefile_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
temp_output_dir = r"C:\GIS\Data\Global\OSM\temp"
os.makedirs(temp_output_dir, exist_ok=True)

def upload_to_s3(local_file, s3_path, retries=3):
    s3_client = boto3.client('s3')
    s3_key = s3_path.replace(f"s3://{s3_bucket}/", "")
    attempt = 0
    while attempt < retries:
        try:
            s3_client.upload_file(local_file, s3_bucket, s3_key)
            logging.info(f"Uploaded {local_file} to s3://{s3_bucket}/{s3_key}")
            return
        except (ClientError, EndpointConnectionError, ConnectionClosedError) as e:
            logging.error(f"Failed to upload file to S3: {s3_path}, {e}. Attempt {attempt + 1} of {retries}")
            attempt += 1
    logging.error(f"All {retries} upload attempts failed for {local_file} to S3 path {s3_path}")

def read_tiles_shapefile():
    logging.info("Reading tiles shapefile from local directory")
    tiles_gdf = gpd.read_file(tiles_shapefile_path)
    logging.info(f"Columns in tiles shapefile: {tiles_gdf.columns}")
    return tiles_gdf

def create_polygon_file(geometry, poly_file_path):
    """
    Create a .poly file from a Shapely geometry.
    """
    poly = mapping(geometry)
    with open(poly_file_path, 'w') as f:
        f.write("polygon\n")
        for coord in poly['coordinates'][0]:
            f.write(f"  {coord[0]} {coord[1]}\n")
        f.write("END\nEND\n")

def run_ogr2ogr_extract(pbf_file, tile_bounds, output_pbf):
    cmd = [
        'ogr2ogr', '-f', 'OSM', output_pbf, pbf_file,
        '-spat', str(tile_bounds[0]), str(tile_bounds[1]), str(tile_bounds[2]), str(tile_bounds[3])
    ]
    logging.debug(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        logging.info(f"Created PBF file {output_pbf} with bounds {tile_bounds}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ogr2ogr: {e}")
        if os.path.exists(output_pbf):
            os.remove(output_pbf)

def s3_file_exists(bucket, key):
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def kill_processes_by_name(process_name):
    """
    Kills all processes with the given name.

    :param process_name: Name of the process to kill
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in proc.info['cmdline']:
                logging.info(f"Killing process {proc.info['pid']} - {proc.info['name']} - {' '.join(proc.info['cmdline'])}")
                proc.terminate()  # or proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def check_pbf_file_not_empty(pbf_file):
    cmd = ['ogrinfo', pbf_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "ERROR" in result.stdout or "ERROR" in result.stderr:
            logging.info(f"No data found in PBF file {pbf_file}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking PBF file with ogrinfo: {e}")
        return False

def process_tile(tile_id, tile_bounds, pbf_file):
    output_pbf = os.path.join(temp_output_dir, f"{tile_id}.osm.pbf")
    s3_output_path = f"{s3_output_dir}/{tile_id}.osm.pbf"
    s3_key = s3_output_path.replace(f"s3://{s3_bucket}/", "")

    if s3_file_exists(s3_bucket, s3_key):
        logging.info(f"File s3://{s3_bucket}/{s3_key} already exists. Skipping processing for tile {tile_id}.")
        return

    run_ogr2ogr_extract(pbf_file, tile_bounds, output_pbf)

    if os.path.exists(output_pbf) and check_pbf_file_not_empty(output_pbf):
        upload_to_s3(output_pbf, s3_output_path)
        os.remove(output_pbf)
    else:
        logging.info(f"Removing empty PBF file {output_pbf}")
        os.remove(output_pbf)

def split_pbf_files(tiles_gdf, pbf_file, pbf_bounds):
    tasks = []
    for idx, tile in tqdm(tiles_gdf.iterrows(), total=tiles_gdf.shape[0]):
        if tile[1].geometry.intersects(box(*pbf_bounds)):
            logging.debug(f"Tile {tile[1]['tile_id']} intersects with PBF bounds {pbf_bounds}")
            tile_id = tile[1]['tile_id']
            tile_bounds = tile[1].geometry.bounds
            tasks.append((tile_id, tile_bounds, pbf_file))
    if not tasks:
        logging.info("No tiles intersect with the PBF bounds.")
        return
    with Pool(os.cpu_count() - 1) as pool:
        pool.starmap(process_tile, tasks)

def filter_tiles_by_region_bounds(tiles_gdf, region_bounds):
    filtered_tiles = tiles_gdf[tiles_gdf.intersects(box(*region_bounds))]
    logging.info(f"Filtered {len(filtered_tiles)} tiles within the region bounds {region_bounds}")
    return filtered_tiles

def main(tile_id=None):
    tiles_gdf = read_tiles_shapefile()

    if tile_id:
        # Process a single tile
        tile = tiles_gdf[tiles_gdf['tile_id'] == tile_id].iloc[0]
        for pbf_file in regional_pbf_files:
            pbf_bounds = bounds_dict[pbf_file]
            if tile.geometry.intersects(box(*pbf_bounds)):
                process_tile(tile['tile_id'], tile.geometry.bounds, pbf_file)
    else:
        # Process all tiles
        for pbf_file in regional_pbf_files:
            pbf_bounds = bounds_dict[pbf_file]
            logging.info(f"Processing PBF file: {pbf_file} with bounds {pbf_bounds}")
            filtered_tiles_gdf = filter_tiles_by_region_bounds(tiles_gdf, pbf_bounds)
            split_pbf_files(filtered_tiles_gdf, pbf_file, pbf_bounds)

    # Kill any lingering processes
    kill_processes_by_name('ogr2ogr')

if __name__ == "__main__":
    tile_id = "00N_110E"  # Change to a specific tile ID to test a single tile, e.g., "00N_110E"
    main(tile_id=tile_id)

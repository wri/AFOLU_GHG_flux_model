import coiled
import logging
import dask_geopandas as dgpd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box
import boto3
import os
import dask
from dask.distributed import Client, LocalCluster
import gc
from rasterio.transform import from_origin
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import warnings

import constants_and_names as cn
import pp_utilities as uu


"""
Script for Processing OSM and GRIP Data for Roads and Canals Using Tiled Shapefiles with Dask and Coiled

This script processes spatial data from OpenStreetMap (OSM) and the Global Roads Inventory Project (GRIP) by 
transforming vector data (roads, canals) into raster data on a global scale. The script is designed to handle 
large datasets by breaking down the processing into smaller chunks (tiles) and utilizing parallel computing 
through Dask and Coiled clusters.

Key Features:
- **Tiling and Chunking**: The script processes data in 10x10 degree tiles, which can be further divided into 
  smaller chunks for parallel processing.
- **Dask and Coiled Integration**: The script supports running on either a local Dask client or a Coiled cluster, 
  enabling scalable processing for large datasets.
- **Rasterization and Fishnet Creation**: Vector features are clipped, assigned to fishnet grids, and rasterized 
  to produce density maps.
- **Automatic Directory Management**: Ensures that required local directories for output are created and managed 
  automatically.
- **Error Handling and Logging**: Comprehensive logging and error handling throughout the script allow for 
  detailed monitoring and troubleshooting during execution.
- **S3 Integration**: Processed raster files are uploaded to an S3 bucket, with the option to delete local 
  copies after successful upload.

Typical Workflow:
1. The script starts by initializing the processing environment, including setting up a Dask client (local or Coiled).
2. It calculates the bounds for each tile, breaks them into smaller chunks, and processes each chunk independently.
3. For each chunk:
   - A raster is read and clipped to the chunk's bounds.
   - Features are read, reprojected, and clipped to the same area.
   - The fishnet grid is created, and features are assigned to cells, where their lengths are calculated.
   - The fishnet is rasterized, saved to a GeoTIFF, and uploaded to S3.
4. The script can either process all tiles for a given feature type or focus on a specific tile and chunk, 
   depending on the user's input parameters.

How to Run:
- The script can be executed via command line with various options, such as specifying a tile ID, feature type, 
  chunk bounds, and client type (local or Coiled).
- Example command:
  `python pp_roads_canals_chunks_rio.py --tile_id 00N_110E --feature_type osm_canals --run_mode default --client coiled --chunk_bounds "112, -4, 114, -2"`

This script is designed to be flexible and efficient, allowing users to process massive spatial datasets with 
high granularity and leverage cloud computing resources for scalability.

Note:
Ensure that AWS credentials are configured correctly for S3 access, and that the required datasets are available 
in the specified S3 bucket.

"""

# Set up general logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up specific logging for Coiled and Dask
logging.getLogger("dask").setLevel(logging.INFO)

# Suppress specific warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS. Results from', UserWarning)

# Ensure local output directories exist
for dataset_key, dataset_info in cn.datasets.items():
    if dataset_key in ['osm', 'grip']:
        for sub_key, sub_dataset in dataset_info.items():
            os.makedirs(sub_dataset['local_processed'], exist_ok=True)
os.makedirs(cn.local_temp_dir, exist_ok=True)
logging.info("Directories and paths set up")

def get_10x10_tile_bounds(tile_id):
    """
    Calculate the bounds for a 10x10 degree tile based on its tile ID.

    Args:
        tile_id (str): The ID of the tile (e.g., '00N_110E').

    Returns:
        tuple: The bounds of the tile in the format (min_x, min_y, max_x, max_y).
    """
    logging.debug(f"Calculating bounds for tile {tile_id}")

    if "S" in tile_id:
        max_y = -1 * (int(tile_id[:2]))
        min_y = max_y - 10
    else:
        max_y = (int(tile_id[:2]))
        min_y = max_y - 10

    if "W" in tile_id:
        min_x = -1 * (int(tile_id[4:7]))
        max_x = min_x + 10
    else:
        min_x = (int(tile_id[4:7]))
        max_x = min_x + 10

    logging.debug(f"Bounds for tile {tile_id}: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
    return min_x, min_y, max_x, max_y  # W, S, E, N

def ensure_crs(gdf, target_crs):
    """
    Ensure that the GeoDataFrame has the correct CRS, and reproject if necessary.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to check.
        target_crs (int or str): The target CRS to ensure.

    Returns:
        GeoDataFrame: The GeoDataFrame with the correct CRS.
    """
    logging.debug(f"Ensuring CRS is {target_crs}")
    if gdf.crs is None:
        logging.warning(f"GeoDataFrame CRS is None, setting it to {target_crs}")
        gdf.set_crs(target_crs, inplace=True)
    elif gdf.crs != target_crs:
        logging.info(f"Reprojecting GeoDataFrame from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf

def get_chunk_bounds(chunk_params):
    """
    Generate the bounds for each chunk within a tile.

    Args:
        chunk_params (list): A list containing [min_x, min_y, max_x, max_y, chunk_size].

    Returns:
        list: A list of chunk bounds in the format [min_x, min_y, max_x, max_y].
    """
    min_x = chunk_params[0]
    min_y = chunk_params[1]
    max_x = chunk_params[2]
    max_y = chunk_params[3]
    chunk_size = chunk_params[4]

    x, y = (min_x, min_y)
    chunks = []

    logging.debug(f"Generating chunk bounds with chunk size {chunk_size}")

    while y < max_y:
        while x < max_x:
            bounds = [x, y, x + chunk_size, y + chunk_size]
            chunks.append(bounds)
            logging.debug(f"Generated chunk bounds: {bounds}")
            x += chunk_size
        x = min_x
        y += chunk_size

    return chunks

def mask_raster(data, profile):
    """
    Mask a raster dataset in memory for values equal to 1.

    Args:
        data (numpy.ndarray): The raster data to mask.
        profile (dict): The raster profile containing metadata.

    Returns:
        numpy.ndarray: The masked raster data as a binary array.
    """
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    return mask.astype(np.uint8)

def create_fishnet_from_raster(data, transform):
    """
    Create a fishnet grid from raster data in memory.

    Args:
        data (numpy.ndarray): The raster data to create the fishnet from.
        transform (Affine): The affine transformation of the raster data.

    Returns:
        Dask GeoDataFrame: The fishnet grid as a Dask GeoDataFrame.
    """
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
                x, y = transform * (col, row)
                x1, y1 = transform * (col + 1, row + 1)
                polygons.append(box(x, y, x1, y1))

    fishnet_gdf = dgpd.from_geopandas(gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326"), npartitions=10)
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def read_tiled_features(tile_id, feature_type):
    """
    Read the tiled features for a specific tile ID and feature type.

    Args:
        tile_id (str): The ID of the tile (e.g., '00N_110E').
        feature_type (str): The type of feature to read (e.g., 'osm_roads').

    Returns:
        Dask GeoDataFrame: The features as a Dask GeoDataFrame.
    """
    try:
        logging.info(f"Reading tiled features for tile {tile_id} and feature type {feature_type}")
        feature_key = feature_type.split('_')
        feature_tile_dir = cn.datasets[feature_key[0]][feature_key[1]]['s3_raw']

        tile_id = '_'.join(tile_id.split('_')[:2])
        s3_file_path = os.path.join(feature_tile_dir, f"{feature_key[1]}_{tile_id}.shp")
        full_s3_path = f"/vsis3/{cn.s3_bucket_name}/{s3_file_path}"

        logging.debug(f"Full S3 path: {full_s3_path}")

        features_gdf = dgpd.read_file(full_s3_path, npartitions=1)
        return features_gdf

    except Exception as e:
        logging.error(f"Error reading {feature_type} shapefile for tile {tile_id}: {e}")
        return dgpd.from_geopandas(gpd.GeoDataFrame(columns=['geometry']), npartitions=1)

def reproject_gdf(gdf, epsg):
    """
    Reproject a GeoDataFrame to the specified EPSG code.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to reproject.
        epsg (int): The target EPSG code.

    Returns:
        GeoDataFrame: The reprojected GeoDataFrame.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame does not have a CRS. Please set a CRS before reprojecting.")

    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)

def assign_segments_to_cells(fishnet_gdf, features_gdf):
    """
    Assign feature segments to fishnet cells and calculate their lengths.

    Args:
        fishnet_gdf (GeoDataFrame): The fishnet grid as a GeoDataFrame.
        features_gdf (GeoDataFrame): The features as a GeoDataFrame.

    Returns:
        GeoDataFrame: The fishnet grid with feature lengths assigned to each cell.
    """
    logging.info("Assigning feature segments to fishnet cells and calculating lengths")

    fishnet_gdf = ensure_crs(fishnet_gdf, 3395)
    features_gdf = ensure_crs(features_gdf, 3395)

    fishnet_gdf = fishnet_gdf.compute() if isinstance(fishnet_gdf, dgpd.GeoDataFrame) else fishnet_gdf
    features_gdf = features_gdf.compute() if isinstance(features_gdf, dgpd.GeoDataFrame) else features_gdf

    logging.debug("Performing clipping of features")
    clipped = gpd.clip(features_gdf, fishnet_gdf)
    logging.info(f"Clipped GeoDataFrame with {len(clipped)} features")

    if len(clipped.index) == 0 or 'geometry' not in clipped.columns:
        logging.warning(
            "Clipping resulted in an empty GeoDataFrame or lacks a geometry column. Skipping further processing.")
        return gpd.GeoDataFrame(columns=['geometry'])

    logging.debug("Calculating lengths of clipped features")
    clipped['length'] = clipped.geometry.length

    fishnet_with_lengths = clipped.dissolve(by=clipped.index, aggfunc='sum')

    logging.info(f"Calculated lengths for {len(fishnet_with_lengths)} features")
    logging.debug(f"Fishnet with feature lengths: {fishnet_with_lengths.head()}")

    return fishnet_with_lengths

def fishnet_to_raster(fishnet_gdf, chunk_raster, output_raster_path):
    """
    Convert a fishnet grid to a raster and save it to a file.

    Args:
        fishnet_gdf (GeoDataFrame): The fishnet grid as a GeoDataFrame.
        chunk_raster (xarray.DataArray): The input raster data.
        output_raster_path (str): The path to save the output raster file.

    Returns:
        None
    """
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")

    fishnet_gdf = fishnet_gdf.to_crs("EPSG:4326")

    transform = chunk_raster.rio.transform()
    out_shape = chunk_raster.shape[1:]

    logging.debug(f"Rasterizing fishnet with shape {out_shape} and transform {transform}")

    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry, fishnet_gdf['length'])],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32
    )

    rasterized /= 1000.0

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    xr_rasterized = xr.DataArray(
        rasterized,
        dims=("y", "x"),
        coords={"y": chunk_raster.y, "x": chunk_raster.x},
    )
    xr_rasterized = xr_rasterized.rio.write_crs("EPSG:4326", inplace=True)
    xr_rasterized = xr_rasterized.rio.write_transform(transform, inplace=True)
    xr_rasterized.rio.set_nodata(0, inplace=True)

    logging.debug("Saving raster to file")
    xr_rasterized.rio.to_raster(output_raster_path, compress='lzw')

    logging.info("Fishnet converted to raster and saved")

def upload_final_output_to_s3(local_output_path, s3_output_path):
    """
    Upload the final output file to S3 and delete the local copy.

    Args:
        local_output_path (str): The path to the local file.
        s3_output_path (str): The S3 path where the file should be uploaded.

    Returns:
        None
    """
    s3_client = boto3.client('s3')
    try:
        s3_output_path = s3_output_path.replace('//', '/')
        logging.info(f"Uploading {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        s3_client.upload_file(local_output_path, cn.s3_bucket_name, s3_output_path)
        logging.info(f"Successfully uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")
        os.remove(local_output_path)
        logging.info(f"Deleted local file: {local_output_path}")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Credentials error: {e}")
    except Exception as e:
        logging.error(f"Failed to upload {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}: {e}")

@dask.delayed
def process_chunk(bounds, feature_type, tile_id):
    """
    Process a single chunk of a tile.

    Args:
        bounds (list): The bounds of the chunk in the format [min_x, min_y, max_x, max_y].
        feature_type (str): The type of feature being processed (e.g., 'osm_roads').
        tile_id (str): The ID of the tile.

    Returns:
        None
    """
    output_dir = cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]]['local_processed']
    bounds_str = "_".join([str(round(x, 2)) for x in bounds])
    local_output_path = os.path.join(output_dir, f"{tile_id}_{bounds_str}_{feature_type}_density.tif")
    s3_output_path = f"{cn.datasets[feature_type.split('_')[0]][feature_type.split('_')[1]][('s3_processed_small')]}/{tile_id}_{bounds_str}_{feature_type}_density.tif"

    logging.info(f"Starting processing of the chunk {bounds_str} for tile {tile_id}")

    try:
        # Ensure the output directory exists
        output_dir_path = os.path.dirname(local_output_path)
        os.makedirs(output_dir_path, exist_ok=True)

        input_s3_path = f'/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix_1km}{tile_id}{cn.peat_pattern}'

        logging.debug(f"Opening raster from {input_s3_path}")
        chunk_raster = rxr.open_rasterio(input_s3_path, masked=True)

        logging.debug(f"Clipping raster with bounds {bounds}")
        chunk_raster = chunk_raster.rio.clip_box(minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3])

        if chunk_raster.isnull().all():
            logging.info(f"No data found in the raster for chunk {bounds_str}. Skipping processing.")
            return

        logging.debug("Masking raster data")
        masked_data = mask_raster(chunk_raster[0].data, chunk_raster.rio)

        if np.all(masked_data == 0):
            logging.info(f"No data found in the masked raster for chunk {bounds_str}. Skipping processing.")
            return

        logging.debug("Creating fishnet from masked raster data")
        fishnet_gdf = create_fishnet_from_raster(masked_data, chunk_raster.rio.transform())

        if len(fishnet_gdf.index) == 0:
            logging.info(f"Fishnet is empty for chunk {bounds_str}. Skipping processing.")
            return

        logging.debug(f"Reading features from tile {tile_id}")
        features_gdf = read_tiled_features(tile_id, feature_type)

        logging.debug("Assigning segments to fishnet cells")
        fishnet_with_lengths = assign_segments_to_cells(fishnet_gdf, features_gdf)

        if len(fishnet_with_lengths.columns) == 0 or 'length' not in fishnet_with_lengths.columns:
            logging.info(f"Skipping export of {local_output_path} as 'length' column is missing.")
            return

        logging.debug("Saving fishnet with lengths to raster")
        fishnet_to_raster(fishnet_with_lengths, chunk_raster, local_output_path)

        if os.path.exists(local_output_path):
            logging.info(f"Output raster saved locally at {local_output_path}")
        else:
            logging.error(f"Failed to save output raster at {local_output_path}")
            return

        logging.debug(f"Uploading final output to S3")
        upload_final_output_to_s3(local_output_path, s3_output_path)

        del chunk_raster, masked_data, fishnet_gdf, features_gdf, fishnet_with_lengths
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing chunk {bounds_str} for tile {tile_id}: {e}", exc_info=True)

def process_tile(tile_key, feature_type, chunk_bounds=None, run_mode='default'):
    """
    Process an entire tile, either in chunks or as a whole.

    Args:
        tile_key (str): The S3 key of the tile.
        feature_type (str): The type of feature being processed (e.g., 'osm_roads').
        chunk_bounds (list, optional): Specific bounds for processing a single chunk.
        run_mode (str, optional): The mode of operation (e.g., 'default', 'test').

    Returns:
        list: A list of Dask delayed tasks for processing the tile.
    """
    logging.info(f"Processing tile {tile_key} with feature type {feature_type}")
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    tile_bounds = get_10x10_tile_bounds(tile_id)
    chunk_size = 2  # 1x1 degree chunks

    chunks = get_chunk_bounds([*tile_bounds, chunk_size])

    if chunk_bounds:
        chunks = [chunk_bounds]

    chunk_tasks = [process_chunk(bounds, feature_type, tile_id) for bounds in chunks]

    logging.info(f"Generated {len(chunk_tasks)} chunk tasks")
    return chunk_tasks

def process_all_tiles(feature_type, run_mode='default'):
    """
    Process all tiles for a given feature type.

    Args:
        feature_type (str): The type of feature being processed (e.g., 'osm_roads').
        run_mode (str, optional): The mode of operation (e.g., 'default', 'test').

    Returns:
        None
    """
    logging.info(f"Processing all tiles for feature type {feature_type}")
    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=cn.s3_bucket_name, Prefix=cn.peat_tiles_prefix_1km)
    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith(cn.peat_pattern):
                    tile_keys.append(tile_key)

    all_tasks = []
    for tile_key in tile_keys:
        all_tasks.extend(process_tile(tile_key, feature_type, run_mode=run_mode))

    logging.info(f"Computing {len(all_tasks)} tasks")
    dask.compute(*all_tasks)

def main(tile_id=None, feature_type='osm_roads', chunk_bounds=None, run_mode='default', client_type='local'):
    """
    Main function for processing tiles or chunks of tiles using Dask.

    Args:
        tile_id (str, optional): The ID of a specific tile to process.
        feature_type (str, optional): The type of feature being processed (e.g., 'osm_roads').
        chunk_bounds (list, optional): Specific bounds for processing a single chunk.
        run_mode (str, optional): The mode of operation (e.g., 'default', 'test').
        client_type (str, optional): The type of Dask client to use ('local' or 'coiled').

    Returns:
        None
    """
    logging.info("Initializing main processing function")
    if client_type == 'coiled':
        client, cluster = uu.setup_coiled_cluster()
        logging.info(f"Coiled cluster initialized: {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logging.info(f"Dask client initialized with {client_type} cluster")

    try:
        if tile_id:
            tile_key = f"{cn.peat_tiles_prefix_1km}{tile_id}{cn.peat_pattern}"
            tasks = process_tile(tile_key, feature_type, chunk_bounds, run_mode)
            logging.info("Computing tasks for specific tile")
            dask.compute(*tasks)
        else:
            logging.info("Processing all tiles")
            process_all_tiles(feature_type, run_mode)
    finally:
        client.close()
        logging.info("Dask client closed")
        if client_type == 'coiled':
            cluster.close()
            logging.info("Coiled cluster closed")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Process OSM and GRIP data for roads and canals using tiled shapefiles.')
    parser.add_argument('--tile_id', type=str, help='Tile ID to process')
    parser.add_argument('--feature_type', type=str, choices=['osm_roads', 'osm_canals', 'grip_roads'],
                        default='osm_roads', help='Type of feature to process')
    parser.add_argument('--chunk_bounds', type=str,
                        help='Specific chunk bounds to process in the format "min_x,min_y,max_x,max_y"', default=None)
    parser.add_argument('--run_mode', type=str, choices=['default', 'test'], default='default',
                        help='Run mode (default or test)')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    chunk_bounds = None
    if args.chunk_bounds:
        chunk_bounds = tuple(map(float, args.chunk_bounds.split(',')))

    if not any(sys.argv[1:]):
        # Default values for running directly from PyCharm or an IDE without command-line arguments
        tile_id = '00N_110E'
        feature_type = 'osm_canals'
        chunk_bounds = (112, -4, 114, -2)  # this chunk has data
        run_mode = 'test'
        client_type = 'local'

        main(tile_id=tile_id, feature_type=feature_type, chunk_bounds=chunk_bounds, run_mode=run_mode,
             client_type=client_type)
    else:
        main(tile_id=args.tile_id, feature_type=args.feature_type, chunk_bounds=chunk_bounds, run_mode=args.run_mode,
             client_type=args.client)

"""
coiled test in WSL for chunk with data:

python pp_roads_canals_chunks_rio.py --tile_id 00N_110E --feature_type osm_canals --run_mode default --client coiled --chunk_bounds "112, -4, 114, -2"

Note that OSM roads had errors when running with 40 workers 32Gib memory 
Recommended running with more workers or more memory or both 
"""

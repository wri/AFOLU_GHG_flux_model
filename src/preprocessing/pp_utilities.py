# pp_utilities.py

import os
import logging
import subprocess
import gc
import re

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

import rasterio
from rasterio.features import rasterize
import rioxarray
import geopandas as gpd
from shapely.geometry import box, Polygon

import psutil
from dask.distributed import Client
import coiled

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup with increased max connections and retries
config = boto3.session.Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)
s3_client = boto3.client('s3', config=config)

# -------------------- S3 Utilities --------------------

import boto3
import botocore.exceptions
import logging

def s3_file_exists(bucket, key):
    """
    Check if a file exists in an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The S3 key (path) of the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        logging.info(f"File exists: s3://{bucket}/{key}")
        return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logging.info(f"File does not exist: s3://{bucket}/{key}")
            return False
        else:
            logging.error(f"Unexpected ClientError when checking file existence: {e}")
            raise  # Re-raise the exception for unexpected ClientErrors
    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.PartialCredentialsError) as e:
        logging.error(f"AWS credentials error: {e}")
        raise  # Re-raise to handle credentials issues at a higher level
    except Exception as e:
        logging.error(f"Unexpected error checking file existence in S3: {e}")
        raise  # Re-raise unexpected exceptions


def list_s3_files(bucket, prefix):
    """
    List all files in a specified S3 bucket and prefix.

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The prefix (path) in the S3 bucket.

    Returns:
        list: List of S3 keys (paths) of the files.
    """
    keys = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")
    return keys

def upload_file_to_s3(local_file_path, bucket_name, s3_file_path):
    """
    Upload a local file to an S3 bucket.

    Args:
        local_file_path (str): Path to the local file.
        bucket_name (str): Name of the S3 bucket.
        s3_file_path (str): S3 key (path) where the file will be uploaded.

    Returns:
        None
    """
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
        logging.info(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}")
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")

def download_file_from_s3(s3_file_path, local_file_path, bucket_name):
    """
    Download a file from S3 to a local path.

    Args:
        s3_file_path (str): S3 key (path) of the file to download.
        local_file_path (str): Local path where the file will be saved.
        bucket_name (str): Name of the S3 bucket.

    Returns:
        None
    """
    try:
        s3_client.download_file(bucket_name, s3_file_path, local_file_path)
        logging.info(f"Downloaded s3://{bucket_name}/{s3_file_path} to {local_file_path}")
    except Exception as e:
        logging.error(f"Error downloading file from S3: {e}")

def download_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name):
    """
    Download the shapefile and its associated files from S3 to a local directory.

    Args:
        s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
        local_dir (str): The local directory where the files will be saved.
        s3_bucket_name (str): The name of the S3 bucket.

    Returns:
        None
    """
    s3 = boto3.client('s3')
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    try:
        for ext in extensions:
            s3_path = f"{s3_prefix}{ext}"
            local_path = os.path.join(local_dir, os.path.basename(s3_prefix) + ext)
            logging.info(f"Attempting to download: s3://{s3_bucket_name}/{s3_path} to {local_path}")
            s3.download_file(s3_bucket_name, s3_path, local_path)
            if os.path.exists(local_path):
                logging.info(f"Downloaded: {local_path}")
            else:
                logging.error(f"Failed to download {s3_path} to {local_path}")
    except Exception as e:
        logging.error(f"Error downloading shapefile from S3: {e}")


def read_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name):
    """
    Read a shapefile from S3 into a GeoDataFrame.

    Args:
        s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
        local_dir (str): The local directory where the files will be saved.
        s3_bucket_name (str): The name of the S3 bucket.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    try:
        download_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name)
        shapefile_path = os.path.join(local_dir, os.path.basename(s3_prefix) + '.shp')
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        logging.error(f"Error reading shapefile from S3: {e}")
        return gpd.GeoDataFrame()  # Return an empty GeoDataFrame in case of error

def get_existing_s3_files(s3_bucket, s3_prefix):
    """
    Get a list of existing files in an S3 bucket and prefix.

    Args:
        s3_bucket (str): The name of the S3 bucket.
        s3_prefix (str): The prefix (path) in the S3 bucket.

    Returns:
        set: A set of S3 keys (paths) of the existing files.
    """
    existing_files = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    existing_files.add(obj['Key'])
    except Exception as e:
        logging.error(f"Error retrieving existing files from S3: {e}")
    return existing_files

# -------------------- File Utilities --------------------

def delete_file_if_exists(file_path):
    """
    Delete a file if it exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        None
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Deleted existing file: {file_path}")
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {e}")

def compress_file(input_file, output_file, nodata_value=None):
    """
    Compress a GeoTIFF file using LZW compression and optionally set a NoData value.

    Args:
        input_file (str): Path to the input GeoTIFF file.
        output_file (str): Path to save the compressed GeoTIFF file.
        nodata_value (float, optional): NoData value to set in the output file.

    Returns:
        None
    """
    try:
        with rasterio.open(input_file) as src:
            nodata = src.nodata
            if nodata_value is not None:
                nodata = nodata_value
            elif nodata is None:
                nodata = 0  # Default NoData value

        subprocess.run(
            ['gdal_translate',
             '-co', 'COMPRESS=LZW',
             '-co', 'TILED=YES',
             '-a_nodata', str(nodata),
             input_file, output_file],
            check=True
        )
        logging.info(f"Compressed file saved to {output_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {input_file}: {e}")

def compress_and_upload_file_to_s3(local_file, s3_bucket, s3_key, nodata_value=None):
    """
    Compress a local file using LZW compression and upload it to S3.

    Args:
        local_file (str): Path to the local file to be compressed and uploaded.
        s3_bucket (str): S3 bucket name.
        s3_key (str): S3 key for the uploaded file.
        nodata_value (float, optional): NoData value to set in the output file.

    Returns:
        None
    """
    compressed_file = local_file.replace('.tif', '_compressed.tif')
    try:
        compress_file(local_file, compressed_file, nodata_value)
        upload_file_to_s3(compressed_file, s3_bucket, s3_key)
        delete_file_if_exists(compressed_file)
    except Exception as e:
        logging.error(f"Error compressing and uploading file {local_file}: {e}")

# -------------------- Raster Utilities --------------------

def log_raster_properties(raster, name):
    """
    Log properties of a raster dataset.

    Args:
        raster (rioxarray.DataArray): The raster data to log properties for.
        name (str): A name to identify the raster in the logs.

    Returns:
        None
    """
    logging.info(f"{name} properties:")
    logging.info(f"  CRS: {raster.rio.crs}")
    logging.info(f"  Transform: {raster.rio.transform()}")
    logging.info(f"  Width: {raster.rio.width}")
    logging.info(f"  Height: {raster.rio.height}")
    logging.info(f"  Count: {raster.rio.count}")
    logging.info(f"  Dtype: {raster.dtype}")

def resample_raster(input_path, output_path, reference_path):
    """
    Resample a raster to match the resolution and extent of a reference raster.

    Args:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the resampled raster file.
        reference_path (str): Path to the reference raster file.

    Returns:
        None
    """
    try:
        input_raster = rioxarray.open_rasterio(input_path, masked=True)
        reference_raster = rioxarray.open_rasterio(reference_path, masked=True)

        log_raster_properties(input_raster, "Input Raster")
        log_raster_properties(reference_raster, "Reference Raster")

        # Reproject and resample
        resampled_raster = input_raster.rio.reproject_match(reference_raster)

        resampled_raster.rio.to_raster(output_path)

        if os.path.exists(output_path):
            logging.info(f"Successfully saved resampled raster to {output_path}")
        else:
            logging.error(f"Failed to save resampled raster to {output_path}")
    except Exception as e:
        logging.error(f"Error in resampling raster: {e}")

def reproject_raster(input_raster_path, output_raster_path, target_crs):
    """
    Reproject a raster to the target CRS using GDAL.

    Args:
        input_raster_path (str): Path to the input raster.
        output_raster_path (str): Path to save the reprojected raster.
        target_crs (str): Target coordinate reference system (e.g., 'EPSG:4326').

    Returns:
        None
    """
    try:
        subprocess.run([
            'gdalwarp',
            '-t_srs', target_crs,
            '-r', 'near',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite',
            input_raster_path,
            output_raster_path
        ], check=True)
        logging.info(f"Reprojected raster saved to {output_raster_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error reprojecting raster: {e}")

def rasterize_shapefile(gdf, output_raster_path, reference_raster_path, fill_value=0, burn_value=1):
    """
    Rasterize a GeoDataFrame to create a raster aligned with a reference raster.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to rasterize.
        output_raster_path (str): Path to save the rasterized output.
        reference_raster_path (str): Path to the reference raster for alignment.
        fill_value (int, optional): Value to fill in the output raster where there are no features. Defaults to 0.
        burn_value (int, optional): Value to burn into the raster where features are present. Defaults to 1.

    Returns:
        None
    """
    try:
        with rasterio.open(reference_raster_path) as ref:
            transform = ref.transform
            out_meta = ref.meta.copy()
            out_shape = (ref.height, ref.width)
            nodata_value = out_meta.get('nodata', fill_value)

        shapes = ((geom, burn_value) for geom in gdf.geometry if geom is not None)

        burned = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=fill_value,
            dtype=out_meta['dtype']
        )

        out_meta.update({
            "driver": "GTiff",
            "height": out_shape[0],
            "width": out_shape[1],
            "transform": transform,
            "nodata": nodata_value,
            "compress": "DEFLATE",
            "tiled": True
        })

        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(burned, 1)

        logging.info(f"Rasterized shapefile saved to {output_raster_path}")

    except Exception as e:
        logging.error(f"Error rasterizing shapefile: {e}")

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def rasterize_shapefile_no_ref(gdf, output_raster_path, bounds, resolution, fill_value=0, burn_value=1, dtype='uint8'):
    """
    Rasterize a GeoDataFrame without a reference raster.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to rasterize.
        output_raster_path (str): Path to save the rasterized output.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for the output raster.
        resolution (float): Pixel resolution (size of a pixel in coordinate units).
        fill_value (int, optional): Value to fill in the output raster where there are no features. Defaults to 0.
        burn_value (int, optional): Value to burn into the raster where features are present. Defaults to 1.
        dtype (str, optional): Data type of the output raster. Defaults to 'uint8'.

    Returns:
        None
    """
    try:
        minx, miny, maxx, maxy = bounds

        # Calculate the number of rows and columns in the raster based on resolution and bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        # Create a transform object for the raster
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Prepare shapes from the GeoDataFrame's geometry for rasterization
        shapes = ((geom, burn_value) for geom in gdf.geometry if geom is not None)

        # Create a blank raster with the specified shape and dtype
        raster_data = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=fill_value,
            dtype=dtype
        )

        # Define the metadata for the output raster
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": "EPSG:4326",  # Assuming the GeoDataFrame is in WGS84 (EPSG:4326)
            "transform": transform,
            "nodata": fill_value,
            "compress": "DEFLATE",
            "tiled": True
        }

        # Write the rasterized data to the output file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)

        logging.info(f"Rasterized shapefile saved to {output_raster_path}")

    except Exception as e:
        logging.error(f"Error rasterizing shapefile: {e}")


def get_hansen_tiles(index_shapefile_prefix, local_temp_dir, s3_bucket_name):
    """
    Get Hansen tile IDs and their bounds from an index shapefile.

    Args:
        index_shapefile_prefix (str): S3 prefix (without extension) of the tile index shapefile.
        local_temp_dir (str): Local directory to download the shapefile.
        s3_bucket_name (str): Name of the S3 bucket.

    Returns:
        dict: Dictionary with tile IDs as keys and bounds as values.
    """
    # Use existing functions to download and read the shapefile
    tiles_gdf = read_shapefile_from_s3(index_shapefile_prefix, local_temp_dir, s3_bucket_name)

    tiles = {}
    for idx, row in tiles_gdf.iterrows():
        tile_id = row['tile_id']
        bounds = row.geometry.bounds
        tiles[tile_id] = bounds

    return tiles

def get_tile_ids_from_raster(raster_path, index_shapefile_path):
    """
    Get the tile IDs that intersect with the bounds of the input raster.

    Args:
        raster_path (str): Path to the input raster file.
        index_shapefile_path (str): Path to the global index shapefile containing tile IDs.

    Returns:
        list: List of tile IDs that intersect with the raster bounds.
    """
    try:
        raster = rioxarray.open_rasterio(raster_path)
        raster_bounds = raster.rio.bounds()
        raster_bbox = box(*raster_bounds)
        index_gdf = gpd.read_file(index_shapefile_path)
        intersecting_tiles = index_gdf[index_gdf.geometry.intersects(raster_bbox)]
        tile_ids = intersecting_tiles["tile_id"].tolist()
    except Exception as e:
        logging.error(f"Error getting tile IDs from raster: {e}")
        tile_ids = []
    return tile_ids

def get_tile_bounds(index_shapefile, tile_id):
    """
    Retrieve the bounds of a specific tile from the global index shapefile.

    Args:
        index_shapefile (str): Path to the global index shapefile.
        tile_id (str): Tile ID to look for.

    Returns:
        tuple: Bounding box of the tile (minx, miny, maxx, maxy).
    """
    try:
        gdf = gpd.read_file(index_shapefile)
        tile = gdf[gdf['tile_id'] == tile_id]
        if tile.empty:
            logging.error(f"Tile {tile_id} not found in index shapefile.")
            return None
        bounds = tile.total_bounds
        return bounds
    except Exception as e:
        logging.error(f"Error retrieving tile bounds for {tile_id}: {e}")
        return None

# -------------------- Dask Utilities --------------------

def shutdown_dask_clients():
    """
    Shut down all active Dask clients on the local machine.

    Returns:
        None
    """
    clients = Client._instances.copy()
    for client in clients:
        try:
            client.shutdown()
        except Exception as e:
            logging.error(f"Error shutting down client: {e}")
    logging.info("All Dask clients have been shut down.")

def close_dask_clusters():
    """
    Close all active Dask clusters on the local machine.

    Returns:
        None
    """
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if 'dask-scheduler' in proc.info['name'] or 'dask-worker' in proc.info['name']:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                p.wait(timeout=5)
                logging.info(f"Terminated {proc.info['name']} with PID {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                logging.error(f"Failed to terminate {proc.info['name']} with PID {proc.info['pid']}: {e}")

def setup_coiled_cluster():
    """
    Set up a Coiled Dask cluster.

    Returns:
        tuple: Dask client and cluster objects.
    """
    coiled_cluster = coiled.Cluster(
        n_workers=40,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="test_coiled_connection",
        account='wri-forest-research',
        worker_memory="32GiB"
    )
    coiled_client = coiled_cluster.get_client()
    return coiled_client, coiled_cluster

# -------------------- Miscellaneous Utilities --------------------

def delete_local_directory(directory_path):
    """
    Delete a local directory and all its contents.

    Args:
        directory_path (str): Path to the directory to delete.

    Returns:
        None
    """
    try:
        if os.path.exists(directory_path):
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory_path)
            logging.info(f"Deleted directory: {directory_path}")
    except Exception as e:
        logging.error(f"Error deleting directory {directory_path}: {e}")

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Path to the directory to create.

    Returns:
        None
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory ensured: {directory_path}")
    except Exception as e:
        logging.error(f"Error creating directory {directory_path}: {e}")

def get_raster_files_from_local(directory):
    """
    Get a list of raster files from a local directory.

    Args:
        directory (str): The local directory path.

    Returns:
        list: List of raster file paths.
    """
    raster_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):
                raster_files.append(os.path.join(root, file))
    return raster_files

def get_raster_files_from_s3(s3_directory):
    """
    Get a list of raster files from an S3 directory.

    Args:
        s3_directory (str): The S3 directory path.

    Returns:
        list: List of raster file paths in S3.
    """
    bucket, prefix = s3_directory.replace("s3://", "").split("/", 1)
    raster_files = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.tif'):
                    raster_files.append(f"s3://{bucket}/{obj['Key']}")
    except Exception as e:
        logging.error(f"Error retrieving raster files from S3: {e}")
    return raster_files

def create_tile_index_from_local(local_directories, output_dir):
    """
    Create tile index shapefiles from local directories.

    Args:
        local_directories (dict): Dictionary of dataset names and local directory paths.
        output_dir (str): Directory where the shapefiles will be saved.

    Returns:
        None
    """
    for dataset_name, local_directory in local_directories.items():
        tile_index = []
        raster_files = get_raster_files_from_local(local_directory)
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                bounds = src.bounds
                tile_id = os.path.basename(raster_file)
                geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                tile_index.append({'tile_id': tile_id, 'geometry': geometry})

        gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:4326")
        output_shapefile = os.path.join(output_dir, f"{dataset_name}_tile_index.shp")
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        logging.info(f"Tile index shapefile created at {output_shapefile}")

def create_tile_index_from_s3(s3_directories, output_dir):
    """
    Create tile index shapefiles from S3 directories.

    Args:
        s3_directories (dict): Dictionary of dataset names and S3 directory paths.
        output_dir (str): Directory where the shapefiles will be saved.

    Returns:
        None
    """
    for dataset_name, s3_directory in s3_directories.items():
        tile_index = []
        raster_files = get_raster_files_from_s3(s3_directory)
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                bounds = src.bounds
                tile_id = os.path.basename(raster_file)
                geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                tile_index.append({'tile_id': tile_id, 'geometry': geometry})

        gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:4326")
        output_shapefile = os.path.join(output_dir, f"{dataset_name}_tile_index.shp")
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        logging.info(f"Tile index shapefile created at {output_shapefile}")

def list_tile_ids(bucket, prefix, pattern=r"(\d{2}[NS]_\d{3}[EW])_peat_mask_processed\.tif"):
    """
    List all tile IDs in a specified S3 directory matching a given pattern.

    Args:
        bucket (str): The S3 bucket name.
        prefix (str): The prefix path in the S3 bucket.
        pattern (str, optional): Regex pattern to match tile IDs. Defaults to a specific peat mask pattern.

    Returns:
        list: List of tile IDs.
    """
    keys = []
    tile_ids = set()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])

        # Extract tile IDs from filenames
        for key in keys:
            match = re.match(pattern, os.path.basename(key))
            if match:
                tile_ids.add(match.group(1))
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")
    return list(tile_ids)

def get_chunk_bounds(minx, miny, maxx, maxy, chunk_size):
    """
    Divide a bounding box into smaller chunks of the specified size.

    Args:
        minx (float): Minimum x-coordinate of the bounding box.
        miny (float): Minimum y-coordinate of the bounding box.
        maxx (float): Maximum x-coordinate of the bounding box.
        maxy (float): Maximum y-coordinate of the bounding box.
        chunk_size (float): Size of each chunk.

    Returns:
        list: List of shapely.geometry.Polygon objects representing the chunks.
    """
    chunks = []
    x_coords = list(range(int(minx), int(maxx), chunk_size))
    y_coords = list(range(int(miny), int(maxy), chunk_size))
    for x in x_coords:
        for y in y_coords:
            chunks.append(box(x, y, x + chunk_size, y + chunk_size))
    return chunks

def export_chunks_to_shapefile(chunk_params, output_filename):
    """
    Export chunk bounds to a shapefile.

    Args:
        chunk_params (tuple): Tuple containing minx, miny, maxx, maxy, and chunk_size.
        output_filename (str): Path to the output shapefile.

    Returns:
        None
    """
    try:
        x_min, y_min, x_max, y_max, chunk_size = chunk_params

        x_range = int((x_max - x_min) / chunk_size)
        y_range = int((y_max - y_min) / chunk_size)

        polygons = []
        for i in range(x_range):
            for j in range(y_range):
                x_start = x_min + i * chunk_size
                y_start = y_min + j * chunk_size
                x_end = x_start + chunk_size
                y_end = y_start + chunk_size
                polygons.append(Polygon([
                    (x_start, y_start),
                    (x_end, y_start),
                    (x_end, y_end),
                    (x_start, y_end)
                ]))

        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
        gdf.to_file(output_filename, driver='ESRI Shapefile')

        logging.info(f"Chunk bounds exported to {output_filename}")

    except Exception as e:
        logging.error(f"Error exporting chunks to shapefile: {e}")

# -------------------- Hansenize Function --------------------

def hansenize(
    input_path,
    output_raster_path,
    bounds,
    s3_bucket,
    s3_prefix,
    run_mode='default',
    nodata_value=None
):
    """
    Processes input shapefiles or rasters to produce a 30-meter resolution raster
    clipped into homogeneous 10x10 degree tiles.

    Args:
        input_path (str): Path to the input shapefile or raster.
        output_raster_path (str): Path to the output raster file.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) to clip the raster.
        s3_bucket (str): S3 bucket name for uploading results.
        s3_prefix (str): S3 prefix for saving results.
        run_mode (str): Mode to run the script ('default' or 'test'). Defaults to 'default'.
        nodata_value (float, optional): NoData value for the output raster. Defaults to None.

    Returns:
        None
    """
    try:
        minx, miny, maxx, maxy = bounds

        # Construct gdalwarp command
        gdalwarp_cmd = [
            'gdalwarp',
            '-t_srs', 'EPSG:4326',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-tr', '0.00025', '0.00025',
            '-tap',
            '-r', 'near',
            '-dstnodata', str(nodata_value) if nodata_value is not None else '-inf',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite',
            input_path,
            output_raster_path
        ]

        logging.info(f"Running gdalwarp command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)

        # Log data statistics
        with rasterio.open(output_raster_path) as output:
            data = output.read(1)
            logging.info(f"Processed raster stats - Min: {data.min()}, Max: {data.max()}, NoData: {output.nodata}")

        # Upload to S3 if not in test mode
        if run_mode != 'test':
            s3_key = os.path.join(s3_prefix, os.path.basename(output_raster_path)).replace("\\", "/")
            upload_file_to_s3(output_raster_path, s3_bucket, s3_key)
            delete_file_if_exists(output_raster_path)

        gc.collect()

    except Exception as e:
        logging.error(f"Error in hansenize function: {e}")

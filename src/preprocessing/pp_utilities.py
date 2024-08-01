import os
import boto3
import re
import rasterio
import geopandas as gpd
import psutil
import rioxarray
from shapely.geometry import box
from dask.distributed import Client
import subprocess
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import gc
import logging
from rasterio.features import rasterize

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup with increased max connections and retries
config = boto3.session.Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)
s3_client = boto3.client('s3', config=config)

def s3_file_exists(bucket, key):
    """
    Check if a file exists in an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The S3 key (path) of the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        logging.info(f"File exists: s3://{bucket}/{key}")
        return True
    except:
        logging.info(f"File does not exist: s3://{bucket}/{key}")
        return False

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
    input_raster = rioxarray.open_rasterio(input_path, masked=True)
    reference_raster = rioxarray.open_rasterio(reference_path, masked=True)

    log_raster_properties(input_raster, "Input Raster")
    log_raster_properties(reference_raster, "Reference Raster")

    clipped_resampled_raster = input_raster.rio.clip_box(*reference_raster.rio.bounds())
    clipped_resampled_raster = clipped_resampled_raster.rio.reproject_match(reference_raster)

    clipped_resampled_raster.rio.to_raster(output_path)

    if os.path.exists(output_path):
        logging.info(f"Successfully saved resampled raster to {output_path}")
    else:
        logging.error(f"Failed to save resampled raster to {output_path}")

def compress_and_upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    """
    Compress and upload all files in a local directory to an S3 bucket.

    Args:
        local_directory (str): Path to the local directory containing files.
        s3_bucket (str): The name of the S3 bucket.
        s3_prefix (str): The prefix (path) in the S3 bucket where files will be uploaded.

    Returns:
        None
    """
    existing_files = get_existing_s3_files(s3_bucket, s3_prefix)
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            compressed_file_path = os.path.join(root, f"compressed_{file}")
            s3_file_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, s3_file_path).replace("\\", "/")
            if s3_key in existing_files:
                logging.info(f"File {s3_key} already exists in S3. Skipping upload.")
            else:
                try:
                    logging.info(f"Compressing {local_file_path}")
                    compress_file(local_file_path, compressed_file_path)
                    logging.info(f"Uploading {compressed_file_path} to s3://{s3_bucket}/{s3_key}")
                    s3_client.upload_file(compressed_file_path, s3_bucket, s3_key)
                    logging.info(f"Successfully uploaded {compressed_file_path} to s3://{s3_bucket}/{s3_key}")
                    os.remove(compressed_file_path)
                except (NoCredentialsError, PartialCredentialsError) as e:
                    logging.error(f"Credentials error: {e}")
                    return
                except Exception as e:
                    logging.error(f"Failed to upload {local_file_path} to s3://{s3_bucket}/{s3_key}: {e}")

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
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                existing_files.add(obj['Key'])
    return existing_files

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
            profile = src.profile
            nodata = src.nodata
            if nodata_value is not None:
                nodata = nodata_value
            elif nodata is None:
                nodata = 3.4028234663852886e+38  # Default NoData value, change as needed

        subprocess.run(
            ['gdal_translate',
             '-co', 'COMPRESS=LZW',
             '-co', 'TILED=YES',
             '-a_nodata', str(nodata),
             '-stats',
             input_file, output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {input_file}: {e}")

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

def list_tile_ids(bucket, prefix):
    """
    List all tile IDs in a specified S3 directory.

    Args:
        bucket (str): The S3 bucket name.
        prefix (str): The prefix path in the S3 bucket.

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
            match = re.match(r"(\d{2}[NS]_\d{3}[EW])_peat_mask_processed\.tif", key.split('/')[-1])
            if match:
                tile_ids.add(match.group(1))

    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")

    return list(tile_ids)

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

def get_raster_files_from_s3(s3_directory):
    """
    Get a list of raster files from an S3 directory.

    Args:
        s3_directory (str): The S3 directory path.

    Returns:
        list: List of raster file paths in S3.
    """
    bucket, prefix = s3_directory.replace("s3://", "").split("/", 1)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    raster_files = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.tif'):
                raster_files.append(f"s3://{bucket}/{obj['Key']}")
    return raster_files

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

def get_tile_ids_from_raster(raster_path, index_shapefile_path):
    """
    Get the tile IDs that intersect with the bounds of the input raster.

    Args:
        raster_path (str): Path to the input raster file.
        index_shapefile_path (str): Path to the global index shapefile containing tile IDs.

    Returns:
        list: List of tile IDs that intersect with the raster bounds.
    """
    # Load the raster and get its bounds
    raster = rioxarray.open_rasterio(raster_path)
    raster_bounds = raster.rio.bounds()

    # Create a bounding box from the raster bounds
    raster_bbox = box(*raster_bounds)

    # Load the global index shapefile
    index_gdf = gpd.read_file(index_shapefile_path)

    # Find the tiles that intersect with the raster bounding box
    intersecting_tiles = index_gdf[index_gdf.geometry.intersects(raster_bbox)]

    # Get the tile IDs from the intersecting tiles
    tile_ids = intersecting_tiles["tile_id"].tolist()

    return tile_ids

def get_chunk_bounds(minx, miny, maxx, maxy, chunk_size):
    """
    Divide a bounding box into smaller chunks of the specified size.

    Args:
        minx (float): Minimum x-coordinate of the bounding box.
        miny (float): Minimum y-coordinate of the bounding box.
        maxx (float): Maximum x-coordinate of the bounding box.
        maxy (float): Maximum y-coordinate of the bounding box.
        chunk_size (int): Size of each chunk.

    Returns:
        list: List of polygons representing the chunks.
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
    minx, miny, maxx, maxy, chunk_size = chunk_params
    chunks = get_chunk_bounds(minx, miny, maxx, maxy, chunk_size)

    gdf = gpd.GeoDataFrame(geometry=chunks, crs="EPSG:4326")
    gdf.to_file(output_filename, driver='ESRI Shapefile')
    logging.info(f"Chunk bounds exported to {output_filename}")

def get_tile_bounds(index_shapefile, tile_id):
    """
    Retrieve the bounds of a specific tile from the global index shapefile.

    Args:
        index_shapefile (str): Path to the global index shapefile.
        tile_id (str): Tile ID to look for.

    Returns:
        tuple: Bounding box of the tile (minx, miny, maxx, maxy).
    """
    gdf = gpd.read_file(index_shapefile)
    tile = gdf[gdf['tile_id'] == tile_id]

    if tile.empty:
        logging.error(f"Tile {tile_id} not found in index shapefile.")
        return None

    bounds = tile.total_bounds
    return bounds

def compress_and_upload_file_to_s3(local_file, s3_bucket, s3_key):
    """
    Compress a local file using LZW compression and upload it to S3.

    Args:
        local_file (str): Path to the local file to be compressed and uploaded.
        s3_bucket (str): S3 bucket name.
        s3_key (str): S3 key for the uploaded file.

    Returns:
        None
    """
    compressed_file = local_file.replace('.tif', '_compressed.tif')
    try:
        subprocess.run(
            ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', local_file, compressed_file],
            check=True
        )
        logging.info(f"Uploading {compressed_file} to s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(compressed_file, s3_bucket, s3_key)
        logging.info(f"Uploaded {compressed_file} to s3://{s3_bucket}/{s3_key}")
        os.remove(compressed_file)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {local_file}: {e}")

def delete_file_if_exists(file_path):
    """
    Delete a file if it exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        None
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        logging.info(f"Deleted existing file: {file_path}")

def hansenize(input_path, output_dir, reference_raster_path, s3_bucket, s3_prefix, tile_id, dataset, run_mode='default', nodata_value=None):
    """
    Process input shapefiles or rasters to produce a 30-meter resolution raster
    clipped into homogeneous 10x10 degree tiles.

    Args:
        input_path (str): Path to the input shapefile or raster.
        output_dir (str): Directory to save the output files.
        reference_raster_path (str): Path to the reference raster for alignment.
        s3_bucket (str): S3 bucket name for uploading results.
        s3_prefix (str): S3 prefix for saving results.
        tile_id (str): ID of the tile being processed.
        dataset (str): Dataset name (e.g., 'dadap' or 'engert').
        run_mode (str): Mode to run the script ('default' or 'test').
        nodata_value (float, optional): NoData value for the output raster.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    output_raster_path = os.path.join(output_dir, f"{dataset}_{tile_id}.tif")
    compressed_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}_compressed.tif")

    try:
        with rasterio.open(reference_raster_path) as ref_ds:
            ref_bounds = ref_ds.bounds
            ref_transform = ref_ds.transform
            ref_crs = ref_ds.crs
            ref_width = ref_ds.width
            ref_height = ref_ds.height

        # Remove the existing output file if it exists
        if os.path.exists(output_raster_path):
            os.remove(output_raster_path)
            logging.info(f"Deleted existing file: {output_raster_path}")

        # Run gdalwarp to reproject and clip the input raster
        gdalwarp_cmd = [
            'gdalwarp',
            '-t_srs', 'EPSG:4326',
            '-te', str(ref_bounds.left), str(ref_bounds.bottom), str(ref_bounds.right), str(ref_bounds.top),
            '-tr', '0.00025', '0.00025',
            '-tap',
            '-r', 'near',
            '-dstnodata', str(nodata_value) if nodata_value else 'None',
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            input_path,
            output_raster_path
        ]

        logging.info(f"Running gdalwarp command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)

        # Compress the output raster
        compress_file(output_raster_path, compressed_output_path)
        logging.info(f"Compressed raster saved to {compressed_output_path}")

        # Upload to S3 if not in test mode
        if run_mode != 'test':
            s3_key = os.path.join(s3_prefix, os.path.basename(compressed_output_path)).replace("\\", "/")
            logging.info(f"Uploading {compressed_output_path} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(compressed_output_path, s3_bucket, s3_key)
            logging.info(f"Uploaded {compressed_output_path} to s3://{s3_bucket}/{s3_key}")

            # Delete local files if not in test mode
            os.remove(output_raster_path)
            os.remove(compressed_output_path)

        gc.collect()

    except Exception as e:
        logging.error(f"Error in hansenize function: {e}")


# AWS S3 setup with increased max connections
config = boto3.session.Config(
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    },
    max_pool_connections=50
)
s3_client = boto3.client('s3', config=config)

def download_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name):
    """
    Download the shapefile and its associated files from S3 to a local directory.

    Parameters:
    s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
    local_dir (str): The local directory where the files will be saved.
    s3_bucket_name (str): The name of the S3 bucket.

    Returns:
    None
    """
    extensions = ['.shp', '.shx', '.dbf', '.prj']
    for ext in extensions:
        s3_path = f"{s3_prefix}{ext}"
        local_path = os.path.join(local_dir, os.path.basename(s3_prefix) + ext)
        try:
            s3_client.download_file(s3_bucket_name, s3_path, local_path)
        except Exception as e:
            logging.error(f"Error downloading {s3_path}: {e}")

def read_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name):
    """
    Read a shapefile from S3 into a GeoDataFrame.

    Parameters:
    s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
    local_dir (str): The local directory where the files will be saved.
    s3_bucket_name (str): The name of the S3 bucket.

    Returns:
    gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    download_shapefile_from_s3(s3_prefix, local_dir, s3_bucket_name)
    shapefile_path = os.path.join(local_dir, os.path.basename(s3_prefix) + '.shp')
    gdf = gpd.read_file(shapefile_path)
    return gdf

def rasterize_shapefile(gdf, tile_bounds, tile_transform, tile_width, tile_height):
    """
    Rasterize a shapefile to create a raster where features are assigned a value of 1.

    Parameters:
    gdf (gpd.GeoDataFrame): The GeoDataFrame to rasterize.
    tile_bounds (tuple): The bounding box of the tile.
    tile_transform (affine.Affine): The affine transform for the tile.
    tile_width (int): The width of the tile in pixels.
    tile_height (int): The height of the tile in pixels.

    Returns:
    np.ndarray: The rasterized data.
    """
    shapes = [(geom, 1) for geom in gdf.geometry if isinstance(geom, Polygon)]
    if not shapes:
        logging.error("No valid geometries found for rasterization.")
        return None
    raster = rasterize(
        shapes,
        out_shape=(tile_height, tile_width),
        transform=tile_transform,
        fill=0,
        dtype=rasterio.uint8
    )
    return raster


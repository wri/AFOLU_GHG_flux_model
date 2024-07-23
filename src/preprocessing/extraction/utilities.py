# utils.py
import geopandas as gpd
import boto3
import logging
import os
import subprocess
import rasterio
from rasterio.features import rasterize

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

def compress_file(input_file, output_file):
    """
    Compress a GeoTIFF file using LZW compression.

    Parameters:
    input_file (str): Path to the input GeoTIFF file.
    output_file (str): Path to save the compressed GeoTIFF file.

    Returns:
    None
    """
    try:
        subprocess.run(
            ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', input_file, output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error compressing file {input_file}: {e}")

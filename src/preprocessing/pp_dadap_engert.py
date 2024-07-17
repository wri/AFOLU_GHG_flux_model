import os
import rioxarray
import boto3
import logging
import time
import subprocess
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
import gc
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box
import geopandas as gpd
from rasterio.vrt import WarpedVRT
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

"""
Script is working well, but something is wrong with Engert units. Working to resolve.
Also need to add documentation when finished 
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'

# Local paths
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
raw_rasters = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/engert_roads/engert_asiapac_ghrdens_1km_resample_30m.tif',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/canals/Dadap_SEA_Drainage/canal_length_data/canal_length_1km_resample_30m.tif'
}
output_prefixes = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m'
}

def delete_local_file(file_path):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            logging.info(f"Deleted local file {file_path}")
            break
        except PermissionError as e:
            if attempt < max_attempts - 1:
                logging.warning(f"Failed to delete {file_path}, retrying... ({e})")
                time.sleep(1)
            else:
                logging.error(f"Could not delete {file_path} after several attempts. ({e})")

def compress_file(input_file, output_file, nodata_value=None):
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

def get_existing_s3_files(s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
    existing_files = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                existing_files.add(obj['Key'])
    return existing_files

def compress_and_upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
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

@delayed
def process_tile(tile_key, dataset, run_mode='default'):
    output_dir = local_temp_dir if run_mode == 'test' else '/tmp/output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    s3_output_dir = output_prefixes[dataset]
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}.tif")
    compressed_output_path = os.path.join(output_dir, f"compressed_{dataset}_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}/{dataset}_{tile_id}.tif"

    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except:
            logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'
    raw_raster_path = f'/vsis3/{s3_bucket_name}/{raw_rasters[dataset]}'

    try:
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_path) as tile_ds:
                tile_bounds = tile_ds.bounds
                logging.info(f"Tile bounds: {tile_bounds}")

                logging.info(f"Raw raster path: {raw_raster_path}")
                with rasterio.open(raw_raster_path) as raw_ds:
                    raw_bounds = raw_ds.bounds
                    logging.info(f"Raw raster bounds: {raw_bounds}")

                    tile_crs = tile_ds.crs

                    # Create a WarpedVRT to reproject the raw raster to the tile's CRS
                    with WarpedVRT(raw_ds, crs=tile_crs, transform=tile_ds.transform, width=tile_ds.width, height=tile_ds.height) as vrt:
                        reprojected_raw = vrt.read(
                            out_shape=(vrt.count, tile_ds.height, tile_ds.width),
                            resampling=Resampling.nearest
                        )

                        # Convert the reprojected data back to a rioxarray DataArray
                        reprojected_data = rioxarray.open_rasterio(vrt)

                        # Clip the reprojected raw raster to the tile bounds
                        clipped_raster = reprojected_data.rio.clip_box(*tile_bounds)

                        # Save the clipped raster
                        clipped_raster.rio.to_raster(local_output_path)

                        # Compress the output
                        compress_file(local_output_path, compressed_output_path)
                        logging.info(f"Compressed and saved locally {compressed_output_path}")

                        if run_mode == 'test':
                            logging.info(f"Test mode: Outputs saved locally at {compressed_output_path}")
                        else:
                            logging.info(f"Uploading {compressed_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
                            s3_client.upload_file(compressed_output_path, s3_bucket_name, s3_output_path)
                            logging.info(f"Uploaded {compressed_output_path} to s3://{s3_bucket_name}/{s3_output_path}")
                            os.remove(local_output_path)
                            os.remove(compressed_output_path)

                        del reprojected_data, clipped_raster
                        gc.collect()
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(dataset, run_mode='default'):
    paginator = boto3.client('s3').get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_tiles_prefix)
    tile_keys = []

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith('_peat_mask_processed.tif'):
                    tile_keys.append(tile_key)

    delayed_tasks = [process_tile(tile_key, dataset, run_mode) for tile_key in tile_keys]
    with ProgressBar():
        dask.compute(*delayed_tasks)

def main(tile_id=None, dataset='engert', run_mode='default'):
    try:
        if tile_id:
            tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
            process_tile(tile_key, dataset, run_mode).compute()
        else:
            process_all_tiles(dataset, run_mode)

        if run_mode != 'test':
            compress_and_upload_directory_to_s3('/tmp/output', s3_bucket_name, output_prefixes[dataset])
    finally:
        logging.info("Processing completed")

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    main(tile_id='00N_110E', dataset='engert', run_mode='test')
    main(tile_id='00N_110E', dataset='dadap', run_mode='test')

    # Process datasets separately
    # main(dataset='engert', run_mode='default')
    # main(dataset='dadap', run_mode='default')

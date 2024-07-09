import os
import logging
import boto3
import rasterio
from rasterio.merge import merge as merge_arrays
import dask
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import atexit
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_bucket = "gfw2-data"
local_temp_dir = "/tmp/aggregated"

# Global variables for Dask cluster and client
cluster = None
client = None

def s3_file_exists(bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=key)
        logging.info(f"File exists: s3://{bucket}/{key}")
        return True
    except:
        logging.info(f"File does not exist: s3://{bucket}/{key}")
        return False

def list_s3_files(bucket, prefix):
    s3 = boto3.client('s3')
    keys = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")
    return keys

def aggregate_tile(tile_path, output_path):
    logging.info(f"Aggregating tile {tile_path} to {output_path}")
    with rasterio.open(tile_path) as src:
        data = src.read(1, masked=True)
        transform = src.transform
        profile = src.profile

        # Resample to 4km resolution
        scale_factor = 30 / 4000  # 30m to 4km
        new_height = int(data.shape[0] * scale_factor)
        new_width = int(data.shape[1] * scale_factor)
        resampled_data = src.read(
            out_shape=(1, new_height, new_width),
            resampling=rasterio.enums.Resampling.average
        )

        # Update profile
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": rasterio.Affine(
                transform.a / scale_factor,
                transform.b,
                transform.c,
                transform.d,
                transform.e / scale_factor,
                transform.f
            )
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(resampled_data, 1)
    logging.info(f"Finished aggregating tile {tile_path}")

def merge_global_tiles(tiles, output_path):
    logging.info(f"Merging global tiles into {output_path} using gdal_merge.py")
    command = ["gdal_merge.py", "-o", output_path] + tiles
    try:
        subprocess.run(command, check=True)
        logging.info(f"Finished merging global tiles into {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error merging global tiles: {e}")

def cleanup():
    global client, cluster
    logging.info("Cleaning up Dask resources")
    if client:
        client.close()
    if cluster:
        cluster.close()

def process_and_aggregate(tile_id, input_prefix, temp_prefix, output_prefix):
    s3_client = boto3.client('s3')
    input_file = os.path.join(input_prefix, f'{tile_id}_soil.tif').replace('\\', '/')
    local_input_path = os.path.join(local_temp_dir, f'{tile_id}.tif')
    temp_output_file = os.path.join(temp_prefix, f'{tile_id}_soil_4km.tif').replace('\\', '/')
    local_temp_output_path = os.path.join(local_temp_dir, f'{tile_id}_4km.tif')

    if not os.path.exists(local_temp_dir):
        os.makedirs(local_temp_dir)

    logging.info(f"Downloading tile from s3://{s3_bucket}/{input_file} to {local_input_path}")
    try:
        s3_client.download_file(s3_bucket, input_file, local_input_path)
    except Exception as e:
        logging.error(f"Failed to download {input_file}: {e}")
        return

    logging.info(f"Aggregating tile {tile_id}")
    aggregate_tile(local_input_path, local_temp_output_path)
    logging.info(f"Uploading aggregated tile {temp_output_file} to S3")
    s3_client.upload_file(local_temp_output_path, s3_bucket, temp_output_file)

    os.remove(local_input_path)
    os.remove(local_temp_output_path)
    logging.info(f"Finished processing tile {tile_id}")

def main(input_prefix, temp_prefix, output_prefix, tile_ids=None):
    global cluster, client
    logging.info("Initializing Dask LocalCluster")
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, heartbeat_interval="1m")
    client = Client(cluster)
    atexit.register(cleanup)
    logging.info(f"Dask LocalCluster initialized with {len(cluster.workers)} workers")

    try:
        available_tile_ids = list_s3_files(s3_bucket, input_prefix)
        logging.info(f"Available tile IDs: {available_tile_ids}")
        available_tile_ids = [os.path.basename(path).replace('.tif', '').replace('_soil', '') for path in available_tile_ids]

        if tile_ids:
            # Ensure provided tile IDs have the correct format
            tile_ids_to_process = [tile_id for tile_id in tile_ids if tile_id in available_tile_ids]
        else:
            tile_ids_to_process = available_tile_ids

        logging.info(f"Tile IDs to process: {tile_ids_to_process}")

        tasks = [dask.delayed(process_and_aggregate)(tile_id, input_prefix, temp_prefix, output_prefix) for tile_id in tile_ids_to_process]
        logging.info(f"Created Dask tasks for {len(tasks)} tiles")
        with ProgressBar():
            dask.compute(*tasks)

        # Merge all the 4km tiles into one global raster
        aggregated_tiles = list_s3_files(s3_bucket, temp_prefix)
        aggregated_tiles = [os.path.join(local_temp_dir, os.path.basename(tile)) for tile in aggregated_tiles]
        logging.info(f"Aggregated tiles: {aggregated_tiles}")

        for tile in aggregated_tiles:
            logging.info(f"Downloading aggregated tile {tile}")
            s3_client.download_file(s3_bucket, tile, tile)

        if aggregated_tiles:
            local_global_path = os.path.join(local_temp_dir, 'global_4km.tif')
            logging.info(f"Merging global tiles into {local_global_path}")
            merge_global_tiles(aggregated_tiles, local_global_path)
            s3_global_output = os.path.join(output_prefix, 'global_4km.tif').replace('\\', '/')
            logging.info(f"Uploading global raster to S3: {s3_global_output}")
            s3_client.upload_file(local_global_path, s3_bucket, s3_global_output)
            os.remove(local_global_path)
            logging.info(f"Uploaded global raster to s3://{s3_bucket}/{s3_global_output}")
    except Exception as e:
        logging.error(f"Error during processing: {e}")
    finally:
        logging.info("Exiting")
        print("exit")

if __name__ == "__main__":
    input_prefix = 'climate/AFOLU_flux_model/organic_soils/outputs/soil/2020/10x10_degrees/20240607/'
    temp_prefix = 'climate/AFOLU_flux_model/organic_soils/outputs/soil/2020/4km_temp/20240607'
    output_prefix = 'climate/AFOLU_flux_model/organic_soils/outputs/soil/2020/global/20240607'

    tile_ids = ['00N_030E', '00N_040E', '00N_040W']  # Replace with the list of tile IDs you want to process
    main(input_prefix, temp_prefix, output_prefix, tile_ids)

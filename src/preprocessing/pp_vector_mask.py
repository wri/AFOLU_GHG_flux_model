import os
import logging
import subprocess
import rasterio
import boto3
import pp_utilities as uu
import constants_and_names as cn

"""
This script reads all tiles in peat_tiles_prefix, resamples them to 0.01 degrees resolution using gdalwarp,
and uploads them to peat_tiles_prefix_1km, while maintaining the same NoData value as the input raster.
TODO: Update this script so that any 1 km cell which intersects with a 30 meter cell is counted
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')

def get_nodata_value(raster_path):
    """
    Retrieve the NoData value from the input raster.

    Parameters:
    raster_path (str): Path to the raster file.

    Returns:
    float: The NoData value of the raster.
    """
    with rasterio.open(raster_path) as src:
        nodata_value = src.nodata
    return nodata_value

def resample_tile(tile_key, tile_id, run_mode='default'):
    """
    Resample a single tile to 0.01 degrees resolution using gdalwarp and upload it to S3.

    Parameters:
    tile_key (str): S3 key of the tile to resample.
    tile_id (str): The tile ID.
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    # Set output directory paths
    output_dir = os.path.join(cn.local_temp_dir, "resampled_tiles")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{tile_id}_peat_mask_processed.tif"
    local_output_path = os.path.join(output_dir, output_filename)
    s3_output_path = f"{cn.peat_tiles_prefix_1km}{output_filename}".replace("\\", "/")

    if run_mode != 'test':
        # Check if the file already exists on S3
        try:
            s3_client.head_object(Bucket=cn.s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping resampling.")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info(f"{s3_output_path} does not exist on S3. Resampling the tile.")
            else:
                logging.error(f"Error checking existence of {s3_output_path} on S3: {e}")
                return

    # Resample the tile using gdalwarp
    raw_raster_path = f"/vsis3/{cn.s3_bucket_name}/{tile_key}"
    nodata_value = get_nodata_value(raw_raster_path)

    gdalwarp_cmd = [
        'gdalwarp',
        '-tr', '0.01', '0.01',  # Target resolution
        '-tap',  # Align the target resolution to the pixel grid
        '-r', 'near',  # Resampling method
        '-co', 'COMPRESS=DEFLATE',  # Compression
        '-co', 'TILED=YES',  # Tiling
        '-dstnodata', str(nodata_value) if nodata_value is not None else '-9999',  # NoData value
        raw_raster_path,
        local_output_path
    ]

    try:
        logging.info(f"Running gdalwarp command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)

        # Upload the resampled file to S3
        if run_mode != 'test':
            uu.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)

        logging.info(f"Resampling and upload of tile {tile_id} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during gdalwarp processing of tile {tile_id}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during resampling of tile {tile_id}: {e}")

    # Cleanup
    if os.path.exists(local_output_path):
        os.remove(local_output_path)

def process_all_tiles(run_mode='default'):
    """
    Process all tiles in peat_tiles_prefix, resampling them to 0.01 degrees resolution.

    Parameters:
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    try:
        # List all tiles in the peat_tiles_prefix
        tile_keys = uu.list_s3_files(cn.s3_bucket_name, cn.peat_tiles_prefix)
        logging.info(f"Found {len(tile_keys)} tiles to process.")

        for tile_key in tile_keys:
            tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
            resample_tile(tile_key, tile_id, run_mode)

    except Exception as e:
        logging.error(f"Error processing all tiles: {e}")

def main(run_mode='default'):
    """
    Main function to orchestrate the processing.

    Parameters:
    run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
    None
    """
    try:
        process_all_tiles(run_mode)

    finally:
        logging.info("Resampling processing completed")

# Example usage
if __name__ == "__main__":
    main(run_mode='default')

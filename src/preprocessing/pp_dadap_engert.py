import os
import logging
import gc
import boto3
import pp_utilities as uu
import constants_and_names as cn

"""
This script processes raster tiles by resampling them to a specified resolution,
clipping them to tile bounds, and uploading the processed tiles to S3.
This script is not currently using dask but I plan to set up a version that uses dask.
TODO: solve weird statistics problem
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')

def process_tile(tile_key, dataset, tile_bounds, run_mode='default'):
    """
    Processes a single tile: resamples, reprojects, clips, and uploads to S3.

    Parameters:
    tile_key (str): S3 key of the tile to process.
    dataset (str): The dataset type (e.g., 'engert' or 'dadap').
    tile_bounds (tuple): Bounding box coordinates for the tile.
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    output_dir = cn.local_temp_dir
    os.makedirs(output_dir, exist_ok=True)

    s3_output_dir = cn.output_prefixes[dataset]
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    local_output_path = os.path.join(output_dir, f"{dataset}_{tile_id}.tif")
    s3_output_path = f"{s3_output_dir}/{dataset}_{tile_id}.tif".replace("\\", "/")

    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=cn.s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")
            else:
                logging.error(f"Error checking existence of {s3_output_path} on S3: {e}")
                return

    logging.info(f"Starting processing of the tile {tile_id}")
    raw_raster_path = f'/vsis3/{cn.s3_bucket_name}/{cn.raw_rasters[dataset]}'
    logging.info(f"Processing raw raster: {raw_raster_path}")

    try:
        if tile_bounds is None:
            logging.error(f"Tile bounds not found for {tile_id}")
            return

        uu.hansenize(
            input_path=raw_raster_path,
            output_raster_path=local_output_path,
            bounds=tile_bounds,
            s3_bucket=cn.s3_bucket_name,
            s3_prefix=s3_output_dir,
            run_mode=run_mode
        )

        logging.info("Processing completed")
    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles(dataset, run_mode='default'):
    """
    Processes all tiles for the given dataset.

    Parameters:
    dataset (str): The dataset type (e.g., 'engert' or 'dadap').
    run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
    None
    """
    try:
        # Ensure the index shapefile is downloaded and get the local path
        index_shapefile_path = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        uu.read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)

        # Retrieve the list of tile IDs
        raw_raster_path = f'/vsis3/{cn.s3_bucket_name}/{cn.raw_rasters[dataset]}'
        tile_ids = uu.get_tile_ids_from_raster(raw_raster_path, index_shapefile_path)
        logging.info(f"Processing {len(tile_ids)} tiles for dataset {dataset}")

        for tile_id in tile_ids:
            tile_key = f"{cn.s3_tiles_prefix}{tile_id}{cn.peat_pattern}"
            tile_bounds = uu.get_tile_bounds(index_shapefile_path, tile_id)
            process_tile(tile_key, dataset, tile_bounds, run_mode)
    except Exception as e:
        logging.error(f"Error processing all tiles: {e}")

def main(tile_id=None, dataset='engert', run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
    tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
    dataset (str, optional): The dataset type (e.g., 'engert' or 'dadap'). Defaults to 'engert'.
    run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
    None
    """
    try:
        if tile_id:
            # Manually specify the tile bounds if a specific tile is requested
            index_shapefile_path = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
            uu.read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir, cn.s3_bucket_name)
            tile_bounds = uu.get_tile_bounds(index_shapefile_path, tile_id)
            tile_key = f"{cn.s3_tiles_prefix}{tile_id}{cn.peat_pattern}"
            process_tile(tile_key, dataset, tile_bounds, run_mode)
        else:
            process_all_tiles(dataset, run_mode)

    finally:
        logging.info("Processing completed")

# Example usage
if __name__ == "__main__":
    # Replace '00N_110E' with the tile ID you want to test
    # main(tile_id='00N_110E', dataset='engert', run_mode='test')
    # main(tile_id='00N_110E', dataset='dadap', run_mode='test')

    # Process datasets separately
    main(dataset='engert', run_mode='default')
    main(dataset='dadap', run_mode='default')
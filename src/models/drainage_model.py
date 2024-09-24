# drainage_model.py

import os
import time
import concurrent.futures
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import numpy as np
import logging
import boto3
import rasterio
from numba import jit
import warnings

# Import utility functions
from utils.logging_utils import setup_logging, print_and_log, timestr
from utils.chunk_utils import (
    get_chunk_bounds,
    check_for_tile,
    prepare_to_download_chunk,
    check_chunk_for_data,
    boundstr,
    calc_chunk_length_pixels,
    xy_to_tile_id
)
from utils.processing_utils import (
    accrete_node,
    create_typed_dicts,
)
from utils.cluster_utils import (
    setup_coiled_cluster,
    shutdown_cluster
)

# Define global variables (adjust as needed)
s3_out_dir = "s3://gfw2-data/climate/AFOLU_flux_model/drainage_outputs"
land_cover = "IPCC_basic_classes"
planted_forest_type_layer = "plantation_type"
planted_forest_tree_crop_layer = "planted_forest_tree_crop"

# Set up basic configuration for logging
logger = setup_logging()

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Geometry is in a geographic CRS.*')

def save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, output_dict, is_final):
    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call

    transform = rasterio.transform.from_bounds(*bounds, width=chunk_length_pixels, height=chunk_length_pixels)

    file_info = f'{tile_id}__{bounds_str}'

    # For every output file, saves from array to local raster, then to s3.
    # Can't save directly to s3, unfortunately, so need to save locally first.
    for key, value in output_dict.items():
        logger.info(f"Processing output for key: {key}")
        try:
            data_array = value[0]
            data_type = value[1]
            data_meaning = value[2]
            year_out = value[3]

            logger.info(f"Data type: {data_type}, Data meaning: {data_meaning}, Year out: {year_out}")

            if not is_final:
                logger.info(f"Saving {bounds_str} in {tile_id} for {year_out}: {timestr()}")

            if is_final:
                file_name = f"{file_info}__{key}.tif"
            else:
                file_name = f"{file_info}__{key}__{timestr()}.tif"

            local_file_path = f"/tmp/{file_name}"

            with rasterio.open(local_file_path, 'w', driver='GTiff', width=chunk_length_pixels, height=chunk_length_pixels, count=1,
                               dtype=data_type, crs='EPSG:4326', transform=transform, compress='lzw', blockxsize=400, blockysize=400) as dst:
                dst.write(data_array, 1)

            s3_path = f"{s3_out_dir}/{data_meaning}/{year_out}/{chunk_length_pixels}_pixels/{time.strftime('%Y%m%d')}"
            logger.info(f"Saving output to {s3_path}...")

            if not is_final:
                logger.info(f"Uploading {bounds_str} in {tile_id} for {year_out} to {s3_path}: {timestr()}")

            s3_client.upload_file(local_file_path, "gfw2-data", Key=f"{s3_path}/{file_name}")

            # Deletes the local raster
            os.remove(local_file_path)

            logger.info(f"Successfully processed and uploaded {file_name}")

        except Exception as e:
            logger.error(f"Error processing key {key} with value {value}: {str(e)}")

    logger.info(f"Completed processing for chunk {bounds_str}.")

def calculate_and_upload_drainage(bounds, is_final):
    bounds_str = boundstr(bounds)  # String form of chunk bounds
    tile_id = xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    no_data_val = 255
    logger.info(f"Processing tile {tile_id} with bounds {bounds_str}")

    try:
        # Dictionary of downloaded layers
        download_dict = {
            f"{land_cover}_2020": f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/40000_pixels/20240205/{tile_id}__IPCC_classes_2020.tif",
            planted_forest_type_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/{tile_id}_plantation_type_oilpalm_woodfiber_other.tif",
            planted_forest_tree_crop_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/{tile_id}.tif",
            "peat": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/{tile_id}_peat_mask_processed.tif",
            "dadap": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m/dadap_{tile_id}.tif",
            "engert": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m/engert_{tile_id}.tif",
            "grip": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/30m/grip_density_{tile_id}.tif",
            "osm_roads": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/30m/roads_density_{tile_id}.tif",
            "osm_canals": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/canals_density_{tile_id}.tif"
        }

        # Checks whether tile exists at all. Doesn't try to download chunk if the tile doesn't exist.
        tile_exists = check_for_tile(download_dict, is_final)

        if tile_exists == 0:
            logger.info(f"Tile {tile_id} does not exist. Skipping.")
            return

        logger.info(f"Tile {tile_id} exists. Proceeding with downloading data.")
        futures = prepare_to_download_chunk(bounds, download_dict, no_data_val)

        if not is_final:
            logger.info(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {timestr()}")

        # Waits for requests to come back with data from S3
        layers = {}
        for future in concurrent.futures.as_completed(futures):
            layer = futures[future]
            try:
                result = future.result()
                layers[layer] = result
                logger.info(f"Downloaded data for layer: {layer}")
            except Exception as e:
                logger.error(f"Error downloading layer {layer}: {e}")

        data_in_chunk = check_chunk_for_data(layers, f"{land_cover}_", bounds_str, tile_id, no_data_val, is_final)

        if data_in_chunk == 0:
            logger.info(f"No data in chunk {bounds_str}. Skipping.")
            return

        logger.info(f"Data present in chunk {bounds_str}. Proceeding with processing.")

        # Initializes empty dictionaries for each type
        uint8_dict_layers = {}
        int16_dict_layers = {}
        float32_dict_layers = {}

        for key, array in layers.items():
            logger.info(f"Processing layer {key} with dtype {array.dtype}")
            if array.dtype == np.uint8:
                uint8_dict_layers[key] = array
            elif array.dtype == np.int16:
                int16_dict_layers[key] = array
            elif array.dtype == np.float32:
                float32_dict_layers[key] = array
            else:
                raise TypeError(f"{key} dtype not in list")

        peat_block = uint8_dict_layers["peat"]
        land_cover_block = uint8_dict_layers[f"{land_cover}_2020"]
        planted_forest_type_block = uint8_dict_layers[planted_forest_type_layer]
        # planted_forest_tree_crop_block = uint8_dict_layers[planted_forest_tree_crop_layer]
        dadap_block = float32_dict_layers["dadap"]
        osm_roads_block = float32_dict_layers["osm_roads"]
        osm_canals_block = float32_dict_layers["osm_canals"]
        engert_block = float32_dict_layers["engert"]
        grip_block = float32_dict_layers["grip"]

        logger.info(f"Creating drainage map in {bounds_str} in {tile_id}: {timestr()}")
        soil_block, state_out = process_soil(
            peat_block, land_cover_block, planted_forest_type_block, dadap_block, osm_roads_block, osm_canals_block, engert_block, grip_block
        )

        out_dict_uint32 = {
            "soil": soil_block,
            "state": state_out
        }

        out_dict_all_dtypes = {}

        for key, value in out_dict_uint32.items():
            data_type = value.dtype.name
            out_pattern = key
            year = 2020  # Hardcoded example year, change as needed
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, f'{year}']

        logger.info(f"Saving and uploading rasters for chunk {bounds_str}.")
        save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes, is_final)

        del out_dict_all_dtypes

        logger.info(f"Completed processing for chunk {bounds_str}.")

    except Exception as e:
        logger.error(f"Failed processing for {bounds_str}: {str(e)}", exc_info=True)

# @jit(nopython=True)
def process_soil(peat_block, land_cover_block, planted_forest_type_block, dadap_block, osm_roads_block, osm_canals_block,
                 engert_block, grip_block):
    rows, cols = peat_block.shape

    soil_block = np.empty((rows, cols), dtype=np.uint32)
    state_out = np.empty((rows, cols), dtype=np.uint32)

    for row in range(rows):
        for col in range(cols):
            peat = peat_block[row, col]
            land_cover = land_cover_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]
            dadap = dadap_block[row, col]
            osm_roads = osm_roads_block[row, col]
            osm_canals = osm_canals_block[row, col]
            engert = engert_block[row, col]
            grip = grip_block[row, col]

            node = 0

            if peat == 1:
                node = accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = accrete_node(node, 1)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node  # 'drained'
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = accrete_node(node, 2)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node  # 'drained'
                elif land_cover == 2 or land_cover == 3:  # 2 = cropland; 3 = settlement
                    node = accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node  # 'drained'
                elif planted_forest_type > 0:  # May need to remap planted forest type for emissions
                    node = accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node  # 'drained'
                else:
                    node = accrete_node(node, 5)
                    soil_block[row, col] = 0  # 'undrained'
                    state_out[row, col] = node  # 'undrained'
            else:
                soil_block[row, col] = 0  # 'undrained'
                node = accrete_node(node, 2)
                state_out[row, col] = node  # 'undrained'

    return soil_block, state_out

def main(chunk_params=None, is_final=False, client_type='local'):
    """
    Main function to process drainage model.

    Args:
        chunk_params (list, optional): List containing [min_x, min_y, max_x, max_y, chunk_size].
        is_final (bool, optional): Determines if the output file names for final versions should be used.
        client_type (str, optional): The type of Dask client to use ('local' or 'coiled').

    Returns:
        None
    """
    logger.info("Initializing main processing function")
    if client_type == 'coiled':
        client, cluster = setup_coiled_cluster()
        logger.info(f"Coiled cluster initialized: {cluster.name}")
    else:
        cluster = LocalCluster()
        client = Client(cluster)
        logger.info(f"Dask client initialized with {client_type} cluster")

    try:
        # Default chunk_params if not provided
        if chunk_params is None:
            # Example chunk_params: [min_x, min_y, max_x, max_y, chunk_size]
            chunk_params = [110.0, -10.0, 120.0, 0.0, 2]  # tile 00N_110E, Indonesia, 25 2-degree chunks

        # Makes list of chunks to analyze
        chunks = get_chunk_bounds(chunk_params)
        logger.info(f"Processing {len(chunks)} chunks")
        # print(chunks)

        # Determines if the output file names for final versions of outputs should be used
        if len(chunks) > 90:
            is_final = True
            logger.info("Running as final model.")

        # Creates list of tasks to run (1 task = 1 chunk)
        delayed_result = [delayed(calculate_and_upload_drainage)(chunk, is_final) for chunk in chunks]

        # Actually runs analysis
        logger.info(f"Computing {len(delayed_result)} tasks")
        results = dask.compute(*delayed_result)
        logger.info(f"Processing results: {results}")

    finally:
        client.close()
        logger.info("Dask client closed")
        if client_type == 'coiled':
            cluster.close()
            logger.info("Coiled cluster closed")

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Process drainage model using Dask and Coiled.'
    )
    parser.add_argument('--chunk_params', type=str,
                        help='Chunk parameters in the format "min_x,min_y,max_x,max_y,chunk_size"', default=None)
    parser.add_argument('--is_final', action='store_true',
                        help='Flag to determine if this is the final run')
    parser.add_argument('--client', type=str, choices=['local', 'coiled'], default='local',
                        help='Dask client type to use (local or coiled)')
    args = parser.parse_args()

    chunk_params = None
    if args.chunk_params:
        chunk_params = list(map(float, args.chunk_params.split(',')))

    if not any(sys.argv[1:]):
        # Default values for running directly from PyCharm or an IDE without command-line arguments
        chunk_params = [110.0, -10.0, 120.0, 0.0, 2]  # tile 00N_110E, Indonesia, 25 2-degree chunks
        is_final = False
        client_type = 'local'

        main(chunk_params=chunk_params, is_final=is_final, client_type=client_type)
    else:
        main(chunk_params=chunk_params, is_final=args.is_final, client_type=args.client)

"""
Example command to run using Coiled:

python drainage_model.py --chunk_params "112,-4,114,-2,2" --client coiled --is_final
"""

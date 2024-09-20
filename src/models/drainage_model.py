# drainage_model.py

import os
import concurrent.futures
import dask
from dask import delayed
from dask.distributed import Client
import numpy as np
import logging

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
from utils.raster_utils import save_and_upload_small_raster_set
from utils.processing_utils import (
    accrete_node,
    create_typed_dicts,
    process_soil,
    calculate_stats,
    calculate_chunk_stats
)
from utils.cluster_utils import (
    setup_coiled_cluster,
    shutdown_cluster
)

# Import additional utilities as needed
from utils.s3_utils import list_rasters_in_folder

# Import boto3 and rasterio within functions to avoid unnecessary global imports
import boto3
import rasterio

def calculate_and_upload_drainage(bounds, is_final, logger, s3_out_dir, land_cover, planted_forest_type_layer):
    """
    Processes a single chunk: downloads necessary layers, processes soil data,
    and uploads the resulting rasters to S3.

    Args:
        bounds (List[float]): Bounding box for the chunk [min_x, min_y, max_x, max_y].
        is_final (bool): Flag indicating if this is the final run.
        logger (logging.Logger): Logger instance.
        s3_out_dir (str): S3 output directory.
        land_cover (str): Land cover layer name.
        planted_forest_type_layer (str): Planted forest type layer name.
    """
    bounds_str = boundstr(bounds)  # String form of chunk bounds
    tile_id = xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = calc_chunk_length_pixels(bounds)  # Chunk length in pixels

    no_data_val = 255
    logger.info(f"Processing tile {tile_id} with bounds {bounds_str}")

    try:
        # Dictionary of downloaded layers
        download_dict = {
            f"{land_cover}_2020": f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/IPCC_basic_classes/2020/40000_pixels/20240205/{tile_id}__IPCC_classes_2020.tif",
            planted_forest_type_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/{tile_id}_plantation_type_oilpalm_woodfiber_other.tif",
            "planted_forest_tree_crop": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/{tile_id}.tif",
            "peat": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/{tile_id}_peat_mask_processed.tif",
            "dadap": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m/dadap_{tile_id}.tif",
            "engert": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m/engert_{tile_id}.tif",
            "grip": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/30m/grip_density_{tile_id}.tif",
            "osm_roads": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/canals_density_{tile_id}.tif",  # Update once roads data is ready
            "osm_canals": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/canals_density_{tile_id}.tif"
        }

        # Check whether tile exists at all. Doesn't try to download chunk if the tile doesn't exist.
        tile_exists = check_for_tile(download_dict, is_final, logger)

        if not tile_exists:
            logger.info(f"Tile {tile_id} does not exist. Skipping.")
            return

        logger.info(f"Tile {tile_id} exists. Proceeding with downloading data.")
        futures = prepare_to_download_chunk(bounds, download_dict, is_final, logger)

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

        # Check if chunk has data
        data_in_chunk = check_chunk_for_data(layers, "any", bounds_str, tile_id, "any", is_final, logger)

        if not data_in_chunk:
            logger.info(f"No data in chunk {bounds_str}. Skipping.")
            return

        logger.info(f"Data present in chunk {bounds_str}. Proceeding with processing.")

        # Separate layers by data type
        typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = create_typed_dicts(layers)

        # Example: Extract specific layers
        try:
            peat_block = typed_dict_uint8["peat"]
            land_cover_block = typed_dict_uint8[f"{land_cover}_2020"]
            planted_forest_type_block = typed_dict_uint8[planted_forest_type_layer]
            dadap_block = typed_dict_float32["dadap"]
            osm_roads_block = typed_dict_float32["osm_roads"]
            osm_canals_block = typed_dict_float32["osm_canals"]
            engert_block = typed_dict_float32["engert"]
            grip_block = typed_dict_float32["grip"]
        except KeyError as e:
            logger.error(f"Missing layer in typed dictionaries: {e}")
            return

        logger.info(f"Creating drainage map in {bounds_str} in {tile_id}: {timestr()}")
        soil_block, state_out = process_soil(
            peat_block, land_cover_block, planted_forest_type_block, dadap_block,
            osm_roads_block, osm_canals_block, engert_block, grip_block
        )

        out_dict_uint32 = {
            "soil": soil_block,
            "state": state_out
        }

        out_dict_all_dtypes = {}
        for key, value in out_dict_uint32.items():
            data_type = value.dtype.name
            out_pattern = key
            year = 2020  # Example year, adjust as needed
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, f'{year}']

        logger.info(f"Saving and uploading rasters for chunk {bounds_str}.")
        save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes, is_final, logger, s3_out_dir, no_data_val)

        del out_dict_all_dtypes

        logger.info(f"Completed processing for chunk {bounds_str}.")

def main():
    # Set up logging
    logger = setup_logging()

    # Define chunk_params
    # Example: [min_x, min_y, max_x, max_y, chunk_size]
    chunk_params = [110.0, -10.0, 120.0, 0.0, 2]  # tile 00N_110E, Indonesia, 25 2-degree chunks
    chunks = get_chunk_bounds(chunk_params)
    logger.info(f"Processing {len(chunks)} chunks")
    print(f"Processing {len(chunks)} chunks")

    # Determine if final run based on number of chunks
    is_final = False
    if len(chunks) > 90:
        is_final = True
        logger.info("Running as final model.")
        print("Running as final model.")

    # Setup cluster (Coiled cluster in this example)
    cluster = setup_coiled_cluster()
    client = cluster.get_client()

    # Define S3 output directory (ensure this is defined appropriately)
    s3_out_dir = "climate/AFOLU_flux_model/drainage_outputs"  # Remove 's3://' prefix since upload functions handle it

    # Define layer names (ensure these are defined or imported appropriately)
    land_cover = "IPCC_basic_classes"  # Example, adjust as needed
    planted_forest_type_layer = "plantation_type"

    # Create Dask delayed tasks
    delayed_result = [
        delayed(calculate_and_upload_drainage)(
            chunk, is_final, logger, s3_out_dir, land_cover, planted_forest_type_layer
        )
        for chunk in chunks
    ]

    # Execute the tasks
    results = dask.compute(*delayed_result)
    print(results)

    # Shutdown cluster
    shutdown_cluster(client, cluster)

if __name__ == "__main__":
    main()

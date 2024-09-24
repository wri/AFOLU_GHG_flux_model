# drainage_model.py

import argparse
import concurrent.futures
import dask
import numpy as np
from dask.distributed import Client
from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu

# Function to calculate drainage status and related outputs
@jit(nopython=True)
def calculate_drainage(in_dict_uint8, in_dict_int16, in_dict_float32):
    # Initialize output dictionaries
    out_dict_uint32 = {}
    out_dict_float32 = {}

    # Extract required input arrays
    peat_block = in_dict_uint8["peat"]
    land_cover_block = in_dict_uint8[f"{cn.land_cover}_2020"]
    planted_forest_type_block = in_dict_uint8[cn.planted_forest_type_layer]
    dadap_block = in_dict_float32["dadap"]
    osm_roads_block = in_dict_float32["osm_roads"]
    osm_canals_block = in_dict_float32["osm_canals"]
    engert_block = in_dict_float32["engert"]
    grip_block = in_dict_float32["grip"]

    # Initialize output arrays
    rows, cols = peat_block.shape
    soil_block = np.zeros((rows, cols), dtype=np.uint32)
    state_out = np.zeros((rows, cols), dtype=np.uint32)

    # Iterate over each pixel
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
                node = nu.accrete_node(node, 1)
                if dadap > 0 or osm_canals > 0:
                    node = nu.accrete_node(node, 1)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif engert > 0 or grip > 0 or osm_roads > 0:
                    node = nu.accrete_node(node, 2)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif land_cover == cn.cropland_code or land_cover == cn.settlement_code:
                    node = nu.accrete_node(node, 3)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                elif planted_forest_type > 0:
                    node = nu.accrete_node(node, 4)
                    soil_block[row, col] = 1  # 'drained'
                    state_out[row, col] = node
                else:
                    node = nu.accrete_node(node, 5)
                    soil_block[row, col] = 0  # 'undrained'
                    state_out[row, col] = node
            else:
                soil_block[row, col] = 0  # 'undrained'
                node = nu.accrete_node(node, 2)
                state_out[row, col] = node

    # Add outputs to dictionaries
    out_dict_uint32["soil"] = soil_block
    out_dict_uint32["state"] = state_out

    return out_dict_uint32, out_dict_float32  # No float32 outputs in this case

def calculate_and_upload_drainage(bounds, download_dict_with_data_types, is_final, no_upload):
    logger = lu.setup_logging()

    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    chunk_stats = []

    # Replace placeholder tile_id in download_dict
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # Check if tile exists
    tile_exists = uu.check_for_tile(updated_download_dict, is_final, logger)
    if not tile_exists:
        return f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}", chunk_stats

    # Prepare to download chunk
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)

    # Wait for downloads to complete
    layers = {}
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        layers[layer] = future.result()

    # Check for data presence
    required_layers = {
        f"{cn.land_cover}_2020": layers[f"{cn.land_cover}_2020"],
        "peat": layers["peat"]
    }
    data_in_chunk = uu.check_chunk_for_data(required_layers, bounds_str, tile_id, "all", is_final, logger)
    if not data_in_chunk:
        return f"Skipped chunk {bounds_str} due to lack of data: {uu.timestr()}", chunk_stats

    # Calculate stats for input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))

    # Create typed dictionaries
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # Run drainage calculation
    lu.print_and_log(f"Calculating drainage in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)
    out_dict_uint32, out_dict_float32 = calculate_drainage(
        typed_dict_uint8, typed_dict_int16, typed_dict_float32
    )

    # Combine outputs into a single dictionary
    out_dict_all_dtypes = {}
    for key, value in out_dict_uint32.items():
        out_dict_all_dtypes[key] = value

    # Calculate stats for output layers
    for key, array in out_dict_all_dtypes.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'output_layer'))

    # Save and upload outputs
    if not no_upload:
        out_no_data_val = 0  # Define NoData value if needed

        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name
            out_pattern = key
            year = 2020  # Example year, adjust if necessary
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, f'{year}']

        uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes,
                                            is_final, logger, out_no_data_val)

    # Clear memory
    del out_dict_all_dtypes

    success_message = f"Success for {bounds_str}: {uu.timestr()}"
    return success_message, chunk_stats

def main(cluster_name, bounding_box, chunk_size, run_local=False, no_stats=False, no_log=False, no_upload=False):
    # Connect to cluster
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Stage info
    stage = 'drainage_model'
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Prepare chunks
    chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    print(f"Processing {len(chunks)} chunks")

    # Determine if run is final
    is_final = False
    if len(chunks) > 20:
        is_final = True
        print("Running as final model.")

    # Accumulate stats and messages
    all_stats = []
    return_messages = []

    sample_tile_id = "00N_000E"

    # Prepare download dictionary
    download_dict = {
        f"{cn.land_cover}_2020": f"{cn.LC_uri}/composite/2020/raw/{sample_tile_id}.tif",
        cn.planted_forest_type_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/{sample_tile_id}_plantation_type_oilpalm_woodfiber_other.tif",
        cn.planted_forest_tree_crop_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/{sample_tile_id}.tif",
        "peat": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/{sample_tile_id}_peat_mask_processed.tif",
        "dadap": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m/dadap_{sample_tile_id}.tif",
        "engert": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m/engert_{sample_tile_id}.tif",
        "grip": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/30m/grip_density_{sample_tile_id}.tif",
        "osm_roads": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_roads_density/30m/roads_density_{sample_tile_id}.tif",
        "osm_canals": f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/canals_density_{sample_tile_id}.tif"
    }

    # Get first tile names and data types
    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Create delayed tasks
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    delayed_results = [dask.delayed(calculate_and_upload_drainage)(chunk, download_dict_with_data_types, is_final, no_upload) for chunk in chunks]

    # Compute tasks
    results = dask.compute(*delayed_results)

    # Process results
    success_count = 0
    skipping_chunk_count = 0

    for result in results:
        return_message, chunk_stats = result

        print(return_message)

        if "Success" in return_message:
            success_count += 1

        if "skipping chunk" in return_message:
            skipping_chunk_count += 1

        if return_message:
            return_messages.append(return_message)

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    # Print counts
    print(f"Number of 'Success' chunks: {success_count}")
    print(f"Number of 'skipping chunk' chunks: {skipping_chunk_count}")

    # Calculate stats if not suppressed
    if not no_stats:
        uu.calculate_chunk_stats(all_stats, stage)

    # End time
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Compile and upload logs
    log_note = "Drainage model run"
    lu.compile_and_upload_log(no_log, client, cluster, stage,
                              len(chunks), chunk_size, start_time, end_time, log_note)

    if not run_local:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate drainage model.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    main(args.cluster_name, args.bounding_box, args.chunk_size, args.run_local, args.no_stats, args.no_log, args.no_upload)

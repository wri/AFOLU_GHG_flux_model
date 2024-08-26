"""
Run from src/LULUCF/
python -m scripts.preprocessing.create_carbon_pools_2000 -cn AFOLU_flux_model_scripts -bb 116 -3 116.25 -2.75 -cs 0.25 --no_stats
"""

import argparse
import concurrent.futures
import coiled
import dask
import os
import numpy as np

from dask.distributed import Client
from dask.distributed import print
from numba import jit

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu


# Function to create initial (year 2000) non-soil carbon pool densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def create_starting_C_densities(in_dict_uint8, in_dict_int16, in_dict_int32, in_dict_float32, mangrove_C_ratio_array):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type.
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_float32 = {}

    # print(in_dict_uint8)
    # print(in_dict_int16)
    # print(in_dict_int32)
    # print(in_dict_float32)

    # Input blocks for remaining inputs, now that they definitely exist (either originally or have been created)
    whrc_agb_2000_block = in_dict_int16["agb_2000"]
    mangrove_agb_2000_block = in_dict_float32["mangrove_agb_2000"]
    r_s_ratio_block = in_dict_float32["r_s_ratio"]
    elevation_block = in_dict_int16["elevation"]
    climate_domain_block = in_dict_int16["climate_domain"]
    precipitation_block = in_dict_int32["precipitation"]
    continent_ecozone_block = in_dict_int16["continent_ecozone"]

    mangrove_in_chunk = True  # Flag for whether chunk has mangrove in it
    whrc_agb_2000_in_chunk = True  # Flag for whether chunk has WHRC AGB 2000 in it

    # Checks if the chunk has various inputs by seeing if the max value is 0.
    # If the max value is 0, it assumed that input doesn't exist.
    if whrc_agb_2000_block.max() == 0:
        whrc_agb_2000_in_chunk = False
    if mangrove_agb_2000_block.max() == 0:
        mangrove_in_chunk = False

    # Output blocks
    # Need to specify the output datatype or it will default to float32
    agc_2000_out_block = np.zeros(in_dict_float32["r_s_ratio"].shape).astype('float32')
    bgc_2000_out_block = np.zeros(in_dict_float32["r_s_ratio"].shape).astype('float32')
    deadwood_c_2000_out_block = np.zeros(in_dict_float32["r_s_ratio"].shape).astype('float32')
    litter_c_2000_out_block = np.zeros(in_dict_float32["r_s_ratio"].shape).astype('float32')

    # Iterates through all pixels in the chunk
    for row in range(continent_ecozone_block.shape[0]):
        for col in range(continent_ecozone_block.shape[1]):

            # Input values for this specific cell
            whrc_agb_2000 = whrc_agb_2000_block[row, col]
            mangrove_agb_2000 = mangrove_agb_2000_block[row, col]
            elevation = elevation_block[row, col]
            climate_domain = climate_domain_block[row, col]
            precipitation = precipitation_block[row, col]
            r_s_ratio = r_s_ratio_block[row, col]
            continent_ecozone = continent_ecozone_block[row, col]

            # If mangrove AGB is present, AGC 2000 is calculated from it, overwriting any AGC that is based on WHRC that is already there
            if (mangrove_in_chunk) and (
                    mangrove_agb_2000 > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
                agc_2000_out_block[row, col] = mangrove_agb_2000 * cn.biomass_to_carbon_mangrove

            # If WHRC AGB is present, AGC 2000 is calculated from it
            elif (whrc_agb_2000_in_chunk) and (
                    whrc_agb_2000 > 0):  # Only uses AGB if chunk exists and there is a value in that pixel
                agc_2000_out_block[row, col] = whrc_agb_2000 * cn.biomass_to_carbon_non_mangrove

            else:
                agc_2000_out_block[row, col] = 0

            # Separate branches for assigning BGC, deadwood C, and litter C ratios depending on whether the pixel has mangroves.
            # Calculation of BGC, deadwood C, and litter C are done after the decision tree assigns the ratios.

            # Mangrove carbon pool ratio branch
            # From IPCC 2013 Wetland Supplement
            if (mangrove_in_chunk) and (mangrove_agb_2000 > 0):  # Only replaces WHRC AGB if mangrove chunk exists and if mangrove value in that pixel
                bgc_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone)][0, 1]
                deadwood_c_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone)][0, 2]
                litter_c_ratio = mangrove_C_ratio_array[np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone)][0, 3]

            # Non-mangrove carbon pool ratio branch
            # Deadwood and litter carbon as fractions of AGC are from
            # https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ar-am-tool-12-v3.0.pdf
            # "Clean Development Mechanism A/R Methodological Tool:
            # Estimation of carbon stocks and change in carbon stocks in dead wood and litter in A/R CDM project activities version 03.0"
            # Tables on pages 18 (deadwood) and 19 (litter).
            # They depend on the climate domain, elevation, and precipitation.
            elif (whrc_agb_2000_in_chunk) and (whrc_agb_2000 > 0):  # Non-mangrove

                # If no mapped R:S, uses the global default value instead
                if r_s_ratio == 0:
                    r_s_ratio = cn.default_r_s
                bgc_ratio = r_s_ratio  # Uses R:S for BGC

                if climate_domain == 1:  # Tropical/subtropical
                    if elevation <= 2000:  # Low elevation
                        if precipitation <= 1000:  # Low precipitation or no precip raster
                            deadwood_c_ratio = cn.tropical_low_elev_low_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_low_precip_litter_c_ratio
                        elif ((precipitation > 1000) and (precipitation <= 1600)):  # Medium precipitation
                            deadwood_c_ratio = cn.tropical_low_elev_med_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_med_precip_litter_c_ratio
                        else:  # High precipitation
                            deadwood_c_ratio = cn.tropical_low_elev_high_precip_deadwood_c_ratio
                            litter_c_ratio = cn.tropical_low_elev_high_precip_litter_c_ratio
                    else:  # High elevation
                        deadwood_c_ratio = cn.tropical_high_elev_deadwood_c_ratio
                        litter_c_ratio = cn.tropical_high_elev_litter_c_ratio
                else:  # Temperate/boreal
                    deadwood_c_ratio = cn.non_tropical_deadwood_c_ratio
                    litter_c_ratio = cn.non_tropical_litter_c_ratio

            else:

                # Ridiculous default BGC, deadwood C, and litter C ratios that will make it very clear if they are being used instead of
                # something being assigned in the decision treea above
                bgc_ratio = -5
                deadwood_c_ratio = -10
                litter_c_ratio = -20

            # Actually calculates BGC, deadwood C, and litter C using the ratios assigned in the above decision tree
            bgc_2000_out_block[row, col] = agc_2000_out_block[row, col] * bgc_ratio
            deadwood_c_2000_out_block[row, col] = agc_2000_out_block[row, col] * deadwood_c_ratio
            litter_c_2000_out_block[row, col] = agc_2000_out_block[row, col] * litter_c_ratio

    # Adds the output arrays to the dictionary with the appropriate data type
    # Outputs need .copy() so that previous intervals' arrays in dicationary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
    out_dict_float32[f"{cn.agc_dens_pattern}_{cn.first_year}"] = agc_2000_out_block.copy()
    out_dict_float32[f"{cn.bgc_dens_pattern}_{cn.first_year}"] = bgc_2000_out_block.copy()
    out_dict_float32[f"{cn.deadwood_c_dens_pattern}_{cn.first_year}"] = deadwood_c_2000_out_block.copy()
    out_dict_float32[f"{cn.litter_c_dens_pattern}_{cn.first_year}"] = litter_c_2000_out_block.copy()

    # return output dictionary/ies
    return out_dict_float32


# All steps for creating starting non-soil carbon pools in a chunk: download chunks, calculate carbon densities, upload to s3
def create_and_upload_starting_C_densities(bounds, is_final, mangrove_C_ratio_array):

    logger = lu.setup_logging()

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    ### Part 1: downloads chunks and checks for data

    # Dictionary of data to download
    download_dict = {

        cn.agb_2000: f"{cn.agb_2000_path}{tile_id}_{cn.agb_2000_pattern}.tif",
        cn.mangrove_agb_2000: f"{cn.mangrove_agb_2000_path}{tile_id}_{cn.mangrove_agb_2000_pattern}.tif",
        cn.elevation: f"{cn.elevation_path}{tile_id}_{cn.elevation_pattern}.tif",
        cn.climate_domain: f"{cn.climate_domain_path}{tile_id}_{cn.climate_domain_pattern}.tif",
        cn.precipitation: f"{cn.precipitation_path}{tile_id}_{cn.precipitation_pattern}.tif",
        cn.r_s_ratio: f"{cn.r_s_ratio_path}{tile_id}_{cn.r_s_ratio_pattern}.tif",
        cn.continent_ecozone: f"{cn.continent_ecozone_path}{tile_id}_{cn.continent_ecozone_pattern}.tif"
    }

    # Checks whether the tile exists at all for any of the inputs (not just the necessary inputs)
    tile_exists = uu.check_for_tile(download_dict, is_final, logger)

    if not tile_exists:
        return f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}"

    futures = uu.prepare_to_download_chunk(bounds, download_dict, is_final, logger)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Dictionary that stores the downloaded data
    layers = {}

    # Waits for requests to come back with data from S3
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        layers[layer] = future.result()

    # print(layers)

    checked_layers = {'agb_2000': layers['agb_2000'],
                      'mangrove_agb_2000': layers['mangrove_agb_2000']}

    # Checks chunk for data. Skips the chunk if it has no data in it.
    # Only one of the checked layers must exist for this chunk.
    data_in_chunk = uu.check_chunk_for_data(checked_layers, bounds_str, tile_id, "any", is_final, logger)

    if not data_in_chunk:
        return f"Skipped chunk {bounds_str} because of a lack of data: {uu.timestr()}"


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Stores the stats for the chunk
    stats = []

    # Calculate stats for the original layers
    for key, array in layers.items():
        stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))


    ### Part 3: Fills in any missing input data chunks so the Numba function has a full dataset to work with.
    ### Missing chunks are filled in here instead of using the typed dicts below just because numpy arrays are easier to work with.
    ### And missing chunks are not filled in earlier (e.g., when downloading chunks)
    ### so that chunk stats are calculated only for the chunks that do exist which is useful for QC.

    # Gets the first tile in each input folder in order to determine the datatype of the input dataset.
    # Needs to check the first tile in each folder because, if the input raster doesn't exist for this chunk,
    # we can't assign a datatype for that input for this chunk.
    # So instead it gets the datatype of the input from a raster that has to exist (the first one in s3).
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Categorizes the files by their data types so that any missing inputs can be filled in
    # with 0s of the correct datatype
    uint8_list, int16_list, int32_list, float32_list = uu.categorize_files_by_dtype(first_tiles)

    # print("uint8_list:", uint8_list)
    # print("int16_list:", int16_list)
    # print("int32_list:", int32_list)
    # print("float32_list:", float32_list)

    filled_layers = uu.fill_missing_input_layers_with_no_data(layers, uint8_list, int16_list, int32_list, float32_list,
                                                              bounds_str, tile_id, is_final, logger)

    ### Part 4: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
    ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
    ### Note: need to add new code if inputs with other data types are added

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(filled_layers)

    # print("uint8_typed_list:", typed_dict_uint8)
    # print("int16_typed_list:", typed_dict_int16)
    # print("int32_typed_list:", typed_dict_int32)
    # print("float32_typed_list:", typed_dict_float32)


    ### Part 5: Creates starting carbon pool densities

    lu.print_and_log(f"Creating starting C densities for {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Create AGC, BGC, deadwood C and litter C densities in 2000
    out_dict_float32 = create_starting_C_densities(
        typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32, mangrove_C_ratio_array
    )

    # Fresh non-Numba-constrained dictionary that stores all output numpy arrays of all datatypes.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    for key, value in out_dict_float32.items():
        out_dict_all_dtypes[key] = value

    # Clear memory of unneeded arrays (output(s) from the numba function)
    del out_dict_float32


    ### Part 6: Calculates min, mean, and max for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculate stats for the output layers from create_starting_C_densities
    for key, array in out_dict_all_dtypes.items():
        stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'output_layer'))


    ### Part 7: Saves numpy arrays as rasters and uploads to s3

    out_no_data_val = 0  # NoData value for output raster (optional)

    # Adds metadata used for uploading outputs to s3 to the dictionary
    for key, value in out_dict_all_dtypes.items():
        data_type = value.dtype.name
        out_pattern = key[:-5]  # Drops the year (2000) from the end of the string

        # Dictionary with metadata for each array
        out_dict_all_dtypes[key] = [value, data_type, out_pattern, cn.first_year]

    uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes,
                                        is_final, logger, out_no_data_val)

    # Clears memory of unneeded arrays
    del out_dict_all_dtypes

    success_message = f"Success for {bounds_str}: {uu.timestr()}"
    return success_message, stats  # Return both the success message and the statistics

def main(cluster_name, bounding_box, chunk_size, run_local=False, no_stats=False, no_log=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'carbon_pool_2000'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Creates numpy array of ratios of BGC, deadwood C, and litter C relative to AGC. Relevant columns must be specified.
    mangrove_C_ratio_array = uu.convert_lookup_table_to_array(cn.rate_ratio_spreadsheet, cn.mangrove_rate_ratio_tab,
                                                           ['gainEcoCon', 'BGC_AGC', 'deadwood_AGC', 'litter_AGC'])

    # Makes list of chunks to analyze
    chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    print("Processing", len(chunks), "chunks")
    # print(chunks)

    # Determines if the output file names for final versions of outputs should be used
    is_final = False
    if len(chunks) > 20:
        is_final = True
        print("Running as final model.")

    # Accumulates all statistics and output messages from chunk analysis
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_stats = []
    return_messages = []

    # Creates list of tasks to run (1 task = 1 chunk)
    delayed_results = [dask.delayed(create_and_upload_starting_C_densities)(chunk, is_final, mangrove_C_ratio_array) for chunk in chunks]

    # Runs analysis and gathers results
    results = dask.compute(*delayed_results)

    # Processes the chunk stats and returned messages
    # Results are the messages from the chunks and chunk stats
    for result in results:
        success_message, chunk_stats = result
        if success_message:
            return_messages.append(success_message)
        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    # Prepares chunk stats spreadsheet: min, mean, max for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag
    if not no_stats:
        uu.calculate_chunk_stats(all_stats, stage)

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    # Prints the returned messages
    for message in return_messages:
        print(message)

    # Creates combined log if not deactivated
    log_note = "Global carbon pool 2000 run"
    lu.compile_and_upload_log(no_log, client, cluster, stage,
                              len(chunks), chunk_size, start_time, end_time, log_note)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create carbon pools in 2000.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.bounding_box, args.chunk_size, args.run_local, args.no_stats, args.no_log)


import os  # Import the OS module to interact with the operating system
import coiled  # Import the Coiled library to manage Dask clusters on the cloud
import dask  # Import Dask for parallel computing
from dask.distributed import Client, \
    LocalCluster  # Import Client for Dask task scheduling, LocalCluster to create a cluster on the local machine
from dask.distributed import print as dask_print  # Import a Dask-aware print function
import dask.config  # Import Dask's configuration management tools
import distributed  # Import additional Dask distributed computing tools
import numpy as np  # Import NumPy for numerical operations
import rasterio  # Import Rasterio to work with geospatial raster data
from numba import jit  # Import the JIT compiler decorator from Numba to speed up functions
import concurrent.futures  # Import the concurrent.futures module for asynchronous execution
import boto3  # Import Boto3 to handle AWS services such as S3
import time  # Import the time module for time-related tasks
import math  # Import the math module for mathematical tasks
import pandas as pd  # Import pandas for data manipulation

# Define a Coiled cloud cluster configuration for scalable distributed processing
coiled_cluster = coiled.Cluster(
    n_workers=20,  # Number of worker machines in the cluster
    use_best_zone=True,  # Optimize for the best AWS zone to minimize latency
    compute_purchase_option="spot_with_fallback",  # Use spot instances with an on-demand fallback
    idle_timeout="10 minutes",  # Timeout to scale down the cluster if no jobs are running
    region="us-east-1",  # AWS region where the cluster is deployed
    name="next_gen_forest_carbon_flux_model",  # Name the cluster for easier management
    account='jterry64',  # Specify the Coiled account linked to AWS
    worker_memory="32GiB"  # Set the memory each worker node should use
)

# Get a client connection to the cluster
coiled_client = coiled_cluster.get_client()

# Setup a local client for running Dask tasks on the local machine without a cluster
local_client = Client()


# Example function definitions with detailed comments:

def timestr():
    """Generate a timestamp string to tag output files."""
    # Format the current time as a string for use in file names or logs
    return time.strftime("%Y%m%d_%H_%M_%S")


def get_chunk_bounds(chunk_params):
    """
    Calculate rectangular chunks within a specified geographic boundary.

    Args:
        chunk_params (list): List of parameters [min_x, min_y, max_x, max_y, chunk_size].

    Returns:
        list: A list of chunk boundaries defined as [x_min, y_min, x_max, y_max].
    """
    # Unpack the bounding box coordinates and the desired chunk size
    min_x, min_y, max_x, max_y, chunk_size = chunk_params
    chunks = []  # Initialize an empty list to store the chunk boundaries
    y = min_y  # Start at the minimum y coordinate
    while y < max_y:  # Iterate over y within the vertical boundary
        x = min_x  # Start at the minimum x coordinate for each new row
        while x < max_x:  # Iterate over x within the horizontal boundary
            # Define the boundary of a chunk and add it to the list
            chunks.append([x, y, x + chunk_size, y + chunk_size])
            x += chunk_size  # Move to the next chunk horizontally
        y += chunk_size  # Move to the next chunk vertically
    return chunks  # Return the list of chunk boundaries


def xy_to_tile_id(x, y):
    """
    Convert geographic coordinates to a standardized tile ID.

    Args:
        x (float): Longitude of the tile.
        y (float): Latitude of the tile.

    Returns:
        str: A string representing the tile ID in 'YYN/S_XXXE/W' format.
    """
    # Determine the tile's latitude and longitude bands based on coordinates
    lat_band = "N" if y >= 0 else "S"
    lon_band = "E" if x >= 0 else "W"
    # Calculate the tile's grid position based on 10-degree blocks
    lat = f"{math.ceil(abs(y) / 10) * 10:02d}{lat_band}"
    lon = f"{math.floor(abs(x) / 10) * 10:03d}{lon_band}"
    return f"{lat}_{lon}"  # Return the formatted tile ID

def get_tile_dataset_rio(uri, bounds, chunk_length):
    """Attempt to open a raster tile and read a specific window."""
    try:
        with rasterio.open(uri) as ds:
            window = rasterio.windows.from_bounds(*bounds, ds.transform)
            data = ds.read(1, window=window)
            return data
    except:
        return np.zeros((chunk_length, chunk_length))

def prepare_regrowth_array(spreadsheet, tab):
    """Prepare an array of regrowth data from a spreadsheet."""
    regrowth_df = pd.read_excel(open(spreadsheet, 'rb'), sheet_name=tab)
    regrowth_df = regrowth_df[['ecozone_code', 'iso_code', 'startH_code', 'Slope_Mg_AGC_ha_yr']]
    regrowth_array = regrowth_df.to_numpy().astype(float)
    return regrowth_array

def previous_and_next_land_cover_map_years(year):
    """Determine the appropriate years of land cover maps to use based on input year."""
    previous_multiple = (year // 5) * 5
    next_multiple = previous_multiple + 5 if year < 2020 else previous_multiple
    return previous_multiple, next_multiple

def classify(regrowth_array, *args):
    """Classify forest state and calculate carbon fluxes using a decision tree approach."""
    forest_height_previous_block, forest_height_current_block, forest_loss_detection_block, driver_block, planted_forest_type_block, peat_block, agc_current_block, bgc_current_block, deadwood_c_current_block, litter_c_current_block, soil_c_current_block, r_s_ratio_block, ecozone_block, iso_block, land_cover_previous_block, land_cover_next_block, burned_area_one_before_block, burned_area_current_block = args
    # The rest of the classify function would be defined here, handling the complex decision logic.

def process_chunk(bounds, start_year, regrowth_array):
    """Process a single chunk of data for all relevant years."""
    futures, layers = {}, {}
    bounds_str = "_".join(map(str, map(round, bounds)))
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    tile_id = xy_to_tile_id(bounds[0], bounds[3])
    # Fetch and process data, classify forest states, calculate fluxes, handle outputs, etc.


# Operates on all pixels in each chunk for a given year
# Inputs are forest heights, various contextual layers used to assign forest state, and current carbon stocks for different pools.
# Outputs are forest state, carbon fluxes, and carbon stocks for different pools.

@jit(nopython=True)  # numba decorator that compiles Python to C++ to accelerate processing
def classify(regrowth_array,
             forest_height_previous_block, forest_height_current_block, forest_loss_detection_block, driver_block,
             planted_forest_type_block, peat_block,
             agc_current_block, bgc_current_block, deadwood_c_current_block, litter_c_current_block,
             soil_c_current_block, r_s_ratio_block, ecozone_block, iso_block,
             land_cover_previous_block, land_cover_next_block, burned_area_current_block, burned_area_current_block):
    # Outputs
    forest_states = np.zeros(forest_height_previous_block.shape)
    emission_factor = np.zeros(forest_height_previous_block.shape)
    removal_factor = np.zeros(forest_height_previous_block.shape)

    agc_flux = np.zeros(forest_height_previous_block.shape)
    bgc_flux = np.zeros(forest_height_previous_block.shape)
    deadwood_c_flux = np.zeros(forest_height_previous_block.shape)
    litter_c_flux = np.zeros(forest_height_previous_block.shape)
    soil_c_flux = np.zeros(forest_height_previous_block.shape)

    # Iterates through all pixels in the chunk
    for row in range(forest_height_previous_block.shape[0]):
        for col in range(forest_height_previous_block.shape[1]):

            # Pixel for each input
            forest_height_previous = forest_height_previous_block[row, col]
            forest_height_current = forest_height_current_block[row, col]
            forest_loss_detection = forest_loss_detection_block[row, col]
            driver = driver_block[row, col]
            planted_forest_type = planted_forest_type_block[row, col]
            peat = peat_block[row, col]

            LC_previous = land_cover_previous_block[row, col]
            LC_next = land_cover_next_block[row, col]
            burned_area_one_before = burned_area_one_before_block[row, col]
            burned_area_current = burned_area_current_block[row, col]

            agc_current = agc_current_block[row, col]
            bgc_current = bgc_current_block[row, col]
            deadwood_c_current = deadwood_c_current_block[row, col]
            litter_c_current = litter_c_current_block[row, col]
            soil_c_current = soil_c_current_block[row, col]

            r_s_ratio = r_s_ratio_block[row, col]
            ecozone = ecozone_block[row, col]
            iso = iso_block[row, col]

            # Various definitions used in decision tree
            grassland_forest_previous = (
                        ((LC_previous >= 2) & (LC_previous <= 48)) | ((LC_previous >= 102) & (LC_previous <= 148)))
            grassland_forest_next = (((LC_next >= 2) & (LC_next <= 48)) | ((LC_next >= 102) & (LC_next <= 148)))
            cropland_previous = (LC_previous == 244)
            cropland_next = (LC_next == 244)
            forestry = (driver == 3)
            non_sdpt_forestry = (
                        forestry & (grassland_forest_previous | grassland_forest_next) & (cropland_previous == 0) & (
                            cropland_next == 0))
            burned_area_recent = ((burned_area_one_before != 0) or (burned_area_one_before != 0))

            # The decision tree that produces all the outputs for a pixel for a given year
            if forest_height_previous >= 5 and forest_height_current >= 5:  # maintained
                forest_states[row, col] = 1
                agc_rf = regrowth_array[np.where(
                    (regrowth_array[:, 0] == ecozone) * (regrowth_array[:, 1] == iso) * (regrowth_array[:, 2] == 2))][
                    0, 3]  # TODO: replace [:,2] == 2 with actual assignment
                removal_factor[row, col] = agc_rf
                agc_flux[row, col] = agc_rf
                bgc_flux[row, col] = agc_rf * r_s_ratio
                deadwood_c_flux[row, col] = agc_rf * 0.06
                litter_c_flux[row, col] = agc_rf * 0.06
                soil_c_flux[row, col] = agc_rf * 0.01
                agc_current_block[row, col] = agc_current + agc_flux[row, col]
                bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
            elif forest_height_previous < 5 and forest_height_current >= 5:  # gain
                forest_states[row, col] = 2
                agc_rf = regrowth_array[np.where(
                    (regrowth_array[:, 0] == ecozone) * (regrowth_array[:, 1] == iso) * (regrowth_array[:, 2] == 1))][
                    0, 3]  # TODO: replace [:,2] == 2 with actual assignment
                removal_factor[row, col] = agc_rf
                agc_flux[row, col] = agc_rf
                bgc_flux[row, col] = agc_rf * r_s_ratio
                deadwood_c_flux[row, col] = agc_rf * 0.09
                litter_c_flux[row, col] = agc_rf * 0.09
                soil_c_flux[row, col] = agc_rf * 0.04
                agc_current_block[row, col] = agc_current + agc_flux[row, col]
                bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
            elif ((forest_height_previous >= 5 and forest_height_current < 5) or forest_loss_detection == 1):  # loss
                if planted_forest_type == 0:  # loss:no SDPT
                    if non_sdpt_forestry == 0:  # loss:no SDPT:no non-SDPT forestry
                        forest_states[row, col] = 311
                        biomass_ef = 0.9
                        dead_litter_ef = 0.3
                        soil_ef = 0.1
                        emission_factor[row, col] = biomass_ef
                        agc_flux[row, col] = (agc_current * biomass_ef) * -1
                        bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                        deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                        litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                        soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                        agc_current_block[row, col] = agc_current + agc_flux[row, col]
                        bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                        deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                        litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                        soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
                    else:  # loss:no SDPT:non-SDPT forestry
                        forest_states[row, col] = 312
                        biomass_ef = 0.7
                        dead_litter_ef = 0.3
                        soil_ef = 0.1
                        emission_factor[row, col] = biomass_ef
                        agc_flux[row, col] = (agc_current * biomass_ef) * -1
                        bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                        deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                        litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                        soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                        agc_current_block[row, col] = agc_current + agc_flux[row, col]
                        bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                        deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                        litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                        soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
                else:  # loss:SDPT
                    if burned_area_recent == 0:  # loss:SDPT:not burned recent
                        if peat == 0:  # loss:SDPT:not burned recent:not peat
                            forest_states[row, col] = 3211
                            biomass_ef = 0.6
                            dead_litter_ef = 0.5
                            soil_ef = 0.2
                            emission_factor[row, col] = biomass_ef
                            agc_flux[row, col] = (agc_current * biomass_ef) * -1
                            bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                            deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                            litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                            soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                            agc_current_block[row, col] = agc_current + agc_flux[row, col]
                            bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                            deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                            litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                            soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
                        else:  # loss:SDPT:not burned recent:peat
                            forest_states[row, col] = 3212
                            biomass_ef = 0.75
                            dead_litter_ef = 0.4
                            soil_ef = 0.1
                            emission_factor[row, col] = biomass_ef
                            agc_flux[row, col] = (agc_current * biomass_ef) * -1
                            bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                            deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                            litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                            soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                            agc_current_block[row, col] = agc_current + agc_flux[row, col]
                            bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                            deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                            litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                            soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
                    else:  # loss:SDPT:burned recent
                        if peat == 0:  # loss:SDPT:burned recent:not peat
                            forest_states[row, col] = 3221
                            biomass_ef = 0.65
                            dead_litter_ef = 0.1
                            soil_ef = 0.3
                            emission_factor[row, col] = biomass_ef
                            agc_flux[row, col] = (agc_current * biomass_ef) * -1
                            bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                            deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                            litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                            soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                            agc_current_block[row, col] = agc_current + agc_flux[row, col]
                            bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                            deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                            litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                            soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
                        else:  # loss:SDPT:burned recent:peat
                            forest_states[row, col] = 3222
                            biomass_ef = 0.9
                            dead_litter_ef = 0.1
                            soil_ef = 0.4
                            emission_factor[row, col] = biomass_ef
                            agc_flux[row, col] = (agc_current * biomass_ef) * -1
                            bgc_flux[row, col] = (bgc_current * biomass_ef) * -1
                            deadwood_c_flux[row, col] = (deadwood_c_current * dead_litter_ef) * -1
                            litter_c_flux[row, col] = (litter_c_current * dead_litter_ef) * -1
                            soil_c_flux[row, col] = (soil_c_current * soil_ef) * -1
                            agc_current_block[row, col] = agc_current + agc_flux[row, col]
                            bgc_current_block[row, col] = bgc_current + bgc_flux[row, col]
                            deadwood_c_current_block[row, col] = deadwood_c_current + deadwood_c_flux[row, col]
                            litter_c_current_block[row, col] = litter_c_current + litter_c_flux[row, col]
                            soil_c_current_block[row, col] = soil_c_current + soil_c_flux[row, col]
            else:  # no forest
                forest_states[row, col] = 0
                emission_factor[row, col] = 0
                removal_factor[row, col] = 0

                agc_flux[row, col] = 0
                bgc_flux[row, col] = 0
                deadwood_c_flux[row, col] = 0
                litter_c_flux[row, col] = 0
                soil_c_flux[row, col] = 0

                agc_current_block[row, col] = agc_current
                bgc_current_block[row, col] = bgc_current
                deadwood_c_current_block[row, col] = deadwood_c_current
                litter_c_current_block[row, col] = litter_c_current
                soil_c_current_block[row, col] = soil_c_current

    return forest_states, emission_factor, removal_factor, agc_flux, bgc_flux, deadwood_c_flux, litter_c_flux, soil_c_flux, agc_current_block, bgc_current_block, deadwood_c_current_block, litter_c_current_block, soil_c_current_block















# Configure analysis parameters, load data, run processing
start_year = 2021
chunk_params = [0, 79.75, 0.25, 80, 0.25]  # small test area
regrowth_array = prepare_regrowth_array(regrowth_spreadsheet, regrowth_tab)
chunks = get_chunk_bounds(chunk_params)
print("Processing", len(chunks), "chunks")
delayed_tasks = [dask.delayed(process_chunk)(chunk, start_year, regrowth_array) for chunk in chunks]
results = dask.compute(*delayed_tasks)
print(results)
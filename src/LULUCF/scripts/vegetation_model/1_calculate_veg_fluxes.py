"""
Calculates annual gross and net fluxes from vegetation by carbon pool (AGC, BGC, deadwood, litter) and gas (CO2, CH4, N2O).
Also, calculates associated non-soil carbon densities. Reports land state node classification and various intermediate
outputs that are useful for QC and potentially as contextual layers (e.g., composite primary forest extent).

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Local test (Dask part does not work because of client.submit()):
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -bb 10 49.75 10.25 50 -cs 0.25 --run_local --no_upload

Coiled small tests:
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -mt standard -mpd test_box -bb 116.25 -2.25 116.5 -2 -cs 0.25

Coiled small tests (1x1 deg chunk needs 32GB worker):
python -m src.utilities.create_cluster -n 1 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -mt standard -mpd test_box -bb -64 -22 -63 -21 -cs 1 --create_zarr

Coiled Cerrado test (174 features):
python -m src.utilities.create_cluster -n 20 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -mt standard -mpd Cerrado -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__Cerrado_center_in.shp --create_zarr

Coiled large shapefile test (1884 features):
python -m src.utilities.create_cluster -n 100 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -mt standard -mpd 1884_features -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in__1884_test_features.shp --create_zarr

Full run:
python -m src.utilities.create_cluster -n 200 -t 1 -m 32 -cn vegetation_model
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes -cn vegetation_model -mt standard -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp --log_note "This is a global run for model v1.0.4 (2016-2024). Hopefully, it is the run used for the published model."

To download all outputs locally:
python src/utilities/download_outputs_local.py v1_test_name 23_-4_24_-3

Using more than 1 thread/worker slows down processing a lot when there are more tasks than workers for the core vegetation model,
which is the situation for large analyses, obviously.
https://app.asana.com/1/25496124013636/task/1206230383901961/comment/1210641504248464?focus=true

#TODO change NoData in flux outputs to something besides 0 because 0 has a meaning for fluxes
#TODO update 1km drivers to correct year. Currently using through 2023.
#TODO add AGC removal factor, AGC emission fraction, and forest age to zarr output (for use in zonal statistics)
#TODO make all outputs have a unit where /PER_HA_OR_PIXEL/ currently is-- change it to /UNIT/ so that non-flux/density outputs have a unit, too
#TODO Check for changes to zarr creation and usage (including 10x10 creation and zonal stats) from working on SOC
#TODO potential change to 3112/3119
#TODO potentially add branches for loss of primary forest (currently just have primary forest remaining primary forest)
#TODO Add veg_ to the start of output patterns to distinguish them from SOC or organic soil
#TODO Delete all references to 5-year intervals (including s3 paths)
#TODO Change all runtimes to decimal hours from HH:MM:SS
"""

import argparse
import concurrent.futures
import dask
import gc
import os
import psutil
import time
import sys
import pandas as pd
import numpy as np
import fsspec
import xarray as xr
import resource
import traceback
import re

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import print
from numba import jit
from datetime import date

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import numba_utilities as nu
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import resize_cluster

# To get enhanced logging from workers so that I can tell why they are lost. I don't know if this works.
# Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
dask.config.set({
    "distributed.logging.distributed": "debug",   # show detailed worker logs
    "distributed.logging.bokeh": "critical",      # silence dashboard noise
})

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# A little testing of it in this script suggests that it doesn't save much, if any time, but leaving it in just in case.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"


# Function to calculate vegetation fluxes and carbon densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def vegetation_fluxes(in_dict_uint8, in_dict_uint16, in_dict_int16, in_dict_int32, in_dict_float32,
                      primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                      model_start_year, end_year, interval_type, interval_year_diff_list, interval_length_list, interval_end_years, is_large_run):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type.
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_uint8 = {}
    out_dict_uint16 = {}
    out_dict_uint32 = {}
    out_dict_float32 = {}

    # The maximum possible number of digits of the state_out in the current decision tree.
    # This needs to be changed if the decision tree is deepened.
    max_digits_state_out = 8

    # Carbon density arrays determined by the starting year of the model (Mg C/ha),
    # but the starting C densities have the same key in the dictionary regardless of the starting year
    agc_dens_block = in_dict_float32[cn.agc_LC_masked_dens_pattern]
    bgc_dens_block = in_dict_float32[cn.bgc_LC_masked_dens_pattern]
    deadwood_c_dens_block = in_dict_float32[cn.deadwood_c_LC_masked_dens_pattern]
    litter_c_dens_block = in_dict_float32[cn.litter_c_LC_masked_dens_pattern]

    # print(agc_dens_block.max())
    # print(bgc_dens_block.max())
    # print(deadwood_c_dens_block.max())
    # print(litter_c_dens_block.max())

    # Root:shoot (unitless)
    r_s_ratio_non_mang_block = in_dict_float32[cn.r_s_ratio_non_mang_pattern]

    # Natural forest regrowth curves (Mg C/ha/yr)
    natrl_forest_curve_0_5_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__0_5_years"]
    natrl_forest_curve_6_10_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__6_10_years"]
    natrl_forest_curve_11_15_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__11_15_years"]
    natrl_forest_curve_16_20_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__16_20_years"]
    natrl_forest_curve_21_40_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__21_40_years"]
    natrl_forest_curve_41_60_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__41_60_years"]
    natrl_forest_curve_61_80_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__61_80_years"]
    natrl_forest_curve_81_100_AGC_RF_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__81_100_years"].astype('float32')

    # Removal factor (Mg C/ha/yr)
    # Because this is used to store the RF from the previous interval,
    # it persists from one interval to the next. Therefore, it must be defined before the first iteration.
    # That way, removal factors can be over-written by those used in the most recent interval.
    agc_rf_pre_dist_out_block = np.zeros(agc_dens_block.shape, dtype='float32')

    # Mangrove extent
    mangrove_extent_1996_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_1996"]
    mangrove_extent_2007_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2007"]
    mangrove_extent_2008_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2008"]
    mangrove_extent_2009_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2009"]
    mangrove_extent_2010_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2010"]
    mangrove_extent_2015_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2015"]
    mangrove_extent_2016_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2016"]
    mangrove_extent_2017_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2017"]
    mangrove_extent_2018_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2018"]
    mangrove_extent_2019_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2019"]
    mangrove_extent_2020_block = in_dict_uint8[f"{cn.mangrove_extent_processed_pattern}_2020"]

    planted_forest_type_block = in_dict_uint8[cn.planted_forest_type_pattern]
    planted_forest_tree_crop_block = in_dict_uint8[cn.planted_forest_tree_crop_pattern]
    planted_forest_AGC_RF_block = in_dict_float32[cn.planted_forest_AGC_removal_factor_pattern]
    planted_forest_AGC_BGC_RF_block = in_dict_float32[cn.planted_forest_AGC_BGC_removal_factor_pattern]
    oil_palm_2000_extent_block = in_dict_uint8[cn.oil_palm_2000_extent_pattern]
    oil_palm_first_year_block = in_dict_int16[cn.oil_palm_first_year_pattern]

    forest_age_start_year_block = in_dict_uint16[cn.forest_age_start_year_pattern]

    drivers_block = in_dict_uint8[cn.drivers_pattern]
    continent_ecozone_block = in_dict_int16[cn.continent_ecozone_pattern]
    climate_zone_block = in_dict_uint8[cn.climate_zone_pattern]
    elevation_block = in_dict_int16[cn.elevation_pattern]
    climate_domain_block = in_dict_int16[cn.climate_domain_pattern]
    precipitation_block = in_dict_int32[cn.precipitation_pattern]


    # Sets a fallback value for continent_ecozone for the chunk in case any pixels fall outside the continent-ecozone boundary.
    # Also, excludes water codes (1022, 2022, 4022, 7022) from the fallback value.
    # Note that these continent-ecozone values need to explicitly be ignored further down when
    # the continent-ecozone pixel is applied.
    # This only excludes certain continent-ecozone values from use in the creation of the fallback value.
    # fallback_value is only used if the chunk doesn't have any continent_ecozone pixels in it at all.
    continent_ecozone_fallback = nu.fallback_conteco_climzone_value(continent_ecozone_block, 2020)

    # Sets a fallback value for climate zone for the chunk in case any pixels fall outside the climate zone boundary.
    # fallback_value is only used if the chunk doesn't have any climate_zone pixels in it at all.
    climate_zone_fallback = nu.fallback_conteco_climzone_value(climate_zone_block, 5)


    ## Test/intermediate outputs blocks

    # Determines the composite primary forest extent based on the model starting year
    if model_start_year == 2000:
        ifl_primary_2000_block = in_dict_uint8[cn.ifl_primary_2000_pattern]
        composite_primary_block = np.where((ifl_primary_2000_block > 0) | (forest_age_start_year_block >=cn.primary_age_threshold), 1, 0).astype(np.uint8)
    elif model_start_year == 2015:
        primary_2001_block = in_dict_uint8[cn.primary_2001_pattern]
        ifl_2016_block = in_dict_uint8[cn.ifl_2016_pattern]
        tcl_block = in_dict_uint8[cn.tree_cover_loss_pattern]

        # Filters tcl_block to only where tcl occurred before 2015 (ignoring 0s)
        pre_2015_tcl_mask_block = ((tcl_block > 0) & (tcl_block < 15)).astype(np.uint8)

        # Masks out any primary forest where TCL occurred before 2015
        primary_2015_block = (primary_2001_block * (1 - pre_2015_tcl_mask_block)).astype(np.uint8)

        # Merges together IFL 2016 and primary 2015 so that if either is 1, it will be in the merged block
        # composite_primary_block = np.maximum(ifl_2016_block, primary_2015_block).astype(np.uint8)
        composite_primary_block = np.where((ifl_2016_block > 0) | (primary_2015_block > 0) | (forest_age_start_year_block >=cn.primary_age_threshold), 1, 0).astype(np.uint8)
    else:
        raise ValueError("invalid start year: must be 2000 or 2015")

    # Saves the starting year composite primary forest to the output dictionary
    out_dict_uint8[f"{cn.composite_primary_forest}_{model_start_year}"] = composite_primary_block.copy()

    # Stores the annual forest disturbance raster blocks for the entire model duration (added to progressively during each interval)
    annual_forest_dist_blocks_all_intervals_so_far = []

    # Stores the burned area blocks for the entire model duration (added to progressively during each interval)
    burned_area_blocks_all_intervals_so_far = []

    # Stores the last year that each pixel did not have tall vegetation composite land cover.
    # 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
    # This is assessed at the pixel level because numba wouldn't allow the needed logical operations on numpy arrays (chunks).
    # Tall vegetation is basd on the composite land cover maps, not the canopy height maps.
    most_recent_year_not_tall_veg_block = np.zeros(agc_dens_block.shape, dtype='uint16')

    # Forest age for each output year of the model
    forest_age_end_of_interval_block = forest_age_start_year_block

    # Maximum height of vegetation since the last interval in which there was not forest
    max_height_since_last_time_not_tall_veg_block = np.zeros(agc_dens_block.shape, dtype='uint8')

    # Tracks whether the height has already decreased more than the signif. height loss threshold compared to
    # maximum vegetation height since the last time the pixel was non-tall vegetation land cover.
    # This prevents a pixel from repeatedly (multiple intervals) being counted as having a height loss disturbance compared to
    # the maximum height; this can only be triggered once since the maximum height is attained.
    # This is used to determine if a forest->forest disturbance based on height loss relative to the max height should be reported.
    # 0=no significant height loss relative to the maximum vegetation height since last non-tall vegetation.
    # 1=height loss relative to the maximum vegetation height occurred in this interval.
    # 2=height loss relative to the maximum vegetation height occurred in a previous interval.
    first_time_sig_loss_from_max_height_block = np.zeros(agc_dens_block.shape, dtype='uint8')

    # Tracks whether there was a partial (non-fire) or full disturbance in a previous interval (not including just fire,
    # which does not count as a partial disturbance in this model if height does not decrease significantly with it).
    # Updated for every interval based on the current interval disturbance status.
    # This is primarily used to determine what age the forest is (which matters for assigning removal factors).
    part_or_full_dist_in_earlier_intervals_block = np.zeros(agc_dens_block.shape, dtype='uint8')

    # Tracks whether a partial disturbance occurs in the current interval due to any cause (not including just fire,
    # which does not count as a partial disturbance in this model if height does not decrease significantly with it).
    # Overwritten for every interval.
    # This is primarily used to determine what age the forest is (which matters for assigning removal factors).
    part_or_full_dist_in_curr_interval_block = np.zeros(agc_dens_block.shape, dtype='uint8')

    # print("interval_end_years:", interval_end_years)

    # All years covered by the model, including the start year of the model
    years_so_far = [model_start_year]

    # Iterates through model intervals
    for i, interval_end_year in enumerate(interval_end_years):

        # print(f"Now at interval ending in {interval_end_year}:")

        # Length of the interval and difference between the start and end years (years)
        interval_length = interval_length_list[i]
        interval_year_diff = interval_year_diff_list[i]
        interval_start_year = interval_end_year - interval_length

        # Model years so far, including the model start year.
        # Eventually used to determine whether current height has decreased significantly from maximum height
        # since last non-tall veg year over multiple intervals (gradual height loss).
        years_so_far = years_so_far + [interval_end_year]

        # Pre-fetches vegetation height data for this chunk and stores in a dictionary or list.
        # Eventually used to determine whether current height has decreased significantly from maximum height since last non-tall veg year over multiple intervals (gradual height loss).
        # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba
        # and speed the code up. I was trying to get vegetation height so far in a variety of ways and it kept being slow.
        # This approach, in conjunction with some pixel-level operations below, seems to not slow down the code.
        vegetation_heights_so_far_block = [
            in_dict_uint8[f"{cn.vegetation_height_pattern}_{year}"]
            for year in years_so_far
        ]
        # print(vegetation_heights_so_far_block)

        # Land cover and vegetation height at the start and end of the interval,
        # e.g., 2005/2010 or 2016/2017
        LC_prev_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year - interval_length}"]
        LC_curr_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year}"]
        veg_h_prev_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year - interval_length}"]
        veg_h_curr_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year}"]

        # Vegetation height from GPW median vegetation height. Original values are rescaled by 10 (reported in dm) to make them ints,
        # so converting them to the m (float) values here.
        GPW_height_prev_block = (in_dict_int16[f"{cn.GPW_MVH_pattern}_{interval_end_year - interval_length}"] / 10).astype('float32')
        GPW_height_curr_block = (in_dict_int16[f"{cn.GPW_MVH_pattern}_{interval_end_year}"] / 10).astype('float32')

        # print(f"{cn.land_cover_pattern}_{interval_end_year - interval_length}:", LC_prev_block)
        # print(f"{cn.land_cover_pattern}_{interval_end_year}:", LC_curr_block)
        # print(f"{cn.vegetation_height_pattern}_{interval_end_year - interval_length}:", veg_h_prev_block)
        # print(f"{cn.vegetation_height_pattern}_{interval_end_year}:", veg_h_curr_block)

        # Creates a list of all the burned area arrays from 2001 to the end of the interval.
        # The values in the array are the year burned starting from 1, e.g., 2001=1, 2008=8, 2017=17.
        # It lists all burned area chunks in all intervals so far.
        # For example, for a 5-year interval 2001-2005, it will get burned area for 2001, 2002, 2003, 2004, and 2005.
        # For annual interval 2015-2016, it will get burned area for 2001, 2002, 2003... 2016.
        # For 2016-2017, it will get burned area for 2001, 2002, 2003... 2017.
        # It works by getting the burned area chunks for the current interval and appending them to a list of
        # chunks from previous intervals.
        # The years included depend on the interval length (5 or 1 years).
        # Note: Stacking the rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
        # So just making a list of numpy arrays instead of a 3D numpy array.
        if interval_length == 5:
            for year in range(interval_end_year - interval_year_diff, interval_end_year+1):  # Iterates through years in interval
                burned_area_for_year_in_interval = f"{cn.burned_area_final_pattern}_{year}"
                year_burned_array = in_dict_uint8[burned_area_for_year_in_interval] * (year - model_start_year)
                burned_area_blocks_all_intervals_so_far.append(year_burned_array)

                # print(year)
                # print("year_burned_array:", year_burned_array)
                # print("year_burned_array max:", np.max(year_burned_array))
                # print("burned_area_blocks_all_intervals_so_far:", burned_area_blocks_all_intervals_so_far)
                # print("burned_area_blocks_all_intervals_so_far max:", np.max(burned_area_blocks_all_intervals_so_far))
        elif interval_length == 1:  # Only the burned area data from the end year of the interval is used
            burned_area_for_year_in_interval = f"{cn.burned_area_final_pattern}_{interval_end_year}"
            year_burned_array = in_dict_uint8[burned_area_for_year_in_interval] * (interval_end_year - model_start_year)
            burned_area_blocks_all_intervals_so_far.append(year_burned_array)

            # print(interval_end_year)
            # print("year_burned_array:", year_burned_array)
            # print("year_burned_array max:", np.max(year_burned_array))
            # print("burned_area_blocks_all_intervals_so_far:", burned_area_blocks_all_intervals_so_far)
            # print("burned_area_blocks_all_intervals_so_far max:", np.max(burned_area_blocks_all_intervals_so_far))
        else:
            raise ValueError("interval_length not valid: must be 1 or 5")

        # print("burned_area_blocks_all_intervals_so_far")
        # print(burned_area_blocks_all_intervals_so_far)
        # print("burned_area_blocks_all_intervals_so_far max for all intervals so far:", np.max(burned_area_blocks_all_intervals_so_far))


        # Creates a list of all the annual Potapov forest disturbance rasters from 2001 to the end of the interval.
        # The values in the list are the disturbance year starting from 1, e.g., 2001=1, 2008=8, 2017=17.
        # It works by getting the annual disturbance chunks for the current interval and appending them to a list of
        # chunks from previous intervals.
        # Only does it for model run using 5-year intervals, as annual disturbance isn't needed for annual interval models.
        # Note: Stacking the rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
        # So just making a list of numpy arrays instead of a 3D numpy array.
        if interval_length == 5:
            for year in range(interval_end_year-interval_year_diff, interval_end_year+1):

                # The name of the disturbance layer in the input dictionary
                annual_disturbance_for_year_in_interval = f"{cn.forest_disturbance_layer_name}_{year}"

                # Replaces the binary annual disturbance array with the year of disturbance (1, 2, 3...23)
                # print(in_dict_uint8[annual_disturbance_for_year_in_interval])
                year_disturb_array = in_dict_uint8[annual_disturbance_for_year_in_interval] * (year - model_start_year)

                # Makes a list of disturbance arrays with the disturbance year.
                # uint8 is okay because the highest value should be 23 (not 2020).
                annual_forest_dist_blocks_all_intervals_so_far.append(year_disturb_array.astype('uint8'))

                # print(year)
                # print("year_disturb_array:", year_disturb_array)
                # print("year_disturb_array max:", np.max(year_disturb_array))
                # print("annual_forest_dist_blocks_all_intervals_so_far:", annual_forest_dist_blocks_all_intervals_so_far)
                # print("annual_forest_dist_blocks_all_intervals_so_far max:", np.max(annual_forest_dist_blocks_all_intervals_so_far))

            # print("annual_forest_dist_blocks_all_intervals_so_far")
            # print(annual_forest_dist_blocks_all_intervals_so_far)
            # print("annual_forest_dist_blocks_all_intervals_so_far max for all intervals so far:", np.max(annual_forest_dist_blocks_all_intervals_so_far))

        # Tracks how many times each pixel was burned during the interval
        times_burned_in_interval_block = np.zeros(agc_dens_block.shape, dtype='uint8')

        # Numpy arrays for outputs that don't depend on previous interval's values
        state_out_block = np.zeros(agc_dens_block.shape, dtype='uint32')  # Land cover state at end of interval

        # Number of years of canopy growth.
        # First digit is pre-disturbance years of growth.
        # Second digit (if it exists) is post-disturbance years of growth
        gain_year_count_out_block = np.zeros(agc_dens_block.shape, dtype='uint8')

        agc_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        bgc_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        deadwood_c_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        litter_c_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')

        ch4_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        n2o_gross_emis_out_block = np.zeros(agc_dens_block.shape, dtype='float32')

        agc_gross_removals_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        bgc_gross_removals_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        deadwood_c_gross_removals_out_block = np.zeros(agc_dens_block.shape, dtype='float32')
        litter_c_gross_removals_out_block = np.zeros(agc_dens_block.shape, dtype='float32')

        # Aboveground carbon emission factors
        agc_ef_out_block = np.zeros(agc_dens_block.shape, dtype='float32')


        # Iterates through all pixels in the chunk
        for row in range(LC_curr_block.shape[0]):
            for col in range(LC_curr_block.shape[1]):

                ### Reads input pixel values for interval
                LC_prev = LC_prev_block[row, col]
                LC_curr = LC_curr_block[row, col]
                veg_h_prev = veg_h_prev_block[row, col]
                veg_h_curr = veg_h_curr_block[row, col]

                GPW_height_prev = GPW_height_prev_block[row, col]
                GPW_height_curr = GPW_height_curr_block[row, col]

                # Mangrove extent years (1 = mangrove, 0 = no mangrove)
                mang_1996 = mangrove_extent_1996_block[row, col]
                mang_2007 = mangrove_extent_2007_block[row, col]
                mang_2008 = mangrove_extent_2008_block[row, col]
                mang_2009 = mangrove_extent_2009_block[row, col]
                mang_2010 = mangrove_extent_2010_block[row, col]
                mang_2015 = mangrove_extent_2015_block[row, col]
                mang_2016 = mangrove_extent_2016_block[row, col]
                mang_2017 = mangrove_extent_2017_block[row, col]
                mang_2018 = mangrove_extent_2018_block[row, col]
                mang_2019 = mangrove_extent_2019_block[row, col]
                mang_2020 = mangrove_extent_2020_block[row, col]

                # Array of entire mangrove timeseries
                mang_timeseries = np.array([mang_1996, mang_2007, mang_2008, mang_2009, mang_2010,
                                            mang_2015, mang_2016, mang_2017, mang_2018, mang_2019, mang_2020]).astype('uint8')

                # Secondary forest removal factors (Mg AGC/ha/yr)
                natrl_forest_curve_0_5_AGC_RF = natrl_forest_curve_0_5_AGC_RF_block[row, col]
                natrl_forest_curve_6_10_AGC_RF = natrl_forest_curve_6_10_AGC_RF_block[row, col]
                natrl_forest_curve_11_15_AGC_RF = natrl_forest_curve_11_15_AGC_RF_block[row, col]
                natrl_forest_curve_16_20_AGC_RF = natrl_forest_curve_16_20_AGC_RF_block[row, col]
                natrl_forest_curve_21_40_AGC_RF = natrl_forest_curve_21_40_AGC_RF_block[row, col]
                natrl_forest_curve_41_60_AGC_RF = natrl_forest_curve_41_60_AGC_RF_block[row, col]
                natrl_forest_curve_61_80_AGC_RF = natrl_forest_curve_61_80_AGC_RF_block[row, col]
                natrl_forest_curve_81_100_AGC_RF = natrl_forest_curve_81_100_AGC_RF_block[row, col]

                planted_forest_type_cell = planted_forest_type_block[row, col]
                planted_forest_tree_crop_cell = planted_forest_tree_crop_block[row, col]
                planted_forest_AGC_RF_cell = planted_forest_AGC_RF_block[row, col]
                planted_forest_AGC_BGC_RF_cell = planted_forest_AGC_BGC_RF_block[row, col]
                planted_forest_BGC_RF_cell = planted_forest_AGC_BGC_RF_cell - planted_forest_AGC_RF_cell

                oil_palm_2000_extent_cell = oil_palm_2000_extent_block[row, col]
                oil_palm_first_year_cell = oil_palm_first_year_block[row, col]

                composite_primary_cell = composite_primary_block[row, col]
                drivers_cell = drivers_block[row, col]
                continent_ecozone_cell = continent_ecozone_block[row, col]
                climate_zone_cell = climate_zone_block[row, col]

                elevation_cell = elevation_block[row, col]
                climate_domain_cell = climate_domain_block[row, col]
                precipitation_cell = precipitation_block[row, col]

                # BGC:AGC for non-mangrove forest
                r_s_ratio_non_mang = r_s_ratio_non_mang_block[row, col]

                # Forest age starts with the previous interval's ending age and is adjusted during the interval
                forest_age_start_of_interval = forest_age_end_of_interval_block[row, col]

                # Tracks whether there was a partial/full disturbance (non-fire) during the current interval
                # or any previous interval
                part_or_full_dist_in_curr_interval = part_or_full_dist_in_curr_interval_block[row, col]
                part_or_full_dist_in_earlier_intervals = part_or_full_dist_in_earlier_intervals_block[row, col]

                # Input carbon densities for the pools using the end of the previous interval (Mg C/ha)
                agc_dens_in = agc_dens_block[row, col]
                bgc_dens_in = bgc_dens_block[row, col]
                deadwood_c_dens_in = deadwood_c_dens_block[row, col]
                litter_c_dens_in = litter_c_dens_block[row, col]

                # Makes a list of carbon densities to save space in the decision tree below.
                # This list is input to flux calculation functions as one argument, rather than a separate argument
                # for each pool (Mg C/ha)
                c_dens_in = [agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in]


                ### Calculates various derived pixel values for interval that are used in the decision tree

                # Applies the continent_ecozone fallback value when there isn't a value for the pixel (0)
                # or the value represents water (the XX22s).
                # We don't want pixels that are in a water ecozone to not be included in land
                # (and therefore not have a removal factor assigned); ecozone shouldn't delineate land.
                # These water codes are also excluded from the creation of continent_ecozone_fallback.
                if continent_ecozone_cell in [0, 1022, 2022, 4022, 7022]:
                    continent_ecozone_cell = continent_ecozone_fallback

                # Applies the climate_zone fallback value when there isn't a value for the pixel
                if climate_zone_cell == 0:
                    climate_zone_cell = climate_zone_fallback

                # Determines the removal factor for primary forests/IFL based on the continent-ecozone combination (Mg AGC/ha/yr)
                primary_forest_AGC_RF = nu.calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RF_array)

                # Determines the emission factor for the cell's driver for partially disturbed forest based on the continent-ecozone combination (unit: fraction AGC lost)
                partial_disturbance_EF_for_driver = nu.calc_partial_disturbance_EFs(drivers_cell, continent_ecozone_cell, partial_disturbance_EF_array)

                # Ratios of deadwood C:AGC and litter C:AGC (for deadwood C and litter C removal factors) for non-mangrove forests
                deadwood_c_ratio_non_mang, litter_c_ratio_non_mang = nu.calc_deadwood_litter_ratios(elevation_cell, climate_domain_cell, precipitation_cell)

                # Replaces pixel without R:S (0) with the global non-mangrove R:S default
                if r_s_ratio_non_mang == 0:
                    r_s_ratio_non_mang = cn.default_r_s_non_mang

                # One-time removal factor for gain of short vegetation (Mg C/ha) based on climate zone
                # and adjusted for vegetation cover fraction
                short_veg_AGC_RF_adj, short_veg_BGC_RF_adj = nu.calc_short_veg_removals(climate_zone_cell, LC_curr)

                # Short veg aboveground and belowground carbon removal factors as a numpy array
                short_veg_AGC_BGC_RF_adj = np.array([short_veg_AGC_RF_adj, short_veg_BGC_RF_adj, 0.0, 0.0]).astype('float32')

                # Sets stating carbon pools under special circumstances:
                # Need to force the pools to float32 because of numba.

                # Tree cover gain: sets starting AGC and BGC pools to 0.
                # This assumes no residual AGC and BGC when tree cover gain occurs.
                # It also assumes that there can be some deadwood and litter C left over.
                c_dens_in_NT_T = [np.float32(0), np.float32(0), deadwood_c_dens_in, litter_c_dens_in]
                # Tree crops (including oil palm): no deadwood or litter carbon
                c_dens_in_tree_crops = [agc_dens_in, bgc_dens_in, np.float32(0), np.float32(0)]
                # Trees outside forests: no deadwood or litter carbon
                c_dens_in_ToF = [agc_dens_in, bgc_dens_in, np.float32(0), np.float32(0)]
                # Cropland: only AGC
                c_dens_in_cropland = [np.float32(cn.cropland_agc_dens), np.float32(0), np.float32(0), np.float32(0)]
                # Short vegetation: no deadwood or litter carbon
                c_dens_in_short_veg = [agc_dens_in, bgc_dens_in, np.float32(0), np.float32(0)]
                # No starting carbon
                c_dens_in_empty = [np.float32(0), np.float32(0), np.float32(0), np.float32(0)]

                # Aboveground removal factors based on stand age at the start of the interval (as opposed to the end) (Mg C/ha/yr).
                # So, for a five-year interval, if the starting age is 39 years, it will use the 20-40 year RF for the entire interval
                # rather than using 20-40 for the first 2 years then 41-60 for the rest of the interval. A fine simplification.
                # Adds 1 to forest_age_start_of_interval to downward adjust the age for the beginning of the interval.
                if 0 <= forest_age_start_of_interval + 1 <= 5:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_0_5_AGC_RF
                elif 6 <= forest_age_start_of_interval + 1 <= 10:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_6_10_AGC_RF
                elif 11 <= forest_age_start_of_interval + 1 <= 15:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_11_15_AGC_RF
                elif 16 <= forest_age_start_of_interval + 1 <= 20:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_16_20_AGC_RF
                elif 21 <= forest_age_start_of_interval + 1 <= 40:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_21_40_AGC_RF
                elif 41 <= forest_age_start_of_interval + 1 <= 60:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_41_60_AGC_RF
                elif 61 <= forest_age_start_of_interval + 1 <= 80:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_61_80_AGC_RF
                elif 81 <= forest_age_start_of_interval + 1 < cn.primary_age_threshold:
                    natrl_forest_age_dependent_agc_rf = natrl_forest_curve_81_100_AGC_RF
                else:  # Use the primary forest/IFL RF for forest >100 years old
                    natrl_forest_age_dependent_agc_rf = primary_forest_AGC_RF

                # Updates whether the cell is primary forest
                if (forest_age_start_of_interval >= cn.primary_age_threshold) or (composite_primary_cell == 1):
                    composite_primary_cell = 1
                else:
                    composite_primary_cell = 0

                # Gef for fire emissions for different gases for forests specifically (grams respective gas/kg dry matter)
                Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest = nu.calc_Gef_forest(climate_domain_cell)

                # Cf for fire emissions for all gases for forests specifically (unitless).
                # Based on driver of loss, not the interval-end land cover.
                Cf_forest = nu.calc_Cf_forest(climate_domain_cell, drivers_cell, composite_primary_cell)

                # Sets all mangrove states to false and only initializes mangrove states if is_ever_mang is True below
                before_mang = mang_gain = mang_loss = mang_remaining_mang = non_mang_remaining_non_mang = after_mang = False

                # Checks whether mangroves are present at all (any year) within the entire timeseries.
                # If mangroves are ever present, we assume that there was never other terrestrial forest type before mangrove gain
                # and that there will not be conversion to other terrestrial forest type after mangrove loss.
                is_ever_mang = np.any(mang_timeseries == 1)

                # Mangrove-specific pre-processing (only if smoothed mangrove was present some year)
                if is_ever_mang:

                    # Years represented by GMWv3.
                    # Changed 2007 to 2004 because 2007 extent represents 2004 in our model.
                    mang_years = np.array((1996, 2004, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020),dtype=np.int32)

                    # Gets the first year of mangrove gain and last year of mangrove loss during the entire timeseries
                    # to determine if this interval is before mangrove establishment or after permanent mangrove loss (ie no fluxes).
                    # Also gets the first year of mangrove loss to establish if mangroves are in "equilibrium". Mangrove are
                    # assumed to NOT be in "equilibrium" after loss, in which case, deadwood and litter removals are calculated.
                        # first_mang_gain_year = 0 if there is extent in first year of the timeseries
                        # first_mang_loss_year = 0 if there is never any loss. Otherwise, it is the first year of temporary mangrove loss.
                        # last_mang_loss_year = 0 if there is extent in last year of the timeseries. Otherwise, it represents permanent mangrove loss.
                    # Note: It is possible to have first_mang_loss_year = 2010 and last_mang_loss_year = 0 if there was
                    # temporary loss in 2010 but recovery and mangrove are present in the final year of the timeseries (2020).
                    first_mang_gain_year, first_mang_loss_year, last_mang_loss_year = nu.mangrove_first_gain_last_loss_years(mang_timeseries, mang_years)
                    # print(f"mang_timeseries: {mang_timeseries}")
                    # print(f"first_mang_gain_year: {first_mang_gain_year}")
                    # print(f"first_mang_loss_year: {first_mang_loss_year}")
                    # print(f"last_mang_loss_year: {last_mang_loss_year}")

                    # Gets mangrove data for this interval (uses most recent year if there is no mangrove extent for a given year)
                    mang_year_index_list = nu.map_years_to_gmwv3_data(interval_end_year, interval_length)
                    mang_interval_timeseries = np.take(mang_timeseries, mang_year_index_list)
                    # print(f"mang_interval_timeseries between {interval_start_year} - {interval_end_year}: {mang_interval_timeseries}")

                    # Determines whether mangroves are present in this interval
                    # Useful if we end up switching to allow conversion of mangroves.
                    # is_mang_in_interval = np.any(mang_interval_timeseries == 1)
                    # print(f"is_mang_in_interval: {is_mang_in_interval}")

                    # Determines whether this interval is mangrove gain, maintenance, permanent loss, before gain, or after permanent loss
                    mang_loss, mang_gain, mang_remaining_mang, non_mang_remaining_non_mang, before_mang, after_mang = (
                        nu.mangrove_states(interval_start_year, interval_end_year, first_mang_gain_year, last_mang_loss_year, mang_interval_timeseries))

                    # Gets the following information to use in decision tree calculations:
                    # 1) mang_loss_in_interval: if there is mangrove loss during this interval (1->0) [True or False]
                    # 2) mang_loss_year_in_interval: the last year there was mangrove loss during the interval [i.e. 2016]
                    # 3) mang_gain_year_count_pre_loss: number of sequestration years in interval before mangrove loss [i.e. 3]
                    # 4) mang_gain_year_count_post_loss: number of sequestration years in interval after mangrove loss [i.e. 1]
                    # If there are multiple years of loss (1->0) in the interval, the last one is reported.
                    # If no loss, then mang_gain_year_count_pre_loss = number of years with mangrove extent during the interval
                    # and mang_gain_year_count_post_loss = 0
                    mang_loss_in_interval, mang_loss_year_in_interval, mang_gain_year_count_pre_loss, mang_gain_year_count_post_loss = (
                        nu.mangrove_gain_year_count_summary(mang_interval_timeseries, interval_start_year, interval_end_year))

                    # Determines the removal factor and ratios of belowground, deadwood, and litter carbon for
                    # mangroves based on the continent-ecozone combination
                    mangrove_AGC_RF, r_s_ratio_mang, deadwood_c_ratio_mang, litter_c_ratio_mang = (
                        nu.calc_mangrove_RF_and_ratios(continent_ecozone_cell, mangrove_C_ratio_array))

                    # print(f"before_mang: {before_mang}")
                    # print(f"mang_gain: {mang_gain}")
                    # print(f"mang_loss: {mang_loss}")
                    # print(f"mang_remaining_mang: {mang_remaining_mang}")
                    # print(f"non_mang_remaining_non_mang: {non_mang_remaining_non_mang}")
                    # print(f"after_mang: {after_mang}")
                    #
                    # print(f"mang_loss_year_in_interval: {mang_loss_year_in_interval}")
                    # print(f"mang_gain_year_count_pre_loss: {mang_gain_year_count_pre_loss}")
                    # print(f"mang_gain_year_count_post_loss: {mang_gain_year_count_post_loss}")
                    #
                    # print(f"continent_ecozone_cell: {continent_ecozone_cell}")
                    # print(f"mangrove_AGC_RF: {mangrove_AGC_RF}")
                    # print(f"r_s_ratio_mang: {r_s_ratio_mang}")
                    # print(f"deadwood_c_ratio_mang: {deadwood_c_ratio_mang}")
                    # print(f"litter_c_ratio_mang: {litter_c_ratio_mang}")


                ### Defines specific land cover classes, including planted tree classification

                # Tree presence at start and end of interval based on canopy heights (not LC composites)
                tree_prev = (veg_h_prev >= cn.tree_threshold)
                tree_curr = (veg_h_curr >= cn.tree_threshold)

                # Tree gain and loss based on canopy heights at start and end of interval (not based on LC composites)
                tree_gain = (not tree_prev and tree_curr)
                tree_loss = (tree_prev and not tree_curr)

                # Gain and loss of vegetation according to Global Pasture Watch vegetation height product (Hunter et al. 2025).
                # Already rescaled from dm to m in the block processing step above.
                GPW_short_veg_prev = (GPW_height_prev >= cn.GPW_short_veg_threshold)
                GPW_short_veg_curr = (GPW_height_curr >= cn.GPW_short_veg_threshold)

                GPW_veg_height_gain = (not GPW_short_veg_prev and GPW_short_veg_curr)
                GPW_veg_height_loss = (GPW_short_veg_prev and not GPW_short_veg_curr)

                # Booleans of vegetation height classes for start (prev) and end (curr) of current interval based on LC composites
                GLAD_bare_ground_LC_prev, GLAD_short_veg_LC_prev, GLAD_tall_veg_LC_prev = nu.classify_GLAD_composite(LC_prev)
                GLAD_bare_ground_LC_curr, GLAD_short_veg_LC_curr, GLAD_tall_veg_LC_curr = nu.classify_GLAD_composite(LC_curr)

                water_LC_curr = (LC_curr >= cn.water_min_code) and (LC_curr <= cn.water_max_code)

                SDPT_planted_trees = (planted_forest_type_cell > 0)  # All SDPT planted trees
                SDPT_oil_palm = (planted_forest_type_cell == cn.SDPT_oil_palm_code)  # Oil palm in SDPT planted trees
                oil_palm_pre_2000 = (oil_palm_2000_extent_cell == 1) # Oil palm that existed in the year 2000, according to that specific map/input


                # Establishes if the interval ends after Descals oil palm planting year. Rules are different for annual and 5-year intervals.
                # Second condition for each used to exclude NoData (0s) from first year of oil palm.
                # If the interval end year is after planting year, the interval is after Descals planting
                if interval_length == 1:
                    oil_palm_year_of_Descals_or_later = (interval_end_year >= oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0)
                # If the interval start year is after planting year, the interval is after Descals planting.
                # This "all or nothing" approach simplifies things compared to saying that an interval can be partially before and partially after Descals planting year.
                elif interval_length == 5:
                    oil_palm_year_of_Descals_or_later = (interval_start_year >= oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0)
                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")

                # All oil palm in the given interval.
                # This excludes oil palm that is in SDPT (of any kind) and Descals extent but precedes Descals extent (i.e. before oil palm in that pixel).
                all_oil_palm = (oil_palm_pre_2000 or oil_palm_year_of_Descals_or_later or (SDPT_oil_palm and (oil_palm_first_year_cell == 0)))

                # All planted trees in the given interval.
                # This excludes oil palm that is in SDPT (of any kind) and Descals extent but precedes Descals extent (i.e. before oil palm in that pixel).
                all_planted_trees = (all_oil_palm or (SDPT_planted_trees and (oil_palm_first_year_cell == 0)))

                # All tree crops in the given interval (including oil palm) (does not including planted forests)
                all_tree_crops = (all_oil_palm or ((planted_forest_tree_crop_cell == 2) and (oil_palm_first_year_cell == 0)))

                # Flag for whether the Descals year of planting is:
                # Annual intervals: planting year one year after the end of the interval
                # 5-year intervals: planting year during the 5-year interval
                # (to determine if forest loss should occur in the year before oil palm is detected)
                if interval_length == 1:  # For annual intervals
                    interval_before_converted_to_oil_palm = (interval_end_year == oil_palm_first_year_cell - interval_length)
                elif interval_length == 5:  # For 5-year intervals
                    interval_before_converted_to_oil_palm = ((interval_start_year < oil_palm_first_year_cell) and
                                                             (interval_end_year > oil_palm_first_year_cell))
                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")

                # Revises carbon densities for tree crops (including oil palm)-- drops deadwood and litter carbon.
                # Because c_dens_in is revised here (outside of the decision tree),
                # c_dens_in_tree_crops doesn't need to be used at individual decision tree nodes.
                if all_tree_crops:
                    c_dens_in = c_dens_in_tree_crops


                ### Identifies disturbed pixels.
                ### This is some of the most convoluted and obscure code in the model.

                # Calculates Potapov annual disturbance raster and burned area metrics for the interval (5-year and annual)
                # 5-year intervals: Burned area and Potapov annual disturbance raster stacks during the interval
                if interval_length == 5:
                    # Value represents year of burned area (not just binary presence/absence).
                    # Note: Stacking the burned area rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                    # So just reading each raster from the list of rasters separately.
                    burned_area_t_4 = burned_area_blocks_all_intervals_so_far[-5][row, col]
                    burned_area_t_3 = burned_area_blocks_all_intervals_so_far[-4][row, col]
                    burned_area_t_2 = burned_area_blocks_all_intervals_so_far[-3][row, col]
                    burned_area_t_1 = burned_area_blocks_all_intervals_so_far[-2][row, col]
                    burned_area_t = burned_area_blocks_all_intervals_so_far[-1][row, col]

                    # The years with burned area during the interval
                    all_burned_area_years_during_interval = np.array([burned_area_t_4, burned_area_t_3,
                                                                      burned_area_t_2, burned_area_t_1, burned_area_t])
                    burned_years_during_interval = all_burned_area_years_during_interval[all_burned_area_years_during_interval != 0]

                    # Number of times burned during interval (for repeat short veg/cropland fire emissions)
                    times_burned_in_interval = np.count_nonzero(all_burned_area_years_during_interval)

                    # Whether the pixel was burned at all during the interval
                    burned_in_curr_interval = (times_burned_in_interval > 0)

                    # The first year with burned area during the interval
                    if burned_in_curr_interval:  # If there were any years with fires during the interval, the first year is reported
                        first_year_burned_during_interval = min(burned_years_during_interval)
                    else:  # If there were no years with fires, 0 is assigned
                        first_year_burned_during_interval = 0

                    # if interval_end_year == 2010:
                    #     print("all_burned_area_years_during_interval:", all_burned_area_years_during_interval)
                    #     print("burned_years_during_interval:", burned_years_during_interval)
                    #     print("times_burned_in_interval:", times_burned_in_interval)
                    #     print("burned_in_curr_interval:", burned_in_curr_interval)
                    #     print("first_year_burned_during_interval:", first_year_burned_during_interval)
                    #
                    #     if times_burned_in_interval == 5:  # To force it to terminate
                    #         sys.quit()

                    # Value represents year of disturbance (not just binary presence/absence).
                    # Note: Stacking the forest disturbance rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                    # So just reading each raster from the list of rasters separately.
                    forest_dist_t_4 = annual_forest_dist_blocks_all_intervals_so_far[-5][row, col]
                    forest_dist_t_3 = annual_forest_dist_blocks_all_intervals_so_far[-4][row, col]
                    forest_dist_t_2 = annual_forest_dist_blocks_all_intervals_so_far[-3][row, col]
                    forest_dist_t_1 = annual_forest_dist_blocks_all_intervals_so_far[-2][row, col]
                    forest_dist_t = annual_forest_dist_blocks_all_intervals_so_far[-1][row, col]

                    # The years with annual disturbance during the interval
                    all_annual_dist_years_during_interval = np.array([forest_dist_t_4, forest_dist_t_3,
                                                                      forest_dist_t_2, forest_dist_t_1, forest_dist_t])
                    annual_dist_years_during_interval = all_annual_dist_years_during_interval[all_annual_dist_years_during_interval != 0]

                    # Number of years with annual disturbance during interval
                    times_annual_dist_in_interval = np.count_nonzero(annual_dist_years_during_interval)

                    # Whether the pixel had annual disturbance at all during the interval
                    annual_dist_in_interval = (times_annual_dist_in_interval > 0)

                    # The first year with annual disturbance during the interval
                    if annual_dist_in_interval:  # If there were any years with annual disturbance during the interval, the first year is reported
                        first_year_annual_dist_during_interval = min(annual_dist_years_during_interval)
                    else:  # If there were no years with annual disturbance, 0 is assigned
                        first_year_annual_dist_during_interval = 0

                    # print("all_annual_dist_years_during_interval:", all_annual_dist_years_during_interval)
                    # print("annual_dist_years_during_interval:", annual_dist_years_during_interval)
                    # print("times_annual_dist_in_interval:", times_annual_dist_in_interval)
                    # print("annual_dist_in_interval:", annual_dist_in_interval)
                    # print("first_year_annual_dist_during_interval:", first_year_annual_dist_during_interval)
                    #
                    # if times_annual_dist_in_interval == 1: # To force it to terminate
                    #     sys.quit()
                # Annual intervals: Burned area for end year of interval only (year t).
                # Use of burned area for end year of interval as opposed to start year is by analogy with
                # Tree Cover Loss due to Fires (TCLF).
                # Annual Potapov disturbance rasters not used for annual intervals.
                elif interval_length == 1:
                    burned_area_t = burned_area_blocks_all_intervals_so_far[-1][row, col]
                    # The first year with burned area during the interval (but for annual interval, the only year)
                    first_year_burned_during_interval = burned_area_t
                    burned_in_curr_interval = (burned_area_t > 0)
                    times_burned_in_interval = 1 if burned_area_t > 0 else 0

                    # Annual Potapov forest disturbance raster not used for annual model
                    times_annual_dist_in_interval = 0
                    annual_dist_in_interval = False
                    first_year_annual_dist_during_interval = 0

                    # if interval_end_year == 2017:
                    #     print("burned_area_t:", burned_area_t)
                    #     print("first_year_burned_during_interval:", first_year_burned_during_interval)
                    #     print("burned_in_curr_interval:", burned_in_curr_interval)
                    #     print("times_burned_in_interval:", times_burned_in_interval)
                    #
                    #     if times_burned_in_interval == 1:  # To force it to terminate
                    #         sys.quit()
                else:
                    raise ValueError("interval_length not valid: must be 1 or 5")

                # Updates whether there was a partial or full disturbance (not including fire) in any previous interval.
                part_or_full_dist_in_earlier_intervals = max(part_or_full_dist_in_earlier_intervals, part_or_full_dist_in_curr_interval)

                ## The most recent year of non-tall vegetation composite land cover in the cell before this interval
                most_recent_year_not_tall_veg = most_recent_year_not_tall_veg_block[row, col]

                # Checks whether to update whether the most recent year of non-tall vegetation land cover.
                # Returns the last year that was non-tall vegetation land cover.
                most_recent_year_not_tall_veg = nu.check_most_recent_year_not_tall_veg(LC_curr, LC_prev, most_recent_year_not_tall_veg, interval_end_year)

                # Calculates the maximum canopy height since the last time a pixel was classified as non-tall vegetation land cover.
                # This is eventually used to determine whether current height has decreased significantly from this maximum height over multiple intervals (gradual height loss).
                # Initializes a fixed-length array for vegetation_height_so_far_cell.
                # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba.
                vegetation_height_so_far_cell = np.empty(len(years_so_far), dtype=np.uint8)

                # Populates the fixed-length array by accessing vegetation_heights_all_years
                # Used to determine whether current height has decreased significantly from this maximum height over multiple intervals (gradual height loss).
                # Suggested by https://chatgpt.com/share/e/6724d803-aca4-800a-928c-11d76d38c0ec to work well with numba
                # and speed the code up. I was trying to get vegetation height so far in a variety of ways and it kept being slow.
                # This approach, in conjunction with some the chunk-level preparation above, seems to not slow down the code.
                for i, year_data in enumerate(vegetation_heights_so_far_block):
                    vegetation_height_so_far_cell[i] = year_data[row, col]

                # Returns the maximum vegetation height since the last year of non-tall vegetation land cover
                max_height_since_last_time_not_tall_veg = nu.calc_max_height_since_last_time_not_tall_veg(most_recent_year_not_tall_veg, vegetation_height_so_far_cell, years_so_far)

                # Height change from maximum height since last time not tall veg land cover.
                # Need to recast to signed int8 from uint8 so that negative values (height gain) stay negative.
                height_change_max_curr = np.int8(max_height_since_last_time_not_tall_veg - veg_h_curr)

                # Is height loss significant in absolute change (m) compared to the maximum height since the last time
                # not tall vegetation land cover?
                sig_height_loss_max_current_abs = (height_change_max_curr >= cn.sig_height_loss_threshold_abs)

                # Tracks whether the height has already decreased more than the signif. height loss threshold compared to
                # maximum vegetation height since the last time the pixel was non-tall vegetation land cover.
                # This prevents a pixel from repeatedly (multiple intervals) being counted as having a disturbance compared to
                # the maximum height; this can only be triggered once since the maximum height is attained.
                # This is used to determine if a forest->forest disturbance based on height loss relative to the max height should be reported.
                # 0=no significant height loss relative to the maximum vegetation height since last non-tall vegetation.
                # 1=height loss relative to the maximum vegetation height occurred in this interval.
                # 2=height loss relative to the maximum vegetation height occurred in a previous interval.
                first_time_sig_loss_from_max_height = first_time_sig_loss_from_max_height_block[row,col]

                # Updates the tracker of whether this is the first time that significant height loss relative to
                # the height maximum has occurred to determine if a forest->forest disturbance should be reported.
                # If this is the first interval in which there has been a significant height decrease relative to the maximum
                # height since the last non-tall vegetation interval, the flag is changed to 1 to show it is being reported in this interval.
                # If this height decrease already occurred in a previous interval, the flag is changed to 2 to show it has
                # already been reported.
                if first_time_sig_loss_from_max_height == 1:
                    first_time_sig_loss_from_max_height = 2

                if sig_height_loss_max_current_abs:
                    if first_time_sig_loss_from_max_height == 0:
                        first_time_sig_loss_from_max_height = 1

                ## Calculates height change during the interval. Need to recast to signed int8 from uint8 so that negative values (height gain) stay negative.
                height_change_prev_curr = np.int8(veg_h_prev - veg_h_curr)

                # Is height loss during the interval significant in absolute change (m)?
                sig_height_loss_prev_curr_abs = (height_change_prev_curr >= cn.sig_height_loss_threshold_abs)

                # Is height gain during the interval significant in absolute change (m)?
                # Significant height gain should not occur using annual intervals.
                # However, I'm not forcing sig_height_gain_prev_curr_abs = 0 when using annual intervals in order to try to catch strange cases.
                sig_height_gain_prev_curr_abs = (height_change_prev_curr <= cn.sig_height_gain_threshold_abs)

                # Whether tall vegetation was partially or fully disturbed in the current interval (not counting fire-only disturbance).
                # Conditions are:
                # 1. Partial dist: Annual disturbance raster detected during interval
                # 2. Partial dist: A significant (>=5 m) height reduction during the current interval
                # 3. Partial dist: A significant (>=5 m) height reduction over several intervals, with the threshold being reached during the current interval
                # 4. Full dist: Full loss of tall vegetation during the interval (height went from >=5 m to < 5 m)
                part_or_full_dist_in_curr_interval = sig_height_loss_prev_curr_abs or (first_time_sig_loss_from_max_height == 1) or annual_dist_in_interval or tree_loss


                ### Starting output pixel values

                # Starting decision tree node value
                # All C pool fluxes and densities are kept in Mg C (as opposed to Mg CO2) until they are saved to
                # the output dictionaries. This simplifies arithmetic.
                RF_AGC_final = 0  # The output AGC RF (pre-disturbance, if applicable) (Mg AGC/ha/yr)
                # Need to force arrays into float32 because numba is so particular about datatypes.
                # Initializes dummy output C gross emissions (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_emis_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output C gross removals (Mg C/ha/interval->later converted to Mg CO2/ha/yr): AGC, BGC, deadwood C, litter C.
                c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')
                # Initializes dummy output non-CO2 fluxes (Mg CO2e/ha/interval->later converted to Mg CO2e/ha/yr): CH4, N2O
                non_co2_flux_out = np.array([0, 0]).astype('float32')

                node = 0
                gain_year_count = 0
                agc_ef_out_cell = 0  # AGC emission factor is reported as 0 unless specified otherwise


                ### Mangrove gain
                # Starting age set to 0. AGC and BGC are set to 0 (c_dens_in_NT_T) but existing deadwood and litter carbon pools are used.
                if mang_gain:
                    node = nu.accrete_node(node, 1) # General mangrove code (1)
                    node = nu.accrete_node(node, 1) # Gain of mangroves (11)
                    RF_AGC_final = mangrove_AGC_RF
                    RF_BGC_final = RF_AGC_final * r_s_ratio_mang

                    if mang_loss_in_interval:
                        state_out = nu.accrete_node(node, 1)  # Gain of mangroves + temp loss in interval (111)
                        # Only AGC and BGC pools are used to calculate emissions for temp mangrove loss
                        mang_c_pools_EF_no_fire = cn.biomass_emissions_only

                        (c_gross_emis_out, c_gross_removals_out, c_dens_out, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = (
                            nu.calc_mang_loss(interval_length, first_mang_gain_year, first_mang_loss_year, interval_start_year,
                                mang_c_pools_EF_no_fire, mang_loss_year_in_interval, mang_gain_year_count_pre_loss, mang_gain_year_count_post_loss,
                                RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio_mang, litter_c_ratio_mang))
                        # print(f"Node code is {state_out}, gain of mangroves with temporary disturbance of mangroves that emits biomass C pools only (111)")

                    else:
                        state_out = nu.accrete_node(node, 2)  # Gain of mangroves, no loss in interval (112)
                        (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                            nu.calc_mang(0, first_mang_gain_year, first_mang_loss_year, interval_end_year,
                                mang_gain_year_count_pre_loss, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio_mang, litter_c_ratio_mang))
                        # print(f"Node code is {state_out}, gain of mangroves, no loss in interval (112)")

                    # print(f"c_dens_in_NT_T: {c_dens_in_NT_T}")
                    # print(f"c_dens_out: {c_dens_out}")
                    # print(f"c_gross_removals_out: {c_gross_removals_out}")
                    # print(f"c_gross_emis_out: {c_gross_emis_out}")
                    # print(f"forest_age_end_of_interval: {forest_age_end_of_interval}")


                ### Mangrove loss
                elif mang_loss:
                    node = nu.accrete_node(node, 1)    # General mangrove code (1)
                    node = nu.accrete_node(node, 2)    # Loss of mangroves (12)
                    RF_AGC_final = mangrove_AGC_RF  #TODO replace RF_AGC_final with 0 for 1-year intervals in calc_mang_loss. RF_AGC_final is currently being output even when there's loss.
                    RF_BGC_final = RF_AGC_final * r_s_ratio_mang

                    # If this interval is permanent mangrove loss, then AGC, BGC, deadwood, and litter are used to calculate emissions
                    # Otherwise, only AGC and BGC are used to calculate emissions for temp mangrove loss
                    if (interval_start_year < last_mang_loss_year) and (last_mang_loss_year <= interval_end_year):
                        mang_c_pools_EF_no_fire = cn.all_non_soil_pools
                    else:
                        mang_c_pools_EF_no_fire = cn.biomass_emissions_only

                    (c_gross_emis_out, c_gross_removals_out, c_dens_out, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = (
                        nu.calc_mang_loss(interval_length, first_mang_gain_year, first_mang_loss_year, interval_start_year,
                            mang_c_pools_EF_no_fire, mang_loss_year_in_interval, mang_gain_year_count_pre_loss, mang_gain_year_count_post_loss,
                            RF_AGC_final, RF_BGC_final, c_dens_in, deadwood_c_ratio_mang, litter_c_ratio_mang))

                    if (interval_start_year < last_mang_loss_year) and (last_mang_loss_year <= interval_end_year):
                        if water_LC_curr:
                            state_out = nu.accrete_node(node, 2)    # Permanent loss of mangroves to water (122)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to water (122)")

                        elif LC_curr == cn.cropland:
                            state_out = nu.accrete_node(node, 3)    # Permanent loss of mangroves to cropland (123)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to cropland (123)")

                        elif LC_curr == cn.builtup:
                            state_out = nu.accrete_node(node, 4)    # Permanent loss of mangroves to settlement (124)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to settlement (124)")

                        elif GLAD_short_veg_LC_curr:
                            state_out = nu.accrete_node(node, 5)    # Permanent loss of mangroves to short vegetation (125)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to short vegetation (125)")

                        elif GLAD_tall_veg_LC_curr:
                            state_out = nu.accrete_node(node, 6)    # Permanent loss of mangroves to tall vegetation (126)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to tall vegetation (126)")

                        else:
                            state_out = nu.accrete_node(node, 7)    # Permanent loss of mangroves to anything else (127)
                            # print(f"Node code is {state_out}, permanent loss of mangroves to anything else (127)")
                    else:
                        state_out = nu.accrete_node(node, 1)  # Temporary loss of mangroves (121)
                        # print(f"Node code is {state_out}, temporary loss of mangroves that emits biomass C pools only (121)")


                    # print(f"c_dens_in: {c_dens_in}")
                    # print(f"c_dens_out: {c_dens_out}")
                    # print(f"c_gross_removals_out: {c_gross_removals_out}")
                    # print(f"c_gross_emis_out: {c_gross_emis_out}")
                    # print(f"mang_c_pools_EF_no_fire: {mang_c_pools_EF_no_fire}")
                    # print(f"forest_age_end_of_interval: {forest_age_end_of_interval}")


                ### Mangrove that is mangrove at end of the interval (includes loss with later gain and mangrove remaining mangrove)
                elif mang_remaining_mang:
                    node = nu.accrete_node(node, 1)  # General mangrove code (1)
                    node = nu.accrete_node(node, 3)  # Mangrove remaining mangrove (13)
                    RF_AGC_final = mangrove_AGC_RF
                    RF_BGC_final = RF_AGC_final * r_s_ratio_mang

                    if mang_loss_in_interval:
                        state_out = nu.accrete_node(node, 1)    # Mangrove remaining mangrove + temp loss in interval (131)
                        mang_c_pools_EF_no_fire = cn.biomass_emissions_only

                        (c_gross_emis_out, c_gross_removals_out, c_dens_out, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = (
                            nu.calc_mang_loss(interval_length, first_mang_gain_year, first_mang_loss_year, interval_start_year,
                                mang_c_pools_EF_no_fire, mang_loss_year_in_interval, mang_gain_year_count_pre_loss, mang_gain_year_count_post_loss,
                                RF_AGC_final, RF_BGC_final, c_dens_in, deadwood_c_ratio_mang, litter_c_ratio_mang))
                        # print(f"Node code is {state_out}, mangrove remaining mangrove with temporary disturbance of mangroves that emits biomass C pools only (131)")

                    else:
                        state_out = nu.accrete_node(node, 2)    # Mangrove remaining mangrove, no loss in interval (132)
                        (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                            nu.calc_mang(forest_age_start_of_interval, first_mang_gain_year, first_mang_loss_year, interval_end_year,
                                mang_gain_year_count_pre_loss, RF_AGC_final, RF_BGC_final, c_dens_in, deadwood_c_ratio_mang, litter_c_ratio_mang))
                        # print(f"Node code is {state_out}, mangrove remaining mangrove (132)")

                    # print(f"c_dens_in: {c_dens_in}")
                    # print(f"c_dens_out: {c_dens_out}")
                    # print(f"c_gross_removals_out: {c_gross_removals_out}")
                    # print(f"c_gross_emis_out: {c_gross_emis_out}")
                    # print(f"forest_age_end_of_interval: {forest_age_end_of_interval}")


                ### Before mangrove or non-mangrove remaining non-mangrove or after permanent mangrove loss
                elif before_mang or non_mang_remaining_non_mang or after_mang:
                    node = nu.accrete_node(node, 1)  # General mangrove code (1)
                    node = nu.accrete_node(node, 0)  # Mangrove pixel before or after mangrove detection (0)

                    c_dens_out = np.asarray(c_dens_in_empty, dtype=np.float32)
                    gain_year_count = 0
                    forest_age_end_of_interval = 0

                    if before_mang:
                        state_out = nu.accrete_node(node, 0)  # Before mangrove gain (mangrove mask) (100)
                        # print(f"Node code is {state_out}, before mangrove gain (100)")

                    elif non_mang_remaining_non_mang:
                        state_out = nu.accrete_node(node, 1)  # Non-mangrove remaining non-mangrove (mangrove mask) (101)
                        # print(f"Node code is {state_out}, non-mangrove remaining non-mangrove (mangrove mask) (101)")

                    elif after_mang:
                        state_out = nu.accrete_node(node, 2)  # After permanent mangrove loss (mangrove mask) (102)
                        # print(f"Node code is {state_out}, after permanent mangrove loss (102)")


                ### Tree gain
                elif tree_gain:  # Terrestrial non-trees converted to trees (2)
                    node = nu.accrete_node(node, 2)
                    if all_planted_trees:  # Gain of planted trees (21)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # Gain of oil palm (incl. SDPT oil palm) (211)
                            state_out = nu.accrete_node(node, 1)
                            RF_AGC_final = cn.oil_palm_agc_rf
                            RF_BGC_final = cn.oil_palm_bgc_rf
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=0, litter_c_ratio=0))
                        else: # Gain of non-oil palm planted trees (212)
                            state_out = nu.accrete_node(node, 2)
                            RF_AGC_final = planted_forest_AGC_RF_cell
                            RF_BGC_final = planted_forest_BGC_RF_cell
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=0, litter_c_ratio=0))
                    else:  # Gain of non-planted trees (22)
                        node = nu.accrete_node(node, 2)
                        if GLAD_tall_veg_LC_curr:  # Gain of terrestrial natural forest (221)
                            state_out = nu.accrete_node(node, 1)
                            RF_AGC_final = natrl_forest_curve_0_5_AGC_RF   # Forces new forest to use the first interval of the age curve
                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_NT_T, deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang))
                        else:  # Gain of trees outside forests (222) (uses c_dens_in_empty because ToF have no residual carbon in any pool)
                            state_out = nu.accrete_node(node, 2)
                            RF_AGC_final = cn.trees_outside_forests_agc_rf_max
                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                            (c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = (
                                nu.calc_NT_T(interval_length, RF_AGC_final, RF_BGC_final, c_dens_in_empty, deadwood_c_ratio=0, litter_c_ratio=0))

                ### Tree loss
                elif tree_loss:  # Trees converted to non-trees (3)
                    node = nu.accrete_node(node, 3)
                    composite_primary_cell = 0   # Sets composite primary forest value to 0 for this entire branch because loss has occurred
                    if all_planted_trees:  # Full loss of planted trees (31)
                        node = nu.accrete_node(node, 1)
                        if all_oil_palm:  # Full loss of oil palm (incl. SDPT) (311->3119/3112)  #TODO This could have a conversion to short veg option (with short veg post-loss removals)
                            node = nu.accrete_node(node, 1)
                            agc_rf_in = cn.oil_palm_agc_rf  # 5-year intervals only
                            bgc_rf_in = cn.oil_palm_bgc_rf  # 5-year intervals only
                            c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                            c_pools_EF_fire_non_CO2 = cn.biomass_emissions_only
                            c_pools_EF_no_fire = cn.biomass_emissions_only
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                            (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                             RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Full loss of non-oil palm planted trees (312)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Full loss of non-oil palm planted trees as cropland (3121)
                                node = nu.accrete_node(node, 1)
                                if planted_forest_tree_crop_cell == 2:  # Full loss of non-oil palm tree crops as cropland (31211->312119/312112)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Full loss of non-oil palm planted forest as cropland (31212->312129/312122)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.all_non_soil_pools
                                    rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                            elif GLAD_short_veg_LC_curr:  # Full loss of non-oil palm planted trees as short vegetation (3122)
                                node = nu.accrete_node(node, 2)
                                if planted_forest_tree_crop_cell == 2:  # Full loss of non-oil palm tree crops as short vegetation (31221->312219/312212)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = short_veg_AGC_BGC_RF_adj
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Full loss of non-oil palm planted forest as short vegetation (31222->312229/312222)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = short_veg_AGC_BGC_RF_adj
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == cn.builtup:  # Full loss of non-oil palm planted trees to settlement (3123)
                                node = nu.accrete_node(node, 3)
                                if planted_forest_tree_crop_cell == 2:  # Full loss of non-oil palm tree crops to settlement (31231->312319/312312)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Full loss of non-oil palm planted forest to settlement (31232->312329/312322)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.all_non_soil_pools
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Full loss of non-oil palm planted trees to anything else (3124)
                                node = nu.accrete_node(node, 4)
                                if planted_forest_tree_crop_cell == 2:  # Full loss of non-oil palm tree crops to anything else (31241->312412) (no fire emissions allowed)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    burned_in_curr_interval = 0  # This particular node can't have fire emissions, so this is forced to 0
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Full loss of non-oil palm planted forest to anything else (31242->312422) (no fire emissions allowed)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = planted_forest_AGC_RF_cell  # 5-year intervals only
                                    bgc_rf_in = planted_forest_BGC_RF_cell  # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.biomass_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    burned_in_curr_interval = 0  # This particular node can't have fire emissions, so this is forced to 0
                                    rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio=0, litter_c_ratio=0)
                    else:  # Full loss of non-planted trees (32)
                        node = nu.accrete_node(node, 2)
                        if GLAD_tall_veg_LC_prev:  # Full loss of natural forest (321)
                            node = nu.accrete_node(node, 1)
                            if LC_curr == cn.cropland:  # Natural forest converted to cropland (3211->32119/32112)
                                node = nu.accrete_node(node, 1)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            elif GLAD_short_veg_LC_curr:  # Natural forest converted to short vegetation (3212)
                                node = nu.accrete_node(node, 2)
                                if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (32121->321219/321212)
                                    node = nu.accrete_node(node, 1)
                                    agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                    bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                    c_pools_EF_no_fire = cn.all_non_soil_pools
                                    rf_post_dist = short_veg_AGC_BGC_RF_adj
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                                else:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (32122->321229/321222)
                                    node = nu.accrete_node(node, 2)
                                    agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                    bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.all_but_bgc_emissions
                                    c_pools_EF_no_fire = cn.biomass_emissions_only
                                    rf_post_dist = short_veg_AGC_BGC_RF_adj
                                    (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                     RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                        node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                        rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                        deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            elif LC_curr == cn.builtup:  # Natural forest converted to settlement (3213->32139/32132)
                                node = nu.accrete_node(node, 3)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                                c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                                c_pools_EF_no_fire = cn.all_non_soil_pools
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                            else:  # Natural forest converted to anything else (wetland/open water/ice, etc.) (3214->32142) (no fire emissions allowed)
                                node = nu.accrete_node(node, 4)
                                agc_rf_in = natrl_forest_age_dependent_agc_rf  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang     # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.biomass_emissions_only  # Fire emissions are treated as non-fire emissions
                                c_pools_EF_fire_non_CO2 = np.array([0, 0, 0, 0]).astype('float32') # This particular node can't have fire emissions-- no non-CO2 emissions
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                burned_in_curr_interval = 0  # This particular node can't have fire emissions, so this is forced to 0
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                        else:  # Full loss of trees outside forests (322)  (slightly compressed variable assignments compared to elsewhere)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Full loss of trees outside forests converted to cropland (3221->32219/32212)
                                node = nu.accrete_node(node, 1)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang  # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([cn.cropland_rf, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif GLAD_short_veg_LC_curr:  # Full loss of trees outside forests converted to short vegetation (3222->32229/32222)
                                node = nu.accrete_node(node, 2)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = short_veg_AGC_BGC_RF_adj
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            elif LC_curr == cn.builtup:  # Full loss of trees outside forests converted to settlement (3223->32239/32232)
                                node = nu.accrete_node(node, 3)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Full loss of trees outside forests converted to anything else (3224->32242) (no fire emissions allowed)
                                node = nu.accrete_node(node, 4)
                                agc_rf_in = cn.trees_outside_forests_agc_rf_max  # 5-year intervals only
                                bgc_rf_in = agc_rf_in * r_s_ratio_non_mang       # 5-year intervals only
                                c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                c_pools_EF_no_fire = cn.biomass_emissions_only
                                burned_in_curr_interval = 0  # This particular node can't have fire emissions, so this is forced to 0
                                rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                 RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                                    node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                    c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                    rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest, Gef_n2o_forest,
                                    deadwood_c_ratio=0, litter_c_ratio=0)

                ### Trees remaining trees
                elif (tree_prev) and (tree_curr):  # Trees remaining trees (4)
                    node = nu.accrete_node(node, 4)
                    if interval_before_converted_to_oil_palm and (not oil_palm_pre_2000): # Non-planted trees with oil palm planted in the next interval (41->419/412)
                        node = nu.accrete_node(node, 1)
                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                        c_pools_EF_fire_CO2 = cn.all_non_soil_pools
                        c_pools_EF_fire_non_CO2 = cn.all_non_soil_pools
                        c_pools_EF_no_fire = cn.all_non_soil_pools
                        # If annual interval, no oil palm removals in the interval of loss
                        if interval_length == 1:
                            rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                        # If 5-year interval, there is one year of oil palm removals in the interval of loss, regardless of the year.
                        # This is for simplicity: the post-conversion oil palm RF can be applied one time, like cropland or short veg RFs.
                        elif interval_length == 5:
                            rf_post_dist = np.array([cn.oil_palm_agc_rf, cn.oil_palm_agc_rf, 0, 0]).astype('float32')
                        else:
                            raise ValueError("interval_length not valid: must be 1 or 5")
                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_NT(
                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                            rf_post_dist, most_recent_year_not_tall_veg, Cf_forest, Gef_ch4_forest,
                            Gef_n2o_forest, deadwood_c_ratio_non_mang, litter_c_ratio_non_mang)
                        composite_primary_cell = 0  # Sets composite primary forest value to 0 for this entire branch because loss has occurred
                    else:  # Trees remaining trees- no conversion to oil palm (42)
                        node = nu.accrete_node(node, 2)
                        if part_or_full_dist_in_curr_interval:  # Trees partially disturbed in the current interval (421)
                            node = nu.accrete_node(node, 1)
                            composite_primary_cell = 0  # Sets composite primary forest value to 0 for this entire branch because disturbance has occurred
                            if all_planted_trees:   # Planted trees partially disturbed in the current interval (4211)
                                node = nu.accrete_node(node, 1)
                                if sig_height_gain_prev_curr_abs:  # Oil palm/planted trees partially disturbed in the current interval with signif. height increase after (42111)
                                    # NOTE: This should only occur with 5-year interval data, not annual data.
                                    node = nu.accrete_node(node, 1)
                                    # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                    if all_oil_palm:  # Oil palm partially disturbed in the current interval with signif. height increase after (421111->4211119/4211112)
                                        node = nu.accrete_node(node, 1)
                                        agc_rf_in = cn.oil_palm_agc_rf
                                        bgc_rf_in = cn.oil_palm_bgc_rf
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([agc_rf_in, bgc_rf_in, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                    else: # Planted trees partially disturbed in the current interval with signif. height increase after (421112->4211129/4211122)
                                        node = nu.accrete_node(node, 2)
                                        agc_rf_in = planted_forest_AGC_RF_cell
                                        bgc_rf_in = planted_forest_BGC_RF_cell
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([agc_rf_in, bgc_rf_in, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                else:  # Oil palm/planted trees partially disturbed in the current interval without signif. height increase after (42112)
                                    # NOTE: All annual interval data is expected to use this branch.
                                    node = nu.accrete_node(node, 2)
                                    # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                    if all_oil_palm: # Oil palm partially disturbed in the current interval without signif. height increase after (421121->4211219/4211212)
                                        node = nu.accrete_node(node, 1)
                                        agc_rf_in = cn.oil_palm_agc_rf
                                        bgc_rf_in = cn.oil_palm_bgc_rf
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                    else: # Planted trees partially disturbed in the current interval without signif. height increase after (421122->4211229/4211222)
                                        node = nu.accrete_node(node, 2)
                                        agc_rf_in = planted_forest_AGC_RF_cell
                                        bgc_rf_in = planted_forest_BGC_RF_cell
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in,
                                            c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Non-planted trees partially disturbed in the current interval (4212)
                                node = nu.accrete_node(node, 2)
                                if GLAD_tall_veg_LC_curr:  # Forest partially disturbed in the current interval (42121)
                                    node = nu.accrete_node(node, 1)
                                    if sig_height_gain_prev_curr_abs:  # Forest partially disturbed in the current interval with signif. height increase after (421211->4212119/4212112)
                                        # NOTE: This should only occur with 5-year interval data, not annual data.
                                        node = nu.accrete_node(node, 1)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        agc_rf_post = natrl_forest_curve_0_5_AGC_RF # Post-dist RF is 0-5 year secondary forest
                                        bgc_rf_post = agc_rf_post * r_s_ratio_non_mang
                                        rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                    else:  # Forest partially disturbed in the current interval without signif. height increase after (421212->4212129/4212122)
                                        # NOTE: All annual interval data is expected to use this branch.
                                        node = nu.accrete_node(node, 2)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = natrl_forest_age_dependent_agc_rf
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                else:  # Trees outside forests partially disturbed in the current interval (42122)
                                    node = nu.accrete_node(node, 2)
                                    if sig_height_gain_prev_curr_abs:  # Trees outside forests partially disturbed in the current interval with signif. height increase after (421221->4212219/4212212)
                                        # NOTE: This should only occur with 5-year interval data, not annual data.
                                        node = nu.accrete_node(node, 1)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        agc_rf_post = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_post = agc_rf_post * r_s_ratio_non_mang
                                        rf_post_dist = np.array([agc_rf_post, bgc_rf_post, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=0, litter_c_ratio=0)
                                    else:  # Trees outside forests partially disturbed in the current interval without signif. height increase after (421222->4212229/4212222)
                                        # NOTE: All annual interval data is expected to use this branch.
                                        node = nu.accrete_node(node, 2)
                                        # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                        agc_rf_in = cn.trees_outside_forests_agc_rf_max
                                        bgc_rf_in = agc_rf_in * r_s_ratio_non_mang
                                        rf_post_dist = np.array([0, 0, 0, 0]).astype('float32')  # No post-disturbance RFs or removals
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        c_pools_EF_no_fire = np.array([partial_disturbance_EF_for_driver, partial_disturbance_EF_for_driver, 0, 0]).astype('float32')
                                        (state_out, c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out,
                                         RF_AGC_final, RF_BGC_final, agc_ef_out_cell, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_non_stand_disturbs(
                                            node, interval_length, burned_in_curr_interval, agc_rf_in, bgc_rf_in, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            c_pools_EF_no_fire, first_year_annual_dist_during_interval, interval_end_year, c_dens_in_ToF,
                                            rf_post_dist, most_recent_year_not_tall_veg,
                                            Cf_forest, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=0, litter_c_ratio=0)
                        else:  # Trees not disturbed in the current interval (422)
                            node = nu.accrete_node(node, 2)
                            if all_planted_trees:  # Oil palm/planted trees not disturbed in the current interval (4221)
                                node = nu.accrete_node(node, 1)
                                # Calculation function only uses the RFs for 5-year intervals but assigning them regardless of interval type for consistency
                                if all_oil_palm:  # Oil palm not disturbed in the current interval (42211->422119/422112)
                                    node = nu.accrete_node(node, 1)
                                    RF_AGC_final = cn.oil_palm_agc_rf
                                    RF_BGC_final = cn.oil_palm_bgc_rf
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_start_of_interval,
                                        first_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                                else: # Planted trees not disturbed in the current interval (42212->422129/422122)
                                    node = nu.accrete_node(node, 2)
                                    RF_AGC_final = planted_forest_AGC_RF_cell
                                    RF_BGC_final = planted_forest_BGC_RF_cell
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_start_of_interval,
                                        first_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                        cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)
                            else:  # Non-planted trees not disturbed in last interval (4222)
                                node = nu.accrete_node(node, 2)
                                if GLAD_tall_veg_LC_curr:  # Natural forest not disturbed in last interval (42221)
                                    node = nu.accrete_node(node, 1)
                                    if (most_recent_year_not_tall_veg > 0) or (part_or_full_dist_in_earlier_intervals > 0):  # Young secondary natural forest (422211->4222119/4222112)
                                        node = nu.accrete_node(node, 1)
                                        RF_AGC_final = natrl_forest_age_dependent_agc_rf
                                        RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                        c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                        c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                        (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                         non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                            node, interval_length, forest_age_start_of_interval, first_year_burned_during_interval,
                                            RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                            interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                            cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                            deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                    else:  # Natural forest undisturbed since model start (422212)
                                        node = nu.accrete_node(node, 2)
                                        if composite_primary_cell == 1:  # Primary forest undisturbed since model start (4222121->42221219/42221212)
                                            node = nu.accrete_node(node, 1)
                                            RF_AGC_final = primary_forest_AGC_RF
                                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                            c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                            c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                            (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                             non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                                node, interval_length, forest_age_start_of_interval, first_year_burned_during_interval,
                                                RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                                cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                                deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                        else: # Old secondary forest undisturbed since model start (4222122->42221229/42221222)
                                            node = nu.accrete_node(node, 2)
                                            RF_AGC_final = natrl_forest_age_dependent_agc_rf
                                            RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                            c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                            c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                            (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                             non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                                node, interval_length, forest_age_start_of_interval, first_year_burned_during_interval,
                                                RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg,
                                                cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest,
                                                deadwood_c_ratio=deadwood_c_ratio_non_mang, litter_c_ratio=litter_c_ratio_non_mang)
                                else:  # Trees outside forests not disturbed in the current interval (42222->422229/422222)
                                    node = nu.accrete_node(node, 2)
                                    RF_AGC_final = cn.trees_outside_forests_agc_rf_max
                                    RF_BGC_final = RF_AGC_final * r_s_ratio_non_mang
                                    c_pools_EF_fire_CO2 = cn.agc_emissions_only
                                    c_pools_EF_fire_non_CO2 = cn.agc_emissions_only
                                    (state_out, c_gross_emis_out, c_gross_removals_out, agc_ef_out_cell,
                                     non_co2_flux_out, c_dens_out, gain_year_count, forest_age_end_of_interval) = nu.calc_T_T_no_disturbs(
                                        node, interval_length, forest_age_start_of_interval, first_year_burned_during_interval,
                                        RF_AGC_final, RF_BGC_final, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2,
                                        interval_end_year, c_dens_in_ToF, most_recent_year_not_tall_veg,
                                        cn.Cf_forest_undisturbed, Gef_co2_forest, Gef_ch4_forest, Gef_n2o_forest, deadwood_c_ratio=0, litter_c_ratio=0)

                ### Non-cropland/non-tree to cropland (without trees)
                elif (LC_prev != cn.cropland) and (LC_curr == cn.cropland):
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    state_out = nu.accrete_node(node, 1)  # Cropland gain (51)
                    c_pools_EF_no_fire = cn.all_non_soil_pools  # Fire not considered in intervals with cropland gain, so no fire option
                    RF_AGC_final = cn.cropland_rf
                    agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                    rf_array = np.array([RF_AGC_final, 0, 0, 0]).astype('float32')
                    forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                    c_gross_emis_out, c_gross_removals_out, c_dens_out = nu.calc_NT_cropland_gain(c_pools_EF_no_fire, c_dens_in, rf_array)
                ### Cropland converted to non-cropland (without trees)
                elif (LC_prev == cn.cropland) and (LC_curr != cn.cropland):
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    node = nu.accrete_node(node, 2)  # Annual cropland loss (52)
                    if GLAD_short_veg_LC_curr:
                        node = nu.accrete_node(node, 1)  # Annual cropland converted to short vegetation (521->5219/5212)
                        c_pools_EF_no_fire = cn.agc_emissions_only  # There should only be AGC in cropland anyway
                        RF_AGC_final = short_veg_AGC_BGC_RF_adj[0]  # Sets the output RF to use the AGC short veg gain RF
                        c_dens_in = c_dens_in_cropland
                        agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                        rf_post_dist = short_veg_AGC_BGC_RF_adj  # Post conversion removals to short veg
                        forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                        (state_out, c_gross_emis_out, c_gross_removals_out,
                         c_dens_out, non_co2_flux_out) = nu.calc_cropland_non_cropland(node, c_dens_in, c_pools_EF_no_fire, times_burned_in_interval, rf_post_dist)
                    elif water_LC_curr:
                        node = nu.accrete_node(node, 2)  # Annual cropland converted to water (522->5222) (no fire option)
                        c_pools_EF_no_fire = cn.agc_emissions_only  # There should only be AGC in cropland anyway
                        c_dens_in = c_dens_in_cropland
                        agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                        rf_post_dist = np.array([0.0, 0.0, 0.0, 0.0]).astype('float32')  # No post-conversion removals
                        forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                        # No fire emissions when cropland is converted to water. Simplest way is to just overwrite the burned count.
                        (state_out, c_gross_emis_out, c_gross_removals_out,
                         c_dens_out, non_co2_flux_out) = nu.calc_cropland_non_cropland(node, c_dens_in, c_pools_EF_no_fire, 0, rf_post_dist)
                    else:
                        node = nu.accrete_node(node, 3)  # Annual cropland converted to anything else (522->5239/5232) (fire option permitted because water is its own branch)
                        c_pools_EF_no_fire = cn.agc_emissions_only  # There should only be AGC in cropland anyway
                        c_dens_in = c_dens_in_cropland
                        agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                        rf_post_dist = np.array([0.0, 0.0, 0.0, 0.0]).astype('float32')  # No post-conversion removals
                        forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                        (state_out, c_gross_emis_out, c_gross_removals_out,
                         c_dens_out, non_co2_flux_out) = nu.calc_cropland_non_cropland(node, c_dens_in, c_pools_EF_no_fire, times_burned_in_interval, rf_post_dist)
                ### Cropland remaining cropland (without trees)
                elif (LC_prev == cn.cropland) and (LC_curr == cn.cropland):
                    node = nu.accrete_node(node, cn.cropland_node)  # General cropland node code (5)
                    node = nu.accrete_node(node, 3)  # Cropland remaining cropland (53->539/532)
                    c_dens_in = c_dens_in_cropland
                    forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_cropland_cropland(node, c_dens_in, times_burned_in_interval)

                ### Non-tree/cropland converted to short vegetation
                ### Requires 1/2) GLAD LC change and 3) GPW height shows sufficient veg at end of interval
                elif (not GLAD_short_veg_LC_prev) and (GLAD_short_veg_LC_curr) and (GPW_short_veg_curr):
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    state_out = nu.accrete_node(node, 1)  # Short vegetation gain (61)
                    rf_array = short_veg_AGC_BGC_RF_adj
                    RF_AGC_final = short_veg_AGC_BGC_RF_adj[0]   # Sets the output RF to the short veg gain RF
                    forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                    c_gross_emis_out, c_gross_removals_out, c_dens_out = nu.calc_short_veg_gain(rf_array)
                ### Short vegetation converted to non-short vegetation, non-forest or non-cropland
                ### Requires 1/2) GLAD LC change, 3) GPW height shows sufficient veg at start of interval, and 4) GPW shows vegetation too short at end of interval
                elif (GLAD_short_veg_LC_prev) and (not GLAD_short_veg_LC_curr) and (GPW_short_veg_prev) and (not GPW_short_veg_curr):
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    node = nu.accrete_node(node, 2)  # Short vegetation loss (62)
                    if water_LC_curr:
                        node = nu.accrete_node(node, 1)  # Short vegetation loss converted to water (621->6212) (no fire option)
                        c_dens_in = c_dens_in_short_veg
                        c_pools_EF_no_fire = cn.biomass_emissions_only
                        agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                        forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                        # No fire emissions when cropland is converted to water. Simplest way is to just overwrite the burned count.
                        (state_out, c_gross_emis_out, c_gross_removals_out,
                         c_dens_out, non_co2_flux_out) = nu.calc_short_veg_loss(node, c_dens_in, c_pools_EF_no_fire, 0)
                    else:
                        node = nu.accrete_node(node, 2)  # Short vegetation loss converted to non-water (622->6229/6222)
                        c_dens_in = c_dens_in_short_veg
                        c_pools_EF_no_fire = cn.biomass_emissions_only
                        agc_ef_out_cell = c_pools_EF_no_fire[0]  # Emission factor used for output geotif
                        forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                        (state_out, c_gross_emis_out, c_gross_removals_out,
                         c_dens_out, non_co2_flux_out) = nu.calc_short_veg_loss(node, c_dens_in, c_pools_EF_no_fire, times_burned_in_interval)
                ### Short vegetation remaining short vegetation
                elif GLAD_short_veg_LC_prev and GLAD_short_veg_LC_curr:
                    node = nu.accrete_node(node, cn.grassland_node)  # General short veg node code (6)
                    node = nu.accrete_node(node, 3)  # Short vegetation remaining short vegetation (63->639/632)
                    c_dens_in = c_dens_in_short_veg
                    forest_age_end_of_interval = 0  # Sets forest age to 0 because there's no forest
                    (state_out, c_gross_emis_out, c_gross_removals_out,
                     c_dens_out, non_co2_flux_out) = nu.calc_short_veg_short_veg(node, c_dens_in, times_burned_in_interval)

                # When decision trees above do not apply (generally, carbon-less, non-vegetated land covers)
                else:

                    state_out = cn.other_landcover_node

                    # If no decision tree branches apply, forest age set to 0 years
                    forest_age_end_of_interval = 0

                    # If no decision tree branches apply, AGC and BGC densities are set to 0 Mg C/ha.
                    # Deadwood and litter C can have residual carbon because otherwise deadwood and litter carbon
                    # not emitted during tree loss will just disappear from the model without ever being emitted.
                    c_dens_out = np.array([0, 0, deadwood_c_dens_in, litter_c_dens_in]).astype('float32')


                ### Populates the output arrays with the calculated fluxes and densities

                # Forces output carbon densities that are very slightly negative or positive (due to rounding errors) to 0 for simplicity
                for i, c_pool_out in enumerate(c_dens_out):
                    if (c_pool_out > -0.0001) and (c_pool_out < 0.0001):
                        c_dens_out[i] = 0

                # Stops model if state_out is more digits than the expected maximum (currently 6).
                # This means that the tree is deeper than expected and max_digits_state_out needs to be increased.
                if state_out > (10 ** max_digits_state_out)-1:
                    raise ValueError("Maximum state_out is greater than the expected number of digits")

                # Converts the state to 8 digits (trailing 0s) for consistency across all nodes
                state_out = nu.pad_to_8_digits(state_out, max_digits_state_out)

                state_out_block[row, col] = state_out

                # Sets the RF from this interval as the RF from the previous interval for the next interval (Mg AGC/ha/yr)
                agc_rf_pre_dist_out_block[row, col] = RF_AGC_final

                # Gross emissions for each C pool for the interval before conversion to CO2 (Mg C/ha/interval)
                agc_gross_emis_out_block[row, col] = c_gross_emis_out[0]
                bgc_gross_emis_out_block[row, col] = c_gross_emis_out[1]
                deadwood_c_gross_emis_out_block[row, col] = c_gross_emis_out[2]
                litter_c_gross_emis_out_block[row, col] = c_gross_emis_out[3]

                # Non-CO2 emissions for the interval (Mg CO2e/ha/interval)
                ch4_gross_emis_out_block[row, col] = non_co2_flux_out[0]
                n2o_gross_emis_out_block[row, col] = non_co2_flux_out[1]

                # Gross removals for each C pool for the interval before conversion to CO2 (Mg C/ha/interval)
                agc_gross_removals_out_block[row, col] = c_gross_removals_out[0]
                bgc_gross_removals_out_block[row, col] = c_gross_removals_out[1]
                deadwood_c_gross_removals_out_block[row, col] = c_gross_removals_out[2]
                litter_c_gross_removals_out_block[row, col] = c_gross_removals_out[3]

                # C density for each C pool at the end of the interval (Mg C/ha)
                agc_dens_block[row, col] = c_dens_out[0]
                bgc_dens_block[row, col] = c_dens_out[1]
                deadwood_c_dens_block[row, col] = c_dens_out[2]
                litter_c_dens_block[row, col] = c_dens_out[3]

                # Test/intermediate outputs
                forest_age_end_of_interval_block[row, col] = forest_age_end_of_interval
                gain_year_count_out_block[row, col] = gain_year_count
                most_recent_year_not_tall_veg_block[row, col] = most_recent_year_not_tall_veg
                max_height_since_last_time_not_tall_veg_block[row, col] = max_height_since_last_time_not_tall_veg
                first_time_sig_loss_from_max_height_block[row, col] = first_time_sig_loss_from_max_height
                part_or_full_dist_in_earlier_intervals_block[row, col] = part_or_full_dist_in_earlier_intervals
                part_or_full_dist_in_curr_interval_block[row, col] = part_or_full_dist_in_curr_interval
                times_burned_in_interval_block[row, col] = times_burned_in_interval
                agc_ef_out_block[row, col] = agc_ef_out_cell
                composite_primary_block[row, col] = composite_primary_cell

        # os.quit()   # For testing the first interval

        ### End of one iteration calculations and outputs

        # Adds the output arrays to the dictionary with the appropriate data type
        # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
        # This applies even for the outputs that aren't reused in the next interval;
        # they will still get overwritten with the final interval's values, I believe.

        out_dict_uint32[f"{cn.land_state_pattern}_{interval_end_year}"] = state_out_block.copy()

        out_dict_float32[f"{cn.agc_rf_pre_dist_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = agc_rf_pre_dist_out_block.copy()

        # Converts carbon pool fluxes from Mg C/ha/interval to Mg CO2/ha/yr.
        # Gross emissions are positive. Gross removals are negative.
        out_dict_float32[f"{cn.agc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (agc_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.bgc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (bgc_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (deadwood_c_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.litter_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (litter_c_gross_emis_out_block * cn.C_to_CO2_numba / interval_length).copy()

        out_dict_float32[f"{cn.agc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (agc_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.bgc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (bgc_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (deadwood_c_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()
        out_dict_float32[f"{cn.litter_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (litter_c_gross_removals_out_block * cn.C_to_CO2_numba / interval_length).copy()

        # Converts non-CO2 emissions from Mg CO2e/ha/interval to Mg CO2e/ha/yr. No conversion of Mg C/ha to Mg CO2 because these are already in Mg CO2e/ha.
        out_dict_float32[f"{cn.ch4_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (ch4_gross_emis_out_block / interval_length).copy()
        out_dict_float32[f"{cn.n2o_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (n2o_gross_emis_out_block / interval_length).copy()

        # Still Mg C/ha at the interval end year
        out_dict_float32[f"{cn.agc_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = agc_dens_block.copy()
        out_dict_float32[f"{cn.bgc_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = bgc_dens_block.copy()
        out_dict_float32[f"{cn.deadwood_c_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = deadwood_c_dens_block.copy()
        out_dict_float32[f"{cn.litter_c_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = litter_c_dens_block.copy()

        # Summative outputs (Mg CO2(e)/ha/yr)
        # Gross emissions across all carbon pools
        out_dict_float32[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.agc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.bgc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.litter_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Gross emissions for non-CO2 emissions
        out_dict_float32[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.ch4_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.n2o_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Gross emissions for all carbon pools and all gases
        out_dict_float32[f"{cn.gross_emis_all_C_pools_all_gases_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
            out_dict_float32[f"{cn.gross_emis_all_C_pools_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
            + out_dict_float32[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
        )

        # Gross removals across all carbon pools
        out_dict_float32[f"{cn.gross_removals_all_C_pools_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.agc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.bgc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.litter_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Net flux for each carbon pool
        out_dict_float32[f"{cn.net_flux_agc_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.agc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.agc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])
        out_dict_float32[f"{cn.net_flux_bgc_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.bgc_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.bgc_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])
        out_dict_float32[f"{cn.net_flux_deadwood_c_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])
        out_dict_float32[f"{cn.net_flux_litter_c_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.litter_c_gross_emis_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.litter_c_gross_removals_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Net flux across all carbon pools but for CO2 only
        out_dict_float32[f"{cn.net_flux_all_C_pools_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.net_flux_agc_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.net_flux_bgc_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.net_flux_deadwood_c_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.net_flux_litter_c_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Net flux across all carbon pools, plus non-pool non-CO2 emissions
        out_dict_float32[f"{cn.net_flux_all_C_pools_all_gases_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.net_flux_all_C_pools_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.gross_emis_all_C_pools_non_CO2_only_pattern}{cn.flux_density_pixel_meaning}_{interval_end_year}"])

        # Carbon density for all non-soil C pools (Mg C)
        out_dict_float32[f"{cn.non_soil_c_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"] = (
                out_dict_float32[f"{cn.agc_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.bgc_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.deadwood_c_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"]
                + out_dict_float32[f"{cn.litter_c_modeled_dens_pattern}{cn.C_density_pixel_meaning}_{interval_end_year}"])

        # Intermediate outputs
        out_dict_uint16[f"{cn.forest_age_output_pattern}_{interval_end_year}"] = forest_age_end_of_interval_block.copy()
        out_dict_uint8[f"{cn.gain_year_count_pattern}_{interval_end_year}"] = gain_year_count_out_block.copy()
        out_dict_uint16[f"{cn.most_recent_year_not_tall_veg}_{model_start_year}_{interval_end_year}"] = most_recent_year_not_tall_veg_block.copy()    # Years represent from model start to current interval end
        out_dict_uint8[f"{cn.max_height_since_last_time_not_tall_veg}_{interval_end_year}"] = max_height_since_last_time_not_tall_veg_block.copy()
        out_dict_uint8[f"{cn.first_time_sig_loss_from_max_height}_{interval_end_year}"] = first_time_sig_loss_from_max_height_block.copy()
        out_dict_uint8[f"{cn.part_or_full_dist_in_earlier_intervals}_{interval_end_year}"] = part_or_full_dist_in_earlier_intervals_block.copy()
        out_dict_uint8[f"{cn.part_or_full_dist_in_curr_interval}_{interval_end_year}"] = part_or_full_dist_in_curr_interval_block.copy()
        out_dict_uint8[f"{cn.times_burned_in_interval}_{interval_end_year}"] = times_burned_in_interval_block.copy()
        out_dict_float32[f"{cn.agc_emission_factor}_{interval_end_year}"] = agc_ef_out_block.copy()
        out_dict_uint8[f"{cn.composite_primary_forest}_{interval_end_year}"] = composite_primary_block.copy()

    return out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32


# Downloads inputs, prepares data, calculates vegetation stocks and fluxes, and uploads outputs to s3
def calculate_and_upload_vegetation_fluxes(bounds, primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                                           download_dict_with_data_types, start_year, end_year, interval_type, interval_year_diff_list,
                                           interval_length_list, interval_end_years, is_large_run, no_upload, create_zarr,
                                           output_folders, stage, model_type, zarr_path=None, outputs_to_zarr=None):

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    chunk_start_time = time.time()

    uu.rename_s3_task_file(stage, bounds, "preprocessing_", is_large_run, logger_worker)

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    # Report the number of retries for the task. Untested.
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694bfc7f-fab0-8332-b903-d5efa84b61c3
    retry_env_var = os.environ.get("DASK_TASK_RETRIES", "0")
    retry_count = int(retry_env_var)

    if retry_count > 0:
        msg = f"Running vegetation flux task for {bounds_str} in {tile_id} (retry #{retry_count}: {uu.timestr()})"
        lu.print_and_log(msg, False, logger_worker)

    # # Can potentially add to prevent indefinite retries. Untested.
    # if retry_count >= 2:
    #     raise RuntimeError(f"Tile {bounds_str} in {tile_id} failed twice — exiting to prevent infinite retries.")


    ### Part 1: Downloads all inputs for chunk.
    ### No checks about whether the chunk has data because the way the chunk_list is constructed,
    ### every chunk is relevant and should be processed, so they don't need to be checked.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # Adds the uri for the global COGS of Global Pasture Watch median vegetation height for each year to the download dictionary
    for year in list(range(2015, 2025)):
        MVH_uri_year = cn.GPW_MVH_uri.replace('YYYY', str(year))
        updated_download_dict[f"{cn.GPW_MVH_pattern}_{year}"] = [MVH_uri_year, 'Int16']

    # print("updated_download_dict:", updated_download_dict)

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    # futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_large_run, logger_worker, False)
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_large_run, logger_worker, False)
    # print(futures)

    lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # Dictionary that stores the dataset name (key) and downloaded data and their statuses (values)
    layers = {}

    # Ensures futures stores Future objects
    # Revised with https://chatgpt.com/share/e/67bde66c-d9a0-800a-a524-a9ef88c641a2 to return status messages for chunks
    # Revised with Claude session 'SOC stock script hang at task 3799'
    try:
        for future in concurrent.futures.as_completed(futures, timeout=cn.download_timeout):
            layer = futures[future]
            data, status = future.result()
            if 'success' not in status:
                lu.print_and_log(f"{status}: {uu.timestr()}", False, logger_worker)
            layers[layer] = data
    except concurrent.futures.TimeoutError:
        raise RuntimeError(f"Download timed out after {cn.download_timeout}s for {bounds_str} ({tile_id}): {uu.timestr()}")

    # Test prints
    # print(layers)
    # print(layers['burned_area_2015'].max())
    # print(layers[cn.climate_zone_pattern].max())
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern])
    # print(layers[cn.planted_forest_AGC_BGC_removal_factor_pattern].max())
    # print(layers[cn.forest_age_start_year_pattern].dtype)
    # print(layers[cn.climate_zone_pattern].dtype)
    # print(layers['GPW_height_2015'].dtype)
    # print("layers['GPW_height_2015']:", layers['GPW_height_2015'])


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))
    # print(chunk_stats)


    ### Part 3: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
    ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
    ### Note: need to add new code if inputs with other data types are added

    lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # print("uint8_typed_list:", typed_dict_uint8)
    # print("uint16_typed_list:", typed_dict_uint16)
    # print("int16_typed_list:", typed_dict_int16)
    # print("int32_typed_list:", typed_dict_int32)
    # print("float32_typed_list:", typed_dict_float32)

    # Frees up a little memory (~15 MB) before moving on
    del updated_download_dict
    del futures
    gc.collect()


    ### Part 4: Calculates vegetation fluxes and densities

    lu.print_and_log(f"Calculating vegetation fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)
    uu.rename_s3_task_file(stage, bounds, "calculating_", is_large_run, logger_worker)
    calc_start = time.time()

    out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32 = vegetation_fluxes(
        typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32,
        primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
        start_year, end_year, interval_type, interval_year_diff_list, interval_length_list, interval_end_years, is_large_run)

    calc_end = time.time()
    lu.print_and_log(f"Done calculating vegetation fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)
    lu.print_and_log(f"Memory usage after numba calculations completed for {bounds_str}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)
    lu.print_and_log(f"Calculated {bounds_str} in {tile_id} in {round(calc_end-calc_start)} seconds: {uu.timestr()}", False, logger_worker)

    # print("out_dict_uint8:", out_dict_uint8)
    # print("out_dict_uint32:", out_dict_uint32)
    # print("out_dict_float32:", out_dict_float32)
    # print(f"Average of {list(out_dict_uint32.keys())[0]} is: {list(out_dict_uint32.values())[0].mean()}")

    # Deletes all unnecessary input dictionaries before moving on
    # Suggested by ChatGPT: https://chatgpt.com/share/e/672bbf2e-ebbc-800a-aae3-3d92f5a1d663
    in_dicts = [layers, typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32]
    [in_dict.clear() for in_dict in in_dicts]

    # Fresh non-Numba-constrained dictionary that stores all numpy arrays.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    # Everything in out_dict also needs to be in cn.veg_core_output_dirs
    # because that has the list of basic output directories which are customized for this run
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    out_dicts = [out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32]

    # Loop through each dictionary and update out_dict_all_dtypes
    for out_dict in out_dicts:
        for key, value in out_dict.items():
            out_dict_all_dtypes[key] = value

        # Clear memory of unneeded arrays
        del out_dict

    # print(out_dict_all_dtypes)


    ### Part 5: Writes outputs to pre-existing global mega-zarr (only if activated)

    zu.populate_zarr(bounds, bounds_str, create_zarr, interval_end_years, is_large_run, logger_worker, zarr_path,
                  out_dict_all_dtypes, outputs_to_zarr, stage, tile_id)


    ### Part 6: Calculates per ha min, per ha mean, per ha max, and per pixel sum for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.
    ### Also useful for a quick sum of outputs without doing zonal stats

    lu.print_and_log(f"Populating chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

    # The relevant pixel area (m^2) file in s3
    pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"

    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    pixel_area_chunk = uu.get_tile_dataset_rio(pixel_area_uri, bounds, chunk_length_pixels, logger_worker, 'Float32')
    pixel_area_chunk = pixel_area_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Calculates stats for the output layers from create_starting_C_densities as a dictionary with chunk attributes
    for key, array_per_ha in out_dict_all_dtypes.items():

        # Converts per hectare values to per pixel values for the output numpy array
        output_per_pixel = array_per_ha * pixel_area_chunk * cn.m2_to_ha

        chunk_stats.append(uu.calculate_stats(array_per_ha, key, bounds_str, tile_id, 'output_layer', output_per_pixel))

    del pixel_area_chunk

    lu.print_and_log(f"Populated chunk stats for outputs in {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)


    ### Part 7: Saves numpy arrays as rasters and uploads to s3

    uu.rename_s3_task_file(stage, bounds, "uploading_", is_large_run, logger_worker)

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if no_upload == False:

        out_no_data_val = 0  # NoData value for output raster (optional)
        upload_start_time = time.time()

        # print("output_folders:", output_folders)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name
            # print("key", key)
            # print("data_type:", data_type)

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, interval_end_year = uu.strip_and_extract_years(key)
            # print("out_pattern:", out_pattern)
            # print("interval_end_year:", interval_end_year)

            # Gets the core filename pattern and pixel meaning
            out_pattern_without_pixel_meaning, pixel_meaning = uu.strip_pixel_meaning(out_pattern)
            # print("out_pattern_without_pixel_meaning:", out_pattern_without_pixel_meaning)

            # Retrieves the relevant output s3 path for this specific output  (list of one element)
            # First, finds the output folders for all intervals with the relevant patterns
            matched_output_s3_folders = [item for item in output_folders if out_pattern_without_pixel_meaning in item]
            # print("matched_output_s3_folders:", matched_output_s3_folders)

            # Adds the starting composite primary forest extent as an output to the upload s3 folder list
            if out_pattern == cn.composite_primary_forest:

                # Creates the starting year composite primary forest s3 folder name; replaces existing year with start year
                starting_composite_primary_forest_s3_folder = matched_output_s3_folders[0].replace(str(start_year+interval_length_list[0]), str(start_year))
                # print("starting_composite_primary_forest_s3_folder:", starting_composite_primary_forest_s3_folder)
                matched_output_s3_folders.append(starting_composite_primary_forest_s3_folder)  # Adds starting year composite primary forest to list
                # print("matched_output_s3_folders with starting one:", matched_output_s3_folders)

            # Second, finds the output folder with the right interval for that pattern
            matched_output_s3_folder_list = [item for item in matched_output_s3_folders if interval_end_year in item]
            # print("matched_output_s3_folder_list:", matched_output_s3_folder_list)

            # Output paths without bucket (s3://gfw2-data).
            # Needs [0] because matched_output_s3_folder_list is a list of all intervals.
            s3_path_without_bucket = f"{matched_output_s3_folder_list[0][cn.full_bucket_prefix_length:]}"
            # print("s3_path_without_bucket:", s3_path_without_bucket)

            # Dictionary with metadata for each array
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, interval_end_year, s3_path_without_bucket]


        # Converts output numpy arrays to local rasters and puts them in a list of files to upload in parallel
        upload_tasks = uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str,
                                                        out_dict_all_dtypes, is_large_run, logger_worker, out_no_data_val)

        lu.print_and_log(f"Upload tasks created for {bounds_str} in {tile_id}. Uploading now: {uu.timestr()}", False, logger_worker)

        # Execute uploads in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(lambda args: uu.upload_raster_to_s3(*args), upload_tasks)

        upload_end_time = time.time()
        lu.print_and_log(f"Uploads completed for {bounds_str} in {tile_id} in {round(upload_end_time - upload_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    chunk_end_time = time.time()
    lu.print_and_log(f"Total chunk processing for {bounds_str} in {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    return_message = f"Success for {bounds_str}: {uu.timestr()}"

    # Removes task tracking file from S3 once task is successful
    uu.delete_s3_task_file(stage, bounds, is_large_run, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"Peak memory for {bounds_str} in {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    return return_message, chunk_stats


# Designed to report task/worker crashes, including memory usage at the time.
# Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
def safe_task_wrapper(*args, **kwargs):
    try:
        result = calculate_and_upload_vegetation_fluxes(*args, **kwargs)
        return result

    except Exception as e:

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()

        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "memory_at_failure": {
                "rss_gb": mem.rss / 1024**3,  # VMS is the important one
                # "vms_gb": mem.vms / 1024**3,
            },
        }


def main(cluster_name, year_range, model_type,
         run_local=False, no_stats=False, no_log=False, no_upload=False, create_zarr=False,
         chunk_shapefile_uri=False, bounding_box=None, chunk_size_deg=None, first_chunks=None,
         run_date=None, model_path_description=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = 'vegetation_fluxes'

    # Runs chunks in batches of specified size.
    # Each batch slows down processing because chunks inevitably lag and that happens more the more batches there are.
    batch_size = 3800  # 5 batches to cover all chunks
    # batch_size = 8  # large-scale testing

    # Determines if arguments for start and end year are valid
    if year_range not in [[cn.first_model_year_5_years, cn.last_model_year_5_years],  # 2000-2020
                          [cn.first_model_year_5_years, cn.last_model_year_annual],  # 2000-2024
                          [cn.first_model_year_annual, cn.last_model_year_annual]]:  # 2015-2024
        print("Year range selection not valid")
        sys.exit()
    else:
        start_year = year_range[0]
        end_year = year_range[1]

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    # Sets date as today if it's not supplied
    if not run_date:
        today = date.today()
        run_date = today.strftime("%Y%m%d")

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Model version: {cn.veg_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Start year: {start_year}; end year: {end_year}")
    main_logger.info(f"Run date: {run_date}")
    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"no_upload: {no_upload}")
    main_logger.info(f"Tolerance for comparison between model and zarr chunk stat metrics: {cn.zarr_difference_tolerance}")

    # Calculates the interval type, difference between start and end years of intervals, and the model output years
    # for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(start_year, end_year, main_logger)

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)
    main_logger.info(f"Chunks to process: {len(chunk_list)}")

    # Determines if the output file names for final versions of outputs should be used
    is_large_run = False
    # is_large_run = True  # large-scale testing
    if len(chunk_list) > 20:
        is_large_run = True
        main_logger.info(f"Running as large-scale run model: {is_large_run}")

    # Whenever the run is large-scale (final), force zarr creation
    if is_large_run:
        create_zarr = True
    main_logger.info(f"Create and populate global mega-zarr: {create_zarr}")

    # This is just a placeholder tile_id that is used to obtain the datatype of each input tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    # It shouldn't really matter what the sample_tile_id is.
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model).
    # These inputs don't depend on the starting year of the model.
    download_dict = {
        cn.r_s_ratio_non_mang_pattern: f"{cn.r_s_ratio_non_mang_dir}{sample_tile_id}_{cn.r_s_ratio_non_mang_pattern}.tif",

        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",

        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_dir}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
        cn.planted_forest_AGC_removal_factor_pattern: f"{cn.planted_forest_AGC_removal_factor_dir}{sample_tile_id}_{cn.planted_forest_AGC_removal_factor_pattern}.tif",
        cn.planted_forest_AGC_BGC_removal_factor_pattern: f"{cn.planted_forest_AGC_BGC_removal_factor_dir}{sample_tile_id}_{cn.planted_forest_AGC_BGC_removal_factor_pattern}.tif",
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_dir}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_dir}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",   # Pattern is before tile_id for this input
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_dir}{sample_tile_id}.tif",

        cn.elevation_pattern: f"{cn.elevation_dir}{sample_tile_id}_{cn.elevation_pattern}.tif",
        cn.climate_domain_pattern: f"{cn.climate_domain_dir}{sample_tile_id}_{cn.climate_domain_pattern}.tif",
        cn.climate_zone_pattern: f"{cn.climate_zone_processed_dir}{sample_tile_id}_{cn.climate_zone_pattern}.tif",
        cn.precipitation_pattern: f"{cn.precipitation_dir}{sample_tile_id}_{cn.precipitation_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_dir}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif",
        cn.pixel_area_pattern: f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{sample_tile_id}.tif"
    }

    # Young natural forest rasters (several age intervals).
    # Each growth interval's rate is in its own folder.
    for growth_interval in cn.natural_forest_growth_curve_intervals:
        download_dict[f"{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years"] = \
            f"{cn.natural_forest_growth_curve_dir}rate_{growth_interval}/{sample_tile_id}_{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years__nibble_{cn.secondary_forest_curve_run_date}.tif"

    # Burned area rasters (every year)-- same code for annual, 5-year model, or hybrid.
    # Each burned area year needs to be in its own folder.
    # Burned area from the start year of the first interval is never used, hence iteration starts with start_year+1.
    for year in range(start_year+1, end_year + 1):  # Annual burned area maps start in 2000
        download_dict[f"{cn.burned_area_final_pattern}_{year}"] = f"{cn.full_bucket_prefix}/{cn.burned_area_final_dir}{year}/{sample_tile_id}_{cn.burned_area_final_pattern}_{year}.tif"

    # Starting carbon pools depend on the starting year of the model
    if start_year == 2000:
        download_dict[cn.agc_LC_masked_dens_pattern] = f"{cn.agc_2000_LC_masked_dir}{sample_tile_id}__{cn.agc_2000_LC_masked_pattern}.tif"
        download_dict[cn.bgc_LC_masked_dens_pattern] = f"{cn.bgc_2000_LC_masked_dir}{sample_tile_id}__{cn.bgc_2000_LC_masked_pattern}.tif"
        download_dict[cn.deadwood_c_LC_masked_dens_pattern] = f"{cn.deadwood_c_2000_LC_masked_dir}{sample_tile_id}__{cn.deadwood_c_2000_LC_masked_pattern}.tif"
        download_dict[cn.litter_c_LC_masked_dens_pattern] = f"{cn.litter_c_2000_LC_masked_dir}{sample_tile_id}__{cn.litter_c_2000_LC_masked_pattern}.tif"
    elif start_year == 2015:
        download_dict[cn.agc_LC_masked_dens_pattern] = f"{cn.agc_2015_LC_masked_dir}{sample_tile_id}__{cn.agc_2015_LC_masked_pattern}.tif"
        download_dict[cn.bgc_LC_masked_dens_pattern] = f"{cn.bgc_2015_LC_masked_dir}{sample_tile_id}__{cn.bgc_2015_LC_masked_pattern}.tif"
        download_dict[cn.deadwood_c_LC_masked_dens_pattern] = f"{cn.deadwood_c_2015_LC_masked_dir}{sample_tile_id}__{cn.deadwood_c_2015_LC_masked_pattern}.tif"
        download_dict[cn.litter_c_LC_masked_dens_pattern] = f"{cn.litter_c_2015_LC_masked_dir}{sample_tile_id}__{cn.litter_c_2015_LC_masked_pattern}.tif"
    else:
        sys.exit('interval_type not found')

    # Starting forest age depends on the starting year of the model
    if start_year == 2000:
        download_dict[f"{cn.forest_age_start_year_pattern}"] = f"{cn.forest_age_2000_gap_filled_dir}{sample_tile_id}__{cn.forest_age_2000_gap_filled_pattern}.tif"
    elif start_year == 2015:
        download_dict[f"{cn.forest_age_start_year_pattern}"] = f"{cn.forest_age_2015_gap_filled_dir}{sample_tile_id}__{cn.forest_age_2015_gap_filled_pattern}.tif"
    else:
        sys.exit('interval_type not found')

    # Source for assigning composite primary forests depend on the starting year of the model
    if start_year == 2000:
        download_dict[f"{cn.ifl_primary_2000_pattern}"] = f"{cn.ifl_primary_2000_dir}{sample_tile_id}_{cn.ifl_primary_2000_pattern}.tif"
    elif start_year == 2015:
        download_dict[f"{cn.primary_2001_pattern}"] = f"{cn.primary_2001_dir}{sample_tile_id}.tif"
        download_dict[f"{cn.ifl_2016_pattern}"] = f"{cn.ifl_2016_dir}{sample_tile_id}.tif"
        download_dict[f"{cn.tree_cover_loss_pattern}"] = f"{cn.tree_cover_loss_dir}{cn.tree_cover_loss_pattern}_{sample_tile_id}.tif"
    else:
        sys.exit('interval_type not found')

    # Land cover and vegetation height timeseries depend on interval_type (better expressed than using the start_year)
    if interval_type == cn.intervals_five_years:
        # Land cover and vegetation height rasters (5-year intervals)
        for year in range(cn.first_model_year_5_years, cn.last_model_year_5_years + 1, cn.five_year_interval_duration):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_5_year_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_5_year_path}{year}/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_{year}.tif"
    elif interval_type == cn.intervals_annual:
        # Land cover and vegetation height rasters (annual intervals)
        for year in range(cn.first_model_year_annual, cn.last_model_year_annual + 1):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_annual_path}{year}/{sample_tile_id}.tif"
    elif interval_type == cn.intervals_hybrid:
        # Land cover and vegetation height rasters (5-year intervals for 2000, 2005, and 2010 only)
        for year in range(cn.first_model_year_5_years, 2010+1, cn.five_year_interval_duration):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_5_year_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_5_year_path}{year}/{sample_tile_id}_{cn.vegetation_height_5_year_pattern}_{year}.tif"
        # Land cover and vegetation height rasters (annual intervals for 2015 onwards)
        for year in range(cn.first_model_year_annual, cn.last_model_year_annual + 1):
            download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_annual_path}{year}/{sample_tile_id}.tif"
            download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_annual_path}{year}/{sample_tile_id}.tif"
    else:
        sys.exit('interval_type not found')

    # GMW mangrove extent timeseries depend on start year (not end year since GMW data ends in 2020)
    for year in cn.mangrove_extent_years:
        download_dict[f"{cn.mangrove_extent_processed_pattern}_{year}"] = f"{cn.mangrove_extent_processed_dir}{year}/{sample_tile_id}__{cn.mangrove_extent_processed_pattern}_{year}.tif"

    # Forest disturbance rasters (every year)-- only for 5-year intervals (including hybrid model)
    # All years need to be in their own folder
    if interval_type in cn.intervals_five_years:
        for year in range(cn.first_model_year_5_years + 1, cn.last_model_year_5_years + 1):  # Annual forest disturbance maps start in 2001 and ends in 2020
            download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.forest_disturbance_annual_dir}{year}/{year}_{sample_tile_id}.tif"
    elif interval_type in cn.intervals_hybrid:
        for year in range(cn.first_model_year_5_years + 1, cn.first_model_year_annual + 1):  # Hybrid model uses annual disturbance data through 2015. Annual data used in 2015-2016 onwards.
            download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.forest_disturbance_annual_dir}{year}/{year}_{sample_tile_id}.tif"
    else:  # Annual model does not use annual disturbance data
        pass

    # Replaces the placeholder parts of the input paths with relevant values
    download_dict = {
        key: value.replace("CHUNK_SIZE", '40000')
        for key, value in download_dict.items()
    }
    download_dict = {
        key: value.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning)
        for key, value in download_dict.items()
    }

    main_logger.info("Download dictionary:")
    for key, item in download_dict.items():
        main_logger.info(f"{key}: {item}")

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    main_logger.info(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    main_logger.info(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    if is_large_run:
        main_logger.info(f"download_dict_with_data_types for {stage}:")
        for key, value in download_dict_with_data_types.items():
            main_logger.info(f"  {key}: {value}")

    # Creates a list of output directories (core and intermediates) for all outputs and intervals based on specifics of the model run
    output_dir_list_core_intermediate = cn.veg_core_output_dirs + cn.veg_intermediate_output_dirs + cn.veg_summative_output_dirs
    output_dir_list = uu.create_output_dir_name_list(output_dir_list_core_intermediate, interval_type, start_year,
                                                     chunk_size_pixels, model_type, cn.veg_model_version_underscore, model_path_description, interval_end_years,
                                                     interval_year_diff_list, run_date, False, "per_ha")
    output_dir_list.sort()  # Alphabetically order the outputs (modifies output_dir_list)
    if is_large_run:
        main_logger.info(f"output_dir_list for {stage}:")
        for item in output_dir_list:
            main_logger.info(f"  {item}")

    # Creates numpy array of IPCC Tier 1 primary forest removal factors by continent-ecozone combination.
    # Needs to by a numpy array for the numba function to use it.
    # Inputs are Mg AGB/ha/yr. Outputs are Mg AGB/ha/yr. Conversion to Mg AGC/ha/yr is done below.
    primary_forest_RF_array = uu.convert_lookup_table_to_array(cn.RF_C_ratio_spreadsheet_full_path,
                                                               cn.IPCC_removal_factor_table_tab,
                                                               ['gainEcoCon', 'growth_primary'])

    # Converts primary forest AGB RFs to AGC RFs (Mg AGB/ha/yr -> Mg AGC/ha/yr)
    primary_forest_RF_array[:, 1] = primary_forest_RF_array[:, 1] * cn.biomass_to_carbon_non_mangrove


    # Creates numpy array of ratios of BGC, deadwood C, and litter C relative to AGC. Relevant columns must be specified.
    mangrove_C_ratio_array = uu.convert_lookup_table_to_array(cn.RF_C_ratio_spreadsheet_full_path, cn.mangrove_rate_ratio_tab,
                                                              ['gainEcoCon', 'AGB_gain_tons_ha_yr', 'BGC_AGC', 'deadwood_AGC', 'litter_AGC'])

    # Creates numpy array of emission factors for partially disturbed forest by driver and continent-ecozone combination
    partial_disturbance_EF_array = uu.convert_lookup_table_to_array(cn.partial_disturbance_emission_factor_table_full_path,
                                                                    cn.partial_disturbance_emission_factor_table_tab,
                                                                    ['gainEcoCon', '1_perm_ag_EF', '2_hard_comm_EF',
                                                                     '3_shift_cult_EF',	'4_logging_EF',	'5_wildfire_EF',
                                                                     '6_sett_infrastr_EF', '7_natrl_dist_EF'])


    ### Step 2: Create empty (metadata-only), global mega-zarr in s3.
    ### Zarr approach from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc

    # Only creates the global mega-zarr if needed (large runs or otherwise specified)
    if create_zarr:

        # Creates s3 paths for the raw mega-zarr
        zarr_path = zu.create_zarr_path(cn.veg_outputs_path_mega_zarr, chunk_size_pixels, interval_type,
                                             model_type, cn.veg_model_version_underscore, model_path_description,
                                             run_date, main_logger)

        # These variables are added to the mega-zarr.
        # Adds the unit to the zarr variable names (uses re.sub to apply to end of string only so that these don't overwrite each other).
        outputs_to_zarr = cn.full_veg_outputs_to_zarr
        outputs_to_zarr_with_unit = [
            re.sub(r"MgC$", f"MgC{cn.C_density_pixel_meaning}", pattern)
            for pattern in outputs_to_zarr
        ]
        outputs_to_zarr_with_unit = [
            re.sub(r"MgCO2$", f"MgCO2{cn.flux_density_pixel_meaning}", pattern)
            for pattern in outputs_to_zarr_with_unit
        ]
        outputs_to_zarr_with_unit = [
            re.sub(r"MgCO2e$", f"MgCO2e{cn.flux_density_pixel_meaning}", pattern)
            for pattern in outputs_to_zarr_with_unit
        ]

        # Creates the global mega-zarr with metadata only
        zu.initialize_global_zarr(zarr_path, outputs_to_zarr_with_unit, len(interval_year_diff_list),
                                  ((cn.end_year_count), chunk_size_pixels, chunk_size_pixels), main_logger)

        # Checks the zarr coordinates and extent
        fs = fsspec.filesystem("s3", anon=False)
        mapper = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(mapper, consolidated=False)
        main_logger.info(f"mega-zarr coords: {ds.coords}")
        main_logger.info(f"y range: {ds.y.values.min()}, {ds.y.values.max()}")
        main_logger.info(f"x range: {ds.x.values.min()}, {ds.x.values.max()}")
        main_logger.info(f"mega-zarr chunk size (years, y, x): {ds.chunksizes}")

    else:
        zarr_path = None
        outputs_to_zarr = False


    ### Step 3: Create 1x1 degree outputs

    # Creates list of tasks to run (1 task = 1 chunk)
    main_logger.info(f"Creating tasks and starting processing: {uu.timestr()}")
    main_logger.info("Workers' logs to be appended after main function log"+ "\n")

    chunk_batches = [chunk_list[i:i + batch_size] for i in range(0, len(chunk_list), batch_size)]
    main_logger.info(f"There are {len(chunk_batches)} batches to process: {uu.timestr()}")

    # Accumulates all output messages and statistics across batches
    # From https://chatgpt.com/share/e/5599b6b0-1aaa-4d54-98d3-c720a436dd9a
    all_results = []
    all_stats = []
    success_count = 0  # Count of successful chunks

    # Iterates through the batches
    for i, chunk_batch in enumerate(chunk_batches):
        main_logger.info(f"Processing batch {i + 1}/{len(chunk_batches)} ({len(chunk_batch)} chunks): {uu.timestr()}")
        main_logger.info("Creating batch task txts in s3...")
        uu.create_s3_task_files(stage, chunk_batch)

        # This approach handles large task lists (graphs) better than [dask.delayed(calculate_and_upload_vegetation_fluxes ... )]
        # safe_vegetation_task is supposed to report task/worker crashes.
        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
        # That chat has a table that explains what different combinations of traceback & memory presence/absence mean for the failure.
        futures = []
        for chunk in chunk_batch:
            future = client.submit(
                        safe_task_wrapper,
                        chunk, primary_forest_RF_array, partial_disturbance_EF_array, mangrove_C_ratio_array,
                        download_dict_with_data_types, start_year, end_year, interval_type, interval_year_diff_list,
                        interval_length_list, interval_end_years, is_large_run, no_upload, create_zarr,
                        output_dir_list, stage, model_type, zarr_path, outputs_to_zarr,
                        retries=1, key=f"vegflux-{chunk}")  # Designed to prevent infinite retries and rerunning completed tasks (happens in global runs)
            futures.append(future)

        batch_results = client.gather(futures)

        for result in batch_results:
            if isinstance(result, dict) and result.get("status") == "failed":
                main_logger.error(
                    "Task failed\n"
                    f"Error: {result['error']}\n"
                    f"Memory at failure (GB): {result['memory_at_failure']}\n"
                    f"Traceback:\n{result['traceback']}"
                )

        all_results.extend(batch_results)

        success_count, batch_stats = uu.count_successful_chunks(chunk_batch, is_large_run, main_logger, batch_results)
        all_stats.extend(batch_stats)

        # Saves stats from batch in Excel locally in case the run fails, but only if there are multiple batches.
        # That way there are some basic chunk stats (not sorted or anything) to fall back on.
        if len(chunk_batches) > 1:

            main_logger.info(f"Writing batch stats locally: {uu.timestr()}")
            df_batch_stats = pd.DataFrame(batch_stats)

            timestamp = uu.timestr()

            # Writes batch output to parquet file if output is large
            if len(df_batch_stats) > 900_000:
            # if len(df_batch_stats) > 7: # large-scale testing
                out_file = f"TEMP_BATCH_{stage}__batch_{i}_{timestamp}.parquet"
                local_path = f"{cn.local_chunk_stats_path}{out_file}"

                # Coerce output to string so there aren't mismatched types
                # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694c44d0-19e8-8330-8098-a7ec93366e44
                for col in ['min_value', 'max_value', 'mean_value', 'sum_value', 'count_value']:
                    if col in df_batch_stats.columns:
                        df_batch_stats[col] = df_batch_stats[col].astype(str)

                df_batch_stats.to_parquet(
                    local_path,
                    engine="pyarrow",
                    index=False
                )

            # Otherwise, writes output to spreadsheet
            else:
                out_file = f"TEMP_BATCH_{stage}__batch_{i}_{timestamp}.xlsx"
                local_path = f"{cn.local_chunk_stats_path}{out_file}"

                with pd.ExcelWriter(local_path) as writer:
                    df_batch_stats.to_excel(
                        writer,
                        sheet_name=f"stats__batch_{i}",
                        index=False
                    )

        del futures
        del batch_results
        client.run(gc.collect)

        uu.stage_duration(start_time, uu.timestr(), f"{stage}, batch {i}", main_logger)


    ### Step 4: Gather worker logs (preliminary, just in case later step goes awry)

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path_prelim = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with preliminary worker log compilation", main_logger)


    ### Step 5: Consolidate chunk stats and export

    # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successful (wasn't skipped).
    if (not no_stats) and (success_count > 0):
        model_chunk_stats_path = uu.compile_1x1_chunk_stats(all_stats, chunk_shapefile_uri, stage, no_upload, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with tile stats", main_logger)


    ### Step 6: Compare model output chunk stats to zarr chunk stats for each variable (only if chunk stats and zarr created)

    # Prepares chunk stats spreadsheet: min, mean, max, and sum for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag and at least one chunk was successful (wasn't skipped).
    if (not no_stats) and create_zarr:

        main_logger.info(f"Starting zarr chunk stats comparison: {uu.timestr()}")

        # Text added to output chunk stats table name(s) (Excel or Parquet)
        comparison_insert = "_original_zarr_comparison"

        # The name of the chunk stats table from the model
        model_chunk_stats_table_name = os.path.basename(model_chunk_stats_path)
        # print(model_chunk_stats_table_name)

        tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
            comparison_insert, main_logger, model_chunk_stats_path)

        # List of dataframes with original and zarr chunk stats and their difference for each dataset-year combination
        all_merged_tables = []

        # Number of chunks with differences between original and zarr exceeding tolerance
        chunks_count_exceeding_total = 0

        # Number of chuinks that have model chunk stats but not corresponding zarr chunk stats
        chunks_without_zarr_stats_total = 0

        # Iterates through select variables/datasets for chunk stats comparison. Can modify as needed.
        outputs_to_compare = [
            cn.gross_emis_all_C_pools_CO2_only_pattern, cn.gross_emis_all_C_pools_non_CO2_only_pattern, cn.gross_emis_all_C_pools_all_gases_pattern,
            cn.gross_removals_all_C_pools_pattern,
            cn.net_flux_all_C_pools_CO2_only_pattern, cn.net_flux_all_C_pools_all_gases_pattern,
            cn.non_soil_c_modeled_dens_pattern, cn.land_state_pattern
        ]

        for var_name in outputs_to_compare:

            main_logger.info(f"Starting {var_name}: {uu.timestr()}")
            var_start_time = time.time()

            # Runs chunk stats for a dataset (all years) in the zarr in parallel
            chunk_stats_variable_year_zarr = zu.run_parallel_stats(
                client=client,
                chunk_list=chunk_list,
                var=var_name,
                zarr_path=zarr_path,
                interval_end_years=interval_end_years
            )
            # print("chunk_stats_variable_year_zarr:", chunk_stats_variable_year_zarr)

            # After all zarr chunk stats is done for the dataset-year combination,
            # the chunk stats from the zarr are compared to the chunk stats from the model.
            # This is done with Pandas dataframes and is not parallelized because it's just table manipulation
            # for each dataset-year combination.
            # The model output vs. zarr comparison is done after each dataset-year combination
            # to get more real-time feedback on how the datasets compare (rather than waiting until after
            # all zarr chunk stats have been calculated to do the metric comparisons).
            chunks_count_exceeding, chunks_without_zarr_stats = zu.compare_dataset_year_chunk_stats(all_merged_tables,
                                                                   chunk_stats_variable_year_zarr,
                                                                   main_logger,
                                                                   tables_to_compare_dict,
                                                                   var_name,
                                                                   zarr_comparison_stats_path)

            # Total number of chunks that have differences in metrics between the model and zarr
            # that exceed the tolerance
            chunks_count_exceeding_total += chunks_count_exceeding
            chunks_without_zarr_stats_total += chunks_without_zarr_stats

            var_end_time = time.time()
            main_logger.info(f"  Processed {var_name} in {round(var_end_time - var_start_time)} seconds: {uu.timestr()}")

        # Counts up chunks that had differences exceeding the tolerance and uploads chunk stats comparisons.
        zu.upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                              main_logger, model_chunk_stats_table_name,
                                              stage, start_time, zarr_comparison_stats_name, zarr_comparison_stats_path)


    ### Step 7: Gather worker logs

    # Collects worker logs before moving to processing that doesn't need the cluster
    if not run_local:

        # Creates combined log from all workers if not deactivated
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        uu.stage_duration(start_time, uu.timestr(), f"{stage} with worker log compilation", main_logger)


    ### Step 8: Resize cluster down to 1 worker for remaining steps since they only need a minimal remainder of the
    ### cluster, not all the workers.

    if not run_local:
        workers = client.scheduler_info()["workers"]
        n_workers = len(workers)

        # Reduces number of workers in the cluster down to 1 if there is more than 10
        if n_workers > 10:
            main_logger.info("Resizing cluster to 1 worker")

            resize_cluster.resize_coiled_cluster(cluster_name, 1)


    ### Step 9: Count output geotifs in s3
    # Iterates through select output folders and counts the number of output rasters (only if uploads enabled and a large run (to save console space))

    main_logger.info(f"Counting geotifs in select output folders. Expecting {len(chunk_list)} in each: {uu.timestr()}")
    keywords = ["gross", "net", "state"]
    output_dir_list_to_count = [
        item for item in output_dir_list
        if any(keyword in item for keyword in keywords)
    ]
    if not no_upload and is_large_run:
        for output_folder in output_dir_list_to_count:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(output_folder)
            main_logger.info(f"Output rasters in {output_folder}: {file_count}")
            if file_count != len(chunk_list):
                main_logger.warning(f"WARNING: Output file count in {output_folder} does not match expectations!")
            # print(geotiff_files)

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with output counts", main_logger)


    ### Step 10: Merge compiled worker log and main log
    if not run_local:

        # Adds the workers' logs to the main log and uploads to s3
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)


    # Closes the Dask client if not running locally
    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate vegetation fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-rd', '--run_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size_deg', type=float, help='Chunk size (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-yr', '--year_range', nargs=2, type=int, default=[cn.first_model_year_annual, cn.last_model_year_annual],
                        help='Starting and ending years for model. Start options: 2000, 2015. End options: 2020, 2024.')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')
    parser.add_argument('--create_zarr', action='store_true', help='Create and populate global mega-zarr with model outputs')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_date = args.run_date
    bounding_box = args.bounding_box
    chunk_size_deg = args.chunk_size_deg
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    year_range = args.year_range
    model_type = args.model_type
    model_path_description = args.model_path_description
    log_note = args.log_note

    run_local = args.run_local
    no_stats = args.no_stats
    no_log = args.no_log
    no_upload = args.no_upload
    create_zarr = args.create_zarr

    # Create the cluster with command line arguments
    main(cluster_name, year_range, model_type, run_local, no_stats, no_log, no_upload, create_zarr, chunk_shapefile_uri,
         bounding_box=bounding_box, chunk_size_deg=chunk_size_deg, first_chunks=first_chunks,
         run_date=run_date, model_path_description=model_path_description, log_note=log_note)
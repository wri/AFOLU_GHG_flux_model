"""
Run from src/LUL

Test:
python -m scripts.utilities.create_cluster -n 1 -m 16 -c 2
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb 10 49.75 10.25 50 -cs 0.25

Full run:
python -m scripts.utilities.create_cluster -n 200 -m 16 -c 2
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb -180 -60 180 80 -cs 1

1 hour of running, 750 Coiled credits, $27 AWS costs, 10,614 tasks out of 50,400.
Each worker tended to use 28-36 GB memory.
boto3.exceptions.S3UploadFailedError: Failed to upload /tmp/30N_090E__99_25_100_26__litter_C_density_MgC_ha_2010_2015.tif
to gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/litter_C_density_MgC_ha/2010_2015/4000_pixels/20240829/30N_090E__99_25_100_26__litter_C_density_MgC_ha_2010_2015.tif:
An error occurred (RequestTimeout) when calling the UploadPart operation (reached max retries: 4):
Your socket connection to the server was not read from or written to within the timeout period. Idle connections will be closed.

"""

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

# Function to calculate LULUCF fluxes and carbon densities
# Operates pixel by pixel, so uses numba (Python compiled to C++).
@jit(nopython=True)
def LULUCF_fluxes(in_dict_uint8, in_dict_int16, in_dict_float32):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type).
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_uint32 = {}
    out_dict_float32 = {}

    end_years = list(range(cn.first_year, cn.last_year + 1, cn.interval_years))[1:]
    # end_years = [2005, 2010]

    # Numpy arrays for outputs that do depend on previous interval's values
    agc_dens_block = in_dict_float32[cn.agc_2000].astype('float32')
    bgc_dens_block = in_dict_float32[cn.bgc_2000].astype('float32')
    deadwood_c_dens_block = in_dict_float32[cn.deadwood_c_2000].astype('float32')
    litter_c_dens_block = in_dict_float32[cn.litter_c_2000].astype('float32')
    soil_c_dens_block = in_dict_int16[cn.soil_c_2000].astype('float32')

    r_s_ratio_block = in_dict_float32[cn.r_s_ratio].astype('float32')

    # Stores the burned area blocks for the entire model duration
    burned_area_blocks_total = []

    # Stores the forest disturbance blocks for the entire model duration
    forest_dist_blocks_total = []

    # Iterates through years
    for year in end_years:

        # print(f"Now at {year}:")

        # Writes the dictionary entries to a chunk for use in the decision tree
        LC_prev_block = in_dict_uint8[f"{cn.land_cover}_{year - cn.interval_years}"]
        LC_curr_block = in_dict_uint8[f"{cn.land_cover}_{year}"]
        veg_h_prev_block = in_dict_uint8[f"{cn.vegetation_height}_{year - cn.interval_years}"]
        veg_h_curr_block = in_dict_uint8[f"{cn.vegetation_height}_{year}"]
        planted_forest_type_block = in_dict_uint8[cn.planted_forest_type_layer]
        planted_forest_tree_crop_block = in_dict_uint8[cn.planted_forest_tree_crop_layer]

        # Creates a list of all the burned area arrays from 2001 to the end of the interval
        for year_offset in range(year-4, year+1):
            year_key = f"{cn.burned_area}_{year_offset}"
            # print(year_key)
            burned_area_blocks_total.append(in_dict_uint8[year_key])

        # Creates a list of all the forest disturbance arrays from 2001 to the end of the interval
        for year_offset in range(year-4, year+1):
            year_key = f"{cn.forest_disturbance}_{year_offset}"
            # print(year_key)
            forest_dist_blocks_total.append(in_dict_uint8[year_key])


        # Numpy arrays for outputs that don't depend on previous interval's values
        state_out = np.zeros(in_dict_float32[cn.agc_2000].shape).astype('uint32')
        agc_flux_out_block = np.zeros(in_dict_float32[cn.agc_2000].shape).astype('float32')
        bgc_flux_out_block = np.zeros(in_dict_float32[cn.agc_2000].shape).astype('float32')
        deadwood_c_flux_out_block = np.zeros(in_dict_float32[cn.agc_2000].shape).astype('float32')
        litter_c_flux_out_block = np.zeros(in_dict_float32[cn.agc_2000].shape).astype('float32')

        # Iterates through all pixels in the chunk
        for row in range(LC_curr_block.shape[0]):
            for col in range(LC_curr_block.shape[1]):

                ### Defining pixel values

                LC_prev = LC_prev_block[row, col]
                LC_curr = LC_curr_block[row, col]
                veg_h_prev = veg_h_prev_block[row, col]
                veg_h_curr = veg_h_curr_block[row, col]
                planted_forest_type = planted_forest_type_block[row, col]
                planted_forest_tree_crop = planted_forest_tree_crop_block[row, col]

                r_s_ratio_cell = r_s_ratio_block[row, col]

                # Note: Stacking the burned area rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # So just reading each raster from the list of rasters separately.
                burned_area_t_4 = burned_area_blocks_total[-5][row, col]
                burned_area_t_3 = burned_area_blocks_total[-4][row, col]
                burned_area_t_2 = burned_area_blocks_total[-3][row, col]
                burned_area_t_1 = burned_area_blocks_total[-2][row, col]
                burned_area_t = burned_area_blocks_total[-1][row, col]
                # Most recent year with burned area during the interval
                burned_area_last = max([burned_area_t_4, burned_area_t_3, burned_area_t_2, burned_area_t_1, burned_area_t])

                # Note: Stacking the forest disturbance rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # So just reading each raster from the list of rasters separately.
                forest_dist_t_4 = forest_dist_blocks_total[-5][row, col]
                forest_dist_t_3 = forest_dist_blocks_total[-4][row, col]
                forest_dist_t_2 = forest_dist_blocks_total[-3][row, col]
                forest_dist_t_1 = forest_dist_blocks_total[-2][row, col]
                forest_dist_t = forest_dist_blocks_total[-1][row, col]
                # Most recent year with forest disturbance during the interval
                forest_dist_last = max([forest_dist_t_4, forest_dist_t_3, forest_dist_t_2, forest_dist_t_1, forest_dist_t])

                # Records the first year of burned area in the pixel, to indicate whether fire was reported at all
                # in the record
                first_burn_in_record = 0

                # Records the first year of forest disturbance in the pixel, to indicate whether disturbance was reported at all
                # in the record
                first_forest_dist_in_record = 0

                # Loops over burned area pixels since 2001 to see if there was a fire.
                # Stops once a fire is detected because all that matters here is that there was a fire at some point.
                for burned_area_year in burned_area_blocks_total:
                    # Update the maximum value for this pixel
                    if burned_area_year[row, col] > 0:
                        first_burn_in_record = burned_area_year[row, col]
                        break

                # if first_burn_in_record > 0:
                #     print("fire", row, col, first_burn_in_record)

                # Loops over forest disturbance pixels since 2001 to see if there was a disturbance.
                # Stops once a disturbance is detected because all that matters here is that there was a disturbance at some point.
                for forest_dist_year in forest_dist_blocks_total:
                    # Update the maximum value for this pixel
                    if forest_dist_year[row, col] > 0:
                        first_forest_dist_in_record = forest_dist_year[row, col]
                        break

                # if first_forest_dist_in_record > 0:
                #     print("disturbance", row, col, first_forest_dist_in_record)

                # Input carbon densities for the pools
                agc_dens_in = agc_dens_block[row, col]
                bgc_dens_in = bgc_dens_block[row, col]
                deadwood_c_dens_in = deadwood_c_dens_block[row, col]
                litter_c_dens_in = litter_c_dens_block[row, col]
                soil_c_dens = soil_c_dens_block[row, col]

                # Makes a list of carbon densities to save space in the decision tree below.
                # This list is input to flux calulation functions as one argument, rather than a separate argument
                # for each pool
                c_dens_in = [agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in]

                ### Defining specific classes

                tree_prev = (veg_h_prev >= cn.tree_threshold)
                tree_curr = (veg_h_curr >= cn.tree_threshold)
                tall_veg_prev = (((LC_prev >= cn.tree_dry_min_height_code) and (LC_prev <= cn.tree_dry_max_height_code)) or
                                 ((LC_prev >= cn.tree_wet_min_height_code) and (LC_prev <= cn.tree_wet_max_height_code)))
                tall_veg_curr = (((LC_curr >= cn.tree_dry_min_height_code) and (LC_curr <= cn.tree_dry_max_height_code)) or
                                 ((LC_curr >= cn.tree_wet_min_height_code) and (LC_curr <= cn.tree_wet_max_height_code)))
                short_med_veg_prev = (((LC_prev >= 2) and (LC_prev <= 26)) or
                                      ((LC_prev >= 102) and (LC_prev <= 126)))
                short_med_veg_curr = (((LC_curr >= 2) and (LC_curr <= 26)) or
                                      ((LC_curr >= 102) and (LC_curr <= 126)))

                sig_height_loss_prev_curr = (veg_h_prev - veg_h_curr >= cn.sig_height_loss_threshold)

                # Starting node value
                node = 0

                # Starting dummy output C fluxes and densities: AGC, BGC, deadwood C, litter C.
                # Should be really obvious if they're being applied.
                # Need to force into float32 because numba is so particular about datatypes.
                c_flux_out = np.array([-1000, -2000, -3000, -3000]).astype('float32')
                c_dens_out = np.array([-10000, -20000, -30000, -30000]).astype('float32')

                ### Tree gain
                if (not tree_prev) and (tree_curr):  # Non-tree converted to tree (1)    ##TODO: Include mangrove exception.
                    node = nu.accrete_node(node, 1)
                    if planted_forest_type == 0:  # New non-SDPT trees (11)
                        node = nu.accrete_node(node, 1)
                        if not tall_veg_curr:  # New trees outside forests (111)
                            state_out[row, col] = nu.accrete_node(node, 1)
                            agc_rf = 2.8
                            c_flux_out, c_dens_out = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                        else:  # New terrestrial natural forest (112)
                            state_out[row, col] = nu.accrete_node(node, 2)
                            agc_rf = 5.6
                            c_flux_out, c_dens_out = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                    else:  # New SDPT trees (12)
                        state_out[row, col] = nu.accrete_node(node, 2)
                        agc_rf = 10
                        c_flux_out, c_dens_out = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)

                ### Tree loss
                elif (tree_prev) and (not tree_curr):  # Tree converted to non-tree (2)    ##TODO: Include forest disturbance condition.  ##TODO: Include mangrove exception.
                    node = 2
                    if planted_forest_type == 0:  # Full loss of non-SDPT trees (21)
                        node = nu.accrete_node(node, 1)
                        if not tall_veg_prev:  # Full loss of trees outside forests (211)
                            node = nu.accrete_node(node, 1)
                            if burned_area_last == 0:  # Full loss of trees outside forests without fire (2111)
                                state_out[row, col] = nu.accrete_node(node, 1)
                                agc_ef = 0.8
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of trees outside forests with fire (2112)
                                state_out[row, col] = nu.accrete_node(node, 2)
                                agc_ef = 0.6
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                        else:  # Full loss of natural forest (212)
                            node = nu.accrete_node(node, 2)
                            if LC_curr == cn.cropland:  # Full loss of natural forest converted to cropland (2121)
                                node = nu.accrete_node(node, 1)
                                if burned_area_last == 0:  # Full loss of natural forest converted to cropland, not burned (21211)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of natural forest converted to cropland, burned (21212)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            elif short_med_veg_curr:  # Full loss of natural forest converted to short or medium vegetation (2122)
                                node = nu.accrete_node(node, 2)
                                if burned_area_last == 0:  # Full loss of natural forest converted to short or medium vegetation, not burned (21221)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.9
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of natural forest converted to short or medium vegetation, burned (21222)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            elif LC_curr == cn.builtup:  # Full loss of natural forest converted to builtup (2123)
                                node = nu.accrete_node(node, 3)
                                if burned_area_last == 0:  # Full loss of natural forest converted to builtup, not burned (21231)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of natural forest converted to builtup, burned (21232)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of natural forest converted to anything else (2124)
                                node = nu.accrete_node(node, 4)
                                if burned_area_last == 0:  # Full loss of natural forest converted to anything else, not burned (21241)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of natural forest converted to anything else, burned (21242)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                    else:  # Full loss of SDPT trees (22)
                        node = nu.accrete_node(node, 2)
                        if LC_curr == cn.cropland:  # Full loss of SDPT converted to cropland (221)
                            node = nu.accrete_node(node, 1)
                            if burned_area_last == 0:  # Full loss of SDPT converted to cropland, not burned (2211)
                                state_out[row, col] = nu.accrete_node(node, 1)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of SPDPT converted to cropland, burned (2212)
                                state_out[row, col] = nu.accrete_node(node, 2)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                        elif short_med_veg_curr:  # Full loss of SDPT converted to short or medium vegetation (222)
                            node = nu.accrete_node(node, 2)
                            if burned_area_last == 0:  # Full loss of SDPT converted to short or medium vegetation, not burned (2221)
                                state_out[row, col] = nu.accrete_node(node, 1)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of SDPT converted to short or medium vegetation, burned (2222)
                                state_out[row, col] = nu.accrete_node(node, 2)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                        elif LC_curr == cn.builtup:  # Full loss of SDPT converted to builtup (223)
                            node = nu.accrete_node(node, 3)
                            if planted_forest_tree_crop == 1:  # Full loss of SDPT planted forest to builtup (2231)
                                node = nu.accrete_node(node, 1)
                                if burned_area_last == 0:  # Full loss of SDPT planted forest converted to builtup, not burned (22311)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of SDPT planted forest converted to builtup, burned (22312)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of SDPT tree crop to builtup (2232)
                                node = nu.accrete_node(node, 2)
                                if burned_area_last == 0:  # Full loss of SDPT tree crop converted to builtup, not burned (22321)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                                else:  # Full loss of SDPT tree crop converted to builtup, burned (22322)
                                    state_out[row, col] = nu.accrete_node(node, 2)
                                    agc_ef = 0.3
                                    c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                        else:  # Full loss of SDPT converted to anything else (224)
                            node = nu.accrete_node(node, 4)
                            if burned_area_last == 0:  # Full loss of SDPT converted to builtup, not burned (2241)
                                state_out[row, col] = nu.accrete_node(node, 1)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Full loss of SDPT converted to builtup, burned (2242)
                                state_out[row, col] = nu.accrete_node(node, 2)
                                agc_ef = 0.3
                                c_flux_out, c_dens_out = nu.calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in)

                ### Trees remaining trees
                elif (tree_prev) and (tree_curr):  # Trees remaining trees (3)    ##TODO: Include mangrove exception.
                    node = nu.accrete_node(node, 3)
                    if forest_dist_last == 0:  # Trees without stand-replacing disturbances in the last interval (31)
                        node = nu.accrete_node(node, 1)
                        if planted_forest_type == 0:  # Non-planted trees without stand-replacing disturbance in the last interval (311)
                            node = nu.accrete_node(node, 1)
                            if not tall_veg_curr:  # Trees outside forests without stand-replacing disturbance in the last interval (3111)
                                node = nu.accrete_node(node, 1)
                                if not sig_height_loss_prev_curr:  # Stable trees outside forests (31111)
                                    state_out[row, col] = nu.accrete_node(node, 1)
                                    agc_rf = 2.8
                                    c_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                                else:  # Partially disturbed trees outside forests (31112)
                                    node = nu.accrete_node(node, 2)
                                    if burned_area_last == 0:  # Partially disturbed trees outside forests without fire (311121)
                                        state_out[row, col] = nu.accrete_node(node, 1)
                                        agc_rf = 1.3
                                        agc_ef = 0.9
                                        c_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                                    else:  # Partially disturbed trees outside forests with fire (311122)
                                        state_out[row, col] = nu.accrete_node(node, 2)
                                        agc_rf = 0.75
                                        agc_ef = 0.33
                                        c_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                            else:  # Natural forest without stand-replacing disturbance in the last interval (3112)
                                state_out[row, col] = nu.accrete_node(node, 2)
                                agc_rf = 2.1
                                c_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)

                                # if not sig_height_loss_prev_curr:    # Stable natural forest (31121)

                    else:
                        state_out[row, col] = 32
                        agc_flux_out_block[row, col] = 5.54
                        agc_dens_block[row, col] = 13.59
                        bgc_flux_out_block[row, col] = 2.83
                        bgc_dens_block[row, col] = 7.34

                else:  # Not covered in above branches
                    state_out[row, col] = 4000000000  # High value for uint32

                # Populates the output arrays with the calculated fluxes and densities
                agc_flux_out_block[row, col] = c_flux_out[0]
                bgc_flux_out_block[row, col] = c_flux_out[1]
                deadwood_c_flux_out_block[row, col] = c_flux_out[2]
                litter_c_flux_out_block[row, col] = c_flux_out[3]

                agc_dens_block[row, col] = c_dens_out[0]
                bgc_dens_block[row, col] = c_dens_out[1]
                deadwood_c_dens_block[row, col] = c_dens_out[2]
                litter_c_dens_block[row, col] = c_dens_out[3]

                # Carbon densities shouldn't be negative.
                # If they are negative, they are turned to an exaggerated value to make it easier to catch.
                if agc_dens_block[row, col] < 0:
                    agc_dens_block[row, col] = -50000
                if bgc_dens_block[row, col] < 0:
                    bgc_dens_block[row, col] = -50000
                if deadwood_c_dens_block[row, col] < 0:
                    deadwood_c_dens_block[row, col] = -50000
                if litter_c_dens_block[row, col] < 0:
                    litter_c_dens_block[row, col] = -50000


        # Adds the output arrays to the dictionary with the appropriate data type
        # Outputs need .copy() so that previous intervals' arrays in dicationary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
        year_range = f"{year - cn.interval_years}_{year}"
        out_dict_uint32[f"{cn.land_state_pattern}_{year_range}"] = state_out.copy()
        out_dict_float32[f"{cn.agc_dens_pattern}_{year_range}"] = agc_dens_block.copy()
        out_dict_float32[f"{cn.bgc_dens_pattern}_{year_range}"] = bgc_dens_block.copy()
        out_dict_float32[f"{cn.deadwood_c_dens_pattern}_{year_range}"] = deadwood_c_dens_block.copy()
        out_dict_float32[f"{cn.litter_c_dens_pattern}_{year_range}"] = litter_c_dens_block.copy()

        out_dict_float32[f"{cn.agc_flux_pattern}_{year_range}"] = agc_flux_out_block.copy()
        out_dict_float32[f"{cn.bgc_flux_pattern}_{year_range}"] = bgc_flux_out_block.copy()
        out_dict_float32[f"{cn.deadwood_c_flux_pattern}_{year_range}"] = deadwood_c_flux_out_block.copy()
        out_dict_float32[f"{cn.litter_c_flux_pattern}_{year_range}"] = litter_c_flux_out_block.copy()

    return out_dict_uint32, out_dict_float32


# Downloads inputs, prepares data, calculates LULUCF stocks and fluxes, and uploads outputs to s3
def calculate_and_upload_LULUCF_fluxes(bounds, download_dict_with_data_types, is_final, no_upload):

    logger = lu.setup_logging()

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)  # Chunk length in pixels (as opposed to decimal degrees)

    # Stores the min, mean, and max chunks for inputs and outputs for the chunk
    chunk_stats = []


    ### Part 1: Checks if tile exists at all, downloads data in chunk if it does exist, and checks if chunk actually has relevant data.
    ### I haven't figured out a good way to check if the chunk has relevant data before downloading,
    ### so inputs are downloaded and then checked.

    # Replaces the placeholder tile_id in the download data dictionary from main with the tile_id for this chunk
    updated_download_dict = uu.replace_tile_id_in_dict(download_dict_with_data_types, tile_id)

    # Checks whether tile exists at all. Doesn't try to download data in chunk if the tile doesn't exist.
    tile_exists = uu.check_for_tile(updated_download_dict, is_final, logger)

    if not tile_exists:
        return f"Skipped chunk {bounds_str} because {tile_id} does not exist for any inputs: {uu.timestr()}", chunk_stats

    # If a particular tile doesn't exist for an input, an array of 0s of the correct size and datatype is returned instead.
    # Thus, this returns a complete set of inputs (missing chunks filled).
    # Note: If running in a local Dask cluster, prints to console may be duplicated. Doesn't happen with a Coiled cluster of the same size (1 worker).
    # Seems to be a problem with local Dask getting overwhelmed by so many futures being created and downloaded from s3.
    futures = uu.prepare_to_download_chunk(bounds, updated_download_dict, chunk_length_pixels, is_final, logger)
    # print(futures)

    # Only prints if not a final run
    if not is_final:
        lu.print_and_log(f"Waiting for requests for data in chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Dictionary that stores the downloaded data
    layers = {}

    # Waits for requests to come back with data from S3
    for future in concurrent.futures.as_completed(futures):
        layer = futures[future]
        layers[layer] = future.result()

    # print(layers)
    # print(layers[soil_c_2000].dtype)

    # List of layers that must be present for the chunk to be run.
    # All of the listed layers must exist for this chunk in order to proceed.
    checked_layers = {cn.agc_2000: layers[cn.agc_2000], cn.bgc_2000: layers[cn.bgc_2000],
                      cn.deadwood_c_2000: layers[cn.deadwood_c_2000], cn.litter_c_2000: layers[cn.litter_c_2000],
                      f"{cn.land_cover}_2000": layers[f"{cn.land_cover}_2000"]}

    # print(f"Layers to check for data: {layers_to_check_for_data}")

    # Checks chunk for data. Skips the chunk if it does not have the required data.
    # data_in_chunk = uu.check_chunk_for_data(checked_layers, bounds_str, tile_id, "all", is_final, logger)
    data_in_chunk = uu.check_chunk_for_data(checked_layers, bounds_str, tile_id, "all", is_final, logger)

    if data_in_chunk == False:
        return f"Skipped chunk {bounds_str} because of a lack of data: {uu.timestr()}", chunk_stats


    ### Part 2: Calculates min, mean, and max for each input chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculates stats for the input layers
    for key, array in layers.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'input_layer'))
    # print(stats)


    ### Part 3: Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
    ### Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32).
    ### Note: need to add new code if inputs with other data types are added

    # Only prints if not a final run
    if not is_final:
        lu.print_and_log(f"Creating typed dictionaries for chunk {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    # Creates the typed dictionaries for all input layers (including those that originally had no data)
    typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32 = nu.create_typed_dicts(layers)

    # print("uint8_typed_list:", typed_dict_uint8)
    # print("int16_typed_list:", typed_dict_int16)
    # print("int32_typed_list:", typed_dict_int32)
    # print("float32_typed_list:", typed_dict_float32)


    ### Part 4: Calculates LULUCF fluxes and densities

    lu.print_and_log(f"Calculating LULUCF fluxes and carbon densities in {bounds_str} in {tile_id}: {uu.timestr()}", is_final, logger)

    out_dict_uint32, out_dict_float32 = LULUCF_fluxes(
        typed_dict_uint8, typed_dict_int16, typed_dict_float32
    )

    # print(out_dict_uint32)
    # print(out_dict_float32)
    # print(f"Average of {list(out_dict_uint32.keys())[0]} is: {list(out_dict_uint32.values())[0].mean()}")

    # Fresh non-Numba-constrained dictionary that stores all numpy arrays.
    # The dictionaries by datatype that are returned from the numba function have limitations on them,
    # e.g., they can't be combined with other datatypes. This prevents the addition of attributes needed for uploading to s3.
    # So the trick here is to copy the numba-exported arrays into normal Python arrays to which we can do anything in Python.
    out_dict_all_dtypes = {}

    # Transfers the dictionaries of numpy arrays for each data type to a new, Pythonic array
    for key, value in out_dict_uint32.items():
        out_dict_all_dtypes[key] = value

    for key, value in out_dict_float32.items():
        out_dict_all_dtypes[key] = value

    # Clear memory of unneeded arrays
    del out_dict_uint32
    del out_dict_float32


    ### Part 5: Calculates min, mean, and max for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculate stats for the output layers from create_starting_C_densities
    for key, array in out_dict_all_dtypes.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'output_layer'))


    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    # Only saves and uploads to s3 if enabled
    #TODO Confirm that the chunk has data in it before proceeding. Model is saving chunks that are in the ocean and
    # don't have data, e.g., [0 -1 1 0]. Actually, that's probably an issue better solved earlier in the model,
    # before the chunk is run. Why was [0 -1 1 0] even run in the first place?  
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name
            out_pattern = key[:-10] # Drops the date range from the end of the string
            year_range = key[-9:]  # Extracts the year range XXXX_YYYY from the file name

            # Dictionary with metadata for each array
            out_dict_all_dtypes[key] = [value, data_type, out_pattern, year_range]

        uu.save_and_upload_small_raster_set(bounds, chunk_length_pixels, tile_id, bounds_str, out_dict_all_dtypes,
                                            is_final, logger, out_no_data_val)

    # Clear memory of unneeded arrays
    del out_dict_all_dtypes

    success_message = f"Success for {bounds_str}: {uu.timestr()}"
    return success_message, chunk_stats  # Return both the success message and the statistics


def main(cluster_name, bounding_box, chunk_size, run_local=False, no_stats=False, no_log=False, no_upload=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_fluxes'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Makes list of chunks to analyze
    chunks = uu.get_chunk_bounds(bounding_box, chunk_size)
    print(f"Processing {len(chunks)} chunks")
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

    # This is just a placeholder tile_id that is used to obtain the datatype of each tile set.
    # It is overwritten when chunks are assigned and analyzed.
    # Using this placeholder allows the full path and tile name to be specified up front, which simplifies things.
    # Otherwise, we'd have just the path but not the file name now and would have to add in the file name later
    # (probably at the chunk level).
    sample_tile_id = "00N_000E"

    # Dictionary of data to download (inputs to model)
    download_dict = {
        f"{cn.land_cover}_2000": f"{cn.LC_uri}/composite/2000/raw/{sample_tile_id}.tif",
        f"{cn.land_cover}_2005": f"{cn.LC_uri}/composite/2005/raw/{sample_tile_id}.tif",
        f"{cn.land_cover}_2010": f"{cn.LC_uri}/composite/2010/raw/{sample_tile_id}.tif",
        f"{cn.land_cover}_2015": f"{cn.LC_uri}/composite/2015/raw/{sample_tile_id}.tif",
        f"{cn.land_cover}_2020": f"{cn.LC_uri}/composite/2020/raw/{sample_tile_id}.tif",

        f"{cn.vegetation_height}_2000": f"{cn.LC_uri}/vegetation_height/2000/{sample_tile_id}_vegetation_height_2000.tif",
        f"{cn.vegetation_height}_2005": f"{cn.LC_uri}/vegetation_height/2005/{sample_tile_id}_vegetation_height_2005.tif",
        f"{cn.vegetation_height}_2010": f"{cn.LC_uri}/vegetation_height/2010/{sample_tile_id}_vegetation_height_2010.tif",
        f"{cn.vegetation_height}_2015": f"{cn.LC_uri}/vegetation_height/2015/{sample_tile_id}_vegetation_height_2015.tif",
        f"{cn.vegetation_height}_2020": f"{cn.LC_uri}/vegetation_height/2020/{sample_tile_id}_vegetation_height_2020.tif",

        cn.agc_2000: f"{cn.agc_2000_path}{sample_tile_id}__{cn.agc_2000_pattern}.tif",
        cn.bgc_2000: f"{cn.bgc_2000_path}{sample_tile_id}__{cn.bgc_2000_pattern}.tif",
        cn.deadwood_c_2000: f"{cn.deadwood_c_2000_path}{sample_tile_id}__{cn.deadwood_c_2000_pattern}.tif",
        cn.litter_c_2000: f"{cn.litter_c_2000_path}{sample_tile_id}__{cn.litter_c_2000_pattern}.tif",
        cn.soil_c_2000: f"s3://gfw2-data/climate/carbon_model/carbon_pools/soil_carbon/intermediate_full_extent/standard/20231108/{sample_tile_id}_soil_C_full_extent_2000_Mg_C_ha.tif",

        cn.r_s_ratio: f"{cn.r_s_ratio_path}{sample_tile_id}_{cn.r_s_ratio_pattern}.tif",

        "drivers": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/tree_cover_loss_drivers/processed/drivers_2022/20230407/{sample_tile_id}_tree_cover_loss_driver_processed.tif",
        cn.planted_forest_type_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/{sample_tile_id}_plantation_type_oilpalm_woodfiber_other.tif",
        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.planted_forest_tree_crop_layer: f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/{sample_tile_id}.tif",
        # Originally from gfw-data-lake, so it's in 400x400 windows
        "peat": f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/{sample_tile_id}_peat_mask_processed.tif",
        # "ecozone": f"s3://gfw2-data/fao_ecozones/v2000/raster/epsg-4326/10/40000/class/gdal-geotiff/{sample_tile_id}.tif",   # Originally from gfw-data-lake, so it's in 400x400 windows
        # "iso": f"s3://gfw2-data/gadm_administrative_boundaries/v3.6/raster/epsg-4326/10/40000/adm0/gdal-geotiff/{sample_tile_id}.tif",  # Originally from gfw-data-lake, so it's in 400x400 windows
        "ifl_primary": f"s3://gfw2-data/climate/carbon_model/ifl_primary_merged/processed/20200724/{sample_tile_id}_ifl_2000_primary_2001_merged.tif"
    }

    for year in range(cn.first_year, cn.last_year + 1):  # Annual burned area maps start in 2000
        download_dict[f"{cn.burned_area}_{year}"] = f"s3://gfw2-data/climate/carbon_model/other_emissions_inputs/burn_year/burn_year_10x10_clip/ba_{year}_{sample_tile_id}.tif"

    for year in range(cn.first_year + 1, cn.last_year + 1):  # Annual forest disturbance maps start in 2001 and ends in 2020
        download_dict[f"{cn.forest_disturbance}_{year}"] = f"{cn.LC_uri}/annual_forest_disturbance/raw/{year}_{sample_tile_id}.tif"

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.

    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed
    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Creates list of tasks to run (1 task = 1 chunk)
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    delayed_results = [dask.delayed(calculate_and_upload_LULUCF_fluxes)(chunk, download_dict_with_data_types, is_final, no_upload) for chunk in chunks]

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
    parser = argparse.ArgumentParser(description="Calculate LULUCF fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.bounding_box, args.chunk_size, args.run_local, args.no_stats, args.no_log, args.no_upload)
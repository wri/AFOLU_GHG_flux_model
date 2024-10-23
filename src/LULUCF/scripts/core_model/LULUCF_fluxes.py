"""
Run from src/LULUCF

Test:
python -m scripts.utilities.create_cluster -n 1 -m 16 -c 2
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb 10 49.75 10.25 50 -cs 0.25

Full run:
python -m scripts.utilities.create_cluster -n 200 -m 16 -c 2
python -m scripts.core_model.LULUCF_fluxes -cn AFOLU_flux_model_scripts -bb -180 -60 180 80 -cs 1
"""

import argparse
import concurrent.futures
import numpy as np

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
def LULUCF_fluxes(in_dict_uint8, in_dict_int16, in_dict_float32, primary_forest_RFs):

    # Separate dictionaries for output numpy arrays of each datatype, named by output data type).
    # This is because a dictionary in a Numba function cannot have arrays with multiple data types, so each dictionary has to store only one data type,
    # just like inputs to the function.
    out_dict_uint8 = {}
    out_dict_uint16 = {}
    out_dict_uint32 = {}
    out_dict_float32 = {}

    interval_end_years = list(range(cn.first_model_year, cn.last_model_year + 1, cn.interval_years))[1:]
    # interval_end_years = [2005, 2010]

    # Numpy arrays for outputs that do depend on previous interval's values
    agc_dens_block = in_dict_float32[cn.agc_2000_pattern].astype('float32')
    bgc_dens_block = in_dict_float32[cn.bgc_2000_pattern].astype('float32')
    deadwood_c_dens_block = in_dict_float32[cn.deadwood_c_2000_pattern].astype('float32')
    litter_c_dens_block = in_dict_float32[cn.litter_c_2000_pattern].astype('float32')
    soil_c_dens_block = in_dict_int16[cn.soil_c_2000_pattern].astype('float32')

    r_s_ratio_block = in_dict_float32[cn.r_s_ratio_pattern].astype('float32')

    natrl_forest_curve_0_5_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__0_5_years"].astype('float32')
    natrl_forest_curve_6_10_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__6_10_years"].astype('float32')
    natrl_forest_curve_11_15_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__11_15_years"].astype('float32')
    natrl_forest_curve_16_20_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__16_20_years"].astype('float32')
    natrl_forest_curve_21_100_block = in_dict_float32[f"{cn.natural_forest_growth_curve_pattern}__21_100_years"].astype('float32')

    # Removal factor (Mg [some unit]/ha/yr) #TODO Units TBD
    # Because this is used to store the RF from the previous interval,
    # it persists from one interval to the next. Therefore, it must be defined before the first iteration.
    # That way, removal factors can be over-written by those used in the latest interval.
    agc_rf_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

    planted_forest_type_block = in_dict_uint8[cn.planted_forest_type_pattern]
    planted_forest_tree_crop_block = in_dict_uint8[cn.planted_forest_tree_crop_pattern]
    planted_forest_removal_factor_block = in_dict_float32[cn.planted_forest_removal_factor_pattern]
    oil_palm_2000_extent_block = in_dict_uint8[cn.oil_palm_2000_extent_pattern]
    oil_palm_first_year_block = in_dict_int16[cn.oil_palm_first_year_pattern]

    ifl_primary_block = in_dict_uint8[cn.ifl_primary_pattern]
    drivers_block = in_dict_uint8[cn.drivers_pattern]
    continent_ecozone_block = in_dict_int16[cn.continent_ecozone_pattern]

    # Stores the burned area blocks for the entire model duration (added to progressively during each interval)
    burned_area_blocks_all_intervals_so_far = []

    # Stores the forest disturbance blocks for the entire model duration (added to progressively during each interval)
    forest_dist_blocks_all_intervals_so_far = []

    # # Stores the vegetation height blocks through the current interval (added to progressively during each interval)
    # vegetation_height_all_intervals_so_far = []

    # Stores the last year that each pixel did not have tall vegetation composite land cover.
    # 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
    # This is assessed at the pixel level because numba wouldn't allow the needed logical operations on numpy arrays (chunks).
    # Tall vegetation is basd on the composite land cover maps, not the canopy height maps.
    most_recent_year_not_forest_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint16')

    # Number of years of regrowth for new forest
    years_of_new_forest_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')

    # Year in which forest loss occurs/is assigned during an interval (0 if no loss)
    year_of_forest_loss_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint16')


    # Iterates through model intervals
    for interval_end_year in interval_end_years:

        # print(f"Now at {interval_end_year}:")

        # Writes the dictionary entries to a chunk for use in the decision tree
        LC_prev_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year - cn.interval_years}"]
        LC_curr_block = in_dict_uint8[f"{cn.land_cover_pattern}_{interval_end_year}"]
        veg_h_prev_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year - cn.interval_years}"]
        veg_h_curr_block = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year}"]

        # Creates a list of all the burned area arrays from 2001 to the end of the interval.
        # It works by getting the burned area chunks for the current interval and appending them to a list of chunks
        # from previous intervals.
        for year_offset in range(interval_end_year-4, interval_end_year+1):
            year_key = f"{cn.burned_area_pattern}_{year_offset}"
            burned_area_blocks_all_intervals_so_far.append(in_dict_uint8[year_key])

        # Creates a list of all the forest disturbance arrays from 2001 to the end of the interval.
        # The values in the list are the disturbance year starting from 1, e.g., 2001=1, 2008=8, 2017=17.
        # It works by getting the annual disturbance chunks for the current interval and appending them to a list of
        # chunks from previous intervals.
        for year_offset in range(interval_end_year-4, interval_end_year+1):
            # The name of the disturbance layer in the input dictionary
            year_key = f"{cn.forest_disturbance_layer_name}_{year_offset}"

            # Replaces the binary annual disturbance array with the year of disturbance (1, 2, 3...2020)
            year_disturb_array = in_dict_uint8[year_key] * (year_offset - cn.first_model_year)

            # Makes a list of disturbance arrays with the disturbance year.
            # uint8 is okay because the highest value should be 20 (not 2020).
            forest_dist_blocks_all_intervals_so_far.append(year_disturb_array.astype('uint8'))

        # Numpy arrays for outputs that don't depend on previous interval's values
        state_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint32')  # Land cover state at end of interval

        # Number of years of canopy growth.
        # First digit is pre-disturbance years of growth.
        # Second digit (if it exists) is post-disturbance years of growth
        gain_year_count_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('uint8')

        agc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        bgc_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        deadwood_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        litter_c_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

        ch4_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        n2o_gross_emis_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

        agc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        bgc_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        deadwood_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')
        litter_c_gross_removals_out_block = np.zeros(in_dict_float32[cn.agc_2000_pattern].shape).astype('float32')

        # Iterates through all pixels in the chunk
        for row in range(LC_curr_block.shape[0]):
            for col in range(LC_curr_block.shape[1]):

                ### Defining pixel values

                LC_prev = LC_prev_block[row, col]
                LC_curr = LC_curr_block[row, col]
                veg_h_prev = veg_h_prev_block[row, col]
                veg_h_curr = veg_h_curr_block[row, col]

                # r_s_ratio_cell = r_s_ratio_block[row, col]
                #
                # # TODO What to do if this is the first interval and there is no previous RF?
                # agc_rf_prev = agc_rf_out_block[row, col]  # The removal factor from the previous interval
                #
                # # Replaces pixel without R:S (0) with the global non-mangrove R:S default #TODO This is the non-mangrove default. Need to adjust if mangrove pixel?
                # if r_s_ratio_cell == 0:
                #     r_s_ratio_cell = cn.default_r_s_non_mang
                #
                # natrl_forest_curve_0_5 = natrl_forest_curve_0_5_block[row, col]
                # natrl_forest_curve_6_10 = natrl_forest_curve_6_10_block[row, col]
                # natrl_forest_curve_11_15 = natrl_forest_curve_11_15_block[row, col]
                # natrl_forest_curve_16_20 = natrl_forest_curve_16_20_block[row, col]
                # natrl_forest_curve_21_100 = natrl_forest_curve_21_100_block[row, col]
                #
                # planted_forest_type_cell = planted_forest_type_block[row, col]
                # planted_forest_tree_crop_cell = planted_forest_tree_crop_block[row, col]
                # planted_forest_removal_factor_cell = planted_forest_removal_factor_block[row, col]
                # oil_palm_2000_extent_cell = oil_palm_2000_extent_block[row, col]
                # oil_palm_first_year_cell = oil_palm_first_year_block[row, col]
                #
                # ifl_primary_cell = ifl_primary_block[row, col]
                # drivers_cell = drivers_block[row, col]
                # continent_ecozone_cell = continent_ecozone_block[row, col]
                #
                # # Note: Stacking the burned area rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # # So just reading each raster from the list of rasters separately.
                # burned_area_t_4 = burned_area_blocks_all_intervals_so_far[-5][row, col]
                # burned_area_t_3 = burned_area_blocks_all_intervals_so_far[-4][row, col]
                # burned_area_t_2 = burned_area_blocks_all_intervals_so_far[-3][row, col]
                # burned_area_t_1 = burned_area_blocks_all_intervals_so_far[-2][row, col]
                # burned_area_t = burned_area_blocks_all_intervals_so_far[-1][row, col]
                # # Most recent year with burned area during the interval
                # burned_area_last = max([burned_area_t_4, burned_area_t_3, burned_area_t_2, burned_area_t_1, burned_area_t])
                # burned_in_last_interval = (burned_area_last > 0)
                #
                # # Note: Stacking the forest disturbance rasters using ndstack, stack, or flatten outside the pixel iteration did not work with numba.
                # # So just reading each raster from the list of rasters separately.
                # forest_dist_t_4 = forest_dist_blocks_all_intervals_so_far[-5][row, col]
                # forest_dist_t_3 = forest_dist_blocks_all_intervals_so_far[-4][row, col]
                # forest_dist_t_2 = forest_dist_blocks_all_intervals_so_far[-3][row, col]
                # forest_dist_t_1 = forest_dist_blocks_all_intervals_so_far[-2][row, col]
                # forest_dist_t = forest_dist_blocks_all_intervals_so_far[-1][row, col]
                # # Most recent year with forest disturbance during the interval
                # forest_dist_last = max([forest_dist_t_4, forest_dist_t_3, forest_dist_t_2, forest_dist_t_1, forest_dist_t])
                #
                # # if forest_dist_last > 0:
                # #     print(forest_dist_last)
                #
                # # Records the first year of burned area in the pixel, to indicate whether fire was reported at all
                # # in the record
                # first_burn_in_record = 0
                #
                # # Records the first year of forest disturbance in the pixel, to indicate whether disturbance was reported at all
                # # in the record
                # first_forest_dist_in_record = 0
                #
                # # Loops over burned area pixels since 2001 to see if there was a fire.
                # # Stops once a fire is detected because all that matters here is that there was a fire at some point.
                # for burned_area_year in burned_area_blocks_all_intervals_so_far:
                #     # Update the maximum value for this pixel
                #     if burned_area_year[row, col] > 0:
                #         first_burn_in_record = burned_area_year[row, col]
                #         break
                #
                # # if first_burn_in_record > 0:
                # #     print("fire", row, col, first_burn_in_record)
                #
                # # Loops over forest disturbance pixels since 2001 to see if there was a disturbance.
                # # Stops once a disturbance is detected because all that matters here is that there was a disturbance at some point (not the specific year).
                # for forest_dist_year in forest_dist_blocks_all_intervals_so_far:
                #     # Update the maximum value for this pixel
                #     if forest_dist_year[row, col] > 0:
                #         first_forest_dist_in_record = forest_dist_year[row, col]
                #         break
                #
                # # if first_forest_dist_in_record > 0:
                # #     print("disturbance", row, col, first_forest_dist_in_record)
                #
                # # Input carbon densities for the pools using the end of the previous interval
                # agc_dens_in = agc_dens_block[row, col]
                # bgc_dens_in = bgc_dens_block[row, col]
                # deadwood_c_dens_in = deadwood_c_dens_block[row, col]
                # litter_c_dens_in = litter_c_dens_block[row, col]
                # soil_c_dens = soil_c_dens_block[row, col]
                #
                # # Makes a list of carbon densities to save space in the decision tree below.
                # # This list is input to flux calculation functions as one argument, rather than a separate argument
                # # for each pool
                # c_dens_in = [agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in]
                #
                # ### Defining specific classes
                #
                # # Based on individual canopy height raster
                # tree_prev = (veg_h_prev >= cn.tree_threshold)
                # tree_curr = (veg_h_curr >= cn.tree_threshold)
                #
                # # Based on composite land covers
                # tall_veg_prev = (((LC_prev >= cn.tree_dry_min_height_code) and (LC_prev <= cn.tree_dry_max_height_code)) or
                #                  ((LC_prev >= cn.tree_wet_min_height_code) and (LC_prev <= cn.tree_wet_max_height_code)))
                # tall_veg_curr = (((LC_curr >= cn.tree_dry_min_height_code) and (LC_curr <= cn.tree_dry_max_height_code)) or
                #                  ((LC_curr >= cn.tree_wet_min_height_code) and (LC_curr <= cn.tree_wet_max_height_code)))
                # med_veg_prev = (((LC_prev >= 25) and (LC_prev <= 26)) or ((LC_prev >= 125) and (LC_prev <= 126)))
                # med_veg_curr = (((LC_curr >= 25) and (LC_curr <= 26)) or ((LC_curr >= 125) and (LC_curr <= 126)))
                # short_veg_prev = (((LC_prev >= 2) and (LC_prev <= 24)) or ((LC_prev >= 102) and (LC_prev <= 124)))
                # short_veg_curr = (((LC_curr >= 2) and (LC_curr <= 24)) or ((LC_curr >= 102) and (LC_curr <= 124)))
                #
                #
                # # Height change during the interval. Need to recast to signed int8 from uint8 so that negative values (height gain) stay negative.
                # height_change_prev_curr = np.int8(veg_h_prev - veg_h_curr)
                #
                # # Is height loss during the interval significant in absolute change (m)?
                # sig_height_loss_prev_curr_abs = (height_change_prev_curr >= cn.sig_height_loss_threshold_abs)
                #
                # tall_veg_gain = (not tree_prev and tree_curr)
                # tall_veg_loss = (tree_prev and not tree_curr)
                #
                # SDPT_planted_trees = (planted_forest_type_cell > 0)  # All SDPT planted trees
                # SDPT_oil_palm = (planted_forest_type_cell == cn.SDPT_oil_palm_code)  # Oil palm in SDPT planted trees
                # oil_palm_after_Descals = (interval_end_year > oil_palm_first_year_cell) and (oil_palm_first_year_cell != 0) # Second condition to exclude NoData (0s) from first year of oil palm
                # oil_palm_pre_2000 = (oil_palm_2000_extent_cell == 1)
                #
                # all_planted_trees = (SDPT_planted_trees or oil_palm_pre_2000 or oil_palm_after_Descals)
                # all_oil_palm = (SDPT_oil_palm or oil_palm_pre_2000 or oil_palm_after_Descals)

                # The most recent year of non-tall vegetation composite land cover in the cell
                most_recent_year_not_forest = most_recent_year_not_forest_block[row, col]

                # Checks whether to update whether the most recent year is non-tall vegetation
                most_recent_year_not_forest = nu.check_most_recent_year_not_forest(LC_curr, LC_prev, most_recent_year_not_forest, interval_end_year)

                # Arrays of vegetation height and corresponding years in all intervals through the current one
                vegetation_height_all_intervals_so_far = []
                years_so_far = []

                # Iterates through all intervals so far to make arrays of vegetation heights and corresponding years
                for year_offset in list(range(cn.first_model_year, interval_end_year + 1, cn.interval_years)):

                    # Vegetation height layer to retrieve
                    year_key = f"{cn.vegetation_height_pattern}_{year_offset}"

                    # Adds vegetation height and corresponding years to arrays from previous intervals
                    vegetation_height_all_intervals_so_far.append(in_dict_uint8[year_key][row, col])
                    years_so_far.append(year_offset)

                most_recent_year_not_forest = interval_end_year-5  #TODO Delete! Testing code.

                # Determines the maximum height so far if the pixel has been forested since the beginning of the model
                if most_recent_year_not_forest == 0:

                    # The maximum vegetation height through all intervals so far
                    vegetation_max_height_since_last_non_forest = max(vegetation_height_all_intervals_so_far)

                # Determines the maximum height so far if the pixel hasn't had forest at least one year since the beginning of the model
                else:

                    # # Need years and heights so far to be numpy arrays for proper subsetting of them
                    # years_so_far_array = np.array([years_so_far])
                    # heights_so_far_array = np.array([vegetation_height_all_intervals_so_far])
                    #
                    # # Selects the heights corresponding to years greater than the interval end year
                    # heights_since_last_time_not_forest = heights_so_far_array[years_so_far_array > most_recent_year_not_forest]

                    heights_since_last_time_not_forest = []

                    # Loops over the years and corresponding heights to only get heights that are after the most recent
                    # non-forest year.
                    # This could be done more elegantly with conditional numpy arrays but that approach
                    # isn't supported in the numba function, unfortunately.
                    for i in range(len(years_so_far)):
                        if years_so_far[i] > most_recent_year_not_forest:
                            heights_since_last_time_not_forest.append(vegetation_height_all_intervals_so_far[i])

                    # The maximum height in the years since the last non-forest interval
                    vegetation_max_height_since_last_non_forest = max(heights_since_last_time_not_forest)

                double_max = vegetation_max_height_since_last_non_forest * 2

                # if (row == 0) and (col == 0):
                #     print("interval end year:", interval_end_year)
                #     print("vegetation height so far:", vegetation_height_all_intervals_so_far)
                #     print("most recent year not forest:", most_recent_year_not_forest)
                #     print("max height so far:", vegetation_max_height_so_far)




                # # Number of years of regrowth for new forest since last time not forest
                # years_of_new_forest = years_of_new_forest_block[row, col]
                #
                # # Calculates the number of years of forest regrowth since the last year of not-tall vegetation.
                # # Can override the pre-existing value.
                # years_of_new_forest = nu.calculate_years_of_new_forest(interval_end_year, most_recent_year_not_forest, tall_veg_curr, years_of_new_forest)
                #
                #
                # # Starting decision tree node value
                # node = 0
                #
                # # Need to force arrays into float32 because numba is so particular about datatypes.
                # state_out = 0
                # gain_year_count = 0
                # agc_rf = 0
                # c_gross_emis_out = np.array([0, 0, 0, 0]).astype('float32')  # Initializes dummy output C gross emissions: AGC, BGC, deadwood C, litter C.
                # c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Initializes dummy output C gross removals: AGC, BGC, deadwood C, litter C.
                # non_co2_flux_out = np.array([0, 0]).astype('float32')  # Initializes dummy output non-CO2 fluxes: CH4, N2O
                # c_dens_out = np.array([0, 0, 0, 0]).astype('float32')  # Initializes dummy output C densities: AGC, BGC, deadwood C, litter C.


                # ### Tree gain
                # if tall_veg_gain:  # Non-tree converted to tree (1)    #TODO: Include mangrove exception.
                #     node = nu.accrete_node(node, 1)
                #     if all_planted_trees:  # New planted trees (11)
                #         node = nu.accrete_node(node, 1)
                #         if SDPT_planted_trees: # New SDPT planted trees (incl. SDPT oil palm) (111)
                #             state_out = nu.accrete_node(node, 1)
                #             agc_rf = planted_forest_removal_factor_cell
                #             c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                #         else: # New non-SDPT oil palm (112)
                #             state_out = nu.accrete_node(node, 2)
                #             agc_rf = cn.oil_palm_agc_rf
                #             c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                #     else:  # New non-planted trees (12)
                #         node = nu.accrete_node(node, 2)
                #         if tall_veg_curr:  # New terrestrial natural forest (121)
                #             state_out = nu.accrete_node(node, 1)
                #             agc_rf = natrl_forest_curve_0_5
                #             c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                #         else:  # New trees outside forests (122)
                #             state_out = nu.accrete_node(node, 2)
                #             agc_rf = cn.trees_outside_forests_agc_rf_max
                #             c_gross_emis_out, c_gross_removals_out, c_dens_out, gain_year_count = nu.calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in)
                #
                # ### Tree loss
                # elif tall_veg_loss:  # Tree converted to non-tree (2)    #TODO: Include mangrove exception.
                #     node = 2
                #     if all_planted_trees:  # Full loss of planted trees (21)
                #         node = nu.accrete_node(node, 1)
                #         if all_oil_palm:  # Full loss of oil palm (211)
                #             node = nu.accrete_node(node, 1)
                #             if burned_in_last_interval:  # Full loss of oil palm (burned) (2111)
                #                 state_out = nu.accrete_node(node, 1)
                #                 agc_rf = 2.2
                #                 ef = cn.biomass_emissions_only
                #                 c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #             else:  # Full loss of oil palm (not burned) (2112)
                #                 state_out = nu.accrete_node(node, 2)
                #                 agc_rf = 2.2
                #                 ef = cn.agc_emissions_only
                #                 c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #         else:  # Full loss of non-oil palm planted trees (212)
                #             node = nu.accrete_node(node, 2)
                #             if LC_curr == cn.cropland:  # Plantation harvested as cropland (2121)
                #                 node = nu.accrete_node(node, 1)
                #                 if burned_in_last_interval:  # Plantation harvested as cropland (burned) (21211)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Plantation harvested as cropland (not burned) (21212)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif short_veg_curr:  # Plantation harvested as short vegetation (2122)
                #                 node = nu.accrete_node(node, 2)
                #                 if burned_in_last_interval:  # Plantation harvested as short vegetation (burned) (21221)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.biomass_emissions_only
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Plantation harvested as short vegetation (not burned) (21222)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_but_bgc_emissions
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif med_veg_curr:  # Plantation harvested as medium vegetation (2123)
                #                 node = nu.accrete_node(node, 3)
                #                 if burned_in_last_interval:  # Plantation harvested as medium vegetation (burned) (21231)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.biomass_emissions_only
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Plantation harvested as medium vegetation (not burned) (21232)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_but_bgc_emissions
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif LC_curr == cn.builtup:  # Plantation converted to settlement (2124)
                #                 node = nu.accrete_node(node, 4)
                #                 if burned_in_last_interval:  # Plantation converted to settlement (burned) (21241)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Plantation converted to settlement (not burned) (21242)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef =  cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             else:  # Plantation converted to anything else (2125)
                #                 node = nu.accrete_node(node, 5)
                #                 if burned_in_last_interval:  # Plantation converted to anything else (burned) (21251)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.biomass_emissions_only
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Plantation converted to anything else (not burned)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_but_bgc_emissions
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #     else:  # Full loss of non-planted trees (22)
                #         node = nu.accrete_node(node, 2)
                #         if tall_veg_prev:  # Full loss of natural forest (221)
                #             node = nu.accrete_node(node, 1)
                #             if LC_curr == cn.cropland:  # Natural forest converted to cropland (2211)
                #                 node = nu.accrete_node(node, 1)
                #                 if burned_in_last_interval:  # Natural forest converted to cropland (burned) (22111)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf_pre = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     agc_rf_post = 4.7
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Natural forest converted to cropland (not burned) (22112)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf_pre = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     agc_rf_post = 4.7
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif short_veg_curr:  # Natural forest converted to short vegetation (2212)
                #                 node = nu.accrete_node(node, 2)
                #                 if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (22121)
                #                     node = nu.accrete_node(node, 1)
                #                     if burned_in_last_interval:  # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (burned) (221211)
                #                         state_out = nu.accrete_node(node, 1)
                #                         agc_rf = 2.2
                #                         ef = cn.biomass_emissions_only
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                     else:  # Natural forest converted to short vegetation with disturbance that emits all non-soil C pools (not burned) (221212)
                #                         agc_rf = 2.2
                #                         ef = cn.biomass_emissions_only
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #                 else:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (22122)
                #                     node = nu.accrete_node(node, 2)
                #                     if burned_in_last_interval:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (burned) (221221)
                #                         state_out = nu.accrete_node(node, 1)
                #                         agc_rf = 2.2
                #                         ef = cn.all_non_soil_pools
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in,0.5, 4.7, 0.26)
                #                     else:  # Natural forest converted to short vegetation with disturbance that emits biomass C pools only (not burned) (221222)
                #                         state_out = nu.accrete_node(node, 2)
                #                         agc_rf = 2.2
                #                         ef = cn.all_non_soil_pools
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif LC_curr == cn.builtup:  # Natural forest converted to settlement (2213)
                #                 node = nu.accrete_node(node, 3)
                #                 if burned_in_last_interval:  # Natural forest converted to settlement (burned) (22131)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Natural forest converted to settlement (not burned) (22132)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             elif med_veg_curr:  # Natural forest converted to medium vegetation (2214)
                #                 node = nu.accrete_node(node, 4)
                #                 if drivers_cell in cn.drivers_non_soil_C: # Natural forest converted to medium vegetation with disturbance that emits all non-soil C pools (22141)
                #                     node = nu.accrete_node(node, 1)
                #                     if burned_in_last_interval:  # Natural forest converted to medium vegetation with disturbance that emits all non-soil C pools (burned) (221411)
                #                         state_out = nu.accrete_node(node, 1)
                #                         agc_rf_pre = 2.2
                #                         ef = cn.biomass_emissions_only
                #                         agc_rf_post = 4.7
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                     else:  # Natural forest converted to medium vegetation with disturbance that emits all non-soil C pools (not burned) (221412)
                #                         agc_rf_pre = 2.2
                #                         ef = cn.biomass_emissions_only
                #                         agc_rf_post = 4.7
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #                 else:  # Natural forest converted to medium vegetation with disturbance that emits biomass C pools only (22142)
                #                     node = nu.accrete_node(node, 2)
                #                     if burned_in_last_interval:  # Natural forest converted to medium vegetation with disturbance that emits biomass C pools only (burned) (221421)
                #                         state_out = nu.accrete_node(node, 1)
                #                         agc_rf_pre = 2.2
                #                         ef = cn.all_non_soil_pools
                #                         agc_rf_post = 4.7
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in,0.5, 4.7, 0.26)
                #                     else:  # Natural forest converted to medium vegetation with disturbance that emits biomass C pools only (not burned) (221422)
                #                         state_out = nu.accrete_node(node, 2)
                #                         agc_rf_pre = 2.2
                #                         ef = cn.all_non_soil_pools
                #                         agc_rf_post = 4.7
                #                         c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf_pre, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #             else:  # Natural forest converted to anything else (wetland/open water/ice, etc.) (2215)
                #                 node = nu.accrete_node(node, 5)
                #                 if burned_in_last_interval:  # Natural forest converted to anything else (burned) (22151)
                #                     state_out = nu.accrete_node(node, 1)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5, 4.7, 0.26)
                #                 else:  # Natural forest converted to anything else (not burned) (22152)
                #                     state_out = nu.accrete_node(node, 2)
                #                     agc_rf = 2.2
                #                     ef = cn.all_non_soil_pools
                #                     c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #         else:  # Full loss of trees outside forests (222)
                #             node = nu.accrete_node(node, 2)
                #             if burned_in_last_interval:  # Full loss of trees outside forests (burned) (2221)
                #                 state_out = nu.accrete_node(node, 1)
                #                 agc_rf = 2.2
                #                 ef = cn.all_non_soil_pools
                #                 c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, 0.5,4.7, 0.26)
                #             else:  # Full loss of trees outside forests (not burned) (2222)
                #                 state_out = nu.accrete_node(node, 2)
                #                 agc_rf = 2.2
                #                 ef = cn.all_non_soil_pools
                #                 c_gross_emis_out, c_gross_removals_out, non_co2_flux_out, c_dens_out, gain_year_count = nu.calc_T_NT(agc_rf, ef, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in)
                #
                # ### Trees remaining trees
                # elif (tree_prev) and (tree_curr):  # Trees remaining trees (3)    ##TODO: Include mangrove exception.
                #     node = nu.accrete_node(node, 3)
                #     if (forest_dist_last > 0) or (sig_height_loss_prev_curr_abs) :  # Partially disturbed trees (31)
                #         state_out = nu.accrete_node(node, 1)
                #
                #     else:
                #         state_out = nu.accrete_node(node, 2)  # Undisturbed trees (32)
                # #         if planted_forest_type_cell == 0:  # Non-planted trees without stand-replacing disturbance in the last interval (311)
                # #             node = nu.accrete_node(node, 1)
                # #             if not tall_veg_curr:  # Trees outside forests without stand-replacing disturbance in the last interval (3111)
                # #                 node = nu.accrete_node(node, 1)
                # #                 if not sig_height_loss_prev_curr:  # Stable trees outside forests (31111)
                # #                     state_out = nu.accrete_node(node, 1)
                # #                     agc_rf = agc_rf_prev
                # #                     c_net_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                # #                 else:  # Partially disturbed trees outside forests (31112)
                # #                     node = nu.accrete_node(node, 2)
                # #                     if burned_area_last == 0:  # Partially disturbed trees outside forests without fire (311121)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = agc_rf_prev
                # #                         agc_ef = 0.9
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                     else:  # Partially disturbed trees outside forests with fire (311122)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = agc_rf_prev
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #             else:  # Natural forest without stand-replacing disturbance in the last interval (3112)
                # #                 node = nu.accrete_node(node, 2)
                # #                 #TODO This doesn't seem like it's working right, e.g., 15.205 41.84 (assigning node 2 instead of node 1 when no height reduction)
                # #                 # sig_height_loss_prev_curr may not be working
                # #                 if not sig_height_loss_prev_curr:    # Stable natural forest (31121)
                # #                     node = nu.accrete_node(node, 1)
                # #                     if first_forest_dist_in_record == 0:   # Natural forest undisturbed since 2000 (311211)  ##TODO: Include "OR non-forest at some point" exception
                # #                         node = nu.accrete_node(node, 1)
                # #                         if ifl_primary_cell == 0:  # Old secondary forest (3112111)
                # #                             state_out = nu.accrete_node(node, 1)
                # #                             agc_rf = agc_rf_prev
                # #                             c_net_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                # #                         else:  # Primary forest (3112112)
                # #                             state_out = nu.accrete_node(node, 2)
                # #                             agc_rf = agc_rf_prev
                # #                             c_net_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                # #                     else:   # Young secondary natural forest (311212)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = agc_rf_prev
                # #                         c_net_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                # #                 else:  # Partially disturbed natural forest (31122)
                # #                     node = nu.accrete_node(node, 2)
                # #                     if burned_area_last == 0: # Partially disturbed natural forest without fire (311221)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = agc_rf_prev
                # #                         agc_ef = 0.6
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #
                # #                         # if interval_end_year == 2010:
                # #                         #     print(state_out)
                # #                         #     print(agc_rf_prev)
                # #                         #     print(agc_rf)
                # #                         #     print(c_net_flux_out)
                # #                         #     os.quit()
                # #
                # #                     else:  # Partially disturbed natural forest with fire (311222)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = agc_rf_prev
                # #                         agc_ef = 0.95
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #         else:  # SDPT planted trees without stand-replacing disturbance in the last interval (312)
                # #             node = nu.accrete_node(node, 2)
                # #             if not sig_height_loss_prev_curr: # Stable SDPT (3121)
                # #                 state_out = nu.accrete_node(node, 1)
                # #                 agc_rf = 1.3
                # #                 c_net_flux_out, c_dens_out = nu.calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in)
                # #             else: # Partially disturbed SDPT (3122)
                # #                 node = nu.accrete_node(node, 2)
                # #                 if burned_area_last == 0: # Partially disturbed SDPT without fire (31221)
                # #                     state_out = nu.accrete_node(node, 1)
                # #                     agc_rf = 1.5
                # #                     agc_ef = 0.1
                # #                     c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                 else: # Partially disturbed SDPT with fire (31222)
                # #                     state_out = nu.accrete_node(node, 2)
                # #                     agc_rf = 1.3
                # #                     agc_ef = 0.65
                # #                     c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #     else: # Trees with stand-replacing disturbance with regrowth in last interval (32)
                # #         node = nu.accrete_node(node, 2)
                # #         if planted_forest_type_cell == 0: # Non-planted terrestrial trees disturbed with regrowth in last interval (321)
                # #             node = nu.accrete_node(node, 1)
                # #             if not tall_veg_curr: # Trees outside forests disturbed and regrown in last interval (3211)
                # #                 node = nu.accrete_node(node, 1)
                # #                 if burned_area_last == 0: # Trees outside forests disturbed without fire and regrown in last interval (32111)
                # #                     state_out = nu.accrete_node(node, 1)
                # #                     agc_rf = 0.75
                # #                     agc_ef = 0.33
                # #                     c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                 else: # Trees outside forests disturbed with fire and regrown in last interval (32112)
                # #                     state_out = nu.accrete_node(node, 2)
                # #                     agc_rf = 0.75
                # #                     agc_ef = 0.95
                # #                     c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #             else: # Natural forest disturbed and regrown in last interval (3212)
                # #                 node = nu.accrete_node(node, 2)
                # #                 if drivers_cell == 1: # Natural forest loss due to commodity-driven deforestation followed by regrowth (32121)
                # #                     node = nu.accrete_node(node, 1)
                # #                     if burned_area_last == 0: # Natural forest->commod. driven (no fire)->regrowth (321211)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                     else: # Natural forest->commod. driven (fire)->regrowth (321212)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #                 elif drivers_cell == 2: # Natural forest loss due to shifting agriculture followed by regrowth (32122)
                # #                     node = nu.accrete_node(node, 2)
                # #                     if burned_area_last == 0: # Natural forest->shifting ag. (no fire)->regrowth (321221)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                     else: # Natural forest->shifting ag. (fire)->regrowth (321222)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #                 elif drivers_cell == 5:  # Natural forest loss due to urbanization followed by regrowth (32123)
                # #                     node = nu.accrete_node(node, 3)
                # #                     if burned_area_last == 0:  # Natural forest->urbanization (no fire)->regrowth (321231)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                     else:  # Natural forest->urbanization (fire)->regrowth (321232)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #                 else:  # Natural forest loss due to forestry/wildfire/unknown followed by regrowth (32124)
                # #                     node = nu.accrete_node(node, 4)
                # #                     if burned_area_last == 0:  # Natural forest->forestry/wildfire/unknown (no fire)->regrowth (321241)
                # #                         state_out = nu.accrete_node(node, 1)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #                     else:  # Natural forest->forestry/wildfire/unknown (fire)->regrowth (321242)
                # #                         state_out = nu.accrete_node(node, 2)
                # #                         agc_rf = 0.75
                # #                         agc_ef = 0.33
                # #                         c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #         else: # Planted forest disturbed with regrowth in last interval (322)
                # #             node = nu.accrete_node(node, 2)
                # #             if burned_area_last == 0:  # SDPT planted forest->not burned->regrowth (3221)
                # #                 state_out = nu.accrete_node(node, 1)
                # #                 agc_rf = 0.75
                # #                 agc_ef = 0.33
                # #                 c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in)
                # #             else:  # SDPT planted forest->burned->regrowth (3222)
                # #                 state_out = nu.accrete_node(node, 2)
                # #                 agc_rf = 0.75
                # #                 agc_ef = 0.33
                # #                 c_net_flux_out, non_co2_flux_out, c_dens_out = nu.calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, 0.5, 4.7, 0.26)
                # #
                # # # If none of the cases above apply, defaults are assigned
                # # else:
                # #     state_out = 4000000000
                # #     gain_year_count = 0
                # #     agc_rf = 0
                # #
                # #     agc_gross_emis_out_block[row, col] = -1000
                # #     bgc_gross_emis_out_block[row, col] = -900
                # #     deadwood_c_gross_emis_out_block[row, col] = -800
                # #     litter_c_gross_emis_out_block[row, col] = -700
                # #
                # #     ch4_gross_emis_out_block[row, col] = 0
                # #     n2o_gross_emis_out_block[row, col] = 0
                # #
                # #     agc_gross_removals_out_block[row, col] = -1000
                # #     bgc_gross_removals_out_block[row, col] = -900
                # #     deadwood_c_gross_removals_out_block[row, col] = -800
                # #     litter_c_gross_removals_out_block[row, col] = -700
                # #
                # #     agc_dens_block[row, col] = -4000
                # #     bgc_dens_block[row, col] = -3600
                # #     deadwood_c_dens_block[row, col] = -3200
                # #     litter_c_dens_block[row, col] = -2800
                #
                #
                # # Populates the output arrays with the calculated fluxes and densities
                # state_out_block[row, col] = state_out
                #
                # agc_rf_out_block[row, col] = agc_rf
                #
                # agc_gross_emis_out_block[row, col] = c_gross_emis_out[0]
                # bgc_gross_emis_out_block[row, col] = c_gross_emis_out[1]
                # deadwood_c_gross_emis_out_block[row, col] = c_gross_emis_out[2]
                # litter_c_gross_emis_out_block[row, col] = c_gross_emis_out[3]
                #
                # ch4_gross_emis_out_block[row, col] = non_co2_flux_out[0]
                # n2o_gross_emis_out_block[row, col] = non_co2_flux_out[1]
                #
                # agc_gross_removals_out_block[row, col] = c_gross_removals_out[0]
                # bgc_gross_removals_out_block[row, col] = c_gross_removals_out[1]
                # deadwood_c_gross_removals_out_block[row, col] = c_gross_removals_out[2]
                # litter_c_gross_removals_out_block[row, col] = c_gross_removals_out[3]
                #
                # agc_dens_block[row, col] = c_dens_out[0]
                # bgc_dens_block[row, col] = c_dens_out[1]
                # deadwood_c_dens_block[row, col] = c_dens_out[2]
                # litter_c_dens_block[row, col] = c_dens_out[3]
                #
                # #Test/intermediate outputs
                # gain_year_count_out_block[row, col] = gain_year_count
                # most_recent_year_not_forest_block[row, col] = most_recent_year_not_forest
                # years_of_new_forest_block[row, col] = years_of_new_forest


        # End of iteration calculations and outputs

        # Calculates net flux. Gross removals is added to gross emissions because gross removals are already negative
        agc_net_flux_out_block = agc_gross_emis_out_block + agc_gross_removals_out_block
        bgc_net_flux_out_block = bgc_gross_emis_out_block + bgc_gross_removals_out_block
        deadwood_c_net_flux_out_block = deadwood_c_gross_emis_out_block + deadwood_c_gross_removals_out_block
        litter_c_net_flux_out_block = litter_c_gross_emis_out_block + litter_c_gross_removals_out_block

        # Adds the output arrays to the dictionary with the appropriate data type
        # Outputs need .copy() so that previous intervals' arrays in dictionary aren't overwritten because arrays in dictionaries are mutable (courtesy of ChatGPT).
        year_range = f"{interval_end_year - cn.interval_years}_{interval_end_year}"

        out_dict_uint32[f"{cn.land_state_pattern}_{year_range}"] = state_out_block.copy()

        out_dict_float32[f"{cn.agc_rf_pattern}_{year_range}"] = agc_rf_out_block.copy()

        #TODO Convert C fluxes to CO2 fluxes somewhere in the process
        # (best place TBD but perhaps out here rather than in the decision tree so only need to apply 44/12 a few times)
        out_dict_float32[f"{cn.agc_gross_emis_pattern}_{year_range}"] = agc_gross_emis_out_block.copy()
        out_dict_float32[f"{cn.bgc_gross_emis_pattern}_{year_range}"] = bgc_gross_emis_out_block.copy()
        out_dict_float32[f"{cn.deadwood_c_gross_emis_pattern}_{year_range}"] = deadwood_c_gross_emis_out_block.copy()
        out_dict_float32[f"{cn.litter_c_gross_emis_pattern}_{year_range}"] = litter_c_gross_emis_out_block.copy()

        out_dict_float32[f"{cn.agc_gross_removals_pattern}_{year_range}"] = agc_gross_removals_out_block.copy()
        out_dict_float32[f"{cn.bgc_gross_removals_pattern}_{year_range}"] = bgc_gross_removals_out_block.copy()
        out_dict_float32[f"{cn.deadwood_c_gross_removals_pattern}_{year_range}"] = deadwood_c_gross_removals_out_block.copy()
        out_dict_float32[f"{cn.litter_c_gross_removals_pattern}_{year_range}"] = litter_c_gross_removals_out_block.copy()

        out_dict_float32[f"{cn.agc_net_flux_pattern}_{year_range}"] = agc_net_flux_out_block.copy()
        out_dict_float32[f"{cn.bgc_net_flux_pattern}_{year_range}"] = bgc_net_flux_out_block.copy()
        out_dict_float32[f"{cn.deadwood_c_net_flux_pattern}_{year_range}"] = deadwood_c_net_flux_out_block.copy()
        out_dict_float32[f"{cn.litter_c_net_flux_pattern}_{year_range}"] = litter_c_net_flux_out_block.copy()

        out_dict_float32[f"{cn.ch4_flux_pattern}_{year_range}"] = ch4_gross_emis_out_block.copy()
        out_dict_float32[f"{cn.n2o_flux_pattern}_{year_range}"] = n2o_gross_emis_out_block.copy()

        out_dict_float32[f"{cn.agc_dens_pattern}_{interval_end_year}"] = agc_dens_block.copy()
        out_dict_float32[f"{cn.bgc_dens_pattern}_{interval_end_year}"] = bgc_dens_block.copy()
        out_dict_float32[f"{cn.deadwood_c_dens_pattern}_{interval_end_year}"] = deadwood_c_dens_block.copy()
        out_dict_float32[f"{cn.litter_c_dens_pattern}_{interval_end_year}"] = litter_c_dens_block.copy()

        # Test/intermediate outputs
        out_dict_uint8[f"{cn.gain_year_count_pattern}_{year_range}"] = gain_year_count_out_block.copy()
        # Years selected to show it represents from model start to end of current interval
        out_dict_uint16[f"most_recent_year_not_forest_{cn.first_model_year}_{interval_end_year}"] = most_recent_year_not_forest_block.copy()
        out_dict_uint8[f"years_of_new_forest_{year_range}"] = years_of_new_forest_block.copy()
        out_dict_uint16[f"years_of_forest_loss_{year_range}"] = year_of_forest_loss_block.copy()

    return out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32


# Downloads inputs, prepares data, calculates LULUCF stocks and fluxes, and uploads outputs to s3
def calculate_and_upload_LULUCF_fluxes(bounds, primary_forest_RFs, download_dict_with_data_types, is_final, no_upload):

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

    # Test prints
    # print(layers)
    # print(layers['burned_area_2002'].max())
    # print(layers[soil_c_2000_pattern].dtype)

    # List of layers that must be present for the chunk to be run.
    # All of the listed layers must exist for this chunk in order to proceed.
    checked_layers = {cn.agc_2000_pattern: layers[cn.agc_2000_pattern], cn.bgc_2000_pattern: layers[cn.bgc_2000_pattern],
                      cn.deadwood_c_2000_pattern: layers[cn.deadwood_c_2000_pattern],
                      cn.litter_c_2000_pattern: layers[cn.litter_c_2000_pattern],
                      f"{cn.land_cover_pattern}_2000": layers[f"{cn.land_cover_pattern}_2000"]}

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

    out_dict_uint8, out_dict_uint16, out_dict_uint32, out_dict_float32 = LULUCF_fluxes(
        typed_dict_uint8, typed_dict_int16, typed_dict_float32, primary_forest_RFs
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
    for key, value in out_dict_uint8.items():
        out_dict_all_dtypes[key] = value

    for key, value in out_dict_uint16.items():
        out_dict_all_dtypes[key] = value

    for key, value in out_dict_uint32.items():
        out_dict_all_dtypes[key] = value

    for key, value in out_dict_float32.items():
        out_dict_all_dtypes[key] = value

    # Clear memory of unneeded arrays
    del out_dict_uint8
    del out_dict_uint16
    del out_dict_uint32
    del out_dict_float32


    ### Part 5: Calculates min, mean, and max for each output chunk.
    ### Useful for QC-- to see if there are any egregiously incorrect or unexpected values.

    # Calculate stats for the output layers from create_starting_C_densities
    for key, array in out_dict_all_dtypes.items():
        chunk_stats.append(uu.calculate_stats(array, key, bounds_str, tile_id, 'output_layer'))


    ### Part 6: Saves numpy arrays as rasters and uploads to s3

    # Only saves arrays to geotifs and uploads them to s3 if enabled
    if not no_upload:

        out_no_data_val = 0  # NoData value for output raster (optional)

        # Adds metadata used for uploading outputs to s3 to the dictionary
        for key, value in out_dict_all_dtypes.items():
            data_type = value.dtype.name

            # Retrieves the file name pattern and date(s) covered for the output file for use in s3 folder construction
            out_pattern, year_range = uu.strip_and_extract_years(key)

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

        cn.agc_2000_pattern: f"{cn.agc_2000_path}{sample_tile_id}__{cn.agc_2000_pattern}.tif",
        cn.bgc_2000_pattern: f"{cn.bgc_2000_path}{sample_tile_id}__{cn.bgc_2000_pattern}.tif",
        cn.deadwood_c_2000_pattern: f"{cn.deadwood_c_2000_path}{sample_tile_id}__{cn.deadwood_c_2000_pattern}.tif",
        cn.litter_c_2000_pattern: f"{cn.litter_c_2000_path}{sample_tile_id}__{cn.litter_c_2000_pattern}.tif",
        cn.soil_c_2000_pattern: f"{cn.soil_c_2000_path}{sample_tile_id}_{cn.soil_c_2000_pattern}.tif",

        cn.r_s_ratio_pattern: f"{cn.r_s_ratio_path}{sample_tile_id}_{cn.r_s_ratio_pattern}.tif",

        cn.drivers_pattern: f"{cn.drivers_path}{sample_tile_id}_{cn.drivers_pattern}.tif",

        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_path}{sample_tile_id}_{cn.planted_forest_type_pattern}.tif",
        cn.planted_forest_removal_factor_pattern: f"{cn.planted_forest_removal_factor_path}{sample_tile_id}_{cn.planted_forest_removal_factor_pattern}.tif",
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_path}{sample_tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_path}{cn.oil_palm_first_year_pattern}_{sample_tile_id}.tif",   # Pattern is before tile_id for this input
        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_path}{sample_tile_id}.tif",

        # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.organic_soil_extent_pattern: f"{cn.organic_soil_extent_path}{sample_tile_id}_{cn.organic_soil_extent_pattern}.tif",
        # "ecozone": f"s3://gfw2-data/fao_ecozones/v2000/raster/epsg-4326/10/40000/class/gdal-geotiff/{sample_tile_id}.tif",   # Originally from gfw-data-lake, so it's in 400x400 windows
        # "iso": f"s3://gfw2-data/gadm_administrative_boundaries/v3.6/raster/epsg-4326/10/40000/adm0/gdal-geotiff/{sample_tile_id}.tif",  # Originally from gfw-data-lake, so it's in 400x400 windows
        cn.ifl_primary_pattern: f"{cn.ifl_primary_path}{sample_tile_id}_{cn.ifl_primary_pattern}.tif",
        cn.continent_ecozone_pattern: f"{cn.continent_ecozone_path}{sample_tile_id}_{cn.continent_ecozone_pattern}.tif"
    }

    # Land cover and vegetation height rasters (5-year intervals)
    for year in range(cn.first_model_year, cn.last_model_year + 1, cn.interval_years):
        download_dict[f"{cn.land_cover_pattern}_{year}"] = f"{cn.land_cover_path}{year}/raw/{sample_tile_id}.tif"
        download_dict[f"{cn.vegetation_height_pattern}_{year}"] = f"{cn.vegetation_height_path}{year}/{sample_tile_id}_{cn.vegetation_height_pattern}_{year}.tif"

    # Burned area rasters (annual)
    # All years need to be in their own folder
    for year in range(cn.first_model_year, cn.last_model_year + 1):  # Annual burned area maps start in 2000
        download_dict[f"{cn.burned_area_pattern}_{year}"] = f"{cn.burned_area_path}{year}/{cn.burned_area_pattern}_{year}_{sample_tile_id}.tif"

    # Forest disturbance rasters (annual)
    # All years need to be in their own folder
    for year in range(cn.first_model_year + 1, cn.last_model_year + 1):  # Annual forest disturbance maps start in 2001 and ends in 2020
        download_dict[f"{cn.forest_disturbance_layer_name}_{year}"] = f"{cn.annual_forest_disturbance_path}{year}/{year}_{sample_tile_id}.tif"

    # Young natural forest rasters (several age intervals)
    # Each growth interval's rate is in its own folder
    for growth_interval in cn.natural_forest_growth_curve_intervals:
        download_dict[f"{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years"] = f"{cn.natural_forest_growth_curve_path}rate_{growth_interval}/{sample_tile_id}_{cn.natural_forest_growth_curve_pattern}__{growth_interval}_years.tif"

    # Returns the first tile in each input so that the datatype can be determined.
    # This is done up front, once per tile set, rather than on each chunk, since
    # all tiles have the same datatype for each input-- it only needs to be done once at the very beginning of the stage.
    print(f"Getting tile_id of first tile in each tile set: {uu.timestr()}")
    first_tiles = uu.first_file_name_in_s3_folder(download_dict)

    # Creates a download dictionary with the datatype of each input in the values.
    # This is supplied to each chunk that is being analyzed.
    # This also serves as a check of whether all inputs are being found (s3 paths correct)
    print(f"Getting datatype of first tile in each tile set: {uu.timestr()}")
    download_dict_with_data_types = uu.add_file_type_to_dict(first_tiles)

    # Creates numpy array of IPCC Tier 1 primary forest removal factors by continent-ecozone combination.
    # Needs to by a numpy array for the numba function to use it.
    primary_forest_RFs = uu.convert_lookup_table_to_array(cn.IPCC_removal_factor_table_full_path,
                                                          cn.IPCC_removal_factor_table_tab,
                                                          ['gainEcoCon', 'growth_primary'])

    # Creates list of tasks to run (1 task = 1 chunk)
    print(f"Creating tasks and starting processing: {uu.timestr()}")

    futures = []
    for chunk in chunks:
        future = client.submit(calculate_and_upload_LULUCF_fluxes, chunk,
                               primary_forest_RFs, download_dict_with_data_types, is_final, no_upload)
        futures.append(future)

    # Collect the results once they are finished
    results = client.gather(futures)

    # Initializes counters for different types of return messages
    success_count = 0
    skipping_chunk_count = 0

    # Processes the chunk stats and returned messages
    # Results are the messages from the chunks and chunk stats
    for result in results:
        return_message, chunk_stats = result

        if "Success" in return_message:
            success_count += 1

        if "Skipped chunk" in return_message:
            skipping_chunk_count += 1

        if return_message:
            return_messages.append(return_message)

        if chunk_stats is not None:
            all_stats.extend(chunk_stats)

    # Prints the returned messages
    for message in return_messages:
        print(message)

    # Print the counts
    print(f"Number of 'Success' chunks: {success_count}")
    print(f"Number of 'Skipped' chunks: {skipping_chunk_count}")
    print(f"Difference between submitted chunks and processed chunks: {len(chunks) - (success_count + skipping_chunk_count)}")

    end_time_1 = uu.timestr()
    print(f"Stage {stage} ended at: {end_time_1}")
    uu.stage_duration(start_time, end_time_1, stage)

    # Prepares chunk stats spreadsheet: min, mean, max for all input and output chunks,
    # and min and max values across all chunks for all inputs and outputs
    # only if not suppressed by the --no_stats flag
    if not no_stats:
        uu.calculate_chunk_stats(all_stats, stage)

    # Ending time for stage
    end_time_2 = uu.timestr()
    print(f"Stage {stage} ended at: {end_time_2}")
    uu.stage_duration(start_time, end_time_2, stage)

    # Creates combined log if not deactivated
    log_note = f"{stage} run"
    lu.compile_and_upload_log(no_log, client, cluster, stage, len(chunks), chunk_size, start_time, end_time_2, success_count, skipping_chunk_count, log_note)

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
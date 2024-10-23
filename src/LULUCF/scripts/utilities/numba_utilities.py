import math
import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types

# Project imports
from . import constants_and_names as cn


# Adds latest decision tree branch to the state node
@jit(nopython=True)
def accrete_node(combo, new):
    combo = combo*10 + new
    return combo


# Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
# Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32)
# Note: need to add new code if inputs with other data types are added
def create_typed_dicts(layers):
    # Initializes empty dictionaries for each type
    uint8_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    # Iterates through the downloaded chunk dictionary and distributes arrays to a separate dictionary for each data type
    for key, array in layers.items():

        # Skips the dictionary entry if it has no data (generally because the chunk doesn't exist for that input)
        if array is None:
            continue

        # If there is data, it puts the data in the corresponding dictionary for that datatype
        if array.dtype == np.uint8:
            uint8_dict_layers[key] = array
        elif array.dtype == np.int16:
            int16_dict_layers[key] = array
        elif array.dtype == np.int32:
            int32_dict_layers[key] = array
        elif array.dtype == np.float32:
            float32_dict_layers[key] = array
        else:
            pass
            # raise TypeError(f"{key} dtype not in list")

    # print(f"uint8 datasets: {uint8_dict_layers.keys()}")
    # print(f"int16 datasets: {int16_dict_layers.keys()}")
    # print(f"int32 datasets: {int32_dict_layers.keys()}")
    # print(f"float32 datasets: {float32_dict_layers.keys()}")

    # Creates numba-compliant typed dict for each type of array
    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')  # Assuming 2D arrays of uint8
    )

    typed_dict_int16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int16, 2, 'C')  # Assuming 2D arrays of int16
    )

    typed_dict_int32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int32, 2, 'C')  # Assuming 2D arrays of int32
    )

    typed_dict_float32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.float32, 2, 'C')  # Assuming 2D arrays of float32
    )

    # Populates the numba-compliant typed dicts
    for key, array in uint8_dict_layers.items():
        typed_dict_uint8[key] = array

    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array

    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array

    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32


# Returns the starting carbon density for each carbon pool
@jit(nopython=True)
def unpack_starting_carbon_densities(c_dens_in):

    agc_dens_in = np.float32(c_dens_in[0])
    bgc_dens_in = np.float32(c_dens_in[1])
    deadwood_c_dens_in = np.float32(c_dens_in[2])
    litter_c_dens_in = np.float32(c_dens_in[3])

    return agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in


# Returns the emission factor for each carbon pool
@jit(nopython=True)
def unpack_stand_replacing_emission_factors(ef):

    agc_ef = np.float32(ef[0])
    bgc_ef = np.float32(ef[1])
    deadwood_c_ef = np.float32(ef[2])
    litter_c_ef = np.float32(ef[3])

    return agc_ef, bgc_ef, deadwood_c_ef, litter_c_ef


# Calculate non-CO2 emissions
#TODO Make sure this is right? Should there be CO2 emissions from fire as well? Had some of that in the forest model?
# Outstanding questions are highlighted in emission factor slides of model schematic.
@jit(nopython=True)
def fire_equations(carbon_in, r_s_ratio_cell, Cf, Gef_ch4, Gef_n2o):
    # Cf is the combustion factor
    # Gef_ch4 and Gef_n2o are the emission factors for their respective gases

    # Default non-CO2 gases to 0
    ch4_flux_out = 0
    n2o_flux_out = 0

    if (isinstance(Cf, (int, float)) and isinstance(Gef_ch4, (int, float)) and isinstance(Gef_n2o, (int, float))):

        # print(f"Carbon in: {carbon_in}; R:S: {r_s_ratio_cell}; Cf: {Cf}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")

        ch4_flux_out = (carbon_in/r_s_ratio_cell) * Cf * Gef_ch4 * cn.g_to_kg * cn.gwp_ch4  # TODO This assumes non-mangrove. Need to make flexible?
        n2o_flux_out = (carbon_in/r_s_ratio_cell) * Cf * Gef_n2o * cn.g_to_kg * cn.gwp_n2o  # TODO This assumes non-mangrove. Need to make flexible?

    # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")

    return ch4_flux_out, n2o_flux_out


# Gross and net fluxes and ending carbon stocks for non-tree converted to tree
@jit(nopython=True)
def calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in):

    # Retrieves the starting densities for each carbon pool from the input array
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Step 1: Calculates the number of years of carbon gain
    gain_year_count = cn.NF_F_gain_year

    # Step 2: Calculates gross removals by carbon pools. Gross removals are negative.
    agc_gross_removals_out = (agc_rf * gain_year_count) * -1
    bgc_gross_removals_out = float(agc_gross_removals_out) * r_s_ratio_cell
    deadwood_c_gross_removals_out = cn.deadwood_c_NT_T_rf
    litter_c_gross_removals_out = cn.litter_c_NT_T_rf

    # Step 3: Calculates gross emissions by carbon pools
    agc_gross_emis_out = 0
    bgc_gross_emis_out = 0
    deadwood_c_gross_emis_out = 0
    litter_c_gross_emis_out = 0

    # Step 4: Calculates ending carbon densities by carbon pool
    agc_dens_out = agc_dens_in - agc_gross_removals_out
    bgc_dens_out = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_removals_out

    # Step 5: Prepares outputs
    # Consolidates all gross fluxes from all carbon pools into arrays to reduce the number of arguments returned to the decision tree
    # Must specify float32 because numba is quite particular about datatypes
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out, gain_year_count


# Gross and net fluxes and ending carbon stocks for tree converted to non-tree with and without fire.
# Non-CO2 gas emissions are only calculated if arguments for fires are supplied.
@jit(nopython=True)
def calc_T_NT(agc_rf, ef_by_pool, forest_dist_last, r_s_ratio_cell, interval_end_year, c_dens_in, Cf=None, Gef_ch4=None, Gef_n2o=None):

    # Retrieves the starting densities for each carbon pool from the input array
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Emission factor for each non-soil carbon pool
    agc_ef, bgc_ef, deadwood_c_ef, litter_c_ef = unpack_stand_replacing_emission_factors(ef_by_pool)

    ## Step 1: Calculates the number of years of carbon gin before loss occurred
    if forest_dist_last > 0:
        # If a forest disturbance was detected, the gain_year_count are the number of years until detection of the last disturbance.
        # There is no growth in the year of disturbance or the years after.
        # The - 1 at the excludes the disturbance year from the gain_year_count since we decided there are no removals in the disturbance year.
        # For example, if the time interval is 2010-2015 and the disturbance is detected in 2013 (t-2),
        # there should be 2 years of growth (years t-4 and t-3, 2011 and 2012).
        # This table illustrates each case for the example interval of 2010-2015.
        # 0 years         11               - ((2015              - 2000)                - 5) - 1   (year t-4)
        # 1 years         12               - ((2015              - 2000)                - 5) - 1   (year t-3)
        # 2 years         13               - ((2015              - 2000)                - 5) - 1   (year t-2)
        # 3 years         14               - ((2015              - 2000)                - 5) - 1   (year t-1)
        # 4 years         15               - ((2015              - 2000)                - 5) - 1   (year t)
        gain_year_count = forest_dist_last - ((interval_end_year - cn.first_model_year) - cn.interval_years) - 1
    else:
        # If a forest disturbance was not detected, the disturbance is assumed to occur in the middle of the interval
        # (year t-2), with removals until then (years t-4 and t-3). There are no removals in the year of assumed
        # disturbance or the years after.
        gain_year_count = math.floor(cn.interval_years/2)

    # Step 2: Calculates gross removals by carbon pools. Gross removals are negative.
    agc_gross_removals_out = (agc_rf * gain_year_count) * -1
    bgc_gross_removals_out = float(agc_gross_removals_out) * r_s_ratio_cell
    deadwood_c_gross_removals_out= cn.deadwood_c_NT_T_rf
    litter_c_gross_removals_out= cn.litter_c_NT_T_rf

    # Step 3: Calculates carbon densities at the year of loss by carbon pool
    agc_pre_disturb = agc_dens_in - agc_gross_removals_out
    bgc_pre_disturb = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_pre_disturb = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_pre_disturb = litter_c_dens_in - litter_c_gross_removals_out

    # Step 4: Calculates gross emissions by carbon pools
    agc_gross_emis_out = agc_pre_disturb * agc_ef
    bgc_gross_emis_out = bgc_pre_disturb * bgc_ef
    deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef
    litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef

    # Step 5: Calculates ending carbon densities by carbon pool.
    # Starts with carbon density in, adds gross removals (subtracts negative value), subtracts emissions
    agc_dens_out = agc_dens_in - agc_gross_removals_out - agc_gross_emis_out
    bgc_dens_out = bgc_dens_in - bgc_gross_removals_out - bgc_gross_emis_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_removals_out - deadwood_c_gross_emis_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_removals_out - litter_c_gross_emis_out

    # Step 6: Calculates non-CO2 gases, if required (as determined by fire-related argument to function)
    ch4_flux_out, n2o_flux_out = fire_equations(agc_dens_in, r_s_ratio_cell, Cf, Gef_ch4, Gef_n2o)  # agc_dens_in can be any set of carbon pools

    # Step 7: Prepares outputs
    # Consolidates all gross fluxes from all carbon pools into arrays to reduce the number of arguments returned to the decision tree
    # Must specify float32 because numba is quite particular about datatypes
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')
    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out, gain_year_count


# Gross and net fluxes and ending carbon stocks for trees remaining trees without disturbances
@jit(nopython=True)
def calc_T_T_undisturbed(agc_rf, r_s_ratio_cell, c_dens_in):

    agc_dens_in = c_dens_in[0]
    bgc_dens_in = c_dens_in[1]
    deadwood_c_dens_in = c_dens_in[2]
    litter_c_dens_in = c_dens_in[3]

    agc_flux_out = (agc_rf * cn.interval_years) * -1
    bgc_flux_out = float(agc_flux_out) * r_s_ratio_cell
    deadwood_c_flux_out= cn.deadwood_c_T_T_rf
    litter_c_flux_out= cn.litter_c_T_T_rf

    agc_dens_out = agc_dens_in - agc_flux_out
    bgc_dens_out = bgc_dens_in - bgc_flux_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_flux_out
    litter_c_dens_out = litter_c_dens_in - litter_c_flux_out

    # Must specify float32 because numba is quite particular about datatypes
    c_fluxes_out = np.array([agc_flux_out, bgc_flux_out, deadwood_c_flux_out, litter_c_flux_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    return c_fluxes_out, c_dens_out


# Gross and net fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances
#TODO include sequence of fluxes for disturbances: pre-disturb removals, emissions, post-disturb removals
@jit(nopython=True)
def calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, Cf=None, Gef_ch4=None, Gef_n2o=None):
    agc_dens_in = c_dens_in[0]
    bgc_dens_in = c_dens_in[1]
    deadwood_c_dens_in = c_dens_in[2]
    litter_c_dens_in = c_dens_in[3]

    agc_flux_out = (agc_rf * cn.interval_years) * -1
    bgc_flux_out = float(agc_flux_out) * r_s_ratio_cell
    deadwood_c_flux_out = cn.deadwood_c_T_T_rf
    litter_c_flux_out = cn.litter_c_T_T_rf

    agc_dens_out = agc_dens_in - agc_flux_out
    bgc_dens_out = bgc_dens_in - bgc_flux_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_flux_out
    litter_c_dens_out = litter_c_dens_in - litter_c_flux_out

    # Must specify float32 because numba is quite particular about datatypes
    c_fluxes_out = np.array([agc_flux_out, bgc_flux_out, deadwood_c_flux_out, litter_c_flux_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    # Calculates non-CO2 gases, if required (as determined by fire-related argument to function)
    ch4_flux_out, n2o_flux_out = fire_equations(agc_dens_in, r_s_ratio_cell, Cf, Gef_ch4, Gef_n2o)  # agc_dens_in can be any set of carbon pools

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return c_fluxes_out, non_co2_fluxes_out, c_dens_out


# Gross and net fluxes and ending carbon stocks for trees remaining trees with stand-replacing disturbances
#TODO include sequence of fluxes for disturbances: pre-disturb removals, emissions, post-disturb removals
@jit(nopython=True)
def calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in, Cf=None, Gef_ch4=None, Gef_n2o=None):

    agc_dens_in = c_dens_in[0]
    bgc_dens_in = c_dens_in[1]
    deadwood_c_dens_in = c_dens_in[2]
    litter_c_dens_in = c_dens_in[3]

    agc_flux_out = (agc_rf * cn.interval_years) * -1
    bgc_flux_out= float(agc_flux_out) * r_s_ratio_cell
    deadwood_c_flux_out= cn.deadwood_c_T_T_rf
    litter_c_flux_out= cn.litter_c_T_T_rf

    agc_dens_out = agc_dens_in - agc_flux_out
    bgc_dens_out = bgc_dens_in - bgc_flux_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_flux_out
    litter_c_dens_out = litter_c_dens_in - litter_c_flux_out

    # Must specify float32 because numba is quite particular about datatypes
    c_fluxes_out = np.array([agc_flux_out, bgc_flux_out, deadwood_c_flux_out, litter_c_flux_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    # Calculates non-CO2 gases, if required (as determined by fire-related argument to function)
    ch4_flux_out, n2o_flux_out = fire_equations(agc_dens_in, r_s_ratio_cell, Cf, Gef_ch4, Gef_n2o)  # agc_dens_in can be any set of carbon pools

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return c_fluxes_out, non_co2_fluxes_out, c_dens_out


# Checks if pixel does not have tall vegetation. If so, updates the value to the most recent year without tall vegetation.
# Tall vegetation/non-tall vegetation is based on the composite land cover maps, not the canopy height maps.
# 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
# Theoretically, checking whether pixels are ever not forest could be done at the chunk level rather
# than at the pixel level (as done here). However, numba doesn't allow the conditional operation
# that would have to be applied to numpy arrays, so I'm doing it at the pixel level instead.
# I have no idea if checking if pixels have ever not been forest is faster or slower at the pixel level
# than at the numpy array level, but the array-level operation isn't even an option.
@jit(nopython=True)
def check_most_recent_year_not_tall_veg(LC_curr, LC_prev, most_recent_year_not_forest, interval_end_year):

    # For the first interval, the land cover in 2000 has to be checked for tall vegetation as well
    if interval_end_year == (cn.first_model_year + cn.interval_years):

        # Criteria for excluding tall vegetation land cover
        not_tall_veg_condition = (
                (LC_prev < cn.tree_dry_min_height_code) |
                ((LC_prev > cn.tree_dry_max_height_code) & (LC_prev < cn.tree_wet_min_height_code))
                | (LC_prev > cn.tree_wet_max_height_code)
        )

        # Sets cell to the model start year wherever land cover is not tall vegetation
        if not_tall_veg_condition == 1:
            most_recent_year_not_forest = cn.first_model_year


    # Checks the current end of interval land cover
    # Criteria for excluding tall vegetation land cover
    not_tall_veg_condition = (
            (LC_curr < cn.tree_dry_min_height_code) |
            ((LC_curr > cn.tree_dry_max_height_code) & (LC_curr < cn.tree_wet_min_height_code))
            | (LC_curr > cn.tree_wet_max_height_code)
    )

    # Sets cell to interval end year wherever land cover is not tall vegetation
    if not_tall_veg_condition == 1:
        most_recent_year_not_forest = interval_end_year

    return most_recent_year_not_forest


# Calculates the number of years of forest regrowth since the last year of not-tall vegetation
@jit(nopython=True)
def calculate_years_of_forest_regrowth(interval_end_year, most_recent_year_not_forest, tall_veg_curr, years_of_forest_regrowth):

    # Determines if the number of years of regrowth should be calculated.
    # Condition 1: The end of the interval must be after the last year that was not tall vegetation,
    # i.e. there was not tall vegetation previously but there is at the end of this interval (indicating regrowth).
    # Condition 2: There must have been some year that was not forest,
    # i.e. the years of regrowth is only relevant when there was not forest some year.
    if (interval_end_year > most_recent_year_not_forest) & (most_recent_year_not_forest > 0):

        # Calculates the number of years of regrowth since the last year that was not forest
        years_of_forest_regrowth = (interval_end_year - most_recent_year_not_forest)

    # Resets the growth year counter in cases where there was tall vegetation and then there wasn't in the next interval.
    # Otherwise, the years counter would continue accruing even if tall veg was lost.
    if not tall_veg_curr:
        years_of_forest_regrowth = 0

    return years_of_forest_regrowth


# Calculates the maximum canopy height since the last time a pixel was classified as not tall vegetation land cover.
# This is used to determine whether current height has decreased significantly from this maximum height.
@jit(nopython=True)
def calc_max_height_since_last_time_not_tall_veg(most_recent_year_not_tall_veg, interval_end_year, in_dict_uint8, row, col):

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

    # Determines the maximum height so far if the pixel has been tall vegetation land cover since the beginning of the model
    if most_recent_year_not_tall_veg == 0:

        # The maximum vegetation height through all intervals so far
        max_height_since_last_time_not_tall_veg = max(vegetation_height_all_intervals_so_far)

    # Determines the maximum height so far if the pixel hasn't had tall vegetation land cover at least one year since the beginning of the model
    else:

        heights_since_last_time_not_tall_veg = []

        # Loops over the years and corresponding heights to only get heights that are after the most recent
        # non-tall vegetation year.
        # This could be done more elegantly with conditional numpy arrays but that approach
        # isn't supported in the numba function, unfortunately.
        # https://chatgpt.com/share/e/6718fb20-48d8-800a-9eb2-d751bd6b1a8f
        for i in range(len(years_so_far)):
            if years_so_far[i] > most_recent_year_not_tall_veg:
                heights_since_last_time_not_tall_veg.append(vegetation_height_all_intervals_so_far[i])

        # In case the pixel is currently non-tall vegetation land cover, so there are no intervals since then
        # and therefore no heights
        if len(heights_since_last_time_not_tall_veg) == 0:

            # Uses the current vegetation height (which would exist when the land cover is not tall vegetation
            # but there is still tall vegetation in the individual tree height layer)
            max_height_since_last_time_not_tall_veg = in_dict_uint8[f"{cn.vegetation_height_pattern}_{interval_end_year}"][row, col]
            # max_height_since_last_time_not_tall_veg = 0

        # When the pixel was previously non-tall vegetation but is now tall vegetation,
        # so there are intervals since then.
        else:

            # The maximum height in the years since the last non-tall vegetation land cover interval
            max_height_since_last_time_not_tall_veg = max(heights_since_last_time_not_tall_veg)

    return max_height_since_last_time_not_tall_veg
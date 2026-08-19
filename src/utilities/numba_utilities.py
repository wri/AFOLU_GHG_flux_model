import math
import numpy as np
from numba import jit
from numba.typed import Dict
from numba.core import types
import sys

# Project imports
from src.utilities import constants_and_names as cn


# Adds latest decision tree branch to the state node
@jit(nopython=True)
def accrete_node(combined, new_digit):
    combined = combined * 10 + new_digit
    return combined


# Makes all output states the same number of digits (currently 7) by padding 0s to the right
@jit(nopython=True)
def pad_to_8_digits(state_out, max_digits_state_out):

    if state_out < 10 ** (max_digits_state_out-1):
        digits = int(np.log10(state_out)) + 1 if state_out > 0 else 1
        pad_zeros = max_digits_state_out - digits
        state_out = state_out * (10 ** pad_zeros)

    return np.uint32(state_out)


# Calculates a backup continent-ecozone value or climate zone value in case pixels don't have one.
# There are many ways that are more efficient or at least succinct to calculate the mode of an array in Python,
# but they don't work with Numba. So, I'm going with this.
# https://chatgpt.com/share/e/67bf7958-351c-800a-bd00-259213586471
# Excludes continent-ecozone codes for water.
@jit(nopython=True)
def fallback_conteco_climzone_value(conteco_or_climzone_block, fallback_value):

    # Flattens the array
    flat = conteco_or_climzone_block.ravel()

    # Manually masks out values to ignore in calculating dominant continent-ecozone: 0, 1022, 2022, 4022, 7022 (water codes)
    valid = []
    for i in range(flat.size):
        v = flat[i]
        if v != 0 and v != 1022 and v != 2022 and v != 4022 and v != 7022:  # Last 4 are water codes
            valid.append(v)

    # If no valid pixels after excluding the codes to ignore, the fallback value is used
    if len(valid) == 0:
        return fallback_value

    # Converts list to NumPy array
    valid_arr = np.array(valid)

    # Gets the most common (mode) continent-ecozone
    counts = np.bincount(valid_arr)
    return np.argmax(counts)


# Creates a separate dictionary for each chunk datatype so that they can be passed to Numba as separate arguments.
# Numba functions can accept (and return) dictionaries of arrays as long as each dictionary only has arrays of one data type (e.g., uint8, float32)
# Note: need to add new code if inputs with other data types are added
def create_typed_dicts(layers):

    # Initializes empty dictionaries for each type
    uint8_dict_layers = {}
    uint16_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    # Iterates through the downloaded chunk dictionary and distributes arrays to a separate dictionary for each data type
    for key, array in layers.items():

        # Skips the dictionary entry if it has no data (generally because the chunk doesn't exist for that input)
        if array is None:
            continue

        # print(key, print(array.dtype))

        # Suggested by https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/672bad5a-cda0-800a-8889-09657ed7e888
        # to optimize memory allocation for numba. Not sure it helps but it doesn't seemt to hurt, so leaving it in.
        contig_array = np.ascontiguousarray(array)

        # If there is data, it puts the data in the corresponding dictionary for that datatype
        if array.dtype == np.uint8:
            uint8_dict_layers[key] = contig_array
        elif array.dtype == np.uint16:
            uint16_dict_layers[key] = contig_array
        elif array.dtype == np.int16:
            int16_dict_layers[key] = contig_array
        elif array.dtype == np.int32:
            int32_dict_layers[key] = contig_array
        elif array.dtype == np.float32:
            float32_dict_layers[key] = contig_array
        else:
            raise TypeError(f"{key} dtype not in list")


    # print(f"uint8 datasets: {uint8_dict_layers.keys()}")
    # print(f"uint16 datasets: {uint16_dict_layers.keys()}")
    # print(f"int16 datasets: {int16_dict_layers.keys()}")
    # print(f"int32 datasets: {int32_dict_layers.keys()}")
    # print(f"float32 datasets: {float32_dict_layers.keys()}")

    # Creates numba-compliant typed dict for each type of array
    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')  # Assuming 2D arrays of uint8
    )

    typed_dict_uint16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint16, 2, 'C')  # Assuming 2D arrays of int16
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

    for key, array in uint16_dict_layers.items():
        typed_dict_uint16[key] = array

    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array

    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array

    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_uint16, typed_dict_int16, typed_dict_int32, typed_dict_float32


# Classifies GLCLU composite as bare ground, short vegetation (<5 m), or tall vegetation (>= 5 m)
@jit(nopython=True)
def classify_GLAD_composite(LC):
    bare_ground = (((LC >= cn.bare_ground_dry_min_code) and (LC <= cn.bare_ground_dry_max_code)) or
                   ((LC >= cn.bare_ground_wet_min_code) and (LC <= cn.bare_ground_wet_max_code)))
    short_veg =   (((LC >= cn.short_veg_dry_min_code) and (LC <= cn.short_veg_dry_max_code)) or
                   ((LC >= cn.short_veg_wet_min_code) and (LC <= cn.short_veg_wet_max_code)))
    tall_veg =    (((LC >= cn.tall_veg_dry_min_code) and (LC <= cn.tall_veg_dry_max_code)) or
                   ((LC >= cn.tall_veg_wet_min_code) and (LC <= cn.tall_veg_wet_max_code)))

    return bare_ground, short_veg, tall_veg


# Checks if pixel does not have tall vegetation. If so, updates the value to the most recent year without tall vegetation.
# Tall vegetation/non-tall vegetation is based on the composite land cover maps, not the canopy height maps.
# 0=Always tall vegetation so far. Other values represent the last year of non-tall vegetation.
# Theoretically, checking whether pixels are ever not forest could be done at the chunk level rather
# than at the pixel level (as done here). However, numba doesn't allow the conditional operation
# that would have to be applied to numpy arrays, so I'm doing it at the pixel level instead.
# I have no idea if checking if pixels have ever not been forest is faster or slower at the pixel level
# than at the numpy array level, but the array-level operation isn't even an option.
@jit(nopython=True)
def check_most_recent_year_not_tall_veg(LC_curr, most_recent_year_not_forest, interval_end_year):

    # Checks the current end of interval land cover
    # Criteria for excluding tall vegetation land cover
    not_tall_veg_condition = (
            (LC_curr < cn.tall_veg_dry_min_code) |
            ((LC_curr > cn.tall_veg_dry_max_code) & (LC_curr < cn.tall_veg_wet_min_code))
            | (LC_curr > cn.tall_veg_wet_max_code)
    )

    # Sets cell to interval end year wherever land cover is not tall vegetation
    if not_tall_veg_condition == 1:
        most_recent_year_not_forest = interval_end_year

    return most_recent_year_not_forest


# Calculates the maximum canopy height since the last time a pixel was classified as not tall vegetation land cover.
# This is used to determine whether current height has decreased significantly from this maximum height.
@jit(nopython=True)
def calc_max_height_since_last_time_not_tall_veg(most_recent_year_not_tall_veg, vegetation_height_so_far_cell, years_so_far_cell):

    # Determines the maximum height so far if the pixel has been tall vegetation land cover since the beginning of the model
    if most_recent_year_not_tall_veg == 0:

        # The maximum vegetation height through all intervals so far
        max_height_since_last_time_not_tall_veg = max(vegetation_height_so_far_cell)

    # Determines the maximum height so far if the pixel hasn't had tall vegetation land cover at least one year since the beginning of the model
    else:

        heights_since_last_time_not_tall_veg = []

        # Loops over the years and corresponding heights to only get heights that are after the most recent
        # non-tall vegetation year.
        # This could be done more elegantly with conditional numpy arrays but that approach
        # isn't supported in the numba function, unfortunately.
        # https://chatgpt.com/share/e/6718fb20-48d8-800a-9eb2-d751bd6b1a8f
        for i in range(len(years_so_far_cell)):
            if years_so_far_cell[i] > most_recent_year_not_tall_veg:
                heights_since_last_time_not_tall_veg.append(vegetation_height_so_far_cell[i])

        # In case the pixel is currently non-tall vegetation land cover, so there are no intervals since then
        # and therefore no heights
        if len(heights_since_last_time_not_tall_veg) == 0:

            # Uses the current vegetation height (which would exist when the land cover is not tall vegetation
            # but there is still tall vegetation in the individual tree height layer)
            max_height_since_last_time_not_tall_veg = vegetation_height_so_far_cell[-1]
            # max_height_since_last_time_not_tall_veg = 0

        # When the pixel was previously non-tall vegetation but is now tall vegetation,
        # so there are intervals since then.
        else:

            # The maximum height in the years since the last non-tall vegetation land cover interval
            max_height_since_last_time_not_tall_veg = max(heights_since_last_time_not_tall_veg)

    return max_height_since_last_time_not_tall_veg


# Calculates the ratio of deadwood C:AGC and litter C:AGC based on climate domain, elevation, and precip
# for natural, terrestrial forests.
# Deadwood and litter carbon as fractions of AGC are from
# https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ar-am-tool-12-v3.0.pdf
# "Clean Development Mechanism A/R Methodological Tool:
# Estimation of carbon stocks and change in carbon stocks in dead wood and litter in A/R CDM project activities version 03.0"
# Tables on pages 18 (deadwood) and 19 (litter).
# They depend on the climate domain, elevation, and precipitation.
@jit(nopython=True)
def calc_deadwood_litter_ratios(elevation, climate_domain, precipitation):

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

    return float(deadwood_c_ratio), float(litter_c_ratio)


# Returns AGC and BGC one-time removal factors for the gain of short-height vegetation (hard-coded values are Mg AGB/ha, but returned as Mg C/ha)
# RF values are from IPCC 2006, V4, Ch. 6, Table 6.4- DEFAULT BIOMASS STOCKS PRESENT ON GRASSLAND, AFTER CONVERSION FROM OTHER LAND USE (no 2019 update).
# short_veg_AGB_RF is from the "peak above-ground biomass" columns.
# short_veg_BGB_RF is the difference between short_veg_AGB_RF and the "total (above-ground and below-ground) non-woody biomass" column.
# Climate zone is from IPCC 2019 Corrigenda map. See constants_and_names.py for more information.
@jit(nopython=True)
def calc_short_veg_removals(climate_zone, LC_curr):

    if climate_zone >= 9:  # Boreal- dry and wet (and polar)
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (8.5 - short_veg_AGB_RF)
    elif climate_zone == 8:  # Cold temperate- dry
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (6.5 - short_veg_AGB_RF)
    elif climate_zone == 7:  # Cold temperate- wet
        short_veg_AGB_RF = 2.4
        short_veg_BGB_RF = (13.6 - short_veg_AGB_RF)
    elif climate_zone == 6:  # Warm temperate- dry
        short_veg_AGB_RF = 1.6
        short_veg_BGB_RF = (6.1 - short_veg_AGB_RF)
    elif climate_zone == 5:  # Warm temperate- wet
        short_veg_AGB_RF = 2.7
        short_veg_BGB_RF = (13.5 - short_veg_AGB_RF)
    elif climate_zone == 4:  # Tropical- dry
        short_veg_AGB_RF = 2.3
        short_veg_BGB_RF = (8.7 - short_veg_AGB_RF)
    elif climate_zone == 2 or climate_zone == 3:  # Tropical- wet/moist
        short_veg_AGB_RF = 6.2
        short_veg_BGB_RF = (16.1 - short_veg_AGB_RF)
    elif climate_zone == 1:  # Tropical- montane (average of tropical dry and tropical wet/moist values)
        short_veg_AGB_RF = (2.3 + 6.2)/2
        short_veg_BGB_RF = (((8.7 + 16.1)/2) - short_veg_AGB_RF)
    else: # Outside climate zone bounds-- apply boreal values
        short_veg_AGB_RF = 1.7
        short_veg_BGB_RF = (8.5-short_veg_AGB_RF)

    short_veg_AGC_RF_raw = short_veg_AGB_RF * cn.biomass_to_carbon_non_mangrove
    short_veg_BGC_RF_raw = short_veg_BGB_RF * cn.biomass_to_carbon_non_mangrove

    # Need the LC_curr GLAD code to be between 0 and 100 to calculate the short vegetation fraction (0-100).
    # This gets the "wet short veg" GLAD code between 0 and 100 so it can be converted to vegetation fraction.
    # Empirically, the relationship between the GLAD LC composite and vegetation fraction is (class*4 + 3)
    # from https://docs.google.com/spreadsheets/d/1UwZNcfXJpQHLwrSLHYyH3Zcahh0GIjcsu7D0F0SncvU/edit?pli=1&gid=0#gid=0
    # for the short veg classes (5-24 and 105-124).
    if LC_curr >= 100:
        veg_fraction = np.float32((((LC_curr - 100) * 4 + 3) / 100))
    else:
        veg_fraction = np.float32(((LC_curr * 4 + 3) / 100))

    # Adjusts the short veg removal factor/C density based on the vegetation fraction in the year of short veg gain
    short_veg_AGC_RF_adj = np.float32((short_veg_AGC_RF_raw * veg_fraction))
    short_veg_BGC_RF_adj = np.float32((short_veg_BGC_RF_raw * veg_fraction))

    return short_veg_AGC_RF_adj, short_veg_BGC_RF_adj


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
def unpack_emission_factors(ef):

    agc_ef = np.float32(ef[0])
    bgc_ef = np.float32(ef[1])
    deadwood_c_ef = np.float32(ef[2])
    litter_c_ef = np.float32(ef[3])

    return agc_ef, bgc_ef, deadwood_c_ef, litter_c_ef


# Returns the removal factor for primary forest/IFL based on the continent-ecozone combination (Mg AGC/ha/yr)
# From https://chatgpt.com/share/e/67340a6f-b8cc-800a-84e3-f98e600001e5
@jit(nopython=True)
def calc_primary_forest_RF(continent_ecozone_cell, primary_forest_RF_array):

    primary_forest_RF_indices = np.where(primary_forest_RF_array[:, 0] == continent_ecozone_cell)

    # Checks if there are matching indices and extracts corresponding primary forest RF
    if primary_forest_RF_indices[0].size > 0:  # If matching continent-ecozone combination...
        primary_forest_RF = primary_forest_RF_array[primary_forest_RF_indices[0][0], 1]  # Uses matching RF
    else:  # If no matching continent-ecozone combination...
        primary_forest_RF = np.mean(primary_forest_RF_array[:, 1]) # Uses average of all primary forest RFs

    return primary_forest_RF


# Returns the aboveground biomass accumulation rate (Mg biomass/ha/yr) for mangroves based on the continent-ecozone combination
# Converts it to a removal factor (Mg AGC/ha/yr) by multiplying it by the carbon fraction for mangrove forests.
# Also returns the ratios of BGC:AGC, deadwood C:AGC and litter C:AGC for mangrove forests
@jit(nopython=True)
def calc_mangrove_RF_and_ratios(continent_ecozone_cell, mangrove_C_ratio_array):

    mangrove_RF_indices = np.where(mangrove_C_ratio_array[:, 0] == continent_ecozone_cell)

    # Checks if there are matching indices and extracts corresponding mangrove RF
    if mangrove_RF_indices[0].size > 0:  # If matching continent-ecozone combination...
        mangrove_BA = mangrove_C_ratio_array[mangrove_RF_indices[0][0], 1]  # Uses matching mangrove biomass accumulation rate
        mangrove_RF = mangrove_BA * cn.biomass_to_carbon_mangrove           # Converts units of biomass to units of carbon (i.e. removal factor)

        mangrove_r_s_ratio = mangrove_C_ratio_array[mangrove_RF_indices[0][0], 2]           # Uses matching mangrove BGC:AGC ratio
        mangrove_deadwood_c_ratio = mangrove_C_ratio_array[mangrove_RF_indices[0][0], 3]    # Uses matching mangrove deadwood:AGC ratio
        mangrove_litter_c_ratio = mangrove_C_ratio_array[mangrove_RF_indices[0][0], 4]      # Uses matching mangrove litter:AGC ratio

    else:  # If no matching continent-ecozone combination...
        mangrove_BA = np.mean(mangrove_C_ratio_array[:, 1])         # Uses average of all mangrove biomass accumulation rates
        mangrove_RF = mangrove_BA * cn.biomass_to_carbon_mangrove

        mangrove_r_s_ratio = np.mean(mangrove_C_ratio_array[:, 2])          # Uses average of all mangrove BGC:AGC ratios
        mangrove_deadwood_c_ratio = np.mean(mangrove_C_ratio_array[:, 3])   # Uses average of all mangrove deadwood:AGC ratio
        mangrove_litter_c_ratio = np.mean(mangrove_C_ratio_array[:, 4])     # Uses average of all mangrove litter:AGC ratio

    return mangrove_RF, mangrove_r_s_ratio, mangrove_deadwood_c_ratio, mangrove_litter_c_ratio


# Returns the first year of mangrove gain (0->1) if there is no mangrove extent at the beginning of the timeseries.
# Also returns first year of mangrove loss (1->0) with or without any subsequent gain.
# Also returns last year of mangrove loss (1->0) when there is no subsequent gain.
@jit(nopython=True)
def mangrove_first_gain_last_loss_years(data_array, year_array):

    length = data_array.shape[0]

    # first mangrove gain year
    if data_array[0] == 1:      #If mangrove extent in starting year, cannot be mangrove gain later on in series so first_gain year = 0
        first_gain = 0
    else:
        first_gain = 0
        for year in range(1, length):
            prev = data_array[year - 1]
            curr = data_array[year]

            if prev == 0 and curr == 1:
                first_gain = year_array[year]
                break

    # first mangrove loss year
    first_loss = 0              #If there is never mangrove loss, first_loss year = 0
    for year in range(1, length):
        prev = data_array[year - 1]
        curr = data_array[year]
        if prev == 1 and curr == 0:
            first_loss = year_array[year]
            break

    #last_mangrove_loss_year
    if data_array[length-1] == 1:
        last_loss = 0       #If mangrove in final year, cannot have permanent mangrove loss so last_loss year = 0
    else:
        last_loss = 0
        suffix_all_zero = 1  # flag: are all values after year zero?

        # Loop backwards over mangrove extent years starting from last year (inclusive), looking for
        # the first time it switches from 0 (curr) to 1 (prev) to get final mangrove loss year
        for year in range(length - 1, 0, -1):
            # If there is mangrove extent in current year, the suffix_all_zero flag is set to False
            if data_array[year] != 0:
                suffix_all_zero = 0
            # If suffix_all_zero flag is True, and mangrove extent in prev year = 1 and in curr year = 0, gets current year.
            if data_array[year-1] == 1 and data_array[year] == 0 and suffix_all_zero == 1:
                last_loss = year_array[year]
                break

    return first_gain, first_loss, last_loss


# Function to get whether the interval is mangrove gain, loss, maintenance, before gain or after loss.
@jit(nopython=True)
def mangrove_states(interval_start_year, interval_end_year, first_mang_gain_year, last_mang_loss_year, mang_interval_timeseries):
    mang_loss = mang_gain = mang_remaining_mang = non_mang_remaining_non_mang = before_mang = after_mang = False

    # Get previous and current mangrove states
    mang_prev = mang_interval_timeseries[0]
    mang_last = mang_interval_timeseries[-1]
    mang_inter = mang_interval_timeseries[1:]

    # Checks whether there are mangroves present at all in this interval (not including the previous interval's end year)
    mang_present = np.any(mang_inter == 1)
    mang_absent = np.all(mang_inter == 0)

    # Mangrove state hierarchy before_mang > after_mang > mang_loss > mang_gain > mang_remaining_mang > non_mang_remaining_non_mang

    # Checks if the end of the current interval is before the first time there is mangrove in this pixel
    # This could be part of non_mang_remaining_non_mang, but keeping separate for zonal stats purposes and to allow for conversion from other classes to mangroves.
    # Note: first_mang_gain_year only exists (is not 0) if there was no mangrove extent in 1996 and there is gain later on in the timeseries.
    before_mang = (first_mang_gain_year != 0) and (first_mang_gain_year > interval_end_year)

    if not before_mang:
        # Checks if the start of the current interval is after the permanent mangrove loss year
        # This could be part of non_mang_remaining_non_mang, but keeping separate for zonal stats and to allow for conversion from mangroves to other classes.
        # Note: last_mang_loss_year only exists (is not 0) if there is no mangrove recovery after loss by the end of the timeseries (2020).
        after_mang = (last_mang_loss_year != 0) and (last_mang_loss_year <= interval_start_year)

    if (not before_mang) and (not after_mang):
        # Mangrove loss if it was mangrove at the end of the previous interval, but not mangrove at the end of the current interval or
        # if it was not mangrove at the end of the previous interval but there was gain and loss before the end of the current interval
        mang_loss = ((mang_prev == 1) and (mang_last == 0)) or ((mang_prev == 0) and mang_present and (mang_last == 0))

    if (not before_mang) and (not after_mang) and (not mang_loss):
        # Mangrove gain if it was not mangrove at the end of the previous interval, but there was mangrove gain remaining mangrove by the end of the interval
        mang_gain = (mang_prev == 0) and (mang_present) and (mang_last == 1)

    if (not before_mang) and (not after_mang) and (not mang_loss) and (not mang_gain):
        # Mangrove remaining mangrove if it was mangrove at the end of the previous interval and mangrove at the end of the current interval.
        mang_remaining_mang = (mang_prev == 1) and (mang_last == 1)

    if (not before_mang) and (not after_mang) and (not mang_loss) and (not mang_gain) and (not mang_remaining_mang):
        # Non-mangrove remaining non-mangrove if it was not mangrove at the end of the previous interval and there is no mangrove presence during time period
        non_mang_remaining_non_mang = (mang_prev == 0) and (mang_absent)

   # Checks that only 1 mangrove state is true
    true_count = int(before_mang) + int(after_mang) + int(mang_loss) + int(mang_gain) + int(mang_remaining_mang) + int(non_mang_remaining_non_mang)
    if true_count != 1:
        raise ValueError(
            f"Invalid classification. Only 1 of the following should be True: "
            f"loss={mang_loss}, gain={mang_gain}, remaining={mang_remaining_mang}, not_remaining = {non_mang_remaining_non_mang}, before_gain={before_mang}, after_loss={after_mang}"
        )

    return mang_loss, mang_gain, mang_remaining_mang, non_mang_remaining_non_mang, before_mang, after_mang


# Returns whether there was mangrove extent loss (1->0), and if so the year of loss
# Also returns the number of years in the interval before and after mangrove extent loss (for pre- and post-loss gain year counts)
@jit(nopython=True)
def mangrove_gain_year_count_summary(data_array, interval_start_year, interval_end_year):

    array_length = data_array.shape[0]
    years_length = (interval_end_year - interval_start_year) + 1

    # Check that data array and year array have the same number of items
    if array_length != years_length:
        raise ValueError("Length mismatch between data_array and years range")

    # Find index of last 1→0 transition, or returns -1 if not found
    loss_index = -1
    for i in range(1, array_length):
        if data_array[i - 1] == 1 and data_array[i] == 0:
            loss_index = i

    if loss_index != -1:
        mang_loss = True
        mang_loss_year = interval_start_year + loss_index

        # sums years with mangrove extent starting after previous interval year and before last loss year
        pre_count = 0
        for k in range(1, loss_index):
            if data_array[k] == 1:
                pre_count += 1

        # sums mangrove extent years after last loss year
        post_count = 0
        for k in range(loss_index + 1, array_length):
            if data_array[k] == 1:
                post_count += 1
    else:
        mang_loss = False
        mang_loss_year = 0

        # If no loss, sums all years with mangrove extent starting after previous interval end year
        pre_count = 0
        for k in range(1, array_length):
            if data_array[k] == 1:
                pre_count += 1
        #If no loss, post_count is 0
        post_count = 0

    # Validate totals against the number of interval years (exclude the first/prior year)
    interval_len = array_length - 1
    total = pre_count + post_count
    if not (0 <= total <= interval_len):
        raise ValueError("Invalid counts: pre+post must be >=0 and <= interval length")

    return mang_loss, mang_loss_year, pre_count, post_count


# Takes in the interval_length and interval_end_year and returns a list of index values to extract the
# corresponding data (including the last year from the previous interval) from the GMWv3 timeseries
@jit(nopython=True)
def map_years_to_gmwv3_data(interval_end_year, interval_length):

    if interval_length == 5:
        if interval_end_year == 2005:
             return np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)    # 2000 (1996), 2001 (1996), 2002 (1996), 2003 (1996), 2004 (2007), 2005 (2007)
        elif interval_end_year == 2010:
             return np.array([1, 1, 1, 2, 3, 4], dtype=np.int64)    # 2005 (2007), 2006 (2007), 2007, 2008, 2009, 2010
        elif interval_end_year == 2015:
             return np.array([4, 4, 4, 4, 4, 5], dtype=np.int64)    # 2010, 2011 (2010), 2012 (2010), 2013 (2010), 2014 (2010), 2015
        elif (interval_end_year == 2020):
             return np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)   # 2015, 2016, 2017, 2018, 2019, 2020

    elif interval_length == 1:
        if interval_end_year == 2015:
             return np.array([4, 5], dtype=np.int64)                # 2014 (2010), 2015
        elif interval_end_year == 2016:
             return np.array([5, 6], dtype=np.int64)                # 2015, 2016
        elif interval_end_year == 2017:
             return np.array([6, 7], dtype=np.int64)                # 2016, 2017
        elif interval_end_year == 2018:
             return np.array([7, 8], dtype=np.int64)                # 2017, 2018
        elif interval_end_year == 2019:
             return np.array([8, 9], dtype=np.int64)                # 2018, 2019
        elif interval_end_year == 2020:
             return np.array([9, 10], dtype=np.int64)               # 2019, 2020
        elif interval_end_year > 2020:
             return np.array([10, 10], dtype=np.int64)              # 2020, 2020

    else:
        raise ValueError("No mangrove index mapping found for for given interval_end_year and interval_length")


# Returns the emission factors for partially disturbed forest by driver based on the continent-ecozone combination (unit: fraction AGC lost)
# From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67feb6f2-c124-800a-9279-61f0e3a67faf
@jit(nopython=True)
def calc_partial_disturbance_EFs(drivers_cell, continent_ecozone_cell, partial_disturbance_EF_array):

    # # For testing-- force driver or continent_ecozone values
    # drivers_cell = 3
    # continent_ecozone_cell = 9999

    # Selects correct driver column. Sets fallback to driver 4 (logging) if driver value isn't legitimate.
    if 1 <= drivers_cell <= 7:
        col_index = drivers_cell  # EF columns are at indexes 1–7 (drivers 1-7)
    else:
        col_index = 4  # Defaults to 5th column (index 4: logging driver)

    partial_disturbance_EF_indices = np.where(partial_disturbance_EF_array[:, 0] == continent_ecozone_cell)

    if partial_disturbance_EF_indices[0].size > 0:
        row_index = partial_disturbance_EF_indices[0][0]
        partial_disturbance_EF = partial_disturbance_EF_array[row_index, col_index]
    else:
        # Manual mean of the specified column (col_index) because numba has all kinds of restrictions!
        # Handles errant NaN in EF tables by calculating EF from the rest of the values
        # Per Claude session 'Flux statistics comparison: chunk vs. flox'
        total = 0.0
        n_rows = partial_disturbance_EF_array.shape[0]
        count = 0
        for i in range(n_rows):
            val = partial_disturbance_EF_array[i, col_index]
            if not np.isnan(val):
                total += val
                count += 1
        partial_disturbance_EF = total / count if count > 0 else 0.0

    return partial_disturbance_EF

# Calculates Cf for fire emissions from forests (as opposed to savanna/grassland or crop residue burning).
# From IPCC 2019 Table 2.6 (unitless).
# There is also a separate Cf for pixels that are burned but show no height reduction ("undisturbed"), cn.Cf_forest_undisturbed
@jit(nopython=True)
def calc_Cf_forest(climate_domain_cell, drivers_cell, ifl_primary_cell, model_type):

    # Groups of drivers with different Cfs
    driver_group_1 = [cn.permanent_agriculture, cn.shifting_cultivation, cn.hard_commodities, cn.wildfire, cn.settlements_and_infrastruct]
    driver_group_2 = [cn.forest_management]
    driver_group_3 = [cn.other_natural_disturbances]

    if model_type == cn.low_EF:  # value-stdev for same row unless otherwise noted
        if climate_domain_cell == 1:  # Tropical/subtropical
            if ifl_primary_cell:  # Tropical/subtropical, primary forest
                Cf_forest = (0.36-0.13)  # Row "All primary tropical forest"
            else:  # Tropical/subtropical, not primary forest
                Cf_forest = (0.55-0.06)  # Row "All secondary tropical forest"
        elif climate_domain_cell == 3:  # Temperate
            if drivers_cell in driver_group_1:  # Temperate, driver group 1
                Cf_forest = (0.51-0.12)  # Row "Felled and burned (land-clearing fire)" temperate forest (- st dev from "Post logging slash burn")
            elif drivers_cell in driver_group_2:  # Temperate, driver group 2
                Cf_forest = (0.62-0.12)  # Row "Post logging slash burn" temperate forest
            elif drivers_cell in driver_group_3:  # Temperate, driver group 3
                Cf_forest = (0.45-0.16)  # Row "all other temperate forest"
            else:  # Temperate, no driver assigned
                Cf_forest = (0.45-0.16)
        elif climate_domain_cell == 2:  # Boreal
            if drivers_cell in driver_group_1:  # Boreal, driver group 1
                Cf_forest = (0.59-0.17)  # Row "Land clearing fire" boreal forest (- st dev from "All boreal forest")
            elif drivers_cell in driver_group_2:  # Boreal, driver group 2
                Cf_forest = (0.33-0.13)  # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Boreal, driver group 3
                Cf_forest = (0.34-0.17)  # Row "All boreal forest"
            else:  # Boreal, no driver assigned
                Cf_forest = (0.34-0.17)  # Row "All boreal forest"
        else:  # Outside ecozone bounds
            if drivers_cell in driver_group_1:  # Outside ecozone bounds, driver group 1
                Cf_forest = (0.59-0.17)  # Row "Land clearing fire" boreal forest (- st dev from "All boreal forest")
            elif drivers_cell in driver_group_2:  # Outside ecozone bounds, driver group 2
                Cf_forest = (0.33-0.14)  # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Outside ecozone bounds, driver group 2
                Cf_forest = (0.34-0.17)  # Row "All boreal forest"
            else:  # Outside ecozone bounds, no driver assigned
                Cf_forest = (0.34-0.17)  # Row "All boreal forest"

    elif model_type == cn.high_EF:  # value+stdev for same row unless otherwise noted
        if climate_domain_cell == 1:  # Tropical/subtropical
            if ifl_primary_cell:  # Tropical/subtropical, primary forest
                Cf_forest = (0.36+0.13)  # Row "All primary tropical forest"
            else:  # Tropical/subtropical, not primary forest
                Cf_forest = (0.55+0.06)  # Row "All secondary tropical forest"
        elif climate_domain_cell == 3:   # Temperate
            if drivers_cell in driver_group_1:  # Temperate, driver group 1
                Cf_forest = (0.51+0.12)     # Row "Felled and burned (land-clearing fire)" temperate forest (+ st dev from "Post logging slash burn")
            elif drivers_cell in driver_group_2:  # Temperate, driver group 2
                Cf_forest = (0.62+0.12)     # Row "Post logging slash burn" temperate forest
            elif drivers_cell in driver_group_3:  # Temperate, driver group 3
                Cf_forest = (0.45+0.16)     # Row "all other temperate forest"
            else:  # Temperate, no driver assigned
                Cf_forest = (0.45+0.16)
        elif climate_domain_cell == 2:  # Boreal
            if drivers_cell in driver_group_1:  # Boreal, driver group 1
                Cf_forest = (0.59+0.17)     # Row "Land clearing fire" boreal forest (+ st dev from "All boreal forest")
            elif drivers_cell in driver_group_2:  # Boreal, driver group 2
                Cf_forest = (0.33+0.13)     # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Boreal, driver group 3
                Cf_forest = (0.34+0.17)     # Row "All boreal forest"
            else:  # Boreal, no driver assigned
                Cf_forest = (0.34+0.17)     # Row "All boreal forest"
        else:  # Outside ecozone bounds
            if drivers_cell in driver_group_1:  # Outside ecozone bounds, driver group 1
                Cf_forest = (0.59+0.17)     # Row "Land clearing fire" boreal forest (+ st dev from "All boreal forest")
            elif drivers_cell in driver_group_2:  # Outside ecozone bounds, driver group 2
                Cf_forest = (0.33+0.14)     # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Outside ecozone bounds, driver group 2
                Cf_forest = (0.34+0.17)     # Row "All boreal forest"
            else:  # Outside ecozone bounds, no driver assigned
                Cf_forest = (0.34+0.17)     # Row "All boreal forest"

    else:    # standard model and any that doesn't change the emission factors
        if climate_domain_cell == 1:  # Tropical/subtropical
            if ifl_primary_cell:  # Tropical/subtropical, primary forest
                Cf_forest = 0.36  # Row "All primary tropical forest"
            else:  # Tropical/subtropical, not primary forest
                Cf_forest = 0.55  # Row "All secondary tropical forest"
        elif climate_domain_cell == 3:   # Temperate
            if drivers_cell in driver_group_1:  # Temperate, driver group 1
                Cf_forest = 0.51     # Row "Felled and burned (land-clearing fire)" temperate forest
            elif drivers_cell in driver_group_2:  # Temperate, driver group 2
                Cf_forest = 0.62     # Row "Post logging slash burn" temperate forest
            elif drivers_cell in driver_group_3:  # Temperate, driver group 3
                Cf_forest = 0.45     # Row "all other temperate forest"
            else:  # Temperate, no driver assigned
                Cf_forest = 0.45
        elif climate_domain_cell == 2:  # Boreal
            if drivers_cell in driver_group_1:  # Boreal, driver group 1
                Cf_forest = 0.59     # Row "Land clearing fire" boreal forest
            elif drivers_cell in driver_group_2:  # Boreal, driver group 2
                Cf_forest = 0.33     # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Boreal, driver group 3
                Cf_forest = 0.34     # Row "All boreal forest"
            else:  # Boreal, no driver assigned
                Cf_forest = 0.34     # Row "All boreal forest"
        else:  # Outside ecozone bounds
            if drivers_cell in driver_group_1:  # Outside ecozone bounds, driver group 1
                Cf_forest = 0.59     # Row "Land clearing fire" boreal forest
            elif drivers_cell in driver_group_2:  # Outside ecozone bounds, driver group 2
                Cf_forest = 0.33     # Row "Post logging slash burn" boreal forest
            elif drivers_cell in driver_group_3:  # Outside ecozone bounds, driver group 2
                Cf_forest = 0.34     # Row "All boreal forest"
            else:  # Outside ecozone bounds, no driver assigned
                Cf_forest = 0.34     # Row "All boreal forest"

    return Cf_forest


# Calculates Gef for fire emissions from forests (as opposed to savanna/grassland or biofuel burning).
# From IPCC 2019 Table 2.5 (g respective gas/kg dry matter)
@jit(nopython=True)
def calc_Gef_forest(climate_domain_cell, model_type):

    if model_type == cn.low_EF:  # value-stdev for same row unless otherwise noted
        if climate_domain_cell == 1:  # Tropical/subtropical
            Gef_CO2_forest = (1580.0-90)   # Row "tropical forest"
            Gef_CH4_forest = (6.8-2.0)      # Row "tropical forest"
            Gef_N2O_forest = (0.2-0.04)      # Row "tropical forest". st dev is my best professional judgement
        elif climate_domain_cell == 2 or climate_domain_cell == 3:   # Boreal/temperate
            Gef_CO2_forest = (1569.0-131)   # Row "extra-tropical forest"
            Gef_CH4_forest = (4.7-1.9)      # Row "extra-tropical forest"
            Gef_N2O_forest = (0.26-0.07)     # Row "extra-tropical forest"
        else:  # Outside ecozone bounds
            Gef_CO2_forest = (1569.0-131)   # Row "extra-tropical forest"
            Gef_CH4_forest = (4.7-1.9)      # Row "extra-tropical forest"
            Gef_N2O_forest = (0.26-0.07)     # Row "extra-tropical forest"

    elif model_type == cn.high_EF:  # value+stdev for same row unless otherwise noted
        if climate_domain_cell == 1:  # Tropical/subtropical
            Gef_CO2_forest = (1580.0+90)   # Row "tropical forest"
            Gef_CH4_forest = (6.8+2.0)      # Row "tropical forest"
            Gef_N2O_forest = (0.2+0.04)      # Row "tropical forest". st dev is my best professional judgement
        elif climate_domain_cell == 2 or climate_domain_cell == 3:   # Boreal/temperate
            Gef_CO2_forest = (1569.0+131)   # Row "extra-tropical forest"
            Gef_CH4_forest = (4.7+1.9)      # Row "extra-tropical forest"
            Gef_N2O_forest = (0.26+0.07)     # Row "extra-tropical forest"
        else:  # Outside ecozone bounds
            Gef_CO2_forest = (1569.0+131)   # Row "extra-tropical forest"
            Gef_CH4_forest = (4.7+1.9)      # Row "extra-tropical forest"
            Gef_N2O_forest = (0.26+0.07)     # Row "extra-tropical forest"
    else:  # standard model and any that doesn't change the emission factors
        if climate_domain_cell == 1:  # Tropical/subtropical
            Gef_CO2_forest = 1580.0   # Row "tropical forest"
            Gef_CH4_forest = 6.8      # Row "tropical forest"
            Gef_N2O_forest = 0.2      # Row "tropical forest"
        elif climate_domain_cell == 2 or climate_domain_cell == 3:   # Boreal/temperate
            Gef_CO2_forest = 1569.0   # Row "extra-tropical forest"
            Gef_CH4_forest = 4.7      # Row "extra-tropical forest"
            Gef_N2O_forest = 0.26     # Row "extra-tropical forest"
        else:  # Outside ecozone bounds
            Gef_CO2_forest = 1569.0   # Row "extra-tropical forest"
            Gef_CH4_forest = 4.7      # Row "extra-tropical forest"
            Gef_N2O_forest = 0.26     # Row "extra-tropical forest"

    return Gef_CO2_forest, Gef_CH4_forest, Gef_N2O_forest


# Calculates non-CO2 emissions (CH4 and N2O) separately.
# Cf_forest is the combustion factor
# Gef_ch4 and Gef_n2o are the emission factors for their respective gases.
# biomass_to_carbon can be hard-coded as non-mangrove because we assume that mangroves don't have fires.
# From IPCC 2019 Eqn. 2.27
@jit(nopython=True)
def non_CO2_fire_equations(carbon_in, Cf, Gef_ch4, Gef_n2o):

    # print(f"Carbon in: {carbon_in}; Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")

    ch4_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_ch4 * cn.g_to_kg * cn.gwp_ch4
    n2o_flux_out = (carbon_in/cn.biomass_to_carbon_non_mangrove) * Cf * Gef_n2o * cn.g_to_kg * cn.gwp_n2o

    # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
    # os.quit()

    return ch4_flux_out, n2o_flux_out


# Gross fluxes and ending carbon stocks for non-tree converted to tree.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_NT_T(agc_rf, bgc_rf, c_dens_in, deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Step 1: Assigns the number of years of carbon gain (years)
    gain_year_count = cn.veg_modeL_increment

    # Step 2: Calculates gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    deadwood_c_gross_removals_out = agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out = agc_gross_removals_out * litter_c_ratio

    # Step 3: Calculates ending carbon densities by carbon pool (Mg C/ha)
    agc_dens_out = agc_dens_in - agc_gross_removals_out
    bgc_dens_out = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_removals_out

    # Step 4: Prepares outputs
    # Consolidates all gross fluxes from all carbon pools into arrays to reduce the number of arguments returned to the decision tree
    # Must specify float32 because numba is quite particular about datatypes
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')  # Mg C/ha/interval
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')  # Mg C/ha

    # Step 5: Increments the forest age (years).
    # Forest age starts at 0 years by definition for NT->T.
    forest_age = 0 + gain_year_count

    return c_gross_removals_out, c_dens_out, gain_year_count, forest_age


# Gross fluxes and ending carbon stocks for mangrove gain or mangrove remaining mangrove without loss in interval.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Applies to mangrove gain and maintenance. Differences are handled by these input arguments: forest_age_start and c_dens_in.
@jit(nopython=True)
def calc_mang(forest_age_start, first_mang_gain_year, first_mang_loss_year, interval_end_year,
              gain_year_count, agc_rf, bgc_rf, c_dens_in, deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Step 1: Assigns deadwood C and litter C ratios for removal factors
    # Deadwood and litter C removals only occur in pixels that are not considered to be in "equilibrium".
    # Thus, we need to check that there is no regrowth occuring.
    # Checking that it is not mangrove gain (i.e. no mangrove extent in 1996 but mangrove gain in subsequent year)
    # Checking that there is mangrove loss through this interval.
    # If conditions are met, the deadwood and litter ratios are set to 0 (no removals while in "equilibrium").
    if (first_mang_gain_year == 0) and ((first_mang_loss_year == 0) or (first_mang_loss_year > interval_end_year)):
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0

    # Step 2: Calculates gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((agc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    bgc_gross_removals_out = float((bgc_rf * gain_year_count) * -1)  #float() necessary for Numba typing
    deadwood_c_gross_removals_out = agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out = agc_gross_removals_out * litter_c_ratio

    # Step 3: Calculates gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # There are no gross emissions in mangrove pixels with no loss, so C pool emissions set to 0.
    agc_gross_emis_out = 0
    bgc_gross_emis_out = 0
    deadwood_c_gross_emis_out = 0
    litter_c_gross_emis_out = 0

    # Step 4: Calculates ending carbon densities by carbon pool (Mg C/ha)
    agc_dens_out = agc_dens_in - agc_gross_removals_out
    bgc_dens_out = bgc_dens_in - bgc_gross_removals_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_removals_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_removals_out

    # Step 5: Prepares outputs
    # Consolidates all gross fluxes from all carbon pools into arrays to reduce the number of arguments returned to the decision tree
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')  # Mg C/ha/interval
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')  # Mg C/ha/interval
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')  # Mg C/ha

    # Step 6: Increments the forest age (years).
    # Forest age starts at 0 years for mangrove gain
    forest_age = forest_age_start + gain_year_count

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out, gain_year_count, forest_age


# Gross fluxes and ending carbon stocks for intervals with temporary or permanent mangrove loss.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
# Model has no gain in the year of mangrove loss.
    # gain_year_count_pre_loss = 0 when there is mangrove loss
    # gain_year_count_post_loss = 1 when there is no mangrove loss
@jit(nopython=True)
def calc_mang_loss(interval_length, first_mang_gain_year, first_mang_loss_year, interval_start_year,
                    c_pools_no_fire, loss_year, gain_year_count_pre_loss, gain_year_count_post_loss,
                    RF_AGC, RF_BGC, c_dens_in, deadwood_c_ratio, litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Carbon pools that are emitted as CO2 (depends on whether temporary or permanent mangrove loss)
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)

    # Step 1: Assigns pre-loss deadwood C and litter C ratios for removal factors
    # Deadwood and litter C removals only occur in pixels that are not considered to be in "equilibrium".
    # Thus, we need to check if mangroves are in "equilibrium" before loss during this interval.
    # Checking that it is not mangrove gain (i.e. no mangrove extent in 1996 but mangrove gain in subsequent year)
    # Checking that the start of the current interval is before any mangrove loss.
    # If conditions are met, the deadwood and litter ratios are set to 0 (no removals while in "equilibrium").
    if (first_mang_gain_year == 0) and ((first_mang_loss_year == 0) or (first_mang_loss_year > interval_start_year)):
        deadwood_c_ratio_pre_loss = 0.0
        litter_c_ratio_pre_loss = 0.0

    # Step 2: Calculates pre-loss gross removals by carbon pools (Mg C/ha/interval) . Gross removals are negative.
    agc_gross_removals_out = float((RF_AGC * gain_year_count_pre_loss) * -1)  # float() necessary for Numba typing
    bgc_gross_removals_out = float((RF_BGC * gain_year_count_pre_loss) * -1)  # float() necessary for Numba typing
    deadwood_c_gross_removals_out = agc_gross_removals_out * deadwood_c_ratio_pre_loss
    litter_c_gross_removals_out = agc_gross_removals_out * litter_c_ratio_pre_loss

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')

    # Step 4: Calculates carbon densities at the year of loss by carbon pool (Mg C/ha). This is not output from the model.
    # C pools pre-loss are the same as input carbon pools because there is no gain before loss.
    agc_pre_loss = agc_dens_in
    bgc_pre_loss = bgc_dens_in
    deadwood_c_pre_loss = deadwood_c_dens_in
    litter_c_pre_loss = litter_c_dens_in
    c_pre_loss = np.array(c_dens_in).astype('float32')


    # Step 5: Calculates emissions for each C pool (Mg C/ha/interval) for stand replacing mangrove loss.
    # Does not consider fire or partial loss (i.e. "disturbance") of mangroves
    if loss_year > 0:
        # If it is only temporary mangrove loss, only AGC and BGC carbon pools are emitted (cn.biomass_emissions_only).
        # If it is permanent mangrove loss, all non-soil carbon pools are emitted (cn.all_non_soil_pools).
        # Not converting to CO2 yet for consistency with all other outputs (Mg C/ha).
        agc_gross_emis_out = agc_pre_loss * agc_ef_CO2
        bgc_gross_emis_out = bgc_pre_loss * bgc_ef_CO2
        deadwood_c_gross_emis_out = deadwood_c_pre_loss * deadwood_c_ef_CO2
        litter_c_gross_emis_out = litter_c_pre_loss * litter_c_ef_CO2
    else:
        raise ValueError(f"Mangrove loss flag is True during interval but loss_year is 0")

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Calculates ending carbon densities by carbon pool.
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions.
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out

    # Step 8: Updates the forest age. Age is set to 0 at time of loss. Increments by the number of years in the interval after loss.
    forest_age_interval_end = gain_year_count_post_loss

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out, agc_ef_CO2, gain_year_count_pre_loss, forest_age_interval_end


# Gross fluxes and ending carbon stocks for trees converted to non-trees with and without fire.
# Non-CO2 gas emissions are only calculated if fire was detected during the interval.
# CO2 emissions are calculated differently depending on if fire was detected during the interval and if a Gef_CO2 is supplied.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_NT(node, burned_in_curr_interval, c_pools_EF_fire_CO2, c_pools_EF_fire_non_CO2, c_pools_EF_no_fire,
              c_dens_in, post_dist_regrowth, Cf_forest, Gef_ch4, Gef_n2o):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->NT, emission factors are binary (1 means full emissions, 0 means no emissions).
    # Carbon pools that are emitted as CO2 if fire was detected.
    if burned_in_curr_interval:
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_EF_fire_CO2)
    else:
        # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_EF_no_fire)


    # Step 1: Calculates carbon densities at the year of loss by carbon pool (Mg C/ha). This is not output from the model.
    # C pools pre-disturbance are the same as input carbon pools because there is no gain before loss.
    agc_pre_disturb = agc_dens_in
    bgc_pre_disturb = bgc_dens_in
    deadwood_c_pre_disturb = deadwood_c_dens_in
    litter_c_pre_disturb = litter_c_dens_in
    c_pre_disturb = np.array(c_dens_in).astype('float32')


    # Step 2: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which pools are emitted is controlled by the ef_CO2 flags.
    agc_gross_emis_out = agc_pre_disturb * agc_ef_CO2
    bgc_gross_emis_out = bgc_pre_disturb * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef_CO2

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 3: Updates gross removals to include one-time post-disturbance regrowth (new LC is cropland or short veg),
    # if applicable (short veg and cropland) (Mg C/ha/interval).
    # Regrowth of short veg and cropland is a one-time value.
    # If no post-loss removals, then removal factors and gross removals are given NoData values
    # because the default assumption for a loss pixel is no removals/RF (and thus NoData).
    if post_dist_regrowth[0] > 0:
        c_gross_removals_out = post_dist_regrowth * -1

        RF_AGC_out = post_dist_regrowth[0]
        RF_BGC_out = post_dist_regrowth[1]

    else:
        RF_AGC_out = np.nan
        RF_BGC_out = np.nan
        c_gross_removals_out = np.array([np.nan, np.nan, np.nan, np.nan]).astype('float32')


    # Step 4: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions (positive value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 5: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if burned_in_curr_interval:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_EF_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    # # For testing
    # if burned_in_prev_interval:
    #
    #     print("agc_dens_in:", agc_dens_in)
    #     print("agc_gross_removals_out:", agc_gross_removals_out)
    #     print("agc_pre_disturb:", agc_pre_disturb)
    #     print("biomass_to_carbon_non_mangrove:", cn.biomass_to_carbon_non_mangrove)
    #     print("Cf_forest:", Cf_forest)
    #     print("Gef_ch4:", Gef_ch4)
    #     print("Gef_n2o:", Gef_n2o)
    #     print("agc_gross_emis_out:", agc_gross_emis_out)
    #     print("c_dens_out:", c_dens_out)
    #     os.quit()


    return (state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out,
            RF_AGC_out, RF_BGC_out, agc_ef_CO2)


# Gross fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances.
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_T_non_stand_disturbs(node, burned_in_curr_interval, c_pools_fire_CO2, c_pools_fire_non_CO2, c_pools_no_fire,
                                interval_end_year, c_dens_in, most_recent_year_not_tall_veg, Cf_forest, Gef_co2,
                                Gef_ch4, Gef_n2o):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted depending on whether fire was detected during the interval.
    # For T->T, emission factors can range between 0 and 1, with 0 meaning no emissions and 1 meaning full emissions.
    if burned_in_curr_interval:      # Carbon pools that are emitted as CO2 if fire was detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)
    else:          # Carbon pools that are emitted as CO2 if fire was not detected.
        agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    # Step 1: Calculates the number of years of carbon gain before the non-stand-replacing disturbance occurred (years).
    # No gain before disturbance (no gain in disturbance year), so in this function gain_year_count_pre_dist = 0 always.
    gain_year_count_pre_dist = 0


    # Step 2: Calculates removal factors by carbon pool (Mg C/ha/interval).
    # Because there are no removals in years with height-based disturbance, RFs and gross removals are reassigned to 0.
    RF_AGC_pre_dist_out = 0
    RF_BGC_pre_dist_out = 0
    agc_gross_removals_out = 0
    bgc_gross_removals_out = 0
    deadwood_c_gross_removals_out = 0
    litter_c_gross_removals_out = 0
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 3: Calculates carbon densities at the year of disturbance by carbon pool (Mg C/ha). This is not output from the model.
    # C pools pre-disturbance are the same as input carbon pools because there is no gain before loss.
    agc_pre_disturb = agc_dens_in
    bgc_pre_disturb = bgc_dens_in
    deadwood_c_pre_disturb = deadwood_c_dens_in
    litter_c_pre_disturb = litter_c_dens_in
    c_pre_disturb = np.array(c_dens_in).astype('float32')


    # Step 4: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if burned_in_curr_interval:

        # Equations divide by C_to_CO2 to put the emissions back in Mg C/ha. They are later converted back to Mg CO2/ha,
        # but we need CO2 fire emissions in Mg C/ha here for consistency with all other outputs.
        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # Emission factor for burned forest is the combustion factor for forest
        agc_ef_CO2 = Cf_forest

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", RF_AGC_pre_dist)
        # print("gain_year_count_pre_dist:", gain_year_count_pre_dist)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # sys.exit()

    # Calculates CO2 emissions from forest loss for each C pool when no fire is detected
    else:

        agc_gross_emis_out = agc_pre_disturb * agc_ef_CO2
        bgc_gross_emis_out = bgc_pre_disturb * bgc_ef_CO2
        deadwood_c_gross_emis_out = deadwood_c_pre_disturb * deadwood_c_ef_CO2
        litter_c_gross_emis_out = litter_c_pre_disturb * litter_c_ef_CO2

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 5: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions (positive value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 6: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if burned_in_curr_interval:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')


    # Step 7: Resets the forest age to 0 because this function always has a partial disturbance
    # either with or without fire-- it doesn't matter. Age is reset either way.
    forest_age_interval_end = 0

    return (state_out, c_gross_emissions_out, c_gross_removals_out, non_co2_fluxes_out, c_dens_out,
            RF_AGC_pre_dist_out, RF_BGC_pre_dist_out, agc_ef_CO2, gain_year_count_pre_dist, forest_age_interval_end)


# Gross fluxes and ending carbon stocks for trees remaining trees with non-stand-replacing disturbances (fires are allowed).
# Carbon pool fluxes and densities are input and output as Mg C/ha(/interval) rather than Mg CO2 for arithmetic simplicity.
@jit(nopython=True)
def calc_T_T_no_disturbs(node, forest_age_interval_start, burned_during_interval, RF_AGC, RF_BGC,
                         c_pools_fire_CO2, c_pools_fire_non_CO2, interval_end_year, c_dens_in,
                         most_recent_year_not_tall_veg, Cf_forest, Gef_co2, Gef_ch4, Gef_n2o, deadwood_c_ratio,
                         litter_c_ratio):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Establishes which carbon pools are emitted if a fire was detected during the interval.
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_fire_CO2)


    # Step 1: Calculates the number of years of carbon gain before a fire occurred (years).
    # Model has no gain in the year of disturbance (including fire),
    # so gain_year_count_pre_dist = 0 when there is fire and = 1 when there is no fire.
    if burned_during_interval > 0:
        gain_year_count_pre_dist = 0  # No removals in a disturbance/fire year, so no removals during annual interval with fire
    else:
        gain_year_count_pre_dist = cn.veg_modeL_increment  # One year of gain when there is no fire


    # Step 2: Assigns deadwood C and litter C ratios for removal factors, if relevant (unitless).
    # Deadwood and litter C removals only occur in pixels that were not tall vegetation at some point (natural forest and mangrove only).
    # Thus, we need to check whether the pixel was non-tall vegetation at some point during the model before the end of this interval.
    # If conditions aren't met, the deadwood and litter ratios are set to 0 (no removals).
    # For simplicity, there are no deadwood or litter removals in loss intervals.
    if most_recent_year_not_tall_veg == 0 or most_recent_year_not_tall_veg == interval_end_year:
        deadwood_c_ratio = 0.0
        litter_c_ratio = 0.0


    # Step 3: Calculates pre-disturbance gross removals by carbon pools (Mg C/ha/interval). Gross removals are negative.
    agc_gross_removals_out = float((RF_AGC * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
    bgc_gross_removals_out = float((RF_BGC * gain_year_count_pre_dist) * -1) #float() necessary for Numba typing
    deadwood_c_gross_removals_out= agc_gross_removals_out * deadwood_c_ratio
    litter_c_gross_removals_out= agc_gross_removals_out * litter_c_ratio

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    # Must specify float32 because numba is quite particular about datatypes.
    c_gross_removals_out = np.array([agc_gross_removals_out, bgc_gross_removals_out, deadwood_c_gross_removals_out, litter_c_gross_removals_out]).astype('float32')


    # Step 4: Calculates carbon densities at the year of fire by carbon pool (Mg C/ha). This is not output from the model.
    # C pools pre-disturbance are the same as input carbon pools because there is no gain before disturbance/fire.
    # Assigning interval start C pools to pre-disturbance C pools reduces the number of calculations and is more explicit
    agc_pre_disturb = agc_dens_in
    bgc_pre_disturb = bgc_dens_in
    deadwood_c_pre_disturb = deadwood_c_dens_in
    litter_c_pre_disturb = litter_c_dens_in
    c_pre_disturb = np.array(c_dens_in).astype('float32')


    # Step 5: Calculates CO2 gross emissions from fire by carbon pools (Mg C/ha/interval).  Which ones are emitted depends on whether fire was detected.

    # Calculates CO2 emissions from fire for each C pool using fire emission factors
    # if a Gef for CO2 is supplied AND if there was fire during the interval.
    # This is used for non-stand replacing forest disturbances (as opposed to entire C pools being combusted).
    # From IPCC 2019 Eqn. 2.27
    if burned_during_interval > 0:

        # Equations divide by C_to_CO2 to put the emissions back in Mg C/ha. They are later converted back to Mg CO2/ha,
        # but we need CO2 fire emissions in Mg C/ha here for consistency with all other outputs.
        agc_gross_emis_out = ((agc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * agc_ef_CO2
        bgc_gross_emis_out = ((bgc_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * bgc_ef_CO2
        deadwood_c_gross_emis_out = ((deadwood_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * deadwood_c_ef_CO2
        litter_c_gross_emis_out = ((litter_c_pre_disturb / cn.biomass_to_carbon_non_mangrove) * Cf_forest * Gef_co2 * cn.g_to_kg) / cn.C_to_CO2 * litter_c_ef_CO2

        # Emission factor for burned forest is the combustion factor for forrest
        agc_ef_CO2 = Cf_forest

        # # For testing CO2 fire emissions
        # print("c_dens_in:", c_dens_in)
        # print("agc_rf:", agc_rf)
        # print("gain_year_count_pre_dist:", gain_year_count_pre_dist)
        # print("agc_pre_disturb:", agc_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_co2: {Gef_co2}")
        # print("AGC emission factor for fire:", agc_ef_CO2)
        # print("agc_gross_emis_out:", agc_gross_emis_out)
        # os.quit()

    # No emissions or emission factor if no fire detected
    else:

        agc_gross_emis_out = 0
        bgc_gross_emis_out = 0
        deadwood_c_gross_emis_out = 0
        litter_c_gross_emis_out = 0
        agc_ef_CO2 = 0

    # Gross emissions as an array
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')


    # Step 6: Calculates ending carbon densities by carbon pool.
    # Starts with carbon density in (list converted to np array), adds gross removals (subtracts negative value), subtracts emissions.
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out


    # Step 7: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if burned_during_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Selects just the carbon pools that have non-CO2 emissions from fire
        c_pools_for_fire_non_CO2 = np.where(c_pools_fire_non_CO2 == 1, c_pre_disturb, 0)

        # Sums the C pools that have non-CO2 fire emissions. We don't track which C pools the CH4 and N2O emissions come from,
        # so the pools are combined.
        c_pools_for_fire_total = np.sum(c_pools_for_fire_non_CO2)

        # Calculates non-CO2 fire emissions using the selected C pools in the year before disturbance
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_pools_for_fire_total, Cf_forest, Gef_ch4, Gef_n2o)

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("c_pre_disturb:", c_pre_disturb)
        # print(f"Cf_forest: {Cf_forest}; Gef_ch4: {Gef_ch4}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf_forest: {Cf_forest}; Gef_n2o: {Gef_n2o}; GWP N2O: {cn.gwp_n2o}")
        # print("c_pools_for_fire_non_CO2:", c_pools_for_fire_non_CO2)
        # print("c_pools_for_fire_total:", c_pools_for_fire_total)
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # os.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')


    # Step 8: Updates the forest age. Age is not affected by fire, so age always increases in this function.
    forest_age_interval_end = forest_age_interval_start + cn.veg_modeL_increment

    return state_out, c_gross_emissions_out, c_gross_removals_out, agc_ef_CO2, non_co2_fluxes_out, c_dens_out, gain_year_count_pre_dist, forest_age_interval_end


# Gross fluxes and ending carbon stocks for non-cropland (without tall vegetation) converted to cropland (without tall vegetation).
@jit(nopython=True)
def calc_NT_cropland_gain(c_pools_no_fire, c_dens_in, RF_array):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    # Carbon pools that are emitted as CO2 if fire was not detected.
    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)

    # Step 1: Calculates CO2 gross emissions by carbon pools (Mg C/ha/interval). Gross emissions are positive.
    # Which pools are emitted is controlled by the ef_CO2 flags.
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2

    # Consolidates outputs into array to reduce the number of arguments returned to the decision tree.
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')

    # Step 2: Calculates gross removals.
    # Gross removals is the annual crop AGC removals after residual carbon is lost (Mg C/ha/interval).
    # Gross removals is negative.
    c_gross_removals_out = -1 * RF_array

    # Step 3: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), subtracts emissions (positive value), subtracts cropland removals (negative value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_emissions_out - c_gross_removals_out

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out


# Gross fluxes and ending carbon stocks for cropland converted to non-cropland (without tall vegetation).
# Removals only if converted to short vegetation. Non-CO2 emissions only if fire.
@jit(nopython=True)
def calc_cropland_non_cropland(node, c_dens_in, c_pools_no_fire, times_burned_in_interval, RF_post_dist,
                               Cf_crop_residue, Gef_CH4_crop_residue, Gef_N2O_crop_residue):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)

    # Step 1: Calculates carbon gross emissions (AGC only) (Mg C/ha/interval)
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')

    # Step 2: Calculates carbon gross removals (Mg C/ha/interval) (only if converted to short vegetation). Gross removals are negative.
    c_gross_removals_out = RF_post_dist * -1  # Would be short vegetation removals (only for AGC and BGC)

    # Step 3: Calculates ending carbon densities (Mg C/ha)
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out - c_gross_emissions_out

    # Step 4: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Residue carbon in cropland is aboveground carbon only. Shouldn't be carbon in any other pools.
        residue_carbon = c_dens_in[0] * cn.cropland_residue_harvest_ratio

        # Calculates non-CO2 fire emissions using aboveground carbon (only cropland pool) for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(residue_carbon, Cf_crop_residue, Gef_CH4_crop_residue, Gef_N2O_crop_residue)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("residue_carbon:", residue_carbon)
        # print("times_burned_in_interval:", np.float32(times_burned_in_interval))
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for cropland remaining cropland (without tall vegetation).
# Carbon densities don't change.
# No CO2 emissions or removals but there are non-CO2 emissions if there is fire (crop residue burning).
@jit(nopython=True)
def calc_cropland_cropland(node, c_dens_in, times_burned_in_interval, Cf_crop_residue, Gef_CH4_crop_residue, Gef_N2O_crop_residue):

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_dens_out = np.array(c_dens_in).astype('float32')
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Residue carbon in cropland is aboveground carbon only. Shouldn't be carbon in any other pools.
        residue_carbon = c_dens_in[0] * cn.cropland_residue_harvest_ratio

        # Calculates non-CO2 fire emissions using aboveground carbon (only cropland pool) for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(residue_carbon, Cf_crop_residue, Gef_CH4_crop_residue, Gef_N2O_crop_residue)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # if times_burned_in_interval > 1:
        #     print("c_dens_in:", c_dens_in)
        #     print("times_burned_in_interval:", np.float32(times_burned_in_interval))
        #     print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        #     print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        #     print(f"ch4_flux_out_single: {ch4_flux_out_single}; n2o_flux_out_single: {n2o_flux_out_single};")
        #     print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        #     # sys.quit()


    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for non-short vegetation/non-forest/non-cropland converted to short vegetation.
@jit(nopython=True)
def calc_short_veg_gain(rf):

    # C densities should already be 0 because the starting LC should have been forced to 0s, but this is safer.
    c_dens_in = [0, 0, 0, 0]

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness

    # Step 2: Calculates gross removals.
    # Gross removals is the annual crop AGC removals after residual carbon is lost (Mg C/ha/interval).
    # Gross removals is negative.
    c_gross_removals_out = -1 * rf

    # Step 3: Calculates ending carbon densities by carbon pool (Mg C/ha).
    # Starts with carbon density in (list converted to np array), subtracts emissions (positive value), subtracts cropland removals (negative value).
    # Ending carbon pools are not affected by non-CO2 emissions in the next step.
    c_dens_out = np.array(c_dens_in).astype('float32') - c_gross_removals_out

    return c_gross_emissions_out, c_gross_removals_out, c_dens_out


# Gross fluxes and ending carbon stocks for short vegetation converted to non-short vegetation, non-forest or non-cropland.
# No CO2 removals. CO2 emissions occur.
# There are non-CO2 emissions where there is fire (biomass burning).
@jit(nopython=True)
def calc_short_veg_loss(node, c_dens_in, c_pools_no_fire, times_burned_in_interval, Cf_grassland, Gef_CH4_grassland, Gef_N2O_grassland):

    # Retrieves the starting densities for each carbon pool from the input array (Mg C/ha)
    agc_dens_in, bgc_dens_in, deadwood_c_dens_in, litter_c_dens_in = unpack_starting_carbon_densities(c_dens_in)

    agc_ef_CO2, bgc_ef_CO2, deadwood_c_ef_CO2, litter_c_ef_CO2 = unpack_emission_factors(c_pools_no_fire)


    # Step 1: Calculates carbon densities (all pools 0 Mg C/ha), carbon gross emissions (AGC only) and carbon gross removals (none)
    agc_gross_emis_out = agc_dens_in * agc_ef_CO2
    bgc_gross_emis_out = bgc_dens_in * bgc_ef_CO2
    deadwood_c_gross_emis_out = deadwood_c_dens_in * deadwood_c_ef_CO2
    litter_c_gross_emis_out = litter_c_dens_in * litter_c_ef_CO2

    agc_dens_out = agc_dens_in - agc_gross_emis_out
    bgc_dens_out = bgc_dens_in - bgc_gross_emis_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_gross_emis_out
    litter_c_dens_out = litter_c_dens_in - litter_c_gross_emis_out

    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')
    c_gross_emissions_out = np.array([agc_gross_emis_out, bgc_gross_emis_out, deadwood_c_gross_emis_out, litter_c_gross_emis_out]).astype('float32')
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Calculates non-CO2 fire emissions using aboveground carbon only for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_dens_in[0], Cf_grassland, Gef_CH4_grassland, Gef_N2O_grassland)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("times_burned_in_interval:", times_burned_in_interval)
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out


# Gross fluxes and ending carbon stocks for short vegetation remaining short vegetation.
# Carbon densities don't change.
# No CO2 emissions or removals but there are non-CO2 emissions if there is fire (biomass burning).
@jit(nopython=True)
def calc_short_veg_short_veg(node, c_dens_in, times_burned_in_interval, Cf_grassland, Gef_CH4_grassland, Gef_N2O_grassland):

    # Step 1: Calculates carbon densities, carbon gross emissions and carbon gross removals (no changes to any)
    c_dens_out = np.array(c_dens_in).astype('float32')
    c_gross_emissions_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness
    c_gross_removals_out = np.array([0, 0, 0, 0]).astype('float32')  # Specified for completeness


    # Step 2: Calculates non-CO2 emissions (if relevant) (Mg CO2e/ha/interval)
    # Default non-CO2 emissions values
    ch4_flux_out = 0
    n2o_flux_out = 0

    # Only assigns fire node code and calculates CH4 and N2O emissions if the pixel burned in the last interval
    if times_burned_in_interval > 0:

        state_out = accrete_node(node, cn.land_state_node_fire_value)

        # Calculates non-CO2 fire emissions using aboveground carbon only for a single year of burning
        ch4_flux_out, n2o_flux_out = non_CO2_fire_equations(c_dens_in[0], Cf_grassland, Gef_CH4_grassland, Gef_N2O_grassland)

        # Multiplies the per-burn emissions by the number of times burned to get total emissions during the interval
        ch4_flux_out = ch4_flux_out * times_burned_in_interval
        n2o_flux_out = n2o_flux_out * times_burned_in_interval

        # # For testing non-CO2 emissions
        # print("c_dens_in:", c_dens_in)
        # print("times_burned_in_interval:", times_burned_in_interval)
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_ch4: {cn.Gef_CH4_crop_residue}; GWP CH4: {cn.gwp_ch4}")
        # print(f"Cf: {cn.Cf_crop_residue}; Gef_n2o: {cn.Gef_N2O_crop_residue}; GWP N2O: {cn.gwp_n2o}")
        # print(f"ch4_flux_out: {ch4_flux_out}; n2o_flux_out: {n2o_flux_out};")
        # sys.quit()

    # Node code if no fire in the last interval. No CH4 and N2O emissions calculated.
    else:

        state_out = accrete_node(node, 2)

    non_co2_fluxes_out = np.array([ch4_flux_out, n2o_flux_out]).astype('float32')

    return state_out, c_gross_emissions_out, c_gross_removals_out, c_dens_out, non_co2_fluxes_out
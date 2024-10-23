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


# Fluxes and stocks for non-tree converted to tree
@jit(nopython=True)
def calc_NT_T(agc_rf, r_s_ratio_cell, c_dens_in):

    agc_dens_in = c_dens_in[0]
    bgc_dens_in = c_dens_in[1]
    deadwood_c_dens_in = c_dens_in[2]
    litter_c_dens_in = c_dens_in[3]

    agc_flux_out = (agc_rf * cn.NF_F_gain_year) * -1
    bgc_flux_out = float(agc_flux_out) * r_s_ratio_cell
    deadwood_c_flux_out= cn.deadwood_c_NT_T_rf
    litter_c_flux_out= cn.litter_c_NT_T_rf

    agc_dens_out = agc_dens_in - agc_flux_out
    bgc_dens_out = bgc_dens_in - bgc_flux_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_flux_out
    litter_c_dens_out = litter_c_dens_in - litter_c_flux_out

    # Must specify float32 because numba is quite particular about datatypes
    c_fluxes_out = np.array([agc_flux_out, bgc_flux_out, deadwood_c_flux_out, litter_c_flux_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    return c_fluxes_out, c_dens_out


# Fluxes and stocks for tree converted to non-tree
@jit(nopython=True)
def calc_T_NT(agc_ef, r_s_ratio_cell, c_dens_in):

    agc_dens_in = c_dens_in[0]
    bgc_dens_in = c_dens_in[1]
    deadwood_c_dens_in = c_dens_in[2]
    litter_c_dens_in = c_dens_in[3]

    agc_flux_out = agc_dens_in * agc_ef
    bgc_flux_out = float(agc_flux_out) * r_s_ratio_cell
    deadwood_c_flux_out = deadwood_c_dens_in * agc_ef
    litter_c_flux_out = litter_c_dens_in * agc_ef

    agc_dens_out = agc_dens_in - agc_flux_out
    bgc_dens_out = bgc_dens_in - bgc_flux_out
    deadwood_c_dens_out = deadwood_c_dens_in - deadwood_c_flux_out
    litter_c_dens_out = litter_c_dens_in - litter_c_flux_out

    # Must specify float32 because numba is quite particular about datatypes
    c_fluxes_out = np.array([agc_flux_out, bgc_flux_out, deadwood_c_flux_out, litter_c_flux_out]).astype('float32')
    c_dens_out = np.array([agc_dens_out, bgc_dens_out, deadwood_c_dens_out, litter_c_dens_out]).astype('float32')

    return c_fluxes_out, c_dens_out


# Fluxes and stocks for trees remaining trees without disturbances
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


# Fluxes and stocks for trees remaining trees with non-stand-replacing disturbances
#TODO include sequence of fluxes for disturbances: pre-disturb removals, emissions, post-disturb removals
@jit(nopython=True)
def calc_T_T_non_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in):

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


# Fluxes and stocks for trees remaining trees with stand-replacing disturbances
#TODO include sequence of fluxes for disturbances: pre-disturb removals, emissions, post-disturb removals
@jit(nopython=True)
def calc_T_T_stand_disturbs(agc_rf, agc_ef, r_s_ratio_cell, c_dens_in):

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

    return c_fluxes_out, c_dens_out
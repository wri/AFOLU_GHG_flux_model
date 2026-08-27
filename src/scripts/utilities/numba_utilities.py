import numpy as np
from numba import jit, njit
from numba.typed import Dict
from numba.core import types

# Project imports
from . import constants_and_names as cn

# ----------------------------------------------------------------------
# UNAMBIGUOUS INTEGER ENCODERS
# ----------------------------------------------------------------------
@njit(inline='always')
def _ndigits(val: int) -> int:
    """Return number‑of‑decimal‑digits in *val* (val ≠ 0)."""
    n = 1
    while val >= 10:
        val //= 10
        n += 1
    return n


@njit(inline='always')
def accrete_node(combo: int, new: int) -> int:
    """
    Append the **full** decimal representation of *new* to *combo*.

    This performs the equivalent of: int(f"{combo}{new}")
    but stays in integer space for Numba speed and works
    for any 0 ≤ new < 10 000 (enough for every branch in the model).

    Examples
    --------
    >>> accrete_node(12, 3)    # 12 → 123
    123
    >>> accrete_node(12, 34)   # 12 → 1234   (no collision!)
    1234
    """
    if new == 0:
        return combo          # appending “0” is a no‑op
    shift = 10 ** _ndigits(new)
    return combo * shift + new


@njit(inline='always')
def pad_to_6_digits(value: int, max_digits: int = 6) -> int:
    """
    Right‑pad *value* with zeroes until it has exactly *max_digits* digits.
    Raises if *value* already exceeds the allowed length.
    """
    if value == 0:
        return 0
    digits = _ndigits(value)
    if digits > max_digits:
        raise ValueError("Node exceeds maximum allowed length")
    return value * (10 ** (max_digits - digits))



def create_typed_dicts(layers):
    """
    Distribute arrays into typed dictionaries by dtype, so that we can pass them
    to Numba in dictionary form.
    """
    uint8_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    for key, array in layers.items():
        if array is None:
            continue
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
            # or raise TypeError(f"{key} dtype not recognized")

    print(f"uint8 datasets: {uint8_dict_layers.keys()}")
    print(f"int16 datasets: {int16_dict_layers.keys()}")
    print(f"int32 datasets: {int32_dict_layers.keys()}")
    print(f"float32 datasets: {float32_dict_layers.keys()}")

    typed_dict_uint8 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')
    )
    typed_dict_int16 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int16, 2, 'C')
    )
    typed_dict_int32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int32, 2, 'C')
    )
    typed_dict_float32 = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.float32, 2, 'C')
    )

    for key, array in uint8_dict_layers.items():
        typed_dict_uint8[key] = array
    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array
    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array
    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32


@jit(nopython=True)
def calculate_drainage_emissions_co2e(
        ef_co2, ef_n2o, ef_ch4_land, ef_ch4_ditch,
        ef_co2_offsite, frac_ditch,
        c_to_co2, n2o_n_to_n2o, gwp_n2o, gwp_ch4
):
    """
    Return drainage components per hectare per year.

    Inputs:
      ef_co2, ef_co2_offsite: t C/ha/yr
      ef_n2o: kg N/ha/yr (as N2O–N)
      ef_ch4_land, ef_ch4_ditch: kg CH4/ha/yr
      frac_ditch: fraction of area occupied by ditches (0–1)

    Returns:
      co2_emissions (t CO2/ha/yr),
      n2o_emissions_co2e (t CO2e/ha/yr),
      ch4_land_emissions_co2e (t CO2e/ha/yr),
      ch4_ditch_emissions_co2e (t CO2e/ha/yr),
      co2_offsite_emissions (t CO2/ha/yr),
      drainage_total_co2e (t CO2e/ha/yr)
    """
    # Guard fraction
    frac_ditch = min(1.0, max(0.0, frac_ditch))

    # CO2 (on-site + off-site DOC): t C -> t CO2
    co2_emissions = ef_co2 * c_to_co2
    co2_offsite_emissions = ef_co2_offsite * c_to_co2

    # N2O: kg N -> kg N2O -> t -> t CO2e
    n2o_emissions_co2e = (ef_n2o * n2o_n_to_n2o * gwp_n2o) / 1000.0

    # CH4 area split (Tier-1): kg -> t -> t CO2e
    ch4_land_emissions_co2e  = (((1.0 - frac_ditch) * ef_ch4_land)  / 1000.0) * gwp_ch4
    ch4_ditch_emissions_co2e = (( frac_ditch        * ef_ch4_ditch) / 1000.0) * gwp_ch4

    # Sum to total CO2e per ha per year
    drainage_total_co2e = (
        co2_emissions
        + co2_offsite_emissions
        + n2o_emissions_co2e
        + ch4_land_emissions_co2e
        + ch4_ditch_emissions_co2e
    )

    return (
        co2_emissions,
        n2o_emissions_co2e,
        ch4_land_emissions_co2e,
        ch4_ditch_emissions_co2e,
        co2_offsite_emissions,
        drainage_total_co2e,
    )



@jit(nopython=True)
def calculate_burned_area_emissions(
        fuel_consumed,
        gef_co2,
        gef_co,
        gef_ch4,
        gwp_ch4
):
    """
    Return burned emissions per hectare.

    fuel_consumed is the IPCC MB*Cf product in t dry matter ha-1.
    """
    burn_co2 = fuel_consumed * gef_co2 * 1e-3
    burn_co = fuel_consumed * gef_co * 1e-3
    burn_ch4 = fuel_consumed * gef_ch4 * 1e-3 * gwp_ch4

    total_burned_emissions_co2e = burn_co2 + burn_ch4
    return burn_co2, burn_co, burn_ch4, total_burned_emissions_co2e

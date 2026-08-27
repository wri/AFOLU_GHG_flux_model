"""
Emission factors for burned area calculations (IPCC Wetlands Supplement aligned).

Keys: ecozone × drainage state
Order per entry: [gef_co2, gef_co, gef_ch4, mass_burnt]
  - gef_* units: g gas per kg dry matter burned (g kg-1)
    (CO2 values are CO2, not CO2-C; boreal/temperate row from Table 2.7 converted with 44/12)
  - mass_burnt: tonnes dry matter per hectare (t d.m. ha-1), i.e., MB*CF Tier-1 product from Table 2.6

Tables:
  - Table 2.7: Emission factors for organic-soil fires (boreal/temperate & tropical)  [CO2-C, CO, CH4]
  - Table 2.6: Fuel consumed (MB*CF) for organic-soil fires by climate, drainage & fire type
"""

import numpy as np
from numba import types
from numba.typed import Dict

ZERO_ARRAY = np.zeros(4, dtype=np.float32)

# ---- Helper constants (CO2-C -> CO2 conversion) ----
_C_TO_CO2 = 44.0 / 12.0

# ---- GEFs from IPCC Table 2.7 ----
# Boreal/Temperate (CO2-C=362±41 -> CO2 in g/kg)
_BORE_TEMP_CO2 = 362.0 * _C_TO_CO2
_BORE_TEMP_CO2_LOW = (362.0 - 41.0) * _C_TO_CO2
_BORE_TEMP_CO2_HIGH = (362.0 + 41.0) * _C_TO_CO2
_BORE_TEMP_CO  = 207.0
_BORE_TEMP_CO_LOW  = 207.0 - 70.0
_BORE_TEMP_CO_HIGH = 207.0 + 70.0
_BORE_TEMP_CH4 = 9.0
_BORE_TEMP_CH4_LOW = 9.0 - 4.0
_BORE_TEMP_CH4_HIGH = 9.0 + 4.0

# Tropical (CO2-C=464 -> CO2; CO=210; CH4=21) – no CI given in Table 2.7
_TROP_CO2 = 464.0 * _C_TO_CO2
_TROP_CO  = 210.0
_TROP_CH4 = 21.0

# ---- Mass burnt MB*CF from IPCC Table 2.6 (t d.m. ha-1) ----
# Boreal/Temperate wildfire (undrained/drained)
_MB_BORE_TEMP_UND = 66.0  # 95% CI 46–86
_MB_BORE_TEMP_UND_LOW, _MB_BORE_TEMP_UND_HIGH = 46.0, 86.0
_MB_BORE_TEMP_DRN = 336.0 # ±4 (SE)
_MB_BORE_TEMP_DRN_LOW, _MB_BORE_TEMP_DRN_HIGH = 332.0, 340.0

# Tropical drained: wildfire vs. prescribed (agricultural)
_MB_TROP_DRN_WF = 353.0   # 95% CI 170–536
_MB_TROP_DRN_WF_LOW, _MB_TROP_DRN_WF_HIGH = 170.0, 536.0
_MB_TROP_DRN_PRESC = 155.0 # 95% CI 82–228 (agricultural management)
_MB_TROP_DRN_PRESC_LOW, _MB_TROP_DRN_PRESC_HIGH = 82.0, 228.0

DEFAULT = {
    # Boreal/Temperate — same GEFs; MB differs by drainage status
    'boreal_drained':      [ _BORE_TEMP_CO2, _BORE_TEMP_CO, _BORE_TEMP_CH4, _MB_BORE_TEMP_DRN ],
    'boreal_undrained':    [ _BORE_TEMP_CO2, _BORE_TEMP_CO, _BORE_TEMP_CH4, _MB_BORE_TEMP_UND ],
    'temperate_drained':   [ _BORE_TEMP_CO2, _BORE_TEMP_CO, _BORE_TEMP_CH4, _MB_BORE_TEMP_DRN ],
    'temperate_undrained': [ _BORE_TEMP_CO2, _BORE_TEMP_CO, _BORE_TEMP_CH4, _MB_BORE_TEMP_UND ],

    # Tropical drained — split into ‘crop_or_plantation’ (prescribed) vs ‘other’ (wildfire)
    'tropical_drained_crop_or_plantation': [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_PRESC ],
    'tropical_drained_other':               [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_WF ],

    # No Tier-1 defaults for tropical undrained organic-soil wildfire fuel consumption (Table 2.6)
    'tropical_undrained':  [ 0.0, 0.0, 0.0, 0.0 ],
    'other':               [ 0.0, 0.0, 0.0, 0.0 ],
}

LOW = {
    'boreal_drained':      [ _BORE_TEMP_CO2_LOW, _BORE_TEMP_CO_LOW, _BORE_TEMP_CH4_LOW, _MB_BORE_TEMP_DRN_LOW ],
    'boreal_undrained':    [ _BORE_TEMP_CO2_LOW, _BORE_TEMP_CO_LOW, _BORE_TEMP_CH4_LOW, _MB_BORE_TEMP_UND_LOW ],
    'temperate_drained':   [ _BORE_TEMP_CO2_LOW, _BORE_TEMP_CO_LOW, _BORE_TEMP_CH4_LOW, _MB_BORE_TEMP_DRN_LOW ],
    'temperate_undrained': [ _BORE_TEMP_CO2_LOW, _BORE_TEMP_CO_LOW, _BORE_TEMP_CH4_LOW, _MB_BORE_TEMP_UND_LOW ],
    'tropical_drained_crop_or_plantation': [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_PRESC_LOW ],
    'tropical_drained_other':               [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_WF_LOW ],
    'tropical_undrained':  [ 0.0, 0.0, 0.0, 0.0 ],
    'other':               [ 0.0, 0.0, 0.0, 0.0 ],
}

HIGH = {
    'boreal_drained':      [ _BORE_TEMP_CO2_HIGH, _BORE_TEMP_CO_HIGH, _BORE_TEMP_CH4_HIGH, _MB_BORE_TEMP_DRN_HIGH ],
    'boreal_undrained':    [ _BORE_TEMP_CO2_HIGH, _BORE_TEMP_CO_HIGH, _BORE_TEMP_CH4_HIGH, _MB_BORE_TEMP_UND_HIGH ],
    'temperate_drained':   [ _BORE_TEMP_CO2_HIGH, _BORE_TEMP_CO_HIGH, _BORE_TEMP_CH4_HIGH, _MB_BORE_TEMP_DRN_HIGH ],
    'temperate_undrained': [ _BORE_TEMP_CO2_HIGH, _BORE_TEMP_CO_HIGH, _BORE_TEMP_CH4_HIGH, _MB_BORE_TEMP_UND_HIGH ],
    'tropical_drained_crop_or_plantation': [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_PRESC_HIGH ],
    'tropical_drained_other':               [ _TROP_CO2, _TROP_CO, _TROP_CH4, _MB_TROP_DRN_WF_HIGH ],
    'tropical_undrained':  [ 0.0, 0.0, 0.0, 0.0 ],
    'other':               [ 0.0, 0.0, 0.0, 0.0 ],
}

def _to_typed(table: dict) -> Dict:
    d = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:])
    for k, vals in table.items():
        d[k] = np.array(vals, dtype=np.float32)
    return d

DEFAULT_TABLE = _to_typed(DEFAULT)
LOW_TABLE = _to_typed(LOW)
HIGH_TABLE = _to_typed(HIGH)

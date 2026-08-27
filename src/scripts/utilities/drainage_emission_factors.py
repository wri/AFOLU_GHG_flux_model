"""
Drainage emission factors lookup tables (IPCC Wetlands Supplement aligned).

Keys: ecozone × land cover
Order per entry: [co2, n2o, ch4_land, ch4_ditch, co2_offsite, frac_ditch]
Units:
  - co2, co2_offsite: tonnes C ha-1 yr-1 (CO2-C)
  - n2o: kg N2O-N ha-1 yr-1
  - ch4_land, ch4_ditch: kg CH4 ha-1 yr-1
  - frac_ditch: unitless fraction of area under ditches

Tables:
  - Table 2.1 (EF_CO2-C), Table 2.5 (EF_N2O), Table 2.3 (EF_CH4_land),
  - Table 2.4 (EF_CH4_ditch & Fracditch), Table 2.2 (DOC off-site CO2-C)
"""

import numpy as np
from numba import types
from numba.typed import Dict

ZERO_ARRAY = np.zeros(6, dtype=np.float32)

# ---- DEFAULT values (exact IPCC Tier-1, using deep/shallow splits where applicable) ----
DEFAULT = {
    # Boreal forests (nutrient classes)
    'boreal_forest_poor':        [0.25, 0.22, 7.0, 217.0, 0.12, 0.025],  # Table 2.1/2.5/2.3/2.4/2.2
    'boreal_forest_rich':        [0.93, 3.2,  2.0, 217.0, 0.12, 0.025],

    # Boreal other land uses
    'boreal_grassland':          [5.7,  9.5,  1.4, 1165.0, 0.12, 0.05],   # deep-drained rich grassland for ditch EF
    'boreal_cropland':           [7.9,  13.0, 0.0, 1165.0, 0.12, 0.05],
    'boreal_extraction':         [2.8,  0.3,  6.1, 542.0,  0.12, 0.05],

    # Temperate forests (single nutrient class at Tier-1)
    'temperate_forest':          [2.6,  2.8,  2.5, 217.0,  0.31, 0.025],

    # Temperate grassland splits (nutrient class + drainage depth influences CH4 & ditch EF)
    'temperate_grassland_poor':  [5.3,  4.3,  1.8, 527.0,  0.31, 0.05],   # shallow-drained
    'temperate_grassland_rich':  [6.1,  8.2, 16.0, 1165.0, 0.31, 0.05],   # deep-drained

    'temperate_cropland':        [7.9,  13.0, 0.0, 1165.0, 0.31, 0.05],
    'temperate_extraction':      [2.8,  0.3,  6.1, 542.0,  0.31, 0.05],

    # Tropical plantations & forest/other
    'tropical_long_rotation':    [15.0, 2.4,  2.7, 2259.0, 0.82, 0.02],   # plantations (unknown/long rotation)
    'tropical_short_rotation':   [20.0, 2.4,  2.7, 2259.0, 0.82, 0.02],   # e.g., acacia
    'tropical_oil_palm':         [11.0, 1.2,  0.0, 2259.0, 0.82, 0.02],
    'tropical_sago_palm':        [1.5,  3.3, 26.2, 2259.0, 0.82, 0.02],

    'tropical_forest':           [5.3,  2.4,  4.9, 2259.0, 0.82, 0.02],
    'tropical_grassland':        [9.6,  5.0,  7.0, 2259.0, 0.82, 0.02],
    'tropical_cropland':         [14.0, 5.0,  7.0, 2259.0, 0.82, 0.02],
    'tropical_extraction':       [2.0,  3.6,  0.0, 2259.0, 0.82, 0.02],   # N2O from Table 2.5; CH4_land ~0 (no Tier-1 value)

    # Proxies for categories not explicitly listed at Tier-1
    "boreal_settlement":   [5.7,  9.5,  1.4, 1165.0, 0.12, 0.05],  # boreal grassland proxy
    "boreal_wetland":      [2.8,  0.3,  6.1,  542.0, 0.12, 0.05],  # boreal extraction proxy
    "boreal_otherland":    [5.7,  9.5,  1.4, 1165.0, 0.12, 0.05],  # boreal grassland proxy

    "temperate_settlement":[6.1,  8.2, 16.0, 1165.0, 0.31, 0.05],  # temperate grassland, deep-drained proxy
    "temperate_wetland":   [6.1,  8.2, 16.0, 1165.0, 0.31, 0.05],  # temperate grassland, deep-drained proxy
    "temperate_otherland": [6.1,  8.2, 16.0, 1165.0, 0.31, 0.05],  # temperate grassland, deep-drained proxy

    "tropical_settlement": [9.6,  5.0,  7.0, 2259.0, 0.82, 0.02],  # tropical grassland proxy
    "tropical_wetland":    [2.0,  3.6,  0.0, 2259.0, 0.82, 0.02],  # tropical extraction proxy
    "tropical_otherland":  [9.6,  5.0,  7.0, 2259.0, 0.82, 0.02],  # tropical grassland proxy

    # Placeholder factors for coastal peat systems (until bespoke values are finalised)
    "coastal_mangrove":    [7.9,  0.0,  0.0,    0.0, 0.0, 0.0],
    "coastal_tidal_marsh": [7.9,  0.0,  0.0,    0.0, 0.0, 0.0],
}

# ---- LOW/HIGH bounds from the same IPCC tables ----
# Note: Where Tier-1 provides no CI (rare), LOW=HIGH=DEFAULT to avoid inventing uncertainty.

LOW = {
    'boreal_forest_poor':        [-0.23, 0.15,  2.9,   41.0, 0.07, 0.025],
    'boreal_forest_rich':        [ 0.54, 1.9,  -1.6,   41.0, 0.07, 0.025],

    'boreal_grassland':          [ 2.9,  4.6,  -1.6,  335.0, 0.07, 0.05],
    'boreal_cropland':           [ 6.5,  8.2,  -2.8,  335.0, 0.07, 0.05],
    'boreal_extraction':         [ 1.1, -0.03,  1.6,  102.0, 0.07, 0.05],

    'temperate_forest':          [ 2.0, -0.57, -0.6,   41.0, 0.19, 0.025],

    'temperate_grassland_poor':  [ 3.7,  1.9,   0.72, 285.0, 0.19, 0.05],
    'temperate_grassland_rich':  [ 5.0,  4.9,   2.4,  335.0, 0.19, 0.05],

    'temperate_cropland':        [ 6.5,  8.2,  -2.8,  335.0, 0.19, 0.05],
    'temperate_extraction':      [ 1.1, -0.03,  1.6,  102.0, 0.19, 0.05],

    'tropical_long_rotation':    [10.0, 1.3,  -0.9,   599.0, 0.56, 0.02],
    'tropical_short_rotation':   [16.0, 1.3,  -0.9,   599.0, 0.56, 0.02],
    'tropical_oil_palm':         [ 5.6, 1.2,   0.0,   599.0, 0.56, 0.02],
    'tropical_sago_palm':        [-2.3, 3.3,   7.2,   599.0, 0.56, 0.02],

    'tropical_forest':           [-0.7, 1.3,   2.3,   599.0, 0.56, 0.02],
    'tropical_grassland':        [ 4.5, 2.3,   0.3,   599.0, 0.56, 0.02],
    'tropical_cropland':         [ 6.6, 2.3,   0.3,   599.0, 0.56, 0.02],
    'tropical_extraction':       [ 0.06,0.2,   0.0,   599.0, 0.56, 0.02],

    "boreal_settlement":         [ 2.9,  4.6,  -1.6,  335.0, 0.07, 0.05],
    "boreal_wetland":            [ 1.1, -0.03,  1.6,  102.0, 0.07, 0.05],
    "boreal_otherland":          [ 2.9,  4.6,  -1.6,  335.0, 0.07, 0.05],

    "temperate_settlement":      [ 5.0,  4.9,   2.4,  335.0, 0.19, 0.05],
    "temperate_wetland":         [ 5.0,  4.9,   2.4,  335.0, 0.19, 0.05],
    "temperate_otherland":       [ 5.0,  4.9,   2.4,  335.0, 0.19, 0.05],

    "tropical_settlement":       [ 4.5, 2.3,   0.3,   599.0, 0.56, 0.02],
    "tropical_wetland":          [ 0.06,0.2,   0.0,   599.0, 0.56, 0.02],
    "tropical_otherland":        [ 4.5, 2.3,   0.3,   599.0, 0.56, 0.02],

    "coastal_mangrove":          [ 5.2, 0.0,   0.0,     0.0, 0.0, 0.0],
    "coastal_tidal_marsh":       [ 5.2, 0.0,   0.0,     0.0, 0.0, 0.0],
}

HIGH = {
    'boreal_forest_poor':        [0.73, 0.28, 11.0,  393.0, 0.19, 0.025],
    'boreal_forest_rich':        [1.3,  4.5,  5.5,   393.0, 0.19, 0.025],

    'boreal_grassland':          [8.6,  14.0, 4.5,  1995.0, 0.19, 0.05],
    'boreal_cropland':           [9.4,  18.0, 2.8,  1995.0, 0.19, 0.05],
    'boreal_extraction':         [4.2,  0.64, 11.0,  981.0, 0.19, 0.05],

    'temperate_forest':          [3.3,  6.1,  5.7,   393.0, 0.46, 0.025],

    'temperate_grassland_poor':  [6.9,  6.8,  2.9,   769.0, 0.46, 0.05],
    'temperate_grassland_rich':  [7.3, 11.0, 29.0, 1995.0, 0.46, 0.05],

    'temperate_cropland':        [9.4, 18.0,  2.8, 1995.0, 0.46, 0.05],
    'temperate_extraction':      [4.2,  0.64, 11.0,  981.0, 0.46, 0.05],

    'tropical_long_rotation':    [21.0, 3.5,  6.3,  3919.0, 1.14, 0.02],
    'tropical_short_rotation':   [24.0, 3.5,  6.3,  3919.0, 1.14, 0.02],
    'tropical_oil_palm':         [17.0, 1.2,  0.0,  3919.0, 1.14, 0.02],
    'tropical_sago_palm':        [ 5.4, 3.3, 45.3, 3919.0, 1.14, 0.02],

    'tropical_forest':           [ 9.5, 3.5,  7.5,  3919.0, 1.14, 0.02],
    'tropical_grassland':        [17.0, 7.7, 13.7, 3919.0, 1.14, 0.02],
    'tropical_cropland':         [26.0, 7.7, 13.7, 3919.0, 1.14, 0.02],
    'tropical_extraction':       [ 7.0, 5.0,  0.0,  3919.0, 1.14, 0.02],

    "boreal_settlement":         [8.6,  14.0, 4.5,  1995.0, 0.19, 0.05],
    "boreal_wetland":            [4.2,  0.64, 11.0,  981.0, 0.19, 0.05],
    "boreal_otherland":          [8.6,  14.0, 4.5,  1995.0, 0.19, 0.05],

    "temperate_settlement":      [7.3, 11.0, 29.0, 1995.0, 0.46, 0.05],
    "temperate_wetland":         [7.3, 11.0, 29.0, 1995.0, 0.46, 0.05],
    "temperate_otherland":       [7.3, 11.0, 29.0, 1995.0, 0.46, 0.05],

    "tropical_settlement":       [17.0, 7.7, 13.7, 3919.0, 1.14, 0.02],
    "tropical_wetland":          [ 7.0, 5.0,  0.0,  3919.0, 1.14, 0.02],
    "tropical_otherland":        [17.0, 7.7, 13.7, 3919.0, 1.14, 0.02],

    "coastal_mangrove":          [11.8, 0.0,  0.0,     0.0, 0.0, 0.0],
    "coastal_tidal_marsh":       [11.8, 0.0,  0.0,     0.0, 0.0, 0.0],
}

def _to_typed(table: dict) -> Dict:
    d = Dict.empty(key_type=types.unicode_type, value_type=types.float32[:])
    for k, vals in table.items():
        d[k] = np.array(vals, dtype=np.float32)
    return d

DEFAULT_TABLE = _to_typed(DEFAULT)
LOW_TABLE = _to_typed(LOW)
HIGH_TABLE = _to_typed(HIGH)
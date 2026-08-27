"""State-code registry and compact combined-state encoding for the core model."""

from __future__ import annotations

import numpy as np


PAD_DIGITS = 8


def _pad_right(code: str) -> str:
    return code.ljust(PAD_DIGITS, "0")


_DRAIN_ROOT = {
    "11": "peat_drained_primary_infra",
    "12": "peat_drained_secondary_infra",
    "13": "peat_drained_cropland_settlement",
    "14": "peat_drained_plantation",
    "15": "peat_drained_extraction",
    "16": "peat_undrained",
    "0": "non_peat",
}

_CLASSIFICATION_SUFFIX_LABELS = {
    "": "",
    "91": "__coastal_mangrove",
    "92": "__coastal_tidal_marsh",
}

_EMISSIONS_BY_SUFFIX = {
    "": {
        "111": "boreal_extraction",
        "1131": "boreal_forest_poor",
        "1132": "boreal_forest_rich",
        "114": "boreal_grassland",
        "115": "boreal_cropland",
        "116": "boreal_settlement",
        "117": "boreal_wetland",
        "118": "boreal_otherland",
        "121": "temperate_extraction",
        "123": "temperate_forest",
        "1241": "temperate_grassland_poor",
        "1242": "temperate_grassland_rich",
        "125": "temperate_cropland",
        "126": "temperate_settlement",
        "127": "temperate_wetland",
        "128": "temperate_otherland",
        "131": "tropical_extraction",
        "1321": "tropical_long_rotation",
        "1322": "tropical_short_rotation",
        "1323": "tropical_oil_palm",
        "133": "tropical_forest",
        "134": "tropical_grassland",
        "135": "tropical_cropland",
        "136": "tropical_settlement",
        "137": "tropical_wetland",
        "138": "tropical_otherland",
    },
    "91": {
        "1191": "boreal_coastal_mangrove",
        "1291": "temperate_coastal_mangrove",
        "1391": "tropical_coastal_mangrove",
        "191": "other_domain_coastal_mangrove",
        "91": "coastal_mangrove",
    },
    "92": {
        "1192": "boreal_coastal_tidal_marsh",
        "1292": "temperate_coastal_tidal_marsh",
        "1392": "tropical_coastal_tidal_marsh",
        "192": "other_domain_coastal_tidal_marsh",
        "92": "coastal_tidal_marsh",
    },
}

_UNDRAINED_ECOZONES = {
    "1": "boreal",
    "2": "temperate",
    "3": "tropical",
    "4": "other_domain",
}


def _build_drained_state_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}

    for root in ("16", "0"):
        base_label = _DRAIN_ROOT[root]
        for suffix, suffix_label in _CLASSIFICATION_SUFFIX_LABELS.items():
            if suffix:
                code = suffix if root == "0" else root + suffix
            else:
                code = root
            mapping[_pad_right(code)] = base_label + suffix_label

    for suffix, suffix_label in _CLASSIFICATION_SUFFIX_LABELS.items():
        coastal_part = suffix_label.lstrip("_")
        class_code = "16" + suffix
        for ecozone_digit, ecozone_label in _UNDRAINED_ECOZONES.items():
            full_code = class_code + ecozone_digit
            if coastal_part:
                meaning = f"peat_undrained__{ecozone_label}_{coastal_part}"
            else:
                meaning = f"peat_undrained__{ecozone_label}"
            mapping[_pad_right(full_code)] = meaning

    for root in ("11", "12", "13", "14", "15"):
        base_label = _DRAIN_ROOT[root]
        for suffix in _CLASSIFICATION_SUFFIX_LABELS:
            class_code = root + suffix
            for emit_code, emit_label in _EMISSIONS_BY_SUFFIX[suffix].items():
                mapping[_pad_right(class_code + emit_code)] = (
                    f"{base_label}__{emit_label}"
                )

    return mapping


DRAINED_STATE_NODE_MEANINGS = _build_drained_state_mapping()

_BURN_STATE_LABELS = {
    "111": "boreal__drained",
    "112": "boreal__undrained",
    "221": "temperate__drained",
    "222": "temperate__undrained",
    "331": "tropical__drained_crop_or_plantation",
    "332": "tropical__drained_other",
    "333": "tropical__undrained",
    "44": "other_domain__other",
}

BURNED_STATE_NODE_MEANINGS = {
    _pad_right(code): label for code, label in _BURN_STATE_LABELS.items()
}

ALL_DRAINED_STATE_CODES = frozenset(DRAINED_STATE_NODE_MEANINGS)
ALL_BURNED_STATE_CODES = frozenset(BURNED_STATE_NODE_MEANINGS)

COMBINED_STATE_DRAINED_BITS = 8
COMBINED_STATE_BURNED_BITS = 4
COMBINED_STATE_DRAINED_MASK = (1 << COMBINED_STATE_DRAINED_BITS) - 1
COMBINED_STATE_BURNED_MASK = (1 << COMBINED_STATE_BURNED_BITS) - 1
COMBINED_STATE_BURNED_SHIFT = COMBINED_STATE_DRAINED_BITS
COMBINED_STATE_HAS_DRAINED_BIT = 12
COMBINED_STATE_HAS_BURNED_BIT = 13

_SORTED_DRAINED_CODES = sorted(ALL_DRAINED_STATE_CODES)
_SORTED_BURNED_CODES = sorted(ALL_BURNED_STATE_CODES)

DRAINED_STATE_CODE_TO_ID = {
    code: index + 1 for index, code in enumerate(_SORTED_DRAINED_CODES)
}
BURNED_STATE_CODE_TO_ID = {
    code: index + 1 for index, code in enumerate(_SORTED_BURNED_CODES)
}
DRAINED_STATE_ID_TO_CODE = {
    index: code for code, index in DRAINED_STATE_CODE_TO_ID.items()
}
BURNED_STATE_ID_TO_CODE = {
    index: code for code, index in BURNED_STATE_CODE_TO_ID.items()
}


def _pack_combined_state_ids(drained_id: int, burned_id: int) -> int:
    """Pack registry identifiers into the stored combined-state value."""

    value = np.uint32(drained_id)
    if drained_id:
        value |= np.uint32(1 << COMBINED_STATE_HAS_DRAINED_BIT)
    if burned_id:
        value |= np.uint32(burned_id << COMBINED_STATE_BURNED_SHIFT)
        value |= np.uint32(1 << COMBINED_STATE_HAS_BURNED_BIT)
    return int(value)


COMBINED_STATE_GROUP_VALUES = np.array(
    sorted(
        _pack_combined_state_ids(drained_id, burned_id)
        for drained_id in [0, *DRAINED_STATE_ID_TO_CODE]
        for burned_id in [0, *BURNED_STATE_ID_TO_CODE]
    ),
    dtype=np.uint32,
)


def pack_combined_state(
    drained_code: np.ndarray,
    burned_code: np.ndarray,
) -> np.ndarray:
    """Pack drained and burned node-code rasters into one uint32 raster."""

    out = np.zeros(drained_code.shape, dtype=np.uint32)
    for code_str, index in DRAINED_STATE_CODE_TO_ID.items():
        code = np.uint32(int(code_str))
        mask = drained_code == code
        if np.any(mask):
            out[mask] |= np.uint32(index)
            out[mask] |= np.uint32(1 << COMBINED_STATE_HAS_DRAINED_BIT)

    for code_str, index in BURNED_STATE_CODE_TO_ID.items():
        code = np.uint32(int(code_str))
        mask = burned_code == code
        if np.any(mask):
            out[mask] |= np.uint32(index << COMBINED_STATE_BURNED_SHIFT)
            out[mask] |= np.uint32(1 << COMBINED_STATE_HAS_BURNED_BIT)

    return out


def unpack_combined_state(
    combined_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a combined-state raster into drained and burned node codes."""

    drained_id = (
        combined_state & np.uint32(COMBINED_STATE_DRAINED_MASK)
    ).astype(np.uint16)
    burned_id = (
        (combined_state >> np.uint32(COMBINED_STATE_BURNED_SHIFT))
        & np.uint32(COMBINED_STATE_BURNED_MASK)
    ).astype(np.uint16)

    drained_out = np.zeros(combined_state.shape, dtype=np.uint32)
    burned_out = np.zeros(combined_state.shape, dtype=np.uint32)

    for index, code_str in DRAINED_STATE_ID_TO_CODE.items():
        mask = drained_id == np.uint16(index)
        if np.any(mask):
            drained_out[mask] = np.uint32(int(code_str))
    for index, code_str in BURNED_STATE_ID_TO_CODE.items():
        mask = burned_id == np.uint16(index)
        if np.any(mask):
            burned_out[mask] = np.uint32(int(code_str))

    return drained_out, burned_out


assert len(DRAINED_STATE_CODE_TO_ID) <= (1 << COMBINED_STATE_DRAINED_BITS)
assert len(BURNED_STATE_CODE_TO_ID) <= (1 << COMBINED_STATE_BURNED_BITS)

# utils/processing_utils.py

import numpy as np
from numba import jit, types
import pandas as pd
import requests
from io import BytesIO
import rasterio
from typing import Tuple, Dict as TypedDict, List
import math
import re
from numba.typed import Dict as NumbaDict
from utils.logging_utils import timestr, print_and_log


@jit(nopython=True)
def accrete_node(combo, new):
    combo = combo * 10 + new
    return combo


def create_typed_dicts(layers: TypedDict[str, np.ndarray]) -> Tuple[NumbaDict, NumbaDict, NumbaDict, NumbaDict]:
    """
    Creates separate Numba-typed dictionaries for each data type.

    Args:
        layers (TypedDict[str, np.ndarray]): Dictionary of layers with their data arrays.

    Returns:
        Tuple[NumbaDict, NumbaDict, NumbaDict, NumbaDict]: Typed dictionaries for uint8, int16, int32, and float32.
    """
    # Initializes empty dictionaries for each type
    uint8_dict_layers = {}
    int16_dict_layers = {}
    int32_dict_layers = {}
    float32_dict_layers = {}

    # Iterates through the downloaded chunk dictionary and distributes arrays to a separate dictionary for each data type
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
            pass  # Handle or log unexpected data types as needed

    # Creates Numba-compliant typed dict for each type of array
    typed_dict_uint8 = NumbaDict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.uint8, 2, 'C')  # Assuming 2D arrays of uint8
    )

    typed_dict_int16 = NumbaDict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int16, 2, 'C')  # Assuming 2D arrays of int16
    )

    typed_dict_int32 = NumbaDict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.int32, 2, 'C')  # Assuming 2D arrays of int32
    )

    typed_dict_float32 = NumbaDict.empty(
        key_type=types.unicode_type,
        value_type=types.Array(types.float32, 2, 'C')  # Assuming 2D arrays of float32
    )

    # Populates the Numba-compliant typed dicts
    for key, array in uint8_dict_layers.items():
        typed_dict_uint8[key] = array

    for key, array in int16_dict_layers.items():
        typed_dict_int16[key] = array

    for key, array in int32_dict_layers.items():
        typed_dict_int32[key] = array

    for key, array in float32_dict_layers.items():
        typed_dict_float32[key] = array

    return typed_dict_uint8, typed_dict_int16, typed_dict_int32, typed_dict_float32


def convert_lookup_table_to_array(spreadsheet: str, sheet_name: str, fields_to_keep: List[str]) -> np.ndarray:
    """
    Creates a numpy array of rates or ratios from a tab in an Excel spreadsheet.

    Args:
        spreadsheet (str): URL or path to the Excel spreadsheet.
        sheet_name (str): Name of the sheet to read.
        fields_to_keep (List[str]): List of column names to retain.

    Returns:
        np.ndarray: Numpy array of the filtered data.
    """
    response = requests.get(spreadsheet)
    response.raise_for_status()  # Ensure we notice bad responses

    excel_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    # Retains only the relevant columns
    filtered_data = excel_df[fields_to_keep]

    # Converts from dataframe to Numpy array
    filtered_array = filtered_data.to_numpy().astype(float)  # Convert to float for Numba compatibility

    return filtered_array


def complete_inputs(existing_input_list: List[str], typed_dict: NumbaDict, datatype: np.dtype, chunk_length_pixels: int,
                    bounds_str: str, tile_id: str, is_final: bool, logger) -> NumbaDict:
    """
    Creates arrays of 0s for any missing inputs and adds them to the corresponding typed dictionary.

    Args:
        existing_input_list (List[str]): List of existing input layer names.
        typed_dict (NumbaDict): Typed dictionary for a specific data type.
        datatype (np.dtype): Numpy data type for the missing arrays.
        chunk_length_pixels (int): Length of the chunk in pixels.
        bounds_str (str): String representation of chunk bounds.
        tile_id (str): Tile ID.
        is_final (bool): Flag indicating if this is a final stage.
        logger (logging.Logger): Logger instance.

    Returns:
        NumbaDict: Updated typed dictionary with missing inputs filled with zeros.
    """
    for dataset_name in existing_input_list:
        if dataset_name not in typed_dict.keys():
            typed_dict[dataset_name] = np.full((chunk_length_pixels, chunk_length_pixels), 0, dtype=datatype)
            print_and_log(f"Created {dataset_name} for chunk {bounds_str} in {tile_id}: {timestr()}", is_final, logger)
    return typed_dict


def calculate_stats(array: np.ndarray, name: str, bounds_str: str, tile_id: str, in_out: str) -> dict:
    """
    Calculates statistics for a chunk (numpy array).

    Args:
        array (np.ndarray): The data array.
        name (str): Layer name.
        bounds_str (str): String representation of chunk bounds.
        tile_id (str): Tile ID.
        in_out (str): Indicates if the layer is input or output.

    Returns:
        dict: Dictionary containing calculated statistics.
    """
    if array is None or not np.any(array):  # Check if the array is None or empty
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'data_type': 'no data'
        }
    else:  # Only calculates stats if there is data in the array
        return {
            'chunk_id': bounds_str,
            'tile_id': tile_id,
            'layer_name': name,
            'in_out': in_out,
            'min_value': np.min(array),
            'mean_value': np.mean(array),
            'max_value': np.max(array),
            'data_type': array.dtype.name
        }


def calculate_chunk_stats(all_stats: List[dict], stage: str):
    """
    Calculates chunk-level stats for all inputs and outputs and saves to Excel spreadsheet.
    Also calculates the min and max value for each input and output across all chunks.

    Args:
        all_stats (List[dict]): List of statistics dictionaries for each chunk.
        stage (str): Current stage of processing.
    """
    # Convert accumulated statistics to a DataFrame
    df_all_stats = pd.DataFrame(all_stats)
    sorted_stats = df_all_stats.sort_values(by=['in_out', 'layer_name']).reset_index(drop=True)

    # Calculate the min and max values for each layer_name
    min_max_stats = df_all_stats.groupby('layer_name').agg(
        min_value=('min_value', 'min'),
        max_value=('max_value', 'max')
    ).reset_index()

    # Write the combined statistics to a single Excel file
    with pd.ExcelWriter(f'chunk_stats/{stage}_chunk_statistics_{timestr()}.xlsx') as writer:
        sorted_stats.to_excel(writer, sheet_name='chunk_stats', index=False)

        # Write the min and max statistics to the second sheet
        min_max_stats.to_excel(writer, sheet_name='min_max_for_layers', index=False)

    print(sorted_stats.head())  # Show first few rows of the stats DataFrame for inspection

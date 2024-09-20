# utils/chunk_utils.py

import math
import re
import numpy as np
import concurrent.futures
import boto3
from numba import types, Dict
from typing import List, Dict as TypedDict


def get_10x10_tile_bounds(tile_id: str) -> List[float]:
    """
    Gets the W, S, E, N bounds of a 10x10 degree tile based on its ID.

    Args:
        tile_id (str): The tile ID in the format '00N_110E'.

    Returns:
        List[float]: A list containing [min_x, min_y, max_x, max_y].
    """
    if "S" in tile_id:
        max_y = -1 * (int(tile_id[:2]))
        min_y = -1 * (int(tile_id[:2]) + 10)
    else:
        max_y = int(tile_id[:2])
        min_y = int(tile_id[:2]) - 10

    if "W" in tile_id:
        max_x = -1 * (int(tile_id[4:7]) - 10)
        min_x = -1 * (int(tile_id[4:7]))
    else:
        max_x = int(tile_id[4:7]) + 10
        min_x = int(tile_id[4:7])

    return [min_x, min_y, max_x, max_y]  # W, S, E, N


def get_chunk_bounds(chunk_params: List[float], chunk_size: float = 2.0) -> List[List[float]]:
    """
    Returns list of all chunk boundaries within a bounding box for chunks of a given size.

    Args:
        chunk_params (List[float]): The bounding box as [min_x, min_y, max_x, max_y].
        chunk_size (float): Size of each chunk in degrees. Defaults to 2.0.

    Returns:
        List[List[float]]: A list of chunk boundaries, each as [min_x, min_y, max_x, max_y].
    """
    min_x, min_y, max_x, max_y = chunk_params
    x, y = min_x, min_y
    chunks = []

    while y < max_y:
        while x < max_x:
            bounds = [
                x,
                y,
                x + chunk_size,
                y + chunk_size,
            ]
            chunks.append(bounds)
            x += chunk_size
        x = min_x
        y += chunk_size

    return chunks


def boundstr(bounds: List[float]) -> str:
    """
    Converts chunk bounds to a string format.

    Args:
        bounds (List[float]): The bounding box as [min_x, min_y, max_x, max_y].

    Returns:
        str: Formatted bounds string.
    """
    bounds_str = "_".join([str(round(x)) for x in bounds])
    return bounds_str


def calc_chunk_length_pixels(bounds: List[float]) -> int:
    """
    Calculates chunk length in pixels based on bounds.

    Args:
        bounds (List[float]): The bounding box as [min_x, min_y, max_x, max_y].

    Returns:
        int: Chunk length in pixels.
    """
    chunk_length_pixels = int((bounds[3] - bounds[1]) * (40000 / 10))
    return chunk_length_pixels


def xy_to_tile_id(top_left_x: float, top_left_y: float) -> str:
    """
    Returns the encompassing tile_id string in the form YYN/S_XXXE/W based on a coordinate.

    Args:
        top_left_x (float): X-coordinate of the top-left corner.
        top_left_y (float): Y-coordinate of the top-left corner.

    Returns:
        str: Tile ID string.
    """
    lat_ceil = math.ceil(top_left_y / 10.0) * 10
    lng_floor = math.floor(top_left_x / 10.0) * 10

    lng = f"{str(lng_floor).zfill(3)}E" if (lng_floor >= 0) else f"{str(-lng_floor).zfill(3)}W"
    lat = f"{str(lat_ceil).zfill(2)}N" if (lat_ceil >= 0) else f"{str(-lat_ceil).zfill(2)}S"

    return f"{lat}_{lng}"


def flatten_list(nested_list: List[List[dict]]) -> List[dict]:
    """
    Flattens a nested list of dictionaries into a single list.

    Args:
        nested_list (List[List[dict]]): Nested list to flatten.

    Returns:
        List[dict]: Flattened list.
    """
    return [x for xs in nested_list for x in xs]


def prepare_to_download_chunk(bounds: List[float], download_dict: dict, is_final: bool, logger) -> dict:
    """
    Prepares futures for downloading chunks using a ThreadPoolExecutor.

    Args:
        bounds (List[float]): Bounding box for the chunk.
        download_dict (dict): Dictionary of layers to download.
        is_final (bool): Flag indicating if this is a final stage.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Dictionary mapping futures to their corresponding keys.
    """
    futures = {}

    bounds_str = boundstr(bounds)
    tile_id = xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = calc_chunk_length_pixels(bounds)

    # Submit requests to S3 for input chunks but don't actually download them yet.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        from utils.processing_utils import get_tile_dataset_rio  # Avoid circular imports

        logger.info(f"Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}")
        print(f"flm: Requesting data in chunk {bounds_str} in {tile_id}: {timestr()}")

        for key, value in download_dict.items():
            future = executor.submit(get_tile_dataset_rio, value, bounds, chunk_length_pixels)
            futures[future] = key

    return futures


def check_for_tile(download_dict: dict, is_final: bool, logger) -> bool:
    """
    Checks if any of the tiles in the download_dict exist in S3.

    Args:
        download_dict (dict): Dictionary of layers to check.
        is_final (bool): Flag indicating if this is a final stage.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if at least one tile exists, False otherwise.
    """
    s3 = boto3.client('s3')
    tile_id_pattern = r"[0-9]{2}[NS]_[0-9]{3}[EW]"  # Example pattern, adjust as needed

    for value in download_dict.values():
        s3_key = value[15:]
        match = re.findall(tile_id_pattern, value)
        if not match:
            continue
        tile_id = match[0]

        # Check if the tile exists
        try:
            s3.head_object(Bucket='gfw2-data', Key=s3_key)
            logger.info(f"flm: Tile id {tile_id} exists for some inputs. Proceeding: {timestr()} ")
            print(f"flm: Tile id {tile_id} exists for some inputs. Proceeding: {timestr()} ")
            return True
        except boto3.exceptions.S3.ClientError:
            continue

    logger.info(f"flm: Tile id does not exist. Skipping chunk: {timestr()}")
    print(f"flm: Tile id does not exist. Skipping chunk: {timestr()}")
    return False


def check_chunk_for_data(required_layers: dict, item_to_check, bounds_str: str, tile_id: str, any_or_all: str,
                         is_final: bool, logger) -> bool:
    """
    Checks whether a chunk has data in it based on the specified condition.

    Args:
        required_layers (dict): Dictionary of required layers.
        item_to_check: (Unused in the function, consider removing if not needed)
        bounds_str (str): String representation of chunk bounds.
        tile_id (str): Tile ID.
        any_or_all (str): Condition to check ('any' or 'all').
        is_final (bool): Flag indicating if this is a final stage.
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if condition is met, False otherwise.
    """
    if any_or_all == "any":
        for array in required_layers.values():
            if array is None:
                continue
            min_val = np.min(array)
            max_val = np.max(array)
            if min_val != max_val:
                logger.info(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                print(f"flm: Data in chunk {bounds_str}. Proceeding: {timestr()}")
                return True

        logger.info(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        print(f"flm: No data in chunk {bounds_str} for assessed inputs: {timestr()}")
        return False

    elif any_or_all == "all":
        for key, array in required_layers.items():
            if array is None:
                logger.info(f"flm: Chunk {bounds_str} does not exist for {key}. Skipping chunk: {timestr()}")
                print(f"flm: Chunk {bounds_str} does not exist for {key}. Skipping chunk: {timestr()}")
                return False
            min_val = np.min(array)
            max_val = np.max(array)
            if min_val == max_val:
                logger.info(f"flm: Chunk {bounds_str} has no data for {key}. Skipping chunk: {timestr()}")
                print(f"flm: Chunk {bounds_str} has no data for {key}. Skipping chunk: {timestr()}")
                return False

        logger.info(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        print(f"flm: Chunk {bounds_str} has data for all assessed inputs: {timestr()}")
        return True

    else:
        raise ValueError("any_or_all argument not valid")

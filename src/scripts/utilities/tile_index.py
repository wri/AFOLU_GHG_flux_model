"""Utilities for loading the shared 10x10 degree tile index shapefile."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import boto3
import geopandas as gpd

LOGGER = logging.getLogger(__name__)
_SHAPEFILE_EXTENSIONS: Iterable[str] = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def _download_shapefile_from_s3(s3_bucket: str, s3_prefix: str, local_dir: str) -> str:
    """Download a shapefile and its sidecars from S3.

    Parameters
    ----------
    s3_bucket : str
        Name of the S3 bucket that stores the shapefile.
    s3_prefix : str
        Object key prefix without the file extension (e.g., ``path/to/file``).
    local_dir : str
        Directory where the shapefile components should be written.

    Returns
    -------
    str
        Local path to the downloaded ``.shp`` file.
    """

    client = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    base_name = os.path.basename(s3_prefix)
    for ext in _SHAPEFILE_EXTENSIONS:
        key = f"{s3_prefix}{ext}"
        local_path = os.path.join(local_dir, f"{base_name}{ext}")
        if os.path.exists(local_path):
            continue
        try:
            client.download_file(s3_bucket, key, local_path)
        except Exception as exc:  # noqa: BLE001 - bubble up after logging
            LOGGER.warning("Failed to download %s from bucket %s: %s", key, s3_bucket, exc)
            raise
    return os.path.join(local_dir, f"{base_name}.shp")


def load_tile_ids_from_s3(
    s3_bucket_name: str,
    s3_prefix: str,
    local_dir: str,
) -> Optional[List[str]]:
    """Load tile identifiers from the shared 10x10 degree index shapefile.

    Parameters
    ----------
    s3_bucket_name : str
        Bucket containing the shapefile assets.
    s3_prefix : str
        Key prefix without the extension (``path/to/file``).
    local_dir : str
        Local directory used for caching the shapefile components.

    Returns
    -------
    list[str] | None
        Sorted list of ``tile_id`` values if the shapefile is available; ``None`` otherwise.
    """

    try:
        shp_path = _download_shapefile_from_s3(s3_bucket_name, s3_prefix, local_dir)
        gdf = gpd.read_file(shp_path)
    except Exception as exc:  # noqa: BLE001 - convert to warning and return None
        LOGGER.warning("Unable to load tile index shapefile %s: %s", s3_prefix, exc)
        return None

    if "tile_id" not in gdf.columns:
        LOGGER.warning("Tile index shapefile %s is missing a 'tile_id' column", s3_prefix)
        return None

    tile_ids = sorted({str(tile_id) for tile_id in gdf["tile_id"] if tile_id})
    if not tile_ids:
        LOGGER.warning("Tile index shapefile %s did not yield any tile identifiers", s3_prefix)
        return None

    return tile_ids


__all__ = ["load_tile_ids_from_s3"]
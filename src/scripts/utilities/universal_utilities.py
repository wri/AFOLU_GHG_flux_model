"""Runtime helpers required by the organic-soils core model."""

from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
import posixpath
import re
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple, Union

import boto3
import coiled
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from botocore.config import Config
from dask.distributed import Client
from rasterio import open as rio_open
from rasterio.windows import from_bounds

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import log_utilities as lu


_DTYPE_MAP = {
    "Byte": np.uint8,
    "UInt16": np.uint16,
    "Int16": np.int16,
    "UInt32": np.uint32,
    "Int32": np.int32,
    "Float32": np.float32,
    "Float64": np.float64,
}


def timestr() -> str:
    import pytz

    return datetime.now(pytz.timezone("US/Eastern")).strftime("%Y%m%d_%H_%M_%S")


def stage_duration(start: str, end: str, stage: str) -> None:
    fmt = "%Y%m%d_%H_%M_%S"
    print(
        f"Elapsed for {stage}: {datetime.strptime(end, fmt) - datetime.strptime(start, fmt)}"
    )


# ----------------------------------------------------------------------
# tiling helpers
# ----------------------------------------------------------------------


def get_chunk_bounds(
    bounding_box: List[float],
    chunk_size: float,
    *,
    as_polygons: bool = False,
) -> List[Union[List[float], "Polygon"]]:
    """Subdivide *bounding_box* into square chunks.

    Parameters
    ----------
    bounding_box : list[float]
        Bounds in ``[W, S, E, N]`` order.
    chunk_size : float
        Size of each chunk in degrees.
    as_polygons : bool, optional
        If ``True`` return :class:`shapely.geometry.Polygon` objects,
        otherwise return numeric bounds.  Defaults to ``False``.

    Returns
    -------
    list
        Sequence of chunk bounds or polygons.
    """
    min_x, min_y, max_x, max_y = bounding_box
    chunks = []
    y = min_y
    while y < max_y:
        x = min_x
        while x < max_x:
            if as_polygons:
                try:
                    from shapely.geometry import box
                except Exception as exc:  # pragma: no cover - import guard
                    raise ImportError(
                        "shapely is required when as_polygons is True"
                    ) from exc
                chunks.append(box(x, y, x + chunk_size, y + chunk_size))
            else:
                chunks.append([x, y, x + chunk_size, y + chunk_size])
            x += chunk_size
        y += chunk_size
    return chunks


def get_10x10_tile_bounds(tile_id: str) -> Tuple[float, float, float, float]:
    """
    Convert a tile_id like '02N_010E' or '10S_050W' to (W,S,E,N) bounds.
    """
    lat_val = int(tile_id[:2])
    lon_val = int(tile_id[4:7])
    south = -lat_val - 10 if "S" in tile_id else lat_val - 10
    north = -lat_val if "S" in tile_id else lat_val
    west = -lon_val if "W" in tile_id else lon_val
    east = west + 10
    return west, south, east, north


def boundstr(bounds: List[float]) -> str:
    return "_".join(str(round(x)) for x in bounds)


def calc_chunk_length_pixels(bounds: List[float]) -> int:
    pixels_per_degree = cn.full_raster_dims / 10
    width_float = (bounds[2] - bounds[0]) * pixels_per_degree
    height_float = (bounds[3] - bounds[1]) * pixels_per_degree
    width = int(round(width_float))
    height = int(round(height_float))
    tolerance = 1e-6
    if (
        width <= 0
        or height <= 0
        or width != height
        or abs(width_float - width) > tolerance
        or abs(height_float - height) > tolerance
    ):
        raise ValueError(
            "Chunk bounds must define a positive square aligned to the "
            f"{1 / pixels_per_degree}-degree model grid: {bounds}."
        )
    return height


def xy_to_tile_id(x: float, y: float) -> str:
    lat = math.ceil(y / 10.0) * 10
    lon = math.floor(x / 10.0) * 10
    lat_part = f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}"
    lon_part = f"{abs(lon):03d}{'E' if lon >= 0 else 'W'}"
    return f"{lat_part}_{lon_part}"


# Creates list of bounding boxes for chunks from a dataframe column structured as W_S_E_N.
# Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]


def process_chunk_id(chunk_id):
    # Split by underscore
    bounding_box = list(map(float, chunk_id.split('_')))
    return bounding_box


# Creates the list of chunks to process given an approach: a bounding box or a shapefile attribute table


def create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger):

    # Makes list of chunks to analyze from the bounding box and chunk size (deg)
    # Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    if bounding_box and chunk_size_deg:

        chunk_size_pixels = int(round(cn.full_raster_dims * chunk_size_deg / 10))

        main_logger.info("Using bounding box and chunk size to determine chunks")
        main_logger.info(f"Chunk source: Bounding box {bounding_box} (W, S, E, N)")
        main_logger.info(f"Chunk size: {chunk_size_deg} degree, {chunk_size_pixels} pixels")
        chunk_list = get_chunk_bounds_from_bounding_box(bounding_box, chunk_size_deg)


    # Makes list of chunks to analyze from an attribute table of a shapefile of 1x1 degree chunks.
    # Attribute table column must be formatted as W_S_E_N.
    # Output list form is [[115.25, -3.75, 115.5, -3.5], [...], [...], ...]
    elif chunk_shapefile_uri:

        chunk_size_pixels = int(cn.full_raster_dims * 1/10)

        main_logger.info("Using chunk list shapefile (and optional number of test chunks) to determine 1x1 deg chunks")
        main_logger.info(f"Chunk source: 1x1 degree tile index shapefile {chunk_shapefile_uri}")
        main_logger.info(f"Chunk size: 1 degree, {chunk_size_pixels} pixels")

        fishnet_1x1_chunk_id_df = fishnet_iso_df[['chunk_id']]  # Creates dataframe

        # If argument for number of chunks in shapefile is supplied, limit to that
        if first_chunks:
            fishnet_1x1_chunk_id_df = fishnet_1x1_chunk_id_df[:first_chunks]

        # Converts dataframe column of chunk bounds to nested list
        chunk_list = fishnet_1x1_chunk_id_df['chunk_id'].apply(process_chunk_id).tolist()

    else:
        main_logger.info("Chunk list cannot be determined")
        sys.exit()

    return chunk_list, chunk_size_pixels


# Creates a dataframe from the attribute table of the 1x1 deg fishnet with GADM iso joined to it


def fishnet_with_GADM_iso(shapefile_uri):

    # Reads the 1x1 deg fishnet with GADM iso joined from S3 to extract "chunk_id" and "iso" fields
    gdf = gpd.read_file(shapefile_uri)

    # Creates a DataFrame of the 1x1def fishnet with "chunk_id" and "iso" fields
    fishnet_df = gdf[['chunk_id', 'iso']]

    return fishnet_df


# ----------------------------------------------------------------------
# Coiled / Dask
# ----------------------------------------------------------------------


def upload_repo_source_to_dask(client: Client) -> None:
    """Upload this repo's ``src`` package so remote workers can import it."""

    import tempfile
    import zipfile
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    if not src_dir.exists():
        print(f"Local source directory not found; skipping Dask source upload: {src_dir}")
        return

    zip_path = Path(tempfile.gettempdir()) / "afolu_ghg_flux_model_src.zip"
    newest_source_mtime = max(
        p.stat().st_mtime
        for p in src_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    if not zip_path.exists() or zip_path.stat().st_mtime < newest_source_mtime:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in src_dir.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                zf.write(path, path.relative_to(repo_root).as_posix())

    client.upload_file(str(zip_path), load=True)
    print(f"Uploaded repo source to Dask workers: {zip_path}")


def connect_to_cluster(
    cluster_name: str = "afolu_cluster",
    n_workers: int = 20,
    region: str = "us-east-1",
    run_local: bool = False,
    worker_memory: str = "32GiB",
):
    """Connect to a running Coiled cluster or run locally.

    Parameters
    ----------
    cluster_name : str
        Name of the cluster to connect to.
    n_workers : int
        Unused. Kept for backwards compatibility.
    region : str
        AWS region for the cluster (unused).
    run_local : bool
        If ``True`` run locally without Dask/Coiled.
    worker_memory : str
        Unused. Kept for backwards compatibility.
    """

    if run_local:
        print("Running locally without Dask/Coiled.")
        return None, None, run_local

    wait_seconds = int(os.environ.get("AFOLU_CLUSTER_READY_WAIT_SECONDS", "0"))
    deadline = time.time() + wait_seconds
    while True:
        all_clusters = coiled.list_clusters(workspace=cn.Coiled_workspace)
        for info in all_clusters:
            if info.get("name") == cluster_name and info.get("current_state", {}).get("state") == "ready":
                print(f"Connecting to running cluster '{cluster_name}'.")
                cluster = coiled.Cluster(name=cluster_name, workspace=cn.Coiled_workspace, shutdown_on_close=False)
                client = Client(cluster)
                upload_repo_source_to_dask(client)
                return cluster, client, run_local

        if time.time() >= deadline:
            break
        print(f"Waiting for Coiled cluster '{cluster_name}' to be ready...")
        time.sleep(30)

    require_cluster = os.environ.get("AFOLU_REQUIRE_CLUSTER", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if require_cluster:
        raise RuntimeError(
            f"Coiled cluster {cluster_name!r} was not ready in workspace "
            f"{cn.Coiled_workspace!r}."
        )
    print(f"Cluster named {cluster_name} not found. Running locally.")
    run_local = True
    return None, None, run_local


# ----------------------------------------------------------------------
# Compatibility alias
# ----------------------------------------------------------------------


def upload_raster_to_s3(file_path: str, bucket: str, s3_key: str) -> None:
    """Upload a raster from ``file_path`` to S3 and remove the local file."""
    s3c = boto3.client("s3")
    try:
        s3c.upload_file(file_path, bucket, s3_key)
    except Exception as exc:  # pragma: no cover - network issues
        raise RuntimeError(f"Upload failed for s3://{bucket}/{s3_key}: {exc}") from exc
    os.remove(file_path)


def list_raster_full_paths_in_s3_folder_and_count(s3_path: str) -> tuple:
    """Return list of GeoTIFF paths under ``s3_path`` and a count."""
    s3c = boto3.client("s3")
    if s3_path.startswith("s3://"):
        parts = s3_path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
    else:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    paginator = s3c.get_paginator("list_objects_v2")
    geotiff_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".tif") or key.endswith(".tiff"):
                geotiff_files.append(f"s3://{bucket}/{key}")
    return geotiff_files, len(geotiff_files)


def split_s3_path(s3_path: str) -> tuple:
    """Split ``s3://bucket/key`` into (bucket, key)."""
    s3_path = s3_path.replace("s3://", "")
    return tuple(s3_path.split("/", 1))


# ----------------------------------------------------------------------
# S3 / raster read helpers
# ----------------------------------------------------------------------


class RequiredInputRasterError(RuntimeError):
    """Raised when a model input that must exist cannot be read."""


def tile_id_set_sha256(tile_ids: Iterable[str]) -> str:
    """Return a stable SHA-256 fingerprint for a tile-ID set."""

    payload = "\n".join(sorted(set(tile_ids))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_tile_set_fingerprint(
    tile_ids: Iterable[str],
    *,
    expected_count: int,
    expected_sha256: str,
    layer_name: str,
) -> set[str]:
    """Return a tile set only if its expected count and fingerprint match."""

    resolved = set(tile_ids)
    fingerprint = tile_id_set_sha256(resolved)
    if len(resolved) != expected_count or fingerprint != expected_sha256:
        raise RequiredInputRasterError(
            f"{layer_name} footprint drifted from the expected reference: "
            f"count={len(resolved)} (expected {expected_count}), "
            f"sha256={fingerprint} (expected {expected_sha256})."
        )
    return resolved


def list_existing_s3_tile_ids(
    uri_template: str,
    tile_ids: Iterable[str],
    *,
    s3_client=None,
) -> set[str]:
    """Return requested tile IDs whose exact objects exist under a template.

    ``uri_template`` must be an S3 URI containing a literal ``{tile_id}``
    placeholder. The containing prefix is listed once and exact expected keys
    are matched, so similarly named objects cannot satisfy the check.
    """

    marker = "{tile_id}"
    requested_tile_ids = sorted(set(tile_ids))
    if not requested_tile_ids:
        return set()
    if not uri_template.startswith("s3://"):
        raise ValueError(f"Tile input must be an S3 URI: {uri_template}")
    if marker not in uri_template:
        raise ValueError(f"Tile URI must contain {marker}: {uri_template}")

    bucket, key_template = split_s3_path(uri_template)
    prefix = key_template.split(marker, 1)[0]
    client = s3_client or boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
    paginator = client.get_paginator("list_objects_v2")
    object_sizes = {
        item["Key"]: int(item.get("Size", 1))
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    }
    expected_by_key = {
        key_template.replace(marker, tile_id): tile_id
        for tile_id in requested_tile_ids
    }
    empty_tile_ids = sorted(
        tile_id
        for key, tile_id in expected_by_key.items()
        if key in object_sizes and object_sizes[key] <= 0
    )
    if empty_tile_ids:
        examples = ", ".join(empty_tile_ids[:5])
        raise RequiredInputRasterError(
            "Tile prefix contains zero-byte raster objects for "
            f"{len(empty_tile_ids)} requested tiles (examples: {examples}): "
            f"{uri_template}"
        )
    return {
        tile_id
        for key, tile_id in expected_by_key.items()
        if key in object_sizes
    }


def list_existing_s3_tile_ids_for_templates(
    uri_templates: Iterable[str],
    tile_ids: Iterable[str],
    *,
    s3_client=None,
) -> dict[str, set[str]]:
    """Resolve exact tile footprints for several S3 templates with one client."""

    client = s3_client or boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
    requested_tile_ids = set(tile_ids)
    return {
        template: list_existing_s3_tile_ids(
            template,
            requested_tile_ids,
            s3_client=client,
        )
        for template in sorted(set(uri_templates))
    }


def validate_required_s3_tile_coverage(
    uri_template: str,
    tile_ids: Iterable[str],
    *,
    layer_name: str,
    s3_client=None,
) -> int:
    """Require every requested tile object to exist before model execution."""

    requested_tile_ids = sorted(set(tile_ids))
    existing_tile_ids = list_existing_s3_tile_ids(
        uri_template,
        requested_tile_ids,
        s3_client=s3_client,
    )
    missing_tile_ids = sorted(set(requested_tile_ids) - existing_tile_ids)
    if missing_tile_ids:
        examples = ", ".join(
            uri_template.replace("{tile_id}", tile_id)
            for tile_id in missing_tile_ids[:5]
        )
        raise RequiredInputRasterError(
            f"Required {layer_name} coverage is incomplete: "
            f"{len(missing_tile_ids)} of {len(requested_tile_ids)} requested tiles are missing. "
            f"Examples: {examples}"
        )
    return len(requested_tile_ids)


def _dtype(gdal_dtype: str) -> np.dtype:
    return _DTYPE_MAP[gdal_dtype]


def open_window_as_array(
    s3_uri: str,
    gdal_dtype: str,
    bounds: List[float],
    chunk_px: int,
    logger,
    is_final: bool = False,
    required: bool = False,
) -> np.ndarray:
    """
    Read a window from an S3 GeoTIFF using GDAL's /vsis3/ driver.

    * If s3_uri is None (placeholder for a missing optional layer) we return an
      all-zero array of the requested dtype. Required layers fail instead.
    * Credentials are obtained automatically from environment variables
      or ~/.aws/credentials.  No s3fs / AWSSession wrapper is used.
    """
    if s3_uri is None and required:
        raise RequiredInputRasterError("Required raster URI is missing.")
    if s3_uri is None:
        return np.zeros((chunk_px, chunk_px), dtype=_dtype(gdal_dtype))

    vsipath = f"/vsis3/{s3_uri[5:]}"  # strip the 's3://'
    try:
        with rasterio.Env():  # GDAL handles auth internally
            with rio_open(vsipath) as ds:
                arr = ds.read(1, window=from_bounds(*bounds, ds.transform))
        expected_shape = (chunk_px, chunk_px)
        if arr.shape != expected_shape:
            message = (
                f"Raster {s3_uri} returned shape {arr.shape}; expected "
                f"{expected_shape} for bounds {bounds}."
            )
            if required:
                raise RequiredInputRasterError(message)
            logger.warning("WARNING: %s -> zeros", message)
            return np.zeros(expected_shape, dtype=_dtype(gdal_dtype))
        return arr
    except Exception as exc:
        if isinstance(exc, RequiredInputRasterError):
            raise
        if required:
            raise RequiredInputRasterError(
                f"Required raster {s3_uri} could not be read: {exc}"
            ) from exc
        logger.warning(f"WARNING: {s3_uri} failed ({exc}) -> zeros")
        return np.zeros((chunk_px, chunk_px), dtype=_dtype(gdal_dtype))


def get_tile_dataset_rio(
    uri: str,
    dtype_str: str,
    bounds: list,
    chunk_px: int,
    is_final: bool,
    logger,
    required: bool = False,
) -> tuple:
    """Return array window from *uri* using rasterio.

    Parameters
    ----------
    uri : str
        Local or S3 path to GeoTIFF.
    dtype_str : str
        GDAL datatype name like ``"Float32"``.
    bounds : list
        Window bounds in ``[W, S, E, N]`` order.
    chunk_px : int
        Output array size in pixels (width/height).
    is_final : bool
        If ``True`` log at info level; otherwise debug.
    logger : logging.Logger

    Returns
    -------
    tuple
        ``(array, True)`` if successful else ``(zeros, False)``.
    """

    dtype = _dtype(dtype_str)
    if uri is None and required:
        raise RequiredInputRasterError("Required raster URI is missing.")
    if uri is None:
        return np.zeros((chunk_px, chunk_px), dtype=dtype), False

    path = uri
    if uri.startswith("s3://"):
        path = f"/vsis3/{uri[5:]}"

    try:
        with rasterio.Env():
            with rasterio.open(path) as ds:
                arr = ds.read(1, window=from_bounds(*bounds, ds.transform))
        expected_shape = (chunk_px, chunk_px)
        if arr.shape != expected_shape:
            raise ValueError(
                f"Raster returned shape {arr.shape}; expected {expected_shape} "
                f"for bounds {bounds}."
            )
        return arr.astype(dtype), True
    except Exception as exc:  # pragma: no cover - network/gdal issues
        if required:
            raise RequiredInputRasterError(
                f"Required raster {uri} could not be read: {exc}"
            ) from exc
        lu.print_and_log(f"WARNING: {uri} failed ({exc}) -> zeros", is_final, logger)
        return np.zeros((chunk_px, chunk_px), dtype=dtype), False


def queue_chunk_downloads(
    bounds,
    typed_dict,
    chunk_px,
    logger,
    max_threads=16,
    is_final=False,
    required_layers=None,
):
    required_layers = set(required_layers or ())
    futs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as ex:
        for k, (uri, dt) in typed_dict.items():
            futs[
                ex.submit(
                    open_window_as_array,
                    uri,
                    dt,
                    bounds,
                    chunk_px,
                    logger,
                    is_final,
                    k in required_layers,
                )
            ] = k
    return futs


# ----------------------------------------------------------------------
# tile existence quick‑check
# ----------------------------------------------------------------------


def check_for_tile(typed_dict, is_final, logger) -> bool:
    s3c = boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
    for uri, _ in typed_dict.values():
        if uri is None:
            continue
        key = uri.removeprefix("s3://gfw2-data/")
        tid_match = re.findall(cn.tile_id_pattern, uri)
        tid = tid_match[0] if tid_match else "unknown"
        try:
            s3c.head_object(Bucket="gfw2-data", Key=key)
            lu.print_and_log(f"Tile {tid} exists, continue", is_final, logger)
            return True
        except Exception:
            pass
    lu.print_and_log("Tile not found, skip chunk", is_final, logger)
    return False


# ----------------------------------------------------------------------
# stats helpers
# ----------------------------------------------------------------------


def calculate_stats(
    arr: np.ndarray | None,
    name,
    bstr,
    tid,
    in_out,
    array_per_pixel: np.ndarray | None = None,
    iv_start: int | None = None,
    iv_end: int | None = None,
):
    """Return per-chunk statistics for ``arr``.

    The returned dictionary contains ``chunk_id`` (``bstr``), ``tile_id``,
    ``layer_name`` (``name``), the extracted ``years`` value, ``in_out`` and the
    basic statistics (``min_value``, ``mean_value``, ``max_value``,
    ``count_value``, ``sum_value`` and ``data_type``). ``years`` is derived from
    :func:`strip_and_extract_years` with static layers reported as ``static`` or
    an interval ``iv_start_iv_end`` when provided.
    """

    _, extracted_year = strip_and_extract_years(name)

    # Explicitly handle static (no year) layers
    if extracted_year == "no year range":
        year_range = f"{iv_start}_{iv_end}" if iv_start and iv_end else "static"
    else:
        year_range = extracted_year

    if arr is None or arr.size == 0 or not np.any(arr):
        return dict(
            chunk_id=bstr,
            tile_id=tid,
            layer_name=name,
            years=year_range,
            in_out=in_out,
            min_value="no data",
            mean_value="no data",
            max_value="no data",
            count_value="no data",
            sum_value="no data",
            data_type="no data",
        )

    sum_value = (
        float(np.sum(array_per_pixel))
        if array_per_pixel is not None and in_out == "output_layer"
        else "N/A- input layer or no per-pixel array supplied"
    )

    return dict(
        chunk_id=bstr,
        tile_id=tid,
        layer_name=name,
        years=year_range,
        in_out=in_out,
        min_value=float(arr.min()),
        mean_value=float(arr.mean()),
        max_value=float(arr.max()),
        count_value=int(np.count_nonzero(arr)),
        sum_value=sum_value,
        data_type=str(arr.dtype),
    )


def build_output_s3_folder(
    data_meaning: str,
    year_out: str,
    chunk_px: int,
    interval_type: str,
    model_type: str = "standard_model",
    date_str: str | None = None,
) -> str:
    date_str = date_str or cn.today_date
    return posixpath.join(
        cn.outputs_path,
        data_meaning,
        model_type,
        f"{interval_type}_intervals",
        year_out,
        f"{chunk_px}_pixels",
        date_str,
    )


def save_and_upload_small_raster_set(
    bounds,
    chunk_px,
    tile_id,
    bstr,
    out_dict,
    is_final,
    logger,
    interval_type,
    model_type="standard_model",
    no_data_val=None,
    date_str=None,
):
    """Save arrays as GeoTIFFs locally and return upload tasks.

    Each entry of ``out_dict`` should be ``key: (array, dtype, data_meaning,
    year_out)``. The function saves the arrays to ``/tmp`` and returns a list
    of ``(local_path, bucket, key)`` tuples for uploading via
    :func:`upload_raster_to_s3`.
    """

    import tempfile
    from rasterio.transform import from_bounds

    upload_tasks = []

    transform = from_bounds(*bounds, width=chunk_px, height=chunk_px)
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    file_info = f"{tile_id}__{bstr}"
    lu.print_and_log(
        f"Saving outputs locally for {bstr} in {tile_id}: {timestr()}",
        is_final,
        logger,
    )

    for key, (arr, dtype, data_meaning, year_out) in out_dict.items():
        fname = (
            f"{file_info}__{key}__{year_out}.tif" if is_final else f"{file_info}__{key}__{year_out}__{timestr()}.tif"
        )
        lpath = os.path.join(temp_dir, fname)

        profile = dict(
            driver="GTiff",
            width=chunk_px,
            height=chunk_px,
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
            blockxsize=400,
            blockysize=400,
        )
        if no_data_val is not None:
            profile["nodata"] = no_data_val

        with rasterio.open(lpath, "w", **profile) as dst:
            dst.write(arr, 1)

        s3_folder = build_output_s3_folder(
            data_meaning, year_out, chunk_px, interval_type, model_type,
            date_str=date_str,
        )
        s3_key = posixpath.join(s3_folder.removeprefix("s3://gfw2-data/"), fname)
        upload_tasks.append((lpath, "gfw2-data", s3_key))

    return upload_tasks


def save_and_upload_single_raster(
    bounds,
    chunk_px,
    tile_id,
    arr,
    dtype,
    outfile,
    outdir,
    is_final,
    logger,
    no_data_val=None,
):
    """Write *arr* to GeoTIFF and upload to ``outdir`` on S3."""
    import tempfile
    from rasterio.transform import from_bounds

    s3c = boto3.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "standard"})
    )
    transform = from_bounds(*bounds, width=chunk_px, height=chunk_px)
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    fname = outfile if is_final else f"{os.path.splitext(outfile)[0]}__{timestr()}.tif"
    lpath = os.path.join(temp_dir, fname)

    profile = dict(
        driver="GTiff",
        width=chunk_px,
        height=chunk_px,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        blockxsize=400,
        blockysize=400,
    )
    if no_data_val is not None:
        profile["nodata"] = no_data_val

    with rasterio.open(lpath, "w", **profile) as dst:
        dst.write(arr, 1)

    s3_key = posixpath.join(outdir.removeprefix("s3://gfw2-data/"), fname)
    s3c.upload_file(lpath, "gfw2-data", s3_key)
    os.remove(lpath)
    lu.print_and_log(f"uploaded {fname} to {outdir}", is_final, logger)


# ----------------------------------------------------------------------
# helpers required by drainage model
# ----------------------------------------------------------------------


def fill_missing_input_layers_with_no_data(
    layers: Dict[str, np.ndarray],
    uint8_list,
    int16_list,
    int32_list,
    float32_list,
    bstr,
    tid,
    is_final,
    logger,
):
    template = next(iter(layers.values()))
    shp = template.shape
    for k in uint8_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.uint8)
    for k in int16_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.int16)
    for k in int32_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.int32)
    for k in float32_list:
        if k not in layers:
            layers[k] = np.zeros(shp, dtype=np.float32)
    return layers


# dtype sniffing helpers -----------------------------------------------------


def first_file_name_in_s3_folder(download_dict):
    s3 = boto3.client("s3")
    first_tiles = {}
    sample_tid = "00N_110E"
    for k, p in download_dict.items():
        dir_path = os.path.dirname(p.replace("{tile_id}", sample_tid))
        bucket, prefix = dir_path[5:].split("/", 1)
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
        if "Contents" in resp and resp["Contents"]:
            first_tiles[k] = f"s3://{bucket}/{resp['Contents'][0]['Key']}"
        else:
            first_tiles[k] = None
    return first_tiles


def get_dtype_from_s3(file_path):
    from osgeo import gdal

    vsipath = f"/vsis3/{file_path[5:]}"
    ds = gdal.Open(vsipath)
    if not ds:
        return None
    return gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)


def add_file_type_to_dict(first_tiles):
    out = {}
    for k, fp in first_tiles.items():
        if fp is None:
            continue
        dt = get_dtype_from_s3(fp)
        if dt:
            out[k] = [fp, dt]
    return out


def get_cluster_info(client, cluster):

    # Retrieves properties of the workers
    workers = client.scheduler_info()["workers"]

    # Retrieves the number of workers
    n_workers = len(workers)

    # Retrieves the number of threads per worker
    first_worker_address = next(iter(workers.keys()))
    nthreads = workers[first_worker_address]["nthreads"]

    # Retrieves scheduler info for other cluster properties
    scheduler_info = (
        cluster.scheduler_info
    )  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info["workers"][
            next(iter(scheduler_info["workers"]))
        ]["memory_limit"]
        worker_memory_gb = worker_memory_bytes / (1024**3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    return worker_memory, n_workers, nthreads

# Returns list of all chunk boundaries within a bounding box for chunks of a given size


def get_chunk_bounds_from_bounding_box(bounding_box, chunk_size):
    min_x = bounding_box[0]
    min_y = bounding_box[1]
    max_x = bounding_box[2]
    max_y = bounding_box[3]

    x, y = (min_x, min_y)
    chunks = []

    # Polygon Size
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


# ----------------------------------------------------------------------
# task tracking helpers (ported from LULUCF utilities)
# ----------------------------------------------------------------------


def count_successful_chunks(chunk_list, is_final, main_logger, results):
    """Summarize success messages returned from chunk workers."""

    all_stats = []
    return_messages = []
    success_count = 0
    skipping_chunk_count = 0
    error_chunk_count = 0
    other_message_count = 0
    for result in results:
        return_message, chunk_stats = result
        if "Success" in return_message:
            success_count += 1
        elif "Skipped chunk" in return_message:
            skipping_chunk_count += 1
        elif "Error" in return_message:
            error_chunk_count += 1
        else:
            other_message_count += 1
        if return_message:
            return_messages.append(return_message)
        chunk_stats = chunk_stats if isinstance(chunk_stats, list) else [chunk_stats]
        if chunk_stats is not None:
            all_stats.extend(chunk_stats)
    if not is_final:
        for message in return_messages:
            main_logger.info(message)
    main_logger.info(f"Number of 'Success' chunks: {success_count}")
    main_logger.info(f"Number of 'Skipped' chunks: {skipping_chunk_count}")
    main_logger.info(f"Number of 'Error' chunks: {error_chunk_count}")
    main_logger.info(f"Number of 'Other message' chunks: {other_message_count}")
    if return_messages and "Success merging" not in return_messages[0]:
        diff = len(chunk_list) - (
            success_count + skipping_chunk_count + error_chunk_count + other_message_count
        )
        main_logger.info(
            f"Difference between submitted chunks and processed chunks: {diff}"
        )
    main_logger.info("\n")
    return success_count, all_stats


def compile_1x1_chunk_stats(
    all_1x1_stats,
    chunk_shapefile_uri,
    stage,
    no_upload,
    main_logger,
    run_name="standard_model",
    run_date=None,
):
    """Aggregate per-chunk statistics into structured categories and upload to S3.

    Parameters
    ----------
    run_name : str, optional
        Model run identifier used to label output paths. Defaults to
        ``"standard_model"`` for backward compatibility.
    run_date : str, optional
        Date string used to mirror model output paths. Defaults to
        ``today_date`` for backward compatibility.
    """

    import boto3
    import pandas as pd
    import geopandas as gpd
    import os
    import posixpath
    from src.scripts.utilities.constants_and_names import (
        local_chunk_stats_path,
        short_bucket_prefix,
        full_bucket_prefix,
        burned_area_final_pattern,
        land_cover_pattern,
        outputs_path,
        today_date,
    )
    from src.scripts.utilities.universal_utilities import timestr

    s3_client = boto3.client("s3")
    main_logger.info(f"Starting stats aggregation: {timestr()}")

    # Create DataFrame from stats
    df_stats = pd.DataFrame(all_1x1_stats)
    df_stats["min_value"] = pd.to_numeric(df_stats["min_value"], errors="coerce")
    df_stats["max_value"] = pd.to_numeric(df_stats["max_value"], errors="coerce")
    df_stats["mean_value"] = pd.to_numeric(df_stats["mean_value"], errors="coerce")

    # Drop rows with no data to reduce Excel file size
    df_stats.dropna(subset=["min_value", "max_value", "mean_value"], how="all", inplace=True)
    df_stats = df_stats[~((df_stats["min_value"] == 0) & (df_stats["max_value"] == 0) & (df_stats["mean_value"] == 0))]

    # Read fishnet for ISO codes
    gdf_chunks = gpd.read_file(chunk_shapefile_uri)[["chunk_id", "iso"]]

    # Merge ISO codes
    df_stats = df_stats.merge(gdf_chunks, on="chunk_id", how="left")
    df_stats["iso"] = df_stats["iso"].fillna("no iso assigned")

    # Separate input and output layers explicitly
    input_layers = df_stats[df_stats["in_out"] == "input_layer"].copy()
    output_layers = df_stats[df_stats["in_out"] == "output_layer"].copy()

    # Explicitly handle annual timeseries inputs separately
    timeseries_layers = f"{burned_area_final_pattern}|{land_cover_pattern}"
    annual_inputs = input_layers[input_layers["layer_name"].str.contains(timeseries_layers, case=False, na=False)]
    other_inputs = input_layers[~input_layers["layer_name"].str.contains(timeseries_layers, case=False, na=False)]

    # Categorize outputs using naming conventions similar to the LULUCF model
    gross_flux_outputs = output_layers[output_layers["layer_name"].str.contains("gross", case=False, na=False)]
    net_flux_outputs = output_layers[output_layers["layer_name"].str.contains("net|flux", case=False, na=False)]
    other_outputs = output_layers[~output_layers["layer_name"].str.contains("flux|gross|net", case=False, na=False)]

    # Summarize min-max per layer
    min_max_summary = df_stats.groupby("layer_name").agg(
        min_value=("min_value", "min"),
        max_value=("max_value", "max"),
        count=("layer_name", "count"),
    ).reset_index()

    # Pixel count validation (aggregation from 1x1 to 10x10 tiles)
    if "count_value" in output_layers.columns:
        output_layers["count_value"] = pd.to_numeric(output_layers["count_value"], errors="coerce").fillna(0)
        pixel_counts = (
            output_layers.groupby(["tile_id", "layer_name"])["count_value"]
            .sum()
            .reset_index()
            .rename(columns={"count_value": "total_pixel_count"})
        )
        pixel_counts["tile_layer"] = pixel_counts["tile_id"] + "__" + pixel_counts["layer_name"] + ".tif"
    else:
        pixel_counts = pd.DataFrame(columns=["tile_id", "layer_name", "tile_layer", "total_pixel_count"])

    # Excel spreadsheet creation
    out_spreadsheet = f"{stage}_1x1_chunk_statistics_{timestr()}.xlsx"

    # Build informative local and S3 directories mirroring raster outputs
    stats_date = run_date or today_date
    local_dir = os.path.join(local_chunk_stats_path, run_name, stats_date)
    s3_dir = posixpath.join(
        outputs_path.removeprefix("s3://gfw2-data/"),
        "chunk_stats",
        run_name,
        stats_date,
    )

    os.makedirs(local_dir, exist_ok=True)
    local_spreadsheet = os.path.join(local_dir, out_spreadsheet)

    main_logger.info(f"Writing stats to Excel: {timestr()}")
    with pd.ExcelWriter(local_spreadsheet) as writer:
        # Inputs
        annual_inputs.to_excel(writer, sheet_name="annual_1x1_inputs", index=False)
        other_inputs.to_excel(writer, sheet_name="other_1x1_inputs", index=False)

        # Outputs grouped like the LULUCF workflow
        gross_flux_outputs.to_excel(writer, sheet_name="gross_outputs_1x1", index=False)
        net_flux_outputs.to_excel(writer, sheet_name="net_outputs_1x1", index=False)
        other_outputs.to_excel(writer, sheet_name="other_outputs_1x1", index=False)

        # Summary stats
        min_max_summary.to_excel(writer, sheet_name="min_max_summary", index=False)
        pixel_counts.to_excel(writer, sheet_name="pixel_counts_summary", index=False)

    main_logger.info(f"Excel file written: {local_spreadsheet} ({timestr()})")

    # Upload to S3 if required
    if not no_upload:
        s3_client.upload_file(
            local_spreadsheet,
            short_bucket_prefix,
            Key=posixpath.join(s3_dir, out_spreadsheet),
        )
        main_logger.info(
            f"Uploaded to {full_bucket_prefix}/{posixpath.join(s3_dir, out_spreadsheet)}: {timestr()}"
        )


def aggregate_10x10_chunk_stats(
    all_10x10_stats,
    stage,
    no_upload,
    main_logger,
    run_name="standard_model",
    run_date=None,
):
    """Write aggregated 10x10 pixel counts to an Excel spreadsheet."""

    s3_client = boto3.client("s3")
    main_logger.info(f"Starting to aggregate and export tile stats: {timestr()}")

    df_all_10x10_stats = pd.DataFrame(all_10x10_stats)

    out_spreadsheet = f"{stage}_10x10_chunk_statistics_{timestr()}.xlsx"
    stats_date = run_date or cn.today_date
    local_dir = os.path.join(cn.local_chunk_stats_path, f"{run_name}_10", stats_date)
    s3_dir = posixpath.join(
        cn.outputs_path.removeprefix("s3://gfw2-data/"),
        "chunk_stats",
        f"{run_name}_10",
        stats_date,
    )
    os.makedirs(local_dir, exist_ok=True)
    local_spreadsheet = os.path.join(local_dir, out_spreadsheet)

    main_logger.info(f"Writing tile stats to spreadsheet: {timestr()}")
    with pd.ExcelWriter(local_spreadsheet) as writer:
        df_all_10x10_stats.to_excel(writer, sheet_name="pix_counts_compa_10x10_1x1", index=False)

    main_logger.info(df_all_10x10_stats.head())

    if not no_upload:
        s3_client.upload_file(
            local_spreadsheet,
            cn.short_bucket_prefix,
            Key=posixpath.join(s3_dir, out_spreadsheet),
        )
        main_logger.info(
            f"Uploaded to {cn.full_bucket_prefix}/{posixpath.join(s3_dir, out_spreadsheet)}: {timestr()}"
        )


def strip_and_extract_years(key: str) -> tuple:
    """Return file pattern and year range extracted from *key*."""

    pattern = re.sub(cn.date_date_range_pattern, "", key)
    m = re.search(cn.date_date_range_pattern, key)
    year_range = m.group()[1:] if m else "no year range"
    return pattern, year_range

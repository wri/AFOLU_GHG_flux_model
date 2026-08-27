from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import dask.array as da
import fsspec
import numpy as np
import xarray as xr
import zarr
from zarr.storage import FsspecStore

from src.scripts.utilities import constants_and_names as cn


GLOBAL_WEST = -180.0
GLOBAL_NORTH = 90.0


def pixel_resolution() -> float:
    return 10.0 / cn.full_raster_dims


def global_grid_shape() -> tuple[int, int]:
    res = pixel_resolution()
    width = int(round(360.0 / res))
    height = int(round(180.0 / res))
    return height, width


@lru_cache(maxsize=1)
def global_coords() -> tuple[np.ndarray, np.ndarray, float]:
    res = pixel_resolution()
    height, width = global_grid_shape()
    # Global pixel-center coordinates must remain float64. At longitudes near
    # +/-180 degrees, float32 cannot represent the 0.00025-degree grid spacing
    # consistently and produces an apparent first-pixel step of
    # 0.0002593994140625 degrees.
    x = GLOBAL_WEST + res * (0.5 + np.arange(width, dtype=np.float64))
    y = GLOBAL_NORTH - res * (0.5 + np.arange(height, dtype=np.float64))
    return x, y, res


def bounds_to_indices(bounds: Sequence[float]) -> tuple[int, int, int, int]:
    west, south, east, north = bounds
    _, _, res = global_coords()
    x0 = int(round((west - GLOBAL_WEST) / res))
    x1 = int(round((east - GLOBAL_WEST) / res))
    y0 = int(round((GLOBAL_NORTH - north) / res))
    y1 = int(round((GLOBAL_NORTH - south) / res))
    return y0, y1, x0, x1


def create_mega_zarr_path(
    base_path: str,
    chunk_size_pixels: int,
    interval_type: str,
    model_type: str,
    run_date: str,
    logger,
) -> str:
    path = "/".join(
        part.strip("/")
        for part in (
            base_path,
            model_type,
            interval_type,
            f"{chunk_size_pixels}_pixels",
            run_date,
            "mega.zarr",
        )
        if part
    )
    logger.info("mega-zarr path: %s", path)
    return path


def make_zarr_store(zarr_path: str, *, read_only: bool = False):
    """Return a zarr v3-compatible store for local or S3 zarr paths."""

    if zarr_path.startswith("s3://"):
        fs = fsspec.filesystem("s3", anon=False, asynchronous=True)
        path = zarr_path.removeprefix("s3://").rstrip("/")
        return FsspecStore(fs, read_only=read_only, path=path)
    return zarr_path


def open_mega_zarr_dataset(zarr_path: str) -> xr.Dataset:
    """Open a drainage mega-zarr using the store API expected by zarr v3."""

    return xr.open_zarr(
        make_zarr_store(zarr_path, read_only=True),
        consolidated=False,
    )


def initialize_global_mega_zarr(
    zarr_path: str,
    outputs_to_zarr: Iterable[str],
    years: Sequence[int],
    chunk_size_pixels: int,
    interval_type: str,
    logger,
) -> None:
    logger.info("Initializing drainage mega-zarr: %s", zarr_path)
    store = make_zarr_store(zarr_path)

    outputs = list(outputs_to_zarr)
    x, y, _ = global_coords()
    year_data = np.asarray(years, dtype=np.int32)
    height, width = global_grid_shape()

    data_vars: dict[str, xr.DataArray] = {}
    for name in outputs:
        dtype = cn.drainage_output_dtypes.get(name, "float32")
        dask_data = da.full(
            (len(years), height, width),
            0,
            dtype=dtype,
            chunks=(1, chunk_size_pixels, chunk_size_pixels),
        )
        data_vars[name] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": year_data, "y": y, "x": x},
            name=name,
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"x": x, "y": y, "year": year_data},
        attrs={
            "model": "organic_soils_drainage",
            "interval_type": interval_type,
            "chunk_size_pixels": chunk_size_pixels,
            "run_status": "initialized",
        },
    )

    ds.to_zarr(
        store=store,
        # Creation must never erase an existing run. Updates use the separate
        # update-existing path and write only requested regions.
        mode="w-",
        compute=False,
        zarr_format=3,
    )

    group = zarr.open_group(store=store, mode="r+")
    for key in group.array_keys():
        arr = group[key]
        if "_FillValue" in arr.attrs:
            del arr.attrs["_FillValue"]

    logger.info("Initialized mega-zarr with %d datasets", len(outputs))


def set_mega_zarr_run_status(
    zarr_path: str,
    status: str,
    **metadata,
) -> None:
    """Record whether a newly-created model store is safe for downstream use."""

    if status not in {"initialized", "running", "complete", "failed"}:
        raise ValueError(f"Unsupported mega-zarr run status: {status!r}")
    store = make_zarr_store(zarr_path)
    group = zarr.open_group(store=store, mode="r+")
    group.attrs["run_status"] = status
    for key, value in metadata.items():
        group.attrs[key] = value


def populate_mega_zarr(
    zarr_path: str,
    outputs: dict[str, np.ndarray],
    outputs_to_zarr: Iterable[str],
    bounds: Sequence[float],
    year_index: int,
) -> None:
    store = make_zarr_store(zarr_path)
    group = zarr.open_group(store=store, mode="r+")
    y0, y1, x0, x1 = bounds_to_indices(bounds)
    for name in outputs_to_zarr:
        if name not in outputs:
            continue
        group[name][year_index, y0:y1, x0:x1] = outputs[name]


def full_model_year_index(interval_type: str) -> list[int]:
    if interval_type == cn.intervals_annual:
        last_year = cn.five_year_inventory_periods[-1][1]
        return list(range(cn.annual_land_cover_start_year, last_year + 1))
    return [end for _, end in cn.five_year_inventory_periods]


def open_zarr_window(
    zarr_path: str,
    dataset: str,
    bounds: Sequence[float],
    year_index: int,
) -> np.ndarray:
    store = make_zarr_store(zarr_path, read_only=True)
    group = zarr.open_group(store=store, mode="r")
    y0, y1, x0, x1 = bounds_to_indices(bounds)
    return group[dataset][year_index, y0:y1, x0:x1]

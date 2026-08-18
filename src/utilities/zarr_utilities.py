import os
import boto3
import fsspec
import pandas as pd
import sys
import dask
from dask.distributed import print
import dask.array as da
import xarray as xr
import gc
import numpy as np
import rasterio
from rasterio.transform import from_origin
import resource
import psutil
import zarr
import time
from bisect import bisect_left, bisect_right

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu

# Creates the s3 paths for the raw and rechunked mega-zarrs
def create_zarr_path(zarr_basic_path, chunk_size_pixels, run_date, main_logger,
                     model_version=None, model_type=None, model_path_description=None):

    # Sets the output zarr location based on the model run
    mega_zarr_path = zarr_basic_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    mega_zarr_path = mega_zarr_path.replace("RUN_DATE", run_date)
    mega_zarr_path = mega_zarr_path.replace("CHUNK_SIZE", str(chunk_size_pixels))

    main_logger.info(f"\n Zarr path to use: {mega_zarr_path}")

    return mega_zarr_path


# Gets the row and column indexes in a global grid for a given lat and long using a given resolution
def latlon_to_global_zarr_indices(lat, lon, resolution):
    lat_max = 90.0
    lon_min = -180.0

    lat_idx = int(round((lat_max - lat) / resolution))
    lon_idx = int(round((lon - lon_min) / resolution))

    return lat_idx, lon_idx


# Creates a Zarr group with individual datasets on S3 with coordinate arrays (x/y/year),
# spatial_ref metadata, and dataset definitions WITHOUT allocating global arrays.
# That is, it doesn't compute anything upfront or locally. It just creates the zarr group
# with datasets inside.
# In addition to x and y dimensions, there is also a time dimension (intervals), which uses an index (not the actual year).
# This zarr-related code from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc
# and maybe some later chats, too.
# The assignment of NoData values is done in
# https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69d50592-48b8-8329-b529-2babe02f7f27
def initialize_global_zarr(store_url, dataset_keys, n_years, chunk_size, main_logger, fill_value=np.nan):

    fs = fsspec.filesystem("s3", anon=False)

    # Checks if zarr already exists at that location. Does not make one if it already exits.
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6945fc55-7d3c-832d-9724-c718ec0abbe3
    if fs.exists(store_url):
        main_logger.info(f"Mega-zarr already exists at {store_url}. Skipping initialization: {uu.timestr()}")
        return

    # Computes dimensions
    lat_size = int(180 / cn.resolution)
    lon_size = int(360 / cn.resolution)

    # Creates coordinate arrays globally and for all years
    lats = np.arange(90.0 - cn.resolution / 2, -90, -cn.resolution)[:lat_size]
    lons = np.arange(-180.0 + cn.resolution / 2, 180, cn.resolution)[:lon_size]
    year_index = np.arange(n_years)  # Can't assign the year dimension the true years (2016...2024). Needs to be the year index

    # Spatial reference (CRS metadata)
    spatial_attrs = {
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": 4326,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }

    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }
    print(f"Using zstd compression (level=3): {uu.timestr()}")

    start_time = time.time()

    # For each dataset, uses Dask arrays filled lazily (no memory blowup)
    data_vars = {}
    encoding = {}

    for key in dataset_keys:
        main_logger.info(f"Creating {key} in global mega-zarr: {uu.timestr()}")

        # Rather than pre-creating an output datatype dictionary, I'm taking the hard-coded route
        # and just assigning the output datatype here for each dataset that goes in the zarr
        if "density" in key:
            dtype = 'float32'
        elif "change" in key:
            dtype = 'float32'
        elif "emis" in key:
            dtype = 'float32'
        elif "removals" in key:
            dtype = 'float32'
        elif "net" in key:
            dtype = 'float32'
        elif "loss" in key:
            dtype = 'float32'
        elif "gain" in key:
            dtype = 'float32'
        elif "factor" in key:
            dtype = 'float32'
        elif cn.land_state_pattern in key:
            dtype = 'uint32'
        elif cn.composite_primary_forest in key:
            dtype = 'uint8'
        elif cn.starting_composite_primary_forest_pattern in key:
            dtype = 'uint8'
        elif cn.forest_age_output_pattern in key:
            dtype = 'uint16'
        else:
            sys.exit(f"Dataset {key} not assigned a data type for addition to global zarr")

        # Should make the fill value/NoData value be NaN instead of 0.
        # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69d50592-48b8-8329-b529-2babe02f7f27
        if dtype == "float32":
            array_fill_value = np.float32(np.nan)
            encoding[key] = {
                "compressors": compressor,
                "fill_value": array_fill_value,
            }
        else:
            array_fill_value = fill_value
            encoding[key] = {
                "compressors": compressor,
            }

        dask_data = da.full(
            (n_years, lat_size, lon_size),
            array_fill_value,
            dtype=dtype,
            chunks=chunk_size
        )

        data_vars[key] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": year_index, "y": lats, "x": lons},
            name=key,
            attrs={"grid_mapping": "spatial_ref"},
        )

    # Constructs dataset
    main_logger.info(f"Constructing megazarr dataset with metadata only: {uu.timestr()}")
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "x": lons,
            "y": lats,
            "year": year_index
        },
    )

    ds["spatial_ref"] = xr.DataArray(
        np.array(0, dtype="int32"),
        attrs=spatial_attrs,
    )

    main_logger.info(f"dataset info: {ds}: {uu.timestr()}")

    # Writes only metadata to s3 (lazy), not values
    main_logger.info(f"Writing metadata for mega-zarr: {uu.timestr()}")
    mapper = fs.get_mapper(store_url)
    ds.to_zarr(
        store=mapper,
        mode="w",
        compute=False,
        encoding=encoding,
        zarr_format=3
    )

    main_logger.info(f"Created metadata for mega-zarr: {uu.timestr()}")

    z = zarr.open_group(mapper, mode="r")
    main_logger.info(f"Mega-zarr group info: {z.info}: {uu.timestr()}")

    # Clean _FillValue in populated zarr
    # Need to remove _FillValue attribute in zarr because it's being encoded in some way that is incompatible with xarray while using zarr v3,
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68f984c6-9aa0-8327-a910-5ad9a8d170fc.
    # There doesn't seem to be a way to create the zarr with a correctly encoded _FillValue in the first place,
    # hence this fix after the fact.
    main_logger.info(f"Cleaning zarr _FillValue from each dataset: {uu.timestr()}")

    # Open Zarr group in read/write mode
    z = zarr.open_group(store=mapper, mode="r+")

    # Loop through all arrays
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            main_logger.info(f"Removing _FillValue from {key}: {uu.timestr()}")
            del arr.attrs["_FillValue"]

    main_logger.info(f"Cleaned _FillValue from Zarr metadata: {uu.timestr()}")

    end_time = time.time()
    main_logger.info(f"Initialized spatial mega-zarr metadata at {store_url} in {round(end_time-start_time)} seconds: {uu.timestr()}")

def initialize_ipcc_global_zarr(store_url, chunk_size, main_logger, fill_value=0):
    dataset_keys = [
        cn.IPCC_class_pattern,
        cn.IPCC_node_pattern,
        cn.IPCC_change_pattern,
        cn.IPCC_summary_pattern,
    ]

    dtype_map = {
        cn.IPCC_class_pattern: "uint8",
        cn.IPCC_node_pattern: "uint16",
        cn.IPCC_change_pattern: "uint8",
        cn.IPCC_summary_pattern: "uint8",
    }

    fs = fsspec.filesystem("s3", anon=False)

    if fs.exists(store_url):
        main_logger.info(f"IPCC zarr already exists at {store_url}. Skipping initialization: {uu.timestr()}")
        return

    lat_size = int(180 / cn.resolution)
    lon_size = int(360 / cn.resolution)

    lats = np.arange(90.0 - cn.resolution / 2, -90, -cn.resolution)[:lat_size]
    lons = np.arange(-180.0 + cn.resolution / 2, 180, cn.resolution)[:lon_size]

    # 10 slots:
    # class/node: 2015-2024
    # change: 2015_2016 through 2023_2024, plus empty index 9
    # summary: 2015_2024 in index 0, empty indices 1-9
    year_index = np.arange(10)

    compressor = {
        "name": "zstd",
        "configuration": {"level": 3}
    }

    spatial_attrs = {
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": 4326,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }

    data_vars = {}
    encoding = {}

    for key in dataset_keys:
        dtype = dtype_map[key]
        dask_data = da.full(
            (10, lat_size, lon_size),
            fill_value,
            dtype=dtype,
            chunks=chunk_size,
        )

        data_vars[key] = xr.DataArray(
            dask_data,
            dims=("year", "y", "x"),
            coords={"year": year_index, "y": lats, "x": lons},
            name=key,
            attrs={"grid_mapping": "spatial_ref"},
        )

        encoding[key] = {
            "compressors": compressor,
        }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"x": lons, "y": lats, "year": year_index},
    )

    ds["spatial_ref"] = xr.DataArray(
        np.array(0, dtype="int32"),
        attrs=spatial_attrs,
    )

    mapper = fs.get_mapper(store_url)
    ds.to_zarr(
        store=mapper,
        mode="w",
        compute=False,
        encoding=encoding,
        zarr_format=3,
    )

    z = zarr.open_group(store=mapper, mode="r+")
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            del arr.attrs["_FillValue"]

    main_logger.info(f"Initialized IPCC global zarr at {store_url}: {uu.timestr()}")


# Populates pre-existing global mega-zarr with select output numpy arrays (out_dict_all_dtypes)
# Accelerated by writing all years at once
# Originally per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/694612e9-0d2c-832f-8b6d-e7cb247ff781
# and modified in Claude session 'NoData handling for chunk stats and fluxes'
def populate_zarr(bounds, bounds_str, create_zarr, interval_end_years, is_large_run, logger_worker, mega_zarr_path,
                  out_dict_all_dtypes, outputs_to_zarr, stage, tile_id, year_in_array_name=False):

    if not create_zarr:
        lu.print_and_log(f"Not writing outputs for {bounds_str} in {tile_id} to global zarr: {uu.timestr()}", False, logger_worker)
        return

    lu.print_and_log(f"Writing select outputs to global zarr for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    zarr_start = time.time()

    # Opens pre-created global mega-zarr
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(mega_zarr_path)
    z = zarr.open(mapper, mode="r+")

    # Computes spatial indices once
    lat_start, lon_start = latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)  # north, west
    lat_end, lon_end = latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)  # south, east

    ny = lat_end - lat_start
    nx = lon_end - lon_start

    # For zarrs where each year is its own separately-named array with a single time slot (starting
    # carbon pools -- the year is baked into the array name itself), there's no shared multi-year array
    # per variable to pre-resolve like the branch below does. Each (variable, year) pair names its own
    # array and has to be resolved and written individually.
    if year_in_array_name:

        for output_to_zarr in outputs_to_zarr:
            for year in interval_end_years:

                _, pattern_with_units_years = add_units_year_to_pattern(output_to_zarr, year)

                if pattern_with_units_years not in z:
                    lu.print_and_log(f"Skipping {pattern_with_units_years}: not found in zarr", False, logger_worker)
                    continue

                zarr_array = z[pattern_with_units_years]

                if pattern_with_units_years in out_dict_all_dtypes:
                    block = out_dict_all_dtypes[pattern_with_units_years].astype(zarr_array.dtype)[np.newaxis, :, :]
                    zarr_array[0:1, lat_start:lat_end, lon_start:lon_end] = block
                else:
                    lu.print_and_log(f"Skipping {pattern_with_units_years}: no data found for this year", False, logger_worker)

    # Zarrs where the year isn't in the array name
    else:

        # Creates list of zarr datasets with unit (but not year)
        outputs_to_zarr_with_pattern = []
        for output_to_zarr in outputs_to_zarr:
            pattern_with_units, pattern_with_units_years = add_units_year_to_pattern(output_to_zarr, 0)
            outputs_to_zarr_with_pattern.append(pattern_with_units)

        # Pre-opens Zarr arrays once rather than repeatedly for each dataset during the for loop
        zarr_arrays = {
            var: z[var]
            for var in outputs_to_zarr_with_pattern
            if var in z
        }

        n_years = len(interval_end_years)

        # Writes each variable as a full time block
        for output_to_zarr_pattern_unit, zarr_array in zarr_arrays.items():

            dtype = zarr_array.dtype

            block = np.empty((n_years, ny, nx), dtype=dtype)

            has_any_data = False

            for i, year in enumerate(interval_end_years):
                pattern_with_units_years = f"{output_to_zarr_pattern_unit}_{year}"

                # Used for output dictionary with years, e.g., vegetation model outputs.
                if pattern_with_units_years in out_dict_all_dtypes:
                    block[i, :, :] = out_dict_all_dtypes[pattern_with_units_years]
                    has_any_data = True
                # In case the output dictionary doesn't have unit/years. Used for starting composite primary forest.
                elif output_to_zarr_pattern_unit in out_dict_all_dtypes:
                    block[i, :, :] = out_dict_all_dtypes[output_to_zarr_pattern_unit]
                    has_any_data = True
                else:
                    # Fills with Zarr fill_value if missing
                    fill = zarr_array.fill_value
                    if fill is None:
                        fill = np.nan
                    block[i, :, :] = fill

            # Only writes if at least one year exists for this variable
            if has_any_data:
                zarr_array[
                0:n_years,
                lat_start:lat_end,
                lon_start:lon_end
                ] = block
            else:
                lu.print_and_log(f"Skipping {output_to_zarr_pattern_unit}: no data found for any year", False, logger_worker)

            del block
            gc.collect()

    zarr_end = time.time()
    lu.print_and_log(f"Wrote outputs to global zarr for {bounds_str} in {tile_id} in {round(zarr_end - zarr_start)} seconds: {uu.timestr()}",False, logger_worker)


def populate_ipcc_zarr(bounds, bounds_str, create_zarr, is_large_run, logger_worker,
                       mega_zarr_path, out_dict, stage, tile_id):
    if not create_zarr:
        lu.print_and_log(f"Not writing IPCC outputs for {bounds_str} in {tile_id} to global zarr: {uu.timestr()}",False, logger_worker)
        return

    lu.print_and_log(f"Writing IPCC outputs to global zarr for {bounds_str} in {tile_id}: {uu.timestr()}", is_large_run, logger_worker)

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(mega_zarr_path)
    z = zarr.open_group(mapper, mode="r+")

    lat_start, lon_start = latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)
    lat_end, lon_end = latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)

    # Annual land use and node code: indices 0-9 map to 2015-2024.
    for i, year in enumerate(cn.LC_years):
        class_key = f"{cn.IPCC_class_pattern}_{year}"
        node_key = f"{cn.IPCC_node_pattern}_{year}"

        if class_key in out_dict:
            z[cn.IPCC_class_pattern][i, lat_start:lat_end, lon_start:lon_end] = out_dict[class_key]

        if node_key in out_dict:
            z[cn.IPCC_node_pattern][i, lat_start:lat_end, lon_start:lon_end] = out_dict[node_key]

    # LU change: indices 1-9 (based on end year) map to intervals, index 0 stays empty.
    for i, (start_year, end_year) in enumerate(zip(cn.LC_years[:-1], cn.LC_years[1:])):
        change_key = f"{cn.IPCC_change_pattern}_{start_year}_{end_year}"

        if change_key in out_dict:
            z[cn.IPCC_change_pattern][i+1, lat_start:lat_end, lon_start:lon_end] = out_dict[change_key]

    # Summary: index 0 stores 2015_2024 summary, indices 1-9 stay empty.
    summary_key = f"{cn.IPCC_summary_pattern}_2015_2024"
    if summary_key in out_dict:
        z[cn.IPCC_summary_pattern][0, lat_start:lat_end, lon_start:lon_end] = out_dict[summary_key]
    else:
        lu.print_and_log(f"WARNING: {summary_key} not found in out_dict for {bounds_str}", False, logger_worker)

    lu.print_and_log( f"Wrote IPCC outputs to global zarr for {bounds_str} in {tile_id}: {uu.timestr()}", False, logger_worker)

# Checks composite ds for each tile against the original geotif to make sure geotifs haven't been flipped north-south
# (as happened for pixel area once).
# This doesn't actually check the final zarr but the two checks included here should be sufficient to
# detect issues in the creation of the global ds from the geotif tile set.
# Per Claude session 'SOC chunk stats mismatch investigation'
def validate_xarray_assembly(ds, tile_uris, main_logger):
    """
    Confirms that open_mfdataset assembled tiles with correct north-south orientation.
    Raises ValueError if any tile's top-left pixel value in the assembled dataset
    doesn't match the value read directly from the source GeoTIF via rasterio.
    """

    var_name = list(ds.data_vars)[0]

    # Check 1: global y-axis should decrease north→south
    y_vals = ds.y.values
    if not np.all(np.diff(y_vals) < 0):
        raise ValueError(
            "Assembled dataset y-coordinates are not monotonically decreasing (north→south). "
            "open_mfdataset may have flipped or misordered latitude bands."
        )

    # Check 2: per-tile northwest corner pixel comparison (single pixel only, but would detect a north-south inversion).
    # Reading northwest pixel of geotifs serially but reading corresponding pixels of global ds in parallel to speed things up.
    # Reading the northwest pixel of each geotif still takes several minutes and doesn't use Dask at all.
    uris = []
    geotif_vals = []
    assembled_selects = []

    main_logger.info(f"Starting pixel-level check: {uu.timestr()}")
    for i, uri in enumerate(tile_uris.values):
        main_logger.info(f"Reading {uri} for y-axis inversion, tile {i} of {len(tile_uris)}")
        with rasterio.open(uri) as src:
            geotif_val = float(src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))[0, 0])
            lat = src.transform.f + src.transform.e * 0.5
            lon = src.transform.c + src.transform.a * 0.5

        uris.append(uri)
        geotif_vals.append(geotif_val)
        assembled_selects.append(ds[var_name].sel(y=lat, x=lon, method='nearest'))

    assembled_vals = dask.compute(*assembled_selects)

    errors = []
    for uri, geotif_val, assembled_val in zip(uris, geotif_vals, assembled_vals):
        if not np.isclose(geotif_val, float(assembled_val), rtol=1e-4):
            errors.append(
                f"  {uri}\n"
                f"    rasterio NW corner: {geotif_val:.6f}\n"
                f"    assembled:          {float(assembled_val):.6f}"
            )
        else:
            main_logger.info(f"Northwest pixels match for {uri}: geotif={geotif_val:.6f}, ds={float(assembled_val):.6f}")

    main_logger.info(f"Ending pixel-level check: {uu.timestr()}")

    if errors:
        raise ValueError(
            f"Tile assembly mismatch for {len(errors)} of {len(tile_uris)} tiles:\n"
            + "\n".join(errors)
        )


# Makes xarray dataframe (I think not a dataset) from list of s3 uris.
# This came from Solomon Negusse and I haven't really changed it.
# He said that an online forum suggested using xr.open_mfdataset to open non-overlapping geotifs.
def make_xarray_chunks(tile_uris, chunk_size, main_logger):

    xarray_chunks = xr.open_mfdataset(
        tile_uris.values.tolist(),
        parallel=True,
        chunks={'x': chunk_size, 'y':chunk_size}
    ).squeeze()

    validate_xarray_assembly(xarray_chunks, tile_uris, main_logger)

    return xarray_chunks


# Removes the zarr FillValue attribute from each dataset, which is necessary to avoid Float32 datatype errors
def remove_FillValue(zarr_path):

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_path)
    z = zarr.open_group(mapper, mode="r+")

    # Loop through each array and remove _FillValue if present
    for key in z.array_keys():
        arr = z[key]
        if "_FillValue" in arr.attrs:
            print(f"   Removing _FillValue from {key}")
            del arr.attrs["_FillValue"]

    print(f"   FillValues removed from {zarr_path}")


# Calculates regular chunk stats in 1x1 deg chunk of dataset-year slice of zarr.
# Chunk stats are calculated using the same function as used on numpy array outputs from models.
def zarr_1x1_deg_stats(bounds, var_name, zarr_path, interval_end_years, nodata_val=np.nan, year_in_array_name=False):

    bounds_str = uu.boundstr(bounds)  # String form of chunk bounds, from e.g., [8, -1, 9, 0] to 8_-1_9_0
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])  # tile_id in YYN/S_XXXE/W

    zarr_stats_raw_all_years = []

    # print(f"Getting stats for {var_name} for {year_idx} for {bounds_str}: {uu.timestr()}")
    start_time = time.time()

    # Bounding box to get stats for, reformatted for zarr extraction
    target_box = {
        "lat_min": bounds[1],
        "lat_max": bounds[3],
        "lon_min": bounds[0],
        "lon_max": bounds[2]
    }

    # print(f"Getting indices for {bounds_str}")
    lat0, lon0 = latlon_to_global_zarr_indices(target_box["lat_max"], target_box["lon_min"], cn.resolution)
    lat1, lon1 = latlon_to_global_zarr_indices(target_box["lat_min"], target_box["lon_max"], cn.resolution)

    fs = fsspec.filesystem("s3", anon=False)

    pattern_with_units, pattern_with_units_years = add_units_year_to_pattern(var_name, 0)
    # print("pattern_with_units:", pattern_with_units)
    # print("pattern_with_units_years:", pattern_with_units_years)

    # Calculates chunk stats on the chunk of the zarr.
    # Rather than encoding rows as input or output layer, they are encoded by whether they are raw or rechunked zarr
    # since all of these are outputs.
    # Chunk stats are dictionaries.
    # print(f"Getting mapper for {bounds_str}")
    zarr_mapper = fs.get_mapper(zarr_path)
    # print(f"Opening zarr for {bounds_str}")
    zarr_group = zarr.open(zarr_mapper, mode="r", use_consolidated=False)
    # print(f"Getting array for {bounds_str}")

    # For zarrs with one shared multi-year array per variable (vegetation, SOC, organic soil),
    # the array is opened once here and sliced by year below. For zarrs where each year
    # is its own separately-named array with a single time slot (e.g., starting carbon pools -- the year is baked
    # into the array name itself), there's no shared array to open up front; each year's array is opened
    # individually inside the loop instead.
    # Per Claude session 'NoData handling for chunk stats and fluxes'
    if not year_in_array_name:
        zarr_chunk_array = zarr_group[pattern_with_units][:, lat0:lat1, lon0:lon1]

    for year_idx, year in enumerate(interval_end_years):

        pattern_with_units, pattern_with_units_years = add_units_year_to_pattern(var_name, year)

        if year_in_array_name:
            zarr_chunk_array_year = zarr_group[pattern_with_units_years][0, lat0:lat1, lon0:lon1]
        else:
            zarr_chunk_array_year = zarr_chunk_array[year_idx]

        # Float zarrs always use NaN as NoData in this codebase (matches the fill_value the zarr
        # was created with -- see initialize_global_zarr). Overriding here means callers don't need
        # to know or track that themselves for float layers.
        # Integer zarrs have no universal "NaN" equivalent, so they keep whatever nodata_val the caller passed in.
        if np.issubdtype(zarr_chunk_array_year.dtype, np.floating):
            nodata_val = np.nan

        zarr_stats_raw_year = uu.calculate_stats(zarr_chunk_array_year, pattern_with_units_years, bounds_str, tile_id,
                                                 'zarr_stats', None, nodata_val)

        zarr_stats_raw_all_years.append(zarr_stats_raw_year)

    return zarr_stats_raw_all_years


# Parallelizes stats calculation in 1x1 deg chunks in zarr for a given dataset-year
def run_parallel_stats(client, chunk_list, var, zarr_path, output_years, nodata_val=np.nan, year_in_array_name=False):

    futures = []

    # Iterates through all chunks in the list for a given dataset-year
    for chunk in chunk_list:
        future = client.submit(zarr_1x1_deg_stats,
                               chunk, var, zarr_path, output_years, nodata_val, year_in_array_name, retries=2)
        futures.append(future)

    # List of dictionaries, where each dictionary is stats for a single chunk
    results = client.gather(futures)

    return results


def ipcc_zarr_1x1_deg_stats(bounds, var, zarr_path):
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])

    lat0, lon0 = latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)
    lat1, lon1 = latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_path)
    z = zarr.open_group(mapper, mode="r")

    stats = []

    if var in {cn.IPCC_class_pattern, cn.IPCC_node_pattern}:
        for i, year in enumerate(cn.LC_years):
            key = f"{var}_{year}"
            arr = z[var][i, lat0:lat1, lon0:lon1]
            stats.append(uu.calculate_ipcc_stats(arr, key, bounds_str, tile_id, "zarr_stats"))
    elif var == cn.IPCC_change_pattern:
        for i, (start_year, end_year) in enumerate(zip(cn.LC_years[:-1], cn.LC_years[1:])):
            key = f"{cn.IPCC_change_pattern}_{start_year}_{end_year}"
            arr = z[var][i+1, lat0:lat1, lon0:lon1]
            stats.append(uu.calculate_ipcc_stats(arr, key, bounds_str, tile_id, "zarr_stats"))
    elif var == cn.IPCC_summary_pattern:
        arr = z[var][0, lat0:lat1, lon0:lon1]
        stats.append(uu.calculate_ipcc_stats(arr, cn.IPCC_summary_pattern, bounds_str, tile_id, "zarr_stats"))
    else:
        raise ValueError(f"Unsupported IPCC zarr variable: {var}")

    return stats


def run_parallel_ipcc_stats(client, chunk_list, var, zarr_path):
    if client is None:
        return [ipcc_zarr_1x1_deg_stats(chunk, var, zarr_path) for chunk in chunk_list]
    futures = [client.submit( ipcc_zarr_1x1_deg_stats, chunk, var, zarr_path, retries=2) for chunk in chunk_list]
    results_nested = client.gather(futures)

    return [item for result in results_nested for item in result]


# Compares chunk stats from model and from zarr for a dataset-year combination
# Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6903d1dd-555c-8321-8547-0aa4772c9878
def compare_dataset_year_chunk_stats(all_merged_tables, chunk_stats_variable_zarr, main_logger,
                                     tables_to_compare_dict, var_name, zarr_comparison_stats_path):

    # Selects relevant model output table
    # The formatting of the year depends on the variable.
    if "gross" in var_name:
        model_table = tables_to_compare_dict[cn.gross_outputs_1x1]
        # year = year
    elif "net" in var_name:
        model_table = tables_to_compare_dict[cn.net_outputs_1x1]
        # year = year
    elif ("loss" in var_name) or ("gain" in var_name):  # For SOC timeseries
        model_table = tables_to_compare_dict[cn.other_outputs_1x1]
        # year = year
    else:
        model_table = tables_to_compare_dict[cn.other_outputs_1x1]
        # For reasons I can't really trace back, the year datatype for C densities is object, not int.
        # So, it needs to be recast to a str or int to match the chunk_stats table.
        # year = str(year)
    # print("model_table:", model_table)

    # Converts zarr chunk stats from list of dictionaries to dataframe.
    # Need to flatten the list because each chunk for each dataset is a list of dictionaries, where each element is a year.
    # So, flattening the list makes all years for all variables and chunks flat, rather than years being nested in each chunk-dataset.
    chunk_stats_variable_zarr_flat = uu.flatten_list(chunk_stats_variable_zarr)
    # print("chunk_stats_variable_zarr_flat:", chunk_stats_variable_zarr_flat)
    zarr_df = pd.DataFrame(chunk_stats_variable_zarr_flat)
    # print("zarr_df:", zarr_df)

    # Subsets model chunk stats to relevant pattern
    subset_model_table = model_table[model_table['pattern'].str.contains(var_name, na=False)].copy()

    # For chunk stat comparisons of starting year data, the geotif chunk stats chunk_name has 'no year range'. Need to replace with the starting year.
    subset_model_table['chunk_name'] = subset_model_table['chunk_name'].str.replace('_no year range', f'_{cn.LC_first_year}', regex=False)
    # print("var_name:", var_name)
    # print("subset_model_table", subset_model_table)
    # print("subset_model_table chunk_name", subset_model_table['chunk_name'].iloc[0])

    # Selects only the needed columns from rechunked_zarr_table
    # main_logger.info(f"    Subsetting zarr table to numeric columns for {var_name}: {uu.timestr()}")
    zarr_subset_table = zarr_df[['chunk_name', 'min_value', 'mean_value', 'max_value', 'count_value']].copy()
    # print("zarr_subset_table", zarr_subset_table)

    # Renames columns in raw_subset to distinguish them after merge
    # main_logger.info(f"    Renaming zarr columns for {var_name}: {uu.timestr()}")
    zarr_subset_table = zarr_subset_table.rename(columns={
        'min_value': 'min_value_zarr',
        'mean_value': 'mean_value_zarr',
        'max_value': 'max_value_zarr',
        'count_value': 'count_value_zarr'
    })
    # print("zarr_subset_table", zarr_subset_table)

    # Converts all zarr value columns to numeric, coercing errors to NaN
    # main_logger.info(f"    Converting zarr columns to numeric for {var_name}: {uu.timestr()}")
    for col in ['min_value_zarr', 'mean_value_zarr', 'max_value_zarr', 'count_value_zarr']:
        zarr_subset_table[col] = pd.to_numeric(zarr_subset_table[col], errors='coerce')

    # Merges with subset_model_table on 'chunk_name', left join (keeps all model output rows)
    # main_logger.info(f"    Merging zarr data to original model data for {var_name}: {uu.timestr()}")
    # print("subset_model_table[chunk_name]:", subset_model_table.iloc[0]['chunk_name'])
    # print("zarr_subset_table[chunk_name]:", zarr_subset_table.iloc[0]['chunk_name'])
    merged_table = subset_model_table.merge(zarr_subset_table, on='chunk_name', how='left')
    # print("merged_table:", merged_table)

    # Calculates differences for four metrics and stores in new columns
    main_logger.info(f"    Calculating differences for {var_name} ({merged_table['count_value_zarr'].sum().item():.0f} pixels in zarr): {uu.timestr()}")
    merged_table['min_value_diff'] = merged_table['min_value'] - merged_table['min_value_zarr']
    merged_table['mean_value_diff'] = merged_table['mean_value'] - merged_table['mean_value_zarr']
    merged_table['max_value_diff'] = merged_table['max_value'] - merged_table['max_value_zarr']
    merged_table['count_value_diff'] = merged_table['count_value'] - merged_table['count_value_zarr']
    # print("merged_table.head():", merged_table.head())


    # Calculates max absolute difference across the four metrics' difference columns
    merged_table['maximum_diff_value'] = merged_table[
        ['min_value_diff', 'mean_value_diff', 'max_value_diff', 'count_value_diff']
    ].abs().max(axis=1)

    # Identifies rows (chunks) which have stats that differ between model and zarr
    mask = merged_table['maximum_diff_value'] > cn.zarr_difference_tolerance
    # print("mask:", mask)

    # Number of rows from model output without matching zarr pixel counts

    # Excludes rows where model 'count_value' is non-numeric (like 'no data') (no model outputs in those chunks).
    # Coerces to numeric and check for valid values.
    valid_count_mask = pd.to_numeric(merged_table['count_value'], errors='coerce').notna()

    # Total comparable rows
    comparable_row_count = valid_count_mask.sum()

    # Of those, how many are missing zarr stats?
    chunks_without_zarr_stats = merged_table.loc[valid_count_mask, 'count_value_diff'].isna().sum()

    main_logger.info(f"    {chunks_without_zarr_stats}/{comparable_row_count} rows with data without pixel count comparison.")

    # Applies the mask to filter those rows
    differences_exceeding_tolerance = merged_table[mask]
    # print("differences_exceeding_tolerance:", differences_exceeding_tolerance)

    # Prints rows that exceed the tolerance for difference between original and zarr chunk stats
    if len(differences_exceeding_tolerance) > 0:
        main_logger.warning(f"    WARNING: {len(differences_exceeding_tolerance)} rows in {var_name} have differences exceeding the tolerance!")

        # Selects chunk_id and all difference to print in the console for easy viewing
        cols_to_print = [
            'chunk_id',
            'min_value_diff',
            'mean_value_diff',
            'max_value_diff',
            'count_value_diff',
            'maximum_diff_value'
        ]

        main_logger.warning(differences_exceeding_tolerance[cols_to_print])

    else:
        main_logger.info(f"    0/{comparable_row_count} rows in {var_name} have metrics with differences exceeding the tolerance.")

    # Adds df for this dataset-year combination to the list of all the dataset-year dfs
    all_merged_tables.append(merged_table)

    # Writes cumulative results (all dataset-year combinations) to Excel file or parquet tables

    # Concatenates all merged dataset-year tables into a single DataFrame
    final_merged_table = pd.concat(all_merged_tables, ignore_index=True)
    # print("final_merged_table:", final_merged_table)

    # Splits output rows based on 'layer_name' containing 'flux', 'gross', or 'net'
    gross_flux_1x1_outputs = final_merged_table[final_merged_table['layer_name'].str.contains('gross', case=False, na=False)]
    net_flux_1x1_outputs = final_merged_table[final_merged_table['layer_name'].str.contains('net|flux', case=False, na=False)]

    # Puts output rows that don't contain 'flux|gross|net' in a separate table
    other_1x1_outputs = final_merged_table[~final_merged_table['layer_name'].str.contains('flux|gross|net', case=False, na=False)]
    # print("other_1x1_outputs:", other_1x1_outputs)

    # Saves output to three tabs in Excel
    if "xlsx" in zarr_comparison_stats_path:
        # Writes to Excel after each iteration of dataset-year to check results more easily (not have to wait until end)
        with pd.ExcelWriter(zarr_comparison_stats_path, engine='openpyxl', mode='w') as writer:

            gross_flux_1x1_outputs.to_excel(writer, sheet_name=cn.gross_outputs_1x1, index=False)
            net_flux_1x1_outputs.to_excel(writer, sheet_name=cn.net_outputs_1x1, index=False)
            other_1x1_outputs.to_excel(writer, sheet_name=cn.other_outputs_1x1, index=False)

    # Saves output to three parquet tables.
    # These must be written in the same order as the file names are created in zu.get_table_names_for_zarr_stats_comparison()
    elif "parquet" in zarr_comparison_stats_path[0]:
        # print("zarr_comparison_stats_path[0]:", zarr_comparison_stats_path[0])
        gross_flux_1x1_outputs.to_parquet(zarr_comparison_stats_path[0], index=False)
        net_flux_1x1_outputs.to_parquet(zarr_comparison_stats_path[1], index=False)
        other_1x1_outputs.to_parquet(zarr_comparison_stats_path[2], index=False)

    else:
        sys.exit("Table type not found")


    # Need to return the combined table so that it can be added to in the next iteration
    return len(differences_exceeding_tolerance), chunks_without_zarr_stats


# Gets the names of the gross, other, and net chunk stats tables that should be compared against
# the zarr, as well as the name of the output comparison tables.
# Works for both Excel and Parquet model chunk stats.
# If the model being compared against output Parquet chunk stats, this will return
# Parquet table names for the comparison chunk stats. 
def get_table_names_for_zarr_stats_comparison(comparison_insert, main_logger, model_chunk_stats_path):

    # Separate logic for naming chunk stat comparison outputs if using Excel or Parquet (very large runs)
    if "xlsx" in model_chunk_stats_path:
        main_logger.info(f"Reading model chunk stats from local file: {model_chunk_stats_path}")
        chunk_stats_model_gross = pd.read_excel(model_chunk_stats_path, sheet_name=cn.gross_outputs_1x1)
        chunk_stats_model_other = pd.read_excel(model_chunk_stats_path, sheet_name=cn.other_outputs_1x1)
        chunk_stats_model_net = pd.read_excel(model_chunk_stats_path, sheet_name=cn.net_outputs_1x1)
        chunk_stats_model_1x1_in_10x10 = pd.read_excel(model_chunk_stats_path, sheet_name=cn.counts_1x1_in_10x10)

        # Name of output Excel spreadsheet with chunk stats comparisons
        name, ext = os.path.splitext(model_chunk_stats_path)
        zarr_comparison_stats_path = f"{name}{comparison_insert}_{uu.timestr()}{ext}"
        zarr_comparison_stats_name = os.path.basename(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    elif "parquet" in model_chunk_stats_path:
        main_logger.info(f"Reading parquet tables from local parquet files: {model_chunk_stats_path}")
        parquet_base = f"{model_chunk_stats_path}__"
        chunk_stats_model_gross = pd.read_parquet(f"{parquet_base}{cn.gross_outputs_1x1}.parquet")
        chunk_stats_model_other = pd.read_parquet(f"{parquet_base}{cn.other_outputs_1x1}.parquet")
        chunk_stats_model_net = pd.read_parquet(f"{parquet_base}{cn.net_outputs_1x1}.parquet")
        chunk_stats_model_1x1_in_10x10 = pd.read_parquet(f"{parquet_base}{cn.counts_1x1_in_10x10}.parquet")

        # Names of output parquet tables with chunk stats comparisons
        zarr_comparison_stats_gross_name = f"{model_chunk_stats_path}__{cn.gross_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_net_name = f"{model_chunk_stats_path}__{cn.net_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_other_name = f"{model_chunk_stats_path}__{cn.other_outputs_1x1}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_1x1_in_10x10_name = f"{model_chunk_stats_path}__{cn.counts_1x1_in_10x10}_{comparison_insert}_{uu.timestr()}.parquet"
        zarr_comparison_stats_path = [zarr_comparison_stats_gross_name, zarr_comparison_stats_net_name,
                                      zarr_comparison_stats_other_name, zarr_comparison_stats_1x1_in_10x10_name]
        zarr_comparison_stats_name = [os.path.basename(stats_path) for stats_path in zarr_comparison_stats_path]
        # print(zarr_comparison_stats_path)
        # print(zarr_comparison_stats_name)

    else:
        sys.exit("Table type not found")
    # The model chunk stat tables
    tables_to_compare_dict = {cn.gross_outputs_1x1: chunk_stats_model_gross,
                              cn.net_outputs_1x1: chunk_stats_model_net,
                              cn.other_outputs_1x1: chunk_stats_model_other,
                              cn.counts_1x1_in_10x10: chunk_stats_model_1x1_in_10x10}
    return tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path


# Adds units and year specifications to core pattern
def add_units_year_to_pattern(core_pattern, year):
    if "emission_factor" in core_pattern:
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
    elif cn.starting_C_pools_LC_masked_source_flag_pattern in core_pattern:
        # Must come before the "density" check below -- this pattern contains "density" as a substring
        # even though it's a classification code, not a density value, and takes no units suffix.
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
    elif "density" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha"
        pattern_with_units_years = f"{core_pattern}_ha_{year}"
    elif "change" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "emis" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "removals" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "removal" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "loss" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "gain" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif "net" in core_pattern:
        pattern_with_units = f"{core_pattern}_ha_yr"
        pattern_with_units_years = f"{core_pattern}_ha_yr_{year}"
    elif cn.land_state_pattern in core_pattern:
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
    elif cn.composite_primary_forest in core_pattern:
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
    elif cn.forest_age_output_pattern in core_pattern:
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
    else:
        pattern_with_units = f"{core_pattern}"
        pattern_with_units_years = f"{core_pattern}_{year}"
        # sys.exit(f"Dataset {core_pattern} not assigned a pattern with units for addition to global zarr")  # Using this led to hard-to-trace errors

    return pattern_with_units, pattern_with_units_years


def upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                       main_logger, model_chunk_stats_table_name, stage,
                                       start_time, zarr_comparison_stats_name, zarr_comparison_stats_path):

    if chunks_count_exceeding_total > 0:
        main_logger.warning(f"WARNING: {chunks_count_exceeding_total} chunks exceeded difference tolerance! Check log!")
    else:
        main_logger.info(f"{chunks_count_exceeding_total} chunks exceeded the difference tolerance for one or more chunk stat metrics.")

    if chunks_without_zarr_stats_total > 0:
        main_logger.warning(f"WARNING: {chunks_without_zarr_stats_total} chunks are missing corresponding zarr chunk stats! Check log!")
    else:
        main_logger.info(f"{chunks_without_zarr_stats_total} chunks were missing corresponding zarr chunk stats.")

    s3_client = boto3.client("s3")

    if "xlsx" in zarr_comparison_stats_path:
        main_logger.info(f"Uploading chunk stats comparison Excel spreadsheet to s3: {uu.timestr()}")
        try:
            s3_client.upload_file(zarr_comparison_stats_path, cn.short_bucket_prefix,
                                  Key=f"{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}")
            main_logger.info(
                f"Chunk stats spreadsheet uploaded to {cn.full_bucket_prefix}/{cn.s3_chunk_stats_path}{zarr_comparison_stats_name}: {uu.timestr()}")
        except Exception as e:
            main_logger.warning(f"Chunk stats upload to s3 failed: {e}. Continuing without halting.")

    elif "parquet" in zarr_comparison_stats_path[0]:  # Because this is a list, so just use the first one to get the pattern
        main_logger.info(f"Uploading chunk stats comparison parquet tables to s3: {uu.timestr()}")

        for parquet_name, parquet_path in zip(zarr_comparison_stats_name, zarr_comparison_stats_path):
            # No zarr stats comparison for 1x1_counts_in_10x10 table, so don't upload that
            if '1x1_counts_in_10x10' in parquet_name:
                continue
            parquet_folder = parquet_path.split('/')[1]   # parquet_YYYYMMDD_HH_MM_SS
            s3_key = f"{cn.s3_chunk_stats_path}{parquet_folder}/{parquet_name}"
            # print(cn.s3_chunk_stats_path)
            # print(parquet_folder)
            # print(parquet_name)
            # print(s3_key)
            main_logger.info(f"Uploading {parquet_path} to {s3_key}: {uu.timestr()}")
            s3_client.upload_file(parquet_path, cn.short_bucket_prefix, Key=s3_key)

    else:
        sys.exit("Table type not found")

    uu.stage_duration(start_time, uu.timestr(), f"{stage} with zarr chunk stats comparison", main_logger)


# Extracts a 10x10° tile from a Zarr store and writes to GeoTIFF on S3
def create_10x10_deg_geotif_from_zarr(var, year_idx, tile_id, raw_path, output_base, model_version, model_type,
                                      model_path_description, no_upload, use_start_year, no_data_val, append_start_year_to_var=False):

    process = psutil.Process(os.getpid())

    logger_worker = lu.setup_logging_worker()

    # Convert tile_id to bounding box (W, S, E, N)
    min_x, min_y, max_x, max_y = uu.get_10x10_tile_bounds(tile_id)

    # Establishes year/year range and units for dataset
    if ("density" in var) and (not cn.starting_C_pools_LC_masked_source_flag_pattern in var):
        per_ha_units = cn.C_density_pixel_meaning
        per_pixel_units = cn.C_per_pixel_pixel_meaning
        coarse_units = cn.C_density_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "emis" in var:
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "removals" in var:
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "net" in var:  # For SOC and vegetation
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif cn.land_state_pattern in var:
        per_ha_units = ""
        per_pixel_units = ""
        coarse_units = ""
        var_per_ha = var
    elif "loss" in var:  # For SOC change
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    elif "gain" in var:  # For SOC change
        per_ha_units = cn.flux_density_pixel_meaning
        per_pixel_units = cn.flux_per_pixel_pixel_meaning
        coarse_units = cn.flux_aggreg_pixel_meaning
        var_per_ha = f"{var}{per_ha_units}"
    else:
        per_ha_units = ""
        per_pixel_units = ""
        coarse_units = ""
        var_per_ha = var

    # If creating outputs from the model start year, it just uses that year.
    # Renames variable to use units and year.
    if use_start_year == True:
        year = cn.LC_first_year
        if append_start_year_to_var:
            var_with_unit = f"{var_per_ha}_{year}"
        else:
            var_with_unit = var_per_ha
    else:      # For timeseries data, uses specified output years (e.g., vegetation, SOC density, SOC change)
        if "SOC_density" in var:
            year = cn.SOC_density_intervals[year_idx]
        elif "SOC_net" in var:
            year = cn.SOC_change_intervals[year_idx]
        elif "SOC_loss" in var:
            year = cn.SOC_change_intervals[year_idx]
        elif "SOC_gain" in var:
            year = cn.SOC_change_intervals[year_idx]
        else:  # Vegetation timeseries
            year = cn.veg_outputs_years[year_idx]
        var_with_unit = var_per_ha  # Doesn't add year to variable/unit name

    # Open Zarr group using fsspec mapper
    fs = fsspec.filesystem("s3", anon=False)
    try:
        model_zarr_store = zarr.open_group(fs.get_mapper(raw_path), mode="r")
    except Exception as e:
        print(f"tile_id: {tile_id}; year: {year}; year_idx: {year_idx}; var: {var}; var_per_ha: {var_per_ha}; var_with_unit: {var_with_unit}--"
              f" zarr path not working. Path is showing up as {raw_path}")
        sys.exit()

    # Determine pixel indices (applies to model outputs and pixel area)
    lat_array_model = model_zarr_store["y"][:]
    lon_array_model = model_zarr_store["x"][:]

    # Get index ranges (applies to model outputs and pixel area)
    y0_model = np.searchsorted(lat_array_model[::-1], max_y, side='right')
    y1_model = np.searchsorted(lat_array_model[::-1], min_y, side='left')
    x0_model = np.searchsorted(lon_array_model, min_x, side='left')
    x1_model = np.searchsorted(lon_array_model, max_x, side='right')

    # Flips y indices since lat is descending
    y0_model, y1_model = len(lat_array_model) - y1_model, len(lat_array_model) - y0_model
    if y0_model > y1_model:
        y0_model, y1_model = y1_model, y0_model

    lu.print_and_log(f"  Extracting {var_with_unit} for {year} for {tile_id}: {uu.timestr()}", False, logger_worker)
    extract_start_time = time.time()

    # Loads model output data block
    if ("SOC_net" in var) or ("SOC_loss" in var) or ("SOC_gain" in var):
        # SOC net, loss, and gain has no data in the zarr for the first time slice because it has one fewer year than SOC density,
        # so change intervals are actually shifted back by 1 year compared to SOC density to account for having one fewer year.
        # All outputs from the vegetation model have the same number of years, so no offsetting is needed.
        data_per_ha = model_zarr_store[var_with_unit][year_idx+1, y0_model:y1_model, x0_model:x1_model]
    else:
        data_per_ha = model_zarr_store[var_with_unit][year_idx, y0_model:y1_model, x0_model:x1_model]

    # Calculates per-pixel output (for numeric outputs only)
    pixel_area_zarr_store = uu.get_pixel_area_store()

    # Determine pixel indices (applies to model outputs and pixel area)
    lat_array_pixel_area = pixel_area_zarr_store["y"][:]
    lon_array_pixel_area = pixel_area_zarr_store["x"][:]

    # Get index ranges (applies to model outputs and pixel area)
    y0_pixel_area = np.searchsorted(lat_array_pixel_area[::-1], max_y, side='right')
    y1_pixel_area = np.searchsorted(lat_array_pixel_area[::-1], min_y, side='left')
    x0_pixel_area = np.searchsorted(lon_array_pixel_area, min_x, side='left')
    x1_pixel_area = np.searchsorted(lon_array_pixel_area, max_x, side='right')

    # Flips y indices since lat is descending
    y0_pixel_area, y1_pixel_area = len(lat_array_pixel_area) - y1_pixel_area, len(lat_array_pixel_area) - y0_pixel_area
    if y0_pixel_area > y1_pixel_area:
        y0_pixel_area, y1_pixel_area = y1_pixel_area, y0_pixel_area

    # Only calculates per-pixel and aggregated geotifs if output is float32 (skips outputs like land_state)
    if model_zarr_store[var_with_unit].dtype == np.float32:
        pixel_area = pixel_area_zarr_store['band_data'][y0_pixel_area:y1_pixel_area, x0_pixel_area:x1_pixel_area]
        # print("y0:", y0_pixel_area)
        # print("y1:", y1_pixel_area)
        # print("x0:", x0_pixel_area)
        # print("x1:", x1_pixel_area)
        # print(pixel_area)
        # sys.quit()

        # Converts per-ha to per-pixel
        data_per_pixel = data_per_ha * pixel_area * cn.m2_to_ha

        # Cleanup. Without this, memory exceeds 24GB/worker and eventually tasks get repeated because of too much memory spillage or something
        del pixel_area

        # Creates 0.04x0.04 deg geotif in Mg CO2(e)/0.04x0.04deg pixel/yr
        # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c/c/69d50592-48b8-8329-b529-2babe02f7f27
        # Should write NaN when there are no valid pixels.

        # Trims fine grid so it splits evenly into coarse blocks
        ny, nx = data_per_pixel.shape
        ny_trim = ny - (ny % cn.global_aggregation_factor)
        nx_trim = nx - (nx % cn.global_aggregation_factor)
        data_fine_trim = data_per_pixel[:ny_trim, :nx_trim]

        # Reshape into coarse blocks
        reshaped = data_fine_trim.reshape(
            ny_trim // cn.global_aggregation_factor, cn.global_aggregation_factor,
            nx_trim // cn.global_aggregation_factor, cn.global_aggregation_factor
        )

        # Sum valid values within each coarse block
        coarse_agg = np.nansum(reshaped, axis=(1, 3)).astype(np.float32)

        # Count how many valid fine pixels contributed to each coarse block
        valid_counts = np.sum(~np.isnan(reshaped), axis=(1, 3))

        # If no fine pixels contributed, restore NoData
        coarse_agg[valid_counts == 0] = np.nan

        # Warning if there are no valid aggregated pixels
        if not np.isfinite(coarse_agg).any():
            logger_worker.warning(f"All-NaN coarse aggregation for {tile_id}, {var}, {year}")

    else:
        data_per_pixel = None
        coarse_agg = None

    extract_end_time = time.time()
    lu.print_and_log(f"  Calculated {var_with_unit} for {year} for {tile_id} in {round(extract_end_time - extract_start_time)} seconds: {uu.timestr()}", False, logger_worker)
    lu.print_and_log(f"  Memory usage after 10x10 extraction for {var_with_unit} for {year} for {tile_id}: {process.memory_info().rss / 1024 ** 2:.2f} MB", False, logger_worker)

    # Name and s3 folder for per-hectare output
    output_path = output_base.replace("PATTERN", var)
    output_path = output_path.replace("START_END", str(year))
    output_path = output_path.replace(cn.model_version_type_description_placeholder, f"version_{model_version}__{model_type}__{model_path_description}")
    output_path_per_ha = output_path.replace("CHUNK_SIZE_pixels", f"{cn.full_raster_dims}_pixels")
    output_path_per_ha = output_path_per_ha.replace("PER_HA_OR_PIXEL", per_ha_units)
    output_name_per_ha = f"{tile_id}__{var}{per_ha_units}_{str(year)}.tif"
    s3_filename_per_ha = f"{output_path_per_ha}{output_name_per_ha}"

    # Hacky way to fix land_state and other unitless outputs that otherwise have in the pay YYYY//40000_pixels.
    # This removes the extra / .
    s3_filename_per_ha = s3_filename_per_ha.replace("//40000", "/40000")

    # Name and s3 folder for per-pixel output
    output_path_per_pixel = output_path.replace("CHUNK_SIZE_pixels", f"{cn.full_raster_dims}_pixels")
    output_path_per_pixel = output_path_per_pixel.replace("PER_HA_OR_PIXEL", per_pixel_units)
    output_name_per_pixel = f"{tile_id}__{var}{per_pixel_units}_{str(year)}.tif"
    s3_filename_per_pixel = f"{output_path_per_pixel}{output_name_per_pixel}"

    # Name and s3 folder for 0.04x0.04 deg output
    output_path_coarse = output_path.replace("CHUNK_SIZE_pixels", f"{cn.global_aggregation_factor}_pixels")
    output_path_coarse = output_path_coarse.replace("PER_HA_OR_PIXEL", coarse_units)
    output_name_coarse = f"{tile_id}__{var}{coarse_units}_{str(year)}.tif"
    s3_filename_coarse = f"{output_path_coarse}{output_name_coarse}"

    # Uploads to s3 if requested
    if no_upload == False:

        # GeoTransform for 0.00025 deg resolution grid (top-left corner)
        transform = from_origin(min_x, max_y, cn.resolution, cn.resolution)

        # GeoTransform for coarse (0.04 deg) resolution grid
        coarse_transform = from_origin(min_x, max_y, cn.global_geotif_resolution, cn.global_geotif_resolution)

        # Writes per-ha geotif to S3
        valid_pixel_count_per_ha = uu.write_single_geotiff_to_s3(var, year, tile_id, data_per_ha, no_data_val, transform, s3_filename_per_ha, logger_worker)

        # Conditionally writes per-pixel output and 0.04x0.04 res output (only if dataset is float32, i.e. numeric output from model).
        if model_zarr_store[var_with_unit].dtype == np.float32:
            valid_pixel_count_per_pixel = uu.write_single_geotiff_to_s3(var, year, tile_id, data_per_pixel, no_data_val, transform, s3_filename_per_pixel, logger_worker)

            # valid_pixel_count_coarse not used. Not doing anything with stats from the aggregated output
            valid_pixel_count_coarse = uu.write_single_geotiff_to_s3(var, year, tile_id, coarse_agg, no_data_val, coarse_transform, s3_filename_coarse, logger_worker)
        else:
            valid_pixel_count_per_pixel = None
            valid_pixel_count_coarse = None

        # # More cleanup. This doesn't actually seem to reduce memory. Leaving it in commented just for reference.
        # del data_per_ha
        # del data_per_pixel

        # Most stats for the 10x10 deg outputs aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats_per_ha = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': valid_pixel_count_per_ha,
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

        chunk_stats_per_pixel = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_pixel,
            'tile_name': output_name_per_pixel,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': valid_pixel_count_per_pixel,
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    else:

        # Most stats for the 10x10 aren't calculated.
        # Only the pixel count is because it is compared to the pixel counts in all the relevant 1x1s.
        # Dictionary is in a list because it's necessary for chunk stats processing later.
        chunk_stats_per_ha = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_ha,
            'tile_name': output_name_per_ha,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'not calculated',
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

        chunk_stats_per_pixel = [{
            'chunk_id': 'N/A',
            'tile_id': tile_id,
            'layer_name': output_name_per_pixel,
            'tile_name': output_name_per_pixel,
            'in_out': 'output_layer',
            'pattern': var,
            'years': year,
            'min_value': 'no data',
            'mean_value': 'no data',
            'max_value': 'no data',
            'count_value': 'not calculated',
            'sum_value': 'no data',
            'data_type': 'no data'
        }]

    tile_end_time = time.time()
    lu.print_and_log(f"  Total chunk processing {var} for {year} for {tile_id} in {round(tile_end_time - extract_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"  Peak memory for {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    return chunk_stats_per_ha, chunk_stats_per_pixel


# Gets indexes of zarr (regardless of its geographic coverage)
# From https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6986043f-c8b0-832c-837f-7329873aa948
def get_index_range(coords, min_val, max_val, descending=False):
    if descending:
        coords = coords[::-1]
        i0 = bisect_left(coords, max_val)
        i1 = bisect_right(coords, min_val)
        return len(coords) - i1, len(coords) - i0
    else:
        i0 = bisect_left(coords, min_val)
        i1 = bisect_right(coords, max_val)
        return i0, i1


def populate_one_zarr_chunk(chunk, var_dir_no_year, zarr_store_url, interval_end_years,
                            var_name_units, var_name_no_units, resolution, main_logger):
    """
    For one spatial chunk, constructs the expected tile path for each year directly,
    stacks them into a (n_years, height, width) block, writes it to the correct zarr
    slice, and returns chunk stats for verification.
    From Claude Code
    """

    logger_worker = lu.setup_logging_worker()

    lu.print_and_log(f"Adding {chunk} to zarr: {uu.timestr()}", False, logger_worker)
    chunk_start_time = time.time()

    bounds_str = uu.boundstr(chunk)
    tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])

    # Read one existing tile to get spatial extent and raster dimensions
    sample_path = None
    for year in interval_end_years:
        candidate = f"{var_dir_no_year.replace('START_END', str(year))}{tile_id}__{bounds_str}__{var_name_units}_{year}.tif"
        fs = fsspec.filesystem("s3", anon=False)
        if fs.exists(candidate.replace("s3://", "")):
            sample_path = candidate
            break

    if sample_path is None:
        return {"status": "skipped", "bounds_str": bounds_str, "reason": "no tile files found", "chunk_stats": []}

    with rasterio.open(sample_path) as src:
        b = src.bounds
        tile_height = src.height
        tile_width  = src.width

    # Compute the slice into the global zarr
    lat_start = int(round((90.0 - b.top)  / resolution))
    lon_start = int(round((b.left + 180.0) / resolution))
    lat_end   = lat_start + tile_height
    lon_end   = lon_start + tile_width

    # Build an empty (n_years, height, width) numpy array
    n_years = len(interval_end_years)
    block = np.full((n_years, tile_height, tile_width), np.float32(np.nan), dtype="float32")

    # Populate the array year by year, constructing each tile path directly
    years_written = []
    for i, year in enumerate(interval_end_years):
        tile_path = f"{var_dir_no_year.replace('START_END', str(year))}{tile_id}__{bounds_str}__{var_name_units}_{year}.tif"
        try:
            with rasterio.open(tile_path) as src:
                data = src.read(1).astype("float32")
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan   # Writes NaN rather than 0 for NoData
                block[i] = data
            years_written.append(year)
        except Exception:
            pass  # Leave year slice as NaN if tile is missing

    # Write the full time block for this chunk to the zarr
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_store_url)
    z = zarr.open_group(mapper, mode="r+", use_consolidated=False)
    z[var_name_units][0:n_years, lat_start:lat_end, lon_start:lon_end] = block

    lu.print_and_log(f"Added {chunk} to zarr: {uu.timestr()}", False, logger_worker)

    # Calculate chunk stats from the zarr for verification
    chunk_stats = zarr_1x1_deg_stats(chunk, var_name_no_units, zarr_store_url, interval_end_years)

    chunk_end_time = time.time()
    lu.print_and_log(f"Total chunk processing for {bounds_str} in {round(chunk_end_time - chunk_start_time)} seconds: {uu.timestr()}", False, logger_worker)

    # To track peak memory usage
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak_kb / 1024 ** 2
    lu.print_and_log(f"Peak memory for {bounds_str} in {tile_id}: {peak_gb:.2f} GB", False, logger_worker)

    return {"status": "success", "bounds_str": bounds_str, "years_written": years_written, "chunk_stats": chunk_stats}

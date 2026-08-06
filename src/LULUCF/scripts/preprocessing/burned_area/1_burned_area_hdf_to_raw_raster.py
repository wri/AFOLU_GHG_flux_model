"""
Burned area preprocessing Step 1:

This script converts monthly stacks of burned area hdfs into annual geotifs of the original extent, projection, and resolution.
Each hdf represents burned area for a given month in a given year, for a given horizontal-vertical (h-v) area.
Annual output rasters show everywhere that was burned in that year (1 for burned).

Run this preprocessing code on the hdfs in s3. The years to run are chosen in main().
hdf processing based on https://chatgpt.com/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e
I have not made this code align with other model components for the most part, e.g., no logs, no output stats, etc.

python -m src.utilities.create_cluster -n 1 -cn AFOLU_flux_model_scripts
python -m src.LULUCF.scripts.preprocessing.burned_area.1_burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

python -m src.utilities.create_cluster -n 100 -cn AFOLU_flux_model_scripts
python -m src.LULUCF.scripts.preprocessing.burned_area.1_burned_area_hdf_to_raw_raster -cn AFOLU_flux_model_scripts

268 h-v stacks for every year 2001-2024, except for 2005, which has 267 h-v stacks (missing h01v08 in original hdf site)
2001-2024: took about 1.5 hours to run, used about 600 Coiled credits, cost about $20 on AWS.

NOTE: Processed 2000 separately later (only last two months available).
Not actually using in the model but inputs for 2000 are expected anyhow.

To download a set of monthly raw h-v hdfs locally from s3 for one year for checking against geotif:
aws s3 cp s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ . --recursive --exclude "*" --include "*A2024*h24v02*"
"""

import argparse
import os
import re
import io
import boto3
import dask.array as da
import dask
import numpy as np
import rasterio
import tempfile
from dask import delayed
from dask.distributed import print
from rasterio.transform import from_bounds
from pyhdf.SD import SD, SDC
from dask import bag as db
from rasterio.crs import CRS
from collections import defaultdict

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu

# S3 Configuration
BUCKET_NAME = "gfw2-data"


MODIS_SINUSOIDAL_PROJ4 = (
    "+proj=sinu +lon_0=0 +datum=WGS84 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
)

s3 = boto3.client("s3")

### Utility Functions
def extract_hv_from_filename(filename):
    match = re.search(r"h(\d{2})v(\d{2})", filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # h, v
    return None, None

def modis_tile_bounds(h, v):
    """Returns MODIS tile boundaries in meters using the sinusoidal grid."""
    tile_size_m = 1111950  # MODIS tile size in meters
    x_min = -20015109 + (h * tile_size_m)
    y_max = 10007555 - (v * tile_size_m)
    x_max = x_min + tile_size_m
    y_min = y_max - tile_size_m
    return x_min, y_min, x_max, y_max

def extract_year_from_filename(filename):
    """Extracts the year (YYYY) from the MODIS burned area filename."""
    match = re.search(r"MCD64A1\.A(\d{4})", filename)
    if match:
        return int(match.group(1))
    return None

def extract_bounds_from_hdf(hdf_s3_key):
    """Extracts spatial bounds and CRS from an HDF file."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
    file_stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        hdf4_file = SD(tmp_path, SDC.READ)
        metadata = hdf4_file.attributes()

        if "WESTBOUNDINGCOORDINATE" in metadata:
            west, east = metadata["WESTBOUNDINGCOORDINATE"], metadata["EASTBOUNDINGCOORDINATE"]
            north, south = metadata["NORTHBOUNDINGCOORDINATE"], metadata["SOUTHBOUNDINGCOORDINATE"]
            crs = MODIS_SINUSOIDAL_PROJ4
            return [west, south, east, north], crs

        # Fallback to MODIS Tile grid calculations
        h, v = extract_hv_from_filename(hdf_s3_key)
        return modis_tile_bounds(h, v), MODIS_SINUSOIDAL_PROJ4

    except Exception as e:
        print(f"Error extracting bounds from {hdf_s3_key}: {e}")
        return None, None
    finally:
        os.remove(tmp_path)

def list_hv_year_files_from_s3(year):
    """Lists and groups all MODIS HDFs by (h, v) for a given year."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    hv_year_dict = defaultdict(list)

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=cn.burned_area_hdf_dir):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                file_year = extract_year_from_filename(key)
                if file_year == year and key.endswith(".hdf"):
                    h, v = extract_hv_from_filename(key)
                    hv_year = f"{year}_h{h}v{v}"
                    hv_year_dict[hv_year].append(key)

    return list(hv_year_dict.items())

def save_geotiff_to_s3(data, s3_key, bounds, crs=MODIS_SINUSOIDAL_PROJ4):
    """Saves raster data as a GeoTIFF to S3."""
    s3 = boto3.client("s3")

    try:
        crs = CRS.from_string(crs)
    except rasterio.errors.CRSError:
        crs = CRS.from_string(MODIS_SINUSOIDAL_PROJ4)

    transform = from_bounds(*bounds, data.shape[1], data.shape[0])

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "count": 1,
        "height": data.shape[0],
        "width": data.shape[1],
        "transform": transform,
        "crs": crs,
        "compress": "DEFLATE",
        "tiled": True,
    }

    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data, 1)
        s3.upload_fileobj(memfile, BUCKET_NAME, s3_key)

### Processing Functions
def read_modis_hdf_s3(hdf_s3_key):
    """Reads an HDF file from S3 and extracts Burn Date band, along with its bounds and CRS."""
    s3 = boto3.client("s3")
    file_stream = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, hdf_s3_key, file_stream)
    file_stream.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        hdf4_file = SD(tmp_path, SDC.READ)
        burn_date_data = hdf4_file.select("Burn Date")[:] if "Burn Date" in hdf4_file.datasets() else np.zeros((2400, 2400), dtype=np.uint8)
        burned_data = np.where(burn_date_data > 0, 1, 0)

        bounds, crs = extract_bounds_from_hdf(hdf_s3_key)
        return da.from_array(burned_data, chunks=(1000, 1000)), bounds, crs

    finally:
        os.remove(tmp_path)

def process_hv_year(hv_year_hdf_files):
    """Processes all HDFs for a given h-v-year."""
    hv_year, hdf_files = hv_year_hdf_files
    tasks = [delayed(read_modis_hdf_s3)(f) for f in hdf_files]
    results = dask.compute(*tasks)

    print(f"  Done with reading {hv_year}")

    dask_arrays, bounds_list, crs_list = zip(*results)
    bounds, crs = bounds_list[0], crs_list[0]
    merged_data = da.stack(dask_arrays, axis=0).max(axis=0)

    print(f"  Done merging dask arrays for {hv_year}")

    s3_key = f"{cn.burned_area_hdf_converted_to_raw_raster_dir}{hv_year}.tif"
    save_geotiff_to_s3(merged_data.compute(), s3_key, bounds, crs)

    print(f"  Done uploading rasters for {hv_year}")

### Main Function
def main(cluster_name, run_local, selected_years):

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Iterates through years. All h-v chunks are parallelized within a year.
    # Years are not parallelized because it was too many tasks for Dask, I think.
    # It was working better to iterate through years rather than try to do all h-v-year combos together.
    for year in selected_years:
        hv_year_files = list_hv_year_files_from_s3(year)
        print(f"Processing {year} with {len(hv_year_files)} h-v tiles: {uu.timestr()}...")
        db.from_sequence(hv_year_files).map(process_hv_year).compute()

    # Closes the Dask client if not running locally
    if not run_local:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cn', '--cluster_name', type=str, required=True)
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    args = parser.parse_args()

    # Sequentially process each year. Upper year is +1 the actual final year to process.
    main(args.cluster_name, args.run_local, list(range(2000, 2025)))










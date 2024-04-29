"""
Load dadap density data, resample to correct grid, and reclassify to binary
"""

import rioxarray
import rasterio
import xarray as xr
import pandas as pd
import pycountry
import os
from dask.distributed import Client
from src.utils import universal_util as uu

# create a dask cluster and make this notebook its client
# client = get_dask_ecs_client(n_workers=100)
# client

# Print the current working directory
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir())

tiles_test = ["00N_110E.tif"]
tiles_full = ["00N_000E.tif", "00N_010E.tif", "00N_020E.tif", "00N_030E.tif", "00N_040E.tif", "00N_040W.tif", "00N_050W.tif", "00N_060W.tif", "00N_070W.tif", "00N_080W.tif", "00N_090E.tif", "00N_090W.tif", "00N_100E.tif", "00N_110E.tif", "00N_120E.tif", "00N_130E.tif", "00N_140E.tif", "00N_150E.tif", "00N_160E.tif", "10N_000E.tif", "10N_010E.tif", "10N_010W.tif", "10N_020E.tif", "10N_020W.tif", "10N_030E.tif", "10N_040E.tif", "10N_050E.tif", "10N_050W.tif", "10N_060W.tif", "10N_070E.tif", "10N_070W.tif", "10N_080E.tif", "10N_080W.tif", "10N_090E.tif", "10N_090W.tif", "10N_100E.tif", "10N_110E.tif", "10N_120E.tif", "10N_130E.tif", "10N_140E.tif", "10S_010E.tif", "10S_020E.tif", "10S_030E.tif", "10S_040E.tif", "10S_040W.tif", "10S_050E.tif", "10S_050W.tif", "10S_060W.tif", "10S_070W.tif", "10S_080W.tif", "10S_100E.tif", "10S_110E.tif", "10S_120E.tif", "10S_130E.tif", "10S_140E.tif", "10S_150E.tif", "10S_160E.tif", "10S_170E.tif", "20N_000E.tif", "20N_010E.tif", "20N_010W.tif", "20N_020E.tif", "20N_020W.tif", "20N_030E.tif", "20N_040E.tif", "20N_050E.tif", "20N_060W.tif", "20N_070E.tif", "20N_070W.tif", "20N_080E.tif", "20N_080W.tif", "20N_090E.tif", "20N_090W.tif", "20N_100E.tif", "20N_100W.tif", "20N_110E.tif", "20N_110W.tif", "20N_120E.tif", "20N_120W.tif", "20S_010E.tif", "20S_020E.tif", "20S_030E.tif", "20S_040E.tif", "20S_050W.tif", "20S_060W.tif", "20S_070W.tif", "20S_080W.tif", "20S_110E.tif", "20S_120E.tif", "20S_130E.tif", "20S_140E.tif", "20S_150E.tif", "20S_160E.tif", "30N_010W.tif", "30N_020W.tif", "30N_030E.tif", "30N_040E.tif", "30N_050E.tif", "30N_060E.tif", "30N_070E.tif", "30N_080E.tif", "30N_080W.tif", "30N_090E.tif", "30N_090W.tif", "30N_100E.tif", "30N_100W.tif", "30N_110E.tif", "30N_110W.tif", "30N_120E.tif", "30N_120W.tif", "30S_010E.tif", "30S_020E.tif", "30S_030E.tif", "30S_060W.tif", "30S_070W.tif", "30S_080W.tif", "30S_090W.tif", "30S_110E.tif", "30S_120E.tif", "30S_130E.tif", "30S_140E.tif", "30S_150E.tif", "30S_170E.tif", "40N_000E.tif", "40N_010E.tif", "40N_010W.tif", "40N_020E.tif", "40N_030E.tif", "40N_040E.tif", "40N_050E.tif", "40N_060E.tif", "40N_070E.tif", "40N_070W.tif", "40N_080E.tif", "40N_080W.tif", "40N_090E.tif", "40N_090W.tif", "40N_100E.tif", "40N_100W.tif", "40N_110E.tif", "40N_110W.tif", "40N_120E.tif", "40N_120W.tif", "40N_130E.tif", "40N_130W.tif", "40N_140E.tif", "40S_070W.tif", "40S_080W.tif", "40S_140E.tif", "40S_160E.tif", "40S_170E.tif", "50N_000E.tif", "50N_010E.tif", "50N_010W.tif", "50N_020E.tif", "50N_030E.tif", "50N_040E.tif", "50N_050E.tif", "50N_060W.tif", "50N_070E.tif", "50N_070W.tif", "50N_080E.tif", "50N_080W.tif", "50N_090E.tif", "50N_090W.tif", "50N_100E.tif", "50N_100W.tif", "50N_110E.tif", "50N_110W.tif", "50N_120E.tif", "50N_120W.tif", "50N_130E.tif", "50N_130W.tif", "50N_140E.tif", "50N_150E.tif", "50S_060W.tif", "50S_070W.tif", "50S_080W.tif", "60N_000E.tif", "60N_010E.tif", "60N_010W.tif", "60N_020E.tif", "60N_020W.tif", "60N_030E.tif", "60N_040E.tif", "60N_050E.tif", "60N_060E.tif", "60N_060W.tif", "60N_070E.tif", "60N_070W.tif", "60N_080E.tif", "60N_080W.tif", "60N_090E.tif", "60N_090W.tif", "60N_100E.tif", "60N_100W.tif", "60N_110E.tif", "60N_110W.tif", "60N_120E.tif", "60N_120W.tif", "60N_130W.tif", "60N_140E.tif", "60N_140W.tif", "60N_150E.tif", "60N_150W.tif", "60N_160E.tif", "60N_160W.tif", "60N_170E.tif", "60N_170W.tif", "60N_180W.tif", "70N_000E.tif", "70N_010E.tif", "70N_010W.tif", "70N_020E.tif", "70N_020W.tif", "70N_030E.tif", "70N_030W.tif", "70N_040E.tif", "70N_050E.tif", "70N_060E.tif", "70N_070E.tif", "70N_080E.tif", "70N_080W.tif", "70N_090E.tif", "70N_090W.tif", "70N_100E.tif", "70N_100W.tif", "70N_110E.tif", "70N_110W.tif", "70N_120E.tif", "70N_120W.tif", "70N_130E.tif", "70N_130W.tif", "70N_140E.tif", "70N_140W.tif", "70N_150E.tif", "70N_150W.tif", "70N_160E.tif", "70N_160W.tif", "70N_170E.tif", "70N_180W.tif", "80N_020E.tif", "80N_060E.tif", "80N_070E.tif", "80N_080E.tif", "80N_090E.tif", "80N_110E.tif", "80N_120E.tif", "80N_120W.tif", "80N_130W.tif", "80N_140E.tif", "80N_140W.tif"]

# specify full or test:
tiles = tiles_test

peatlands_uri = "s3://gfw-data-lake/gfw_peatlands/v20230302/raster/epsg-4326/10/40000/is/geotiff/"
pixel_area_uri = "s3://gfw-data-lake/gfw_pixel_area/v20150327/raster/epsg-4326/10/40000/m2/geotiff/"
dadap_uri = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/raw/Dadap_SEA_Drainage/canal_length_data/canal_length_1km.tif"

s3_base_dir = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/"
dadap_pattern = "dadap_density"
dadap_dir = os.path.join(s3_base_dir,"inputs/processed/dadap_density/")

def get_tile_dataset(uri, tile, name, template=None):
    try:
        return rioxarray.open_rasterio(uri + tile, chunks=4000, default_name=name).squeeze("band")
    except rasterio.errors.RasterioIOError as e:
        if template is not None:
            return xr.zeros_like(template)
        else:
            raise e

def coords(tile_id):
    NS = tile_id.split("_")[0][-1:]
    EW = tile_id.split("_")[1][-1:]

    if NS == 'S':
        ymax =-1*int(tile_id.split("_")[0][:2])
    else:
        ymax = int(str(tile_id.split("_")[0][:2]))

    if EW == 'W':
        xmin = -1*int(str(tile_id.split("_")[1][:3]))
    else:
        xmin = int(str(tile_id.split("_")[1][:3]))


    ymin = str(int(ymax) - 10)
    xmax = str(int(xmin) + 10)

    return xmin, ymin, xmax, ymax

def get_dataset(uri, name, template=None):
    try:
        return rioxarray.open_rasterio(uri, chunks=4000, default_name=name).squeeze("band")
    except rasterio.errors.RasterioIOError as e:
        if template is not None:
            return xr.zeros_like(template)
        else:
            raise e

def print_raster_info(raster):
    """
    Print detailed information about a rioxarray object.

    :param raster: rioxarray object to get information from
    """

    # Basic raster information
    print("Raster Information:")
    print(f"  Name: {raster.name if hasattr(raster, 'name') else 'Not set'}")  # Default name for the raster
    print(f"  Dimensions: {raster.dims}")  # Names of the dimensions (e.g., x, y, band)
    print(f"  Shape: {raster.shape}")  # Shape of the raster (e.g., (1, 512, 512))
    print(f"  CRS: {raster.rio.crs}")  # Coordinate Reference System
    print(f"  Bounds: {raster.rio.bounds()}")  # Spatial extent (xmin, ymin, xmax, ymax)

    # Additional raster metadata
    print("Raster Metadata:")
    print(f"  Count: {raster.rio.count}")  # Number of bands in the raster
    print(f"  Data Type: {type(raster)}")  # Data type of the raster bands
    print(f"  NoData Value: {raster.rio.nodata}")  # NoData value indicating missing data

dadap_density = get_dataset(dadap_uri, "dadap_density", template=None)
print_raster_info(dadap_density)


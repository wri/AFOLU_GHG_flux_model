import boto3
import rasterio
import geopandas as gpd
from shapely.geometry import box
import os

"""
this script accepts an s3 directory input and creates shapefile footprints of all 
rasters within the directory
"""
#todo automatically upload to s3

s3_bucket = "gfw2-data"

def get_raster_files_from_s3(s3_directory, s3_client):
    bucket, prefix = s3_directory.replace("s3://", "").split("/", 1)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    raster_files = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.tif'):
                raster_files.append(f"s3://{bucket}/{obj['Key']}")
    return raster_files

def create_tile_index(s3_directories, output_dir):
    s3_client = boto3.client('s3')

    for dataset_name, s3_directory in s3_directories.items():
        tile_index = []
        raster_files = get_raster_files_from_s3(s3_directory, s3_client)
        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                bounds = src.bounds
                tile_id = os.path.basename(raster_file)
                geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                tile_index.append({'tile_id': tile_id, 'geometry': geometry})

        gdf = gpd.GeoDataFrame(tile_index, crs="EPSG:4326")
        output_shapefile = os.path.join(output_dir, f"{dataset_name}_tile_index.shp")
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        print(f"Tile index shapefile created at {output_shapefile}")

if __name__ == "__main__":
    # Dictionary of dataset names and their corresponding S3 paths
    s3_directories = {
        "dadap": f"s3://{s3_bucket}/climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m/",
        "engert": f"s3://{s3_bucket}/climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m/",
        "grip": f"s3://{s3_bucket}/climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/30m/",
        "osm_canals": f"s3://{s3_bucket}/climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/"
    }
    output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\indices"  # Output directory for the shapefiles
    create_tile_index(s3_directories, output_dir)

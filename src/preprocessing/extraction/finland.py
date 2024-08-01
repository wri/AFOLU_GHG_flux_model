import geopandas as gpd
import rasterio
from shapely.geometry import box
import gc
import logging
import os
import boto3
from utilities import download_shapefile_from_s3, read_shapefile_from_s3, rasterize_shapefile, compress_file, delete_file_if_exists

"""
This script processes raster tiles for Finland extraction areas, converting vector data to raster format and uploading results to S3.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_bucket_name = 'gfw2-data'
s3_finland_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/extraction/Finland_turvetuotantoalueet/turvetuotantoalueet_jalkikaytto'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_tile_index_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/index/Global_Peatlands'

# Local paths
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"  # Update to your own temporary directory
os.makedirs(local_temp_dir, exist_ok=True)

def filter_finland(gdf):
    """
    Filters the Finland shapefile data to include only relevant extraction areas.

    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame containing Finland shapefile data.

    Returns:
    GeoDataFrame: Filtered GeoDataFrame with only relevant extraction areas.
    """
    # Example filtering, adjust according to actual requirements
    return gdf[gdf['attribute_name'] == 'desired_value']

def process_tile(tile_key, finland_bounds):
    """
    Processes a single tile: reads the raster, rasterizes the shapefile data, saves and compresses the result, and uploads it to S3.

    Parameters:
    tile_key (str): The S3 key for the tile.
    finland_bounds (shapely.geometry.box): The bounding box of the Finland shapefile.

    Returns:
    None
    """
    tile_id = '_'.join(os.path.basename(tile_key).split('_')[:2])
    logging.info(f"Processing tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'
    local_output_path = os.path.join(local_temp_dir, f"finland_extraction_{tile_id}.tif")
    compressed_output_path = os.path.join(local_temp_dir, f"compressed_finland_extraction_{tile_id}.tif")
    s3_output_path = f"climate/AFOLU_flux_model/organic_soils/inputs/processed/extraction/{tile_id}.tif"

    if os.path.exists(local_output_path):
        logging.info(f"Output file for tile {tile_id} already exists. Skipping processing.")
        return

    try:
        with rasterio.Env(GDAL_SHAPE_RESTORE_SHX='YES', AWS_SESSION=boto3.Session()):
            logging.info(f"Opening input raster for tile {tile_id}")
            with rasterio.open(s3_input_path) as src:
                tile_bounds = box(*src.bounds)
                if not tile_bounds.intersects(finland_bounds):
                    logging.info(f"Skipping tile {tile_id} as it does not overlap with Finland bounds")
                    return

                tile_transform = src.transform
                tile_width = src.width
                tile_height = src.height

                # Read the shapefile from S3
                logging.info(f"Reading Finland shapefile for tile {tile_id}")
                gdf = read_shapefile_from_s3(s3_finland_shapefile_prefix, local_temp_dir, s3_bucket_name)

                # Ensure the Finland data is in WGS 1984
                if gdf.crs != "EPSG:4326":
                    logging.info(f"Reprojecting Finland data to WGS 1984 for tile {tile_id}")
                    gdf = gdf.to_crs(epsg=4326)

                gdf = gdf.to_crs(src.crs)

                # Filter Finland GeoDataFrame to correct attributes
                filtered_gdf = filter_finland(gdf)

                # Rasterize the shapefile
                logging.info(f"Rasterizing shapefile for tile {tile_id}")
                raster_data = rasterize_shapefile(filtered_gdf, tile_bounds.bounds, tile_transform, tile_width, tile_height)

                if raster_data is None:
                    logging.error(f"Rasterization failed for tile {tile_id}")
                    return

                # Save the rasterized shapefile as a GeoTIFF
                logging.info(f"Saving rasterized data for tile {tile_id}")
                profile = src.profile
                profile.update(dtype=rasterio.uint8, compress='lzw', count=1, nodata=0)

                with rasterio.open(local_output_path, 'w', **profile) as dst:
                    dst.write(raster_data, 1)

                # Compress the file
                logging.info(f"Compressing rasterized file for tile {tile_id}")
                compress_file(local_output_path, compressed_output_path)

                # Upload the compressed file to S3
                logging.info(f"Uploading compressed file for tile {tile_id} to S3")
                s3_client.upload_file(compressed_output_path, s3_bucket_name, s3_output_path)

                # Remove local files
                delete_file_if_exists(local_output_path)
                delete_file_if_exists(compressed_output_path)

                del gdf, raster_data
                gc.collect()
                logging.info(f"Finished processing tile {tile_id}")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_all_tiles():
    """
    Processes all tiles that intersect with the Finland bounds.

    Returns:
    None
    """
    logging.info("Reading Finland shapefile")
    finland_gdf = read_shapefile_from_s3(s3_finland_shapefile_prefix, local_temp_dir, s3_bucket_name)

    # Ensure the Finland data is in WGS 1984
    if finland_gdf.crs != "EPSG:4326":
        logging.info("Reprojecting Finland data to WGS 1984")
        finland_gdf = finland_gdf.to_crs(epsg=4326)

    finland_bounds = box(*finland_gdf.total_bounds)

    logging.info("Reading tile index shapefile")
    tile_index_gdf = read_shapefile_from_s3(s3_tile_index_shapefile_prefix, local_temp_dir, s3_bucket_name)

    logging.info("Filtering relevant tiles")
    relevant_tiles = tile_index_gdf[tile_index_gdf.geometry.intersects(finland_bounds)]
    tile_ids = relevant_tiles['tile_id'].tolist()

    for tile_id in tile_ids:
        tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
        process_tile(tile_key, finland_bounds)

def main():
    """
    Main function to initiate the processing of all relevant tiles.

    Returns:
    None
    """
    logging.info("Starting processing of all tiles")
    process_all_tiles()
    logging.info("Finished processing all tiles")

if __name__ == "__main__":
    main()

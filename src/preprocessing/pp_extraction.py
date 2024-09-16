import os
import logging
import boto3
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import box
import gc

# Import your custom modules (ensure these are correctly set up in your environment)
import pp_utilities as uu  # Utilities module with helper functions
import constants_and_names as cn  # Module containing constants like paths and S3 prefixes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')

def read_shapefile_from_s3(s3_prefix, local_dir):
    """
    Read a shapefile from S3 into a GeoDataFrame.

    Args:
        s3_prefix (str): The S3 prefix (path without the file extension) for the shapefile.
        local_dir (str): The local directory where the files will be saved.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame.
    """
    try:
        extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
        for ext in extensions:
            s3_path = f"{s3_prefix}{ext}"
            local_path = os.path.join(local_dir, os.path.basename(s3_prefix) + ext)
            logging.info(f"Downloading {s3_path} to {local_path}")
            s3_client.download_file(cn.s3_bucket_name, s3_path, local_path)
        shapefile_path = os.path.join(local_dir, os.path.basename(s3_prefix) + '.shp')
        gdf = gpd.read_file(shapefile_path)
        logging.info(f"Shapefile {shapefile_path} successfully loaded with {len(gdf)} features")
    except Exception as e:
        logging.error(f"Error reading shapefile from S3: {e}")
        gdf = gpd.GeoDataFrame()  # Return an empty GeoDataFrame in case of error
    return gdf

def get_tile_bounds(tile_id):
    """
    Retrieve the bounds of a specific tile from the global peatlands index shapefile.

    Args:
        tile_id (str): Tile ID to look for.

    Returns:
        tuple: Bounding box of the tile (minx, miny, maxx, maxy).
    """
    try:
        index_shapefile = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        if not os.path.exists(index_shapefile):
            logging.info("Global peatlands index not found locally. Downloading...")
            read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir)
        gdf = gpd.read_file(index_shapefile)
        tile = gdf[gdf['tile_id'] == tile_id]
        if tile.empty:
            logging.error(f"Tile {tile_id} not found in index shapefile.")
            return None
        bounds = tile.geometry.bounds.iloc[0]
        logging.info(f"Tile bounds for {tile_id}: {bounds}")
    except Exception as e:
        logging.error(f"Error retrieving tile bounds for {tile_id}: {e}")
        bounds = None
    return bounds

def rasterize_shapefile_with_ref(gdf, output_raster_path, transform, width, height, fill_value=0, burn_value=1, dtype='uint8', tile_id=None):
    """
    Rasterize a GeoDataFrame using a reference raster's transform and dimensions.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to rasterize.
        output_raster_path (str): Path to save the rasterized output.
        transform (Affine): Affine transform for the output raster.
        width (int): Width of the output raster.
        height (int): Height of the output raster.
        fill_value (int, optional): Value to fill in the output raster where there are no features. Defaults to 0.
        burn_value (int, optional): Value to burn into the raster where features are present. Defaults to 1.
        dtype (str, optional): Data type of the output raster. Defaults to 'uint8'.
        tile_id (str, optional): Tile ID for logging purposes.

    Returns:
        None
    """
    try:
        # Prepare shapes
        shapes = [(geom, burn_value) for geom in gdf.geometry if geom.is_valid and not geom.is_empty]

        if not shapes:
            logging.warning(f"No shapes to rasterize for tile {tile_id}.")
            return  # Early exit since there's nothing to rasterize

        logging.info(f"Rasterizing {len(shapes)} shapes for tile {tile_id}.")

        # Create a blank raster with the specified shape and dtype
        raster_data = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=fill_value,
            dtype=dtype
        )

        # Define the metadata for the output raster
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype,
            "crs": gdf.crs,
            "transform": transform,
            "nodata": fill_value,
            "compress": "DEFLATE",
            "tiled": True
        }

        # Write the rasterized data to the output file
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)

        logging.info(f"Rasterized shapefile saved to {output_raster_path}")

    except Exception as e:
        logging.error(f"Error rasterizing shapefile for tile {tile_id}: {e}")
        raise  # Re-raise the exception to be caught in the calling function

def process_tile(tile_id, run_mode='default'):
    """
    Processes a single tile: rasterizes the vector data for the tile, clips to bounds, and uploads to S3.

    Parameters:
        tile_id (str): ID of the tile to process.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    output_dir = cn.datasets['extraction']['finland']['local_processed']
    os.makedirs(output_dir, exist_ok=True)

    s3_output_dir = cn.datasets['extraction']['finland']['s3_processed']
    local_output_path = os.path.join(output_dir, f"{tile_id}_extraction.tif")
    s3_output_path = os.path.join(s3_output_dir, f"{tile_id}_extraction.tif").replace("\\", "/")

    if run_mode != 'test':
        try:
            s3_client.head_object(Bucket=cn.s3_bucket_name, Key=s3_output_path)
            logging.info(f"{s3_output_path} already exists on S3. Skipping processing.")
            return
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.info(f"{s3_output_path} does not exist on S3. Processing the tile.")
            else:
                logging.error(f"Error checking existence of {s3_output_path} on S3: {e}")
                return

    logging.info(f"Starting processing of the tile {tile_id}")

    try:
        # Get tile bounds and CRS from the input raster tile
        s3_input_raster_path = f"/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix}{tile_id}_peat_mask_processed.tif"
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_raster_path) as src:
                tile_bounds = src.bounds
                tile_transform = src.transform
                tile_width = src.width
                tile_height = src.height
                tile_crs = src.crs

        # Load the Finland shapefile data
        local_raw_prefix = os.path.join(cn.local_temp_dir, 'turvetuotantoalueet_jalkikaytto')
        local_raw_path = local_raw_prefix + '.shp'

        if not os.path.exists(local_raw_path):
            logging.info(f"Shapefile not found locally, downloading from S3.")
            uu.download_shapefile_from_s3(
                cn.datasets['extraction']['finland']['s3_raw'],
                cn.local_temp_dir,
                cn.s3_bucket_name
            )

        gdf = gpd.read_file(local_raw_path)

        # Reproject Finland data to match the tile's CRS
        if gdf.crs != tile_crs:
            logging.info(f"Reprojecting Finland data to match tile CRS for tile {tile_id}")
            gdf = gdf.to_crs(tile_crs)

        # Fix invalid geometries
        gdf['geometry'] = gdf['geometry'].buffer(0)
        gdf = gdf.make_valid()
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[~gdf.geometry.is_empty & ~gdf.geometry.isna()]

        # Clip GeoDataFrame to tile bounds using geopandas.clip function
        tile_box = box(*tile_bounds)
        gdf_tile = gpd.clip(gdf, tile_box)

        # Check and convert if necessary
        logging.info(f"Type of gdf_tile after clipping: {type(gdf_tile)}")
        if isinstance(gdf_tile, gpd.GeoSeries):
            logging.info("Converting GeoSeries to GeoDataFrame")
            gdf_tile = gdf_tile.to_frame(name='geometry')
            gdf_tile = gpd.GeoDataFrame(gdf_tile, geometry='geometry', crs=gdf.crs)

        # Ensure 'geometry' column exists
        if 'geometry' not in gdf_tile.columns:
            logging.error(f"'geometry' column missing in gdf_tile for tile {tile_id}")
            return

        if gdf_tile.empty:
            logging.info(f"No data in tile {tile_id} after clipping. Skipping.")
            return
        else:
            logging.info(f"Number of geometries in tile {tile_id}: {len(gdf_tile)}")

        # Fix invalid geometries in gdf_tile
        gdf_tile['geometry'] = gdf_tile['geometry'].buffer(0)
        gdf_tile = gdf_tile.make_valid()
        gdf_tile = gdf_tile.explode(index_parts=False).reset_index(drop=True)
        gdf_tile = gdf_tile[~gdf_tile.geometry.is_empty & ~gdf_tile.geometry.isna()]

        # Drop any remaining invalid geometries
        invalid_geoms = gdf_tile[~gdf_tile.is_valid]
        if not invalid_geoms.empty:
            logging.warning(f"Dropping {len(invalid_geoms)} invalid geometries in tile {tile_id}.")
            gdf_tile = gdf_tile[gdf_tile.is_valid]

        if gdf_tile.empty:
            logging.warning(f"All geometries in tile {tile_id} are invalid after cleaning. Skipping.")
            return

        # Rasterize the shapefile using the tile's transform and dimensions
        rasterize_shapefile_with_ref(
            gdf_tile,
            local_output_path,
            transform=tile_transform,
            width=tile_width,
            height=tile_height,
            fill_value=0,
            burn_value=1,
            dtype="uint8",
            tile_id=tile_id
        )

        # Check raster data before uploading
        with rasterio.open(local_output_path) as src:
            data = src.read(1)
            non_zero_count = np.count_nonzero(data)
            logging.info(f"Tile {tile_id} raster has {non_zero_count} non-zero pixels.")

            if non_zero_count == 0:
                logging.warning(f"Raster for tile {tile_id} contains no data. Skipping upload.")
                uu.delete_file_if_exists(local_output_path)
                return  # Skip uploading empty rasters

        if run_mode != 'test':
            # Upload to S3
            uu.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)
            logging.info(f"Uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")

            # Remove local file
            uu.delete_file_if_exists(local_output_path)
            logging.info(f"Intermediate output raster {local_output_path} removed")

        logging.info(f"Tile {tile_id} processed successfully")

        del gdf, gdf_tile
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")


def main(tile_id=None, run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.

    Parameters:
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
        run_mode (str, optional): The mode to run the script ('default' or 'test'). Defaults to 'default'.

    Returns:
        None
    """
    try:
        logging.info("Starting main processing routine for Finland peat extraction dataset")

        if tile_id:
            process_tile(tile_id, run_mode)
        else:
            # Load the Finland shapefile
            finland_shapefile_name = 'turvetuotantoalueet_jalkikaytto'
            finland_shapefile_path = os.path.join(cn.local_temp_dir, finland_shapefile_name + '.shp')
            if not os.path.exists(finland_shapefile_path):
                logging.info(f"Finland shapefile not found locally, downloading from S3.")
                uu.download_shapefile_from_s3(
                    cn.datasets['extraction']['finland']['s3_raw'],
                    cn.local_temp_dir,
                    cn.s3_bucket_name
                )
            gdf_finland = gpd.read_file(finland_shapefile_path)

            # Ensure Finland GeoDataFrame has valid geometries
            gdf_finland['geometry'] = gdf_finland['geometry'].buffer(0)
            logging.info(f"After buffer(0), type of gdf_finland: {type(gdf_finland)}")

            # Explode multi-part geometries
            gdf_finland = gdf_finland.explode(index_parts=False)
            logging.info(f"After explode, type of gdf_finland: {type(gdf_finland)}")

            # Check if gdf_finland is a GeoSeries and convert if necessary
            if isinstance(gdf_finland, gpd.GeoSeries):
                logging.info("Converting GeoSeries to GeoDataFrame")
                gdf_finland = gpd.GeoDataFrame(geometry=gdf_finland)
                gdf_finland.crs = gdf_tiles.crs

            gdf_finland = gdf_finland.reset_index(drop=True)
            gdf_finland = gdf_finland[~gdf_finland.geometry.is_empty & ~gdf_finland.geometry.isna()]

            # Load tile index shapefile
            index_shapefile = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
            if not os.path.exists(index_shapefile):
                logging.info("Global peatlands index not found locally. Downloading...")
                read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir)
            gdf_tiles = gpd.read_file(index_shapefile)

            # Reproject Finland data to match tiles CRS if necessary
            if gdf_finland.crs != gdf_tiles.crs:
                logging.info(f"Reprojecting Finland data to match tiles CRS")
                gdf_finland = gdf_finland.to_crs(gdf_tiles.crs)

            # Perform spatial join to find tiles intersecting with Finland
            tiles_intersecting_finland = gpd.sjoin(gdf_tiles, gdf_finland, how='inner', predicate='intersects')
            tile_ids = tiles_intersecting_finland['tile_id'].unique()

            logging.info(f"Found {len(tile_ids)} tiles intersecting with Finland dataset.")

            for tid in tile_ids:
                process_tile(tid, run_mode)

    except Exception as e:
        logging.error(f"Error in main processing routine: {e}")
    finally:
        logging.info("Processing completed")



if __name__ == "__main__":
    # Example usage
    # To process a specific tile, provide the tile_id
    # main(tile_id='70N_020E', run_mode='default')
    main(tile_id=None, run_mode='default')


    # To process all tiles, uncomment the following line
    # main(run_mode='default')

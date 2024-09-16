import os
import logging
import boto3
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import box
import gc
import rasterio.warp
import rasterio.mask
from rasterio.vrt import WarpedVRT

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

def process_finland(dataset, tile_id=None, run_mode='default'):
    """
    Process the Finland dataset.

    Parameters:
        dataset (str): The dataset to process ('finland').
        tile_id (str, optional): Tile ID to process a specific tile. Defaults to None.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    try:
        logging.info("Starting processing routine for Finland peat extraction dataset")

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

        # Check if gdf_finland is a GeoSeries and convert if necessary
        if isinstance(gdf_finland, gpd.GeoSeries):
            logging.info("Converting GeoSeries to GeoDataFrame")
            gdf_finland = gpd.GeoDataFrame(geometry=gdf_finland)
            gdf_finland.crs = gdf_tiles.crs

        gdf_finland = gdf_finland.reset_index(drop=True)
        gdf_finland = gdf_finland[~gdf_finland.geometry.is_empty & ~gdf_finland.geometry.isna()]

        # Perform spatial join to find tiles intersecting with Finland
        tiles_intersecting_finland = gpd.sjoin(gdf_tiles, gdf_finland, how='inner', predicate='intersects')
        tile_ids = tiles_intersecting_finland['tile_id'].unique()

        logging.info(f"Found {len(tile_ids)} tiles intersecting with Finland dataset.")

        if tile_id:
            if tile_id in tile_ids:
                process_finland_tile(tile_id, gdf_finland, run_mode)
            else:
                logging.info(f"Tile {tile_id} does not intersect with Finland dataset. Skipping.")
        else:
            for tid in tile_ids:
                process_finland_tile(tid, gdf_finland, run_mode)

    except Exception as e:
        logging.error(f"Error processing Finland dataset: {e}")

def process_finland_tile(tile_id, gdf_finland, run_mode='default'):
    """
    Processes a single tile for the Finland dataset.

    Parameters:
        tile_id (str): ID of the tile to process.
        gdf_finland (GeoDataFrame): The Finland GeoDataFrame.
        run_mode (str): The mode to run the script ('default' or 'test').

    Returns:
        None
    """
    output_dir = cn.datasets['extraction']['finland']['local_processed']
    os.makedirs(output_dir, exist_ok=True)

    s3_output_dir = cn.datasets['extraction']['finland']['s3_processed']
    local_output_path = os.path.join(output_dir, f"{tile_id}_finland_extraction.tif")
    s3_output_path = os.path.join(s3_output_dir, f"{tile_id}_finland_extraction.tif").replace("\\", "/")

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

        # Reproject Finland data to match the tile's CRS
        if gdf_finland.crs != tile_crs:
            logging.info(f"Reprojecting Finland data to match tile CRS for tile {tile_id}")
            gdf_finland_tile = gdf_finland.to_crs(tile_crs)
        else:
            gdf_finland_tile = gdf_finland

        # Clip GeoDataFrame to tile bounds using geopandas.clip function
        tile_box = box(*tile_bounds)
        gdf_tile = gpd.clip(gdf_finland_tile, tile_box)

        # Check and convert if necessary
        logging.info(f"Type of gdf_tile after clipping: {type(gdf_tile)}")
        if isinstance(gdf_tile, gpd.GeoSeries):
            logging.info("Converting GeoSeries to GeoDataFrame")
            gdf_tile = gdf_tile.to_frame(name='geometry')
            gdf_tile = gpd.GeoDataFrame(gdf_tile, geometry='geometry', crs=gdf_finland_tile.crs)

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

        del gdf_tile
        gc.collect()

    except Exception as e:
        logging.error(f"Error processing tile {tile_id}: {e}")

def process_raster_dataset(dataset, tile_id=None, run_mode='default'):
    """
    Process raster datasets (Ireland and Russia).
    """
    try:
        # Load the raster dataset from S3
        s3_raster_path = cn.datasets['extraction'][dataset]['s3_raw']
        local_raster_path = os.path.join(cn.local_temp_dir, f"{dataset}_raw.tif")

        # Download the raster if not already present locally
        if not os.path.exists(local_raster_path):
            logging.info(f"Downloading {dataset} raster dataset from S3.")
            s3_client.download_file(cn.s3_bucket_name, s3_raster_path, local_raster_path)

        # Open the raster dataset
        with rasterio.open(local_raster_path) as raster_dataset:
            raster_crs = raster_dataset.crs

            # Check and set CRS if missing
            if raster_crs is None:
                logging.warning(f"No CRS found for {dataset} raster dataset. Setting CRS manually.")
                # Set the CRS here (replace with the correct CRS)
                raster_crs = raster_dataset.crs = 'EPSG:29902'  # Replace with the correct CRS

            logging.info(f"{dataset.capitalize()} raster CRS: {raster_crs}")
            logging.info(f"Raster dataset bounds: {raster_dataset.bounds}")

            # Reproject raster bounds to EPSG:4326 (tile index CRS)
            raster_bounds_4326 = rasterio.warp.transform_bounds(
                raster_crs,
                'EPSG:4326',
                *raster_dataset.bounds,
                densify_pts=21
            )
            raster_bbox_4326 = box(*raster_bounds_4326)
            logging.info(f"Raster bounds in EPSG:4326: {raster_bounds_4326}")

        # Load tile index shapefile (in EPSG:4326)
        index_shapefile = os.path.join(cn.local_temp_dir, os.path.basename(cn.index_shapefile_prefix) + '.shp')
        if not os.path.exists(index_shapefile):
            logging.info("Global peatlands index not found locally. Downloading...")
            read_shapefile_from_s3(cn.index_shapefile_prefix, cn.local_temp_dir)
        gdf_tiles = gpd.read_file(index_shapefile)
        logging.info(f"Tile index shapefile CRS: {gdf_tiles.crs}")

        # Ensure tile index is in EPSG:4326
        if gdf_tiles.crs != 'EPSG:4326':
            logging.info("Reprojecting tile index to EPSG:4326")
            gdf_tiles = gdf_tiles.to_crs('EPSG:4326')

        # Create a GeoDataFrame for raster bounds in EPSG:4326
        gdf_raster_bbox = gpd.GeoDataFrame({'geometry': [raster_bbox_4326]}, crs='EPSG:4326')

        # Identify intersecting tiles
        tiles_intersecting_raster = gpd.sjoin(gdf_tiles, gdf_raster_bbox, how='inner', predicate='intersects')
        tile_ids = tiles_intersecting_raster['tile_id'].unique()

        logging.info(f"Found {len(tile_ids)} tiles intersecting with {dataset} dataset: {tile_ids}")

        if tile_id:
            if tile_id in tile_ids:
                process_raster_tile(dataset, tile_id, local_raster_path, run_mode)
            else:
                logging.info(f"Tile {tile_id} does not intersect with {dataset} dataset. Skipping.")
        else:
            for tid in tile_ids:
                process_raster_tile(dataset, tid, local_raster_path, run_mode)

    except Exception as e:
        logging.error(f"Error processing {dataset} dataset: {e}")

def process_raster_tile(dataset, tile_id, local_raster_path, run_mode='default'):
    """
    Process a single tile for raster datasets, including 'hansenize' step.
    """
    try:
        logging.info(f"Processing tile {tile_id} for dataset {dataset}")

        # Prepare output paths
        output_dir = cn.datasets['extraction'][dataset]['local_processed']
        os.makedirs(output_dir, exist_ok=True)
        local_output_path = os.path.join(output_dir, f"{tile_id}_{dataset}_extraction.tif")
        s3_output_dir = cn.datasets['extraction'][dataset]['s3_processed']
        s3_output_path = os.path.join(s3_output_dir, f"{tile_id}_extraction.tif").replace("\\", "/")

        # Check if the output already exists
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

        # Get tile properties from the peat tile (EPSG:4326)
        s3_input_raster_path = f"/vsis3/{cn.s3_bucket_name}/{cn.peat_tiles_prefix}{tile_id}_peat_mask_processed.tif"
        with rasterio.Env(AWS_SESSION=boto3.Session()):
            with rasterio.open(s3_input_raster_path) as peat_tile:
                tile_bounds = peat_tile.bounds
                tile_crs = peat_tile.crs
                tile_transform = peat_tile.transform
                tile_width = peat_tile.width
                tile_height = peat_tile.height

        logging.info(f"Tile CRS: {tile_crs}")

        # Open the source raster dataset
        with rasterio.open(local_raster_path) as src_raster:
            # Create a WarpedVRT to match the tile properties (hansenize step)
            with WarpedVRT(
                src_raster,
                crs=tile_crs,
                resampling=rasterio.warp.Resampling.nearest,
                transform=tile_transform,
                width=tile_width,
                height=tile_height,
                nodata=0,
                dtype='uint8'
            ) as vrt:
                # Read the data
                data = vrt.read(1)

                # Mask the data with the tile bounds
                data, _ = rasterio.mask.mask(
                    vrt,
                    [box(*tile_bounds)],
                    crop=True,
                    all_touched=True,
                    invert=False
                )

                # If the data is empty, skip processing
                if not data.any():
                    logging.info(f"No data in tile {tile_id} after masking. Skipping.")
                    return

                # Ensure data is uint8 and nodata is set to 0
                data = data.astype('uint8')
                data[data == vrt.nodata] = 0

                # Update metadata
                out_meta = vrt.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": tile_height,
                    "width": tile_width,
                    "transform": tile_transform,
                    "crs": tile_crs,
                    "count": 1,
                    "dtype": 'uint8',
                    "compress": "DEFLATE",
                    "tiled": True,
                    "nodata": 0
                })

                # Write the output raster
                with rasterio.open(local_output_path, "w", **out_meta) as dest:
                    dest.write(data, 1)

        logging.info(f"Tile {tile_id} processed and saved to {local_output_path}")

        if run_mode != 'test':
            # Upload to S3
            uu.upload_file_to_s3(local_output_path, cn.s3_bucket_name, s3_output_path)
            logging.info(f"Uploaded {local_output_path} to s3://{cn.s3_bucket_name}/{s3_output_path}")

            # Remove local file
            uu.delete_file_if_exists(local_output_path)
            logging.info(f"Intermediate output raster {local_output_path} removed")

    except Exception as e:
        logging.error(f"Error processing tile {tile_id} for dataset {dataset}: {e}")

def main(dataset='finland', tile_id=None, run_mode='default'):
    """
    Main function to orchestrate the processing based on provided arguments.
    """
    try:
        logging.info(f"Starting main processing routine for {dataset} peat extraction dataset")

        if dataset == 'finland':
            process_finland(dataset, tile_id, run_mode)
        elif dataset in ['ireland', 'russia']:
            process_raster_dataset(dataset, tile_id, run_mode)
        else:
            logging.error(f"Dataset '{dataset}' is not recognized. Please choose 'finland', 'ireland', or 'russia'.")

    except Exception as e:
        logging.error(f"Error in main processing routine: {e}")
    finally:
        logging.info("Processing completed")


if __name__ == "__main__":
    # Example usage
    # Process Finland dataset
    # main(dataset='finland', tile_id=None, run_mode='default')
    # Process Ireland dataset
    # main(dataset='ireland', tile_id=None, run_mode='default')
    # Process Russia dataset
    main(dataset='russia', tile_id=None, run_mode='default')

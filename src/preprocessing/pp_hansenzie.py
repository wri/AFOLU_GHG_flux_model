import rasterio
import os
import subprocess
import logging
import gc
"""
This is a temporary script I am using to text flexibility of hansen function with mini functions for
merging or clipping
"""

def clip_raster_to_tile(input_path, output_path, bounds, nodata_value=None):
    """
    Clips a raster to the given bounds.

    Parameters:
    input_path (str): Path to the input raster file.
    output_path (str): Path to save the clipped raster file.
    bounds (tuple): Bounding box (minx, miny, maxx, maxy) to clip the raster.
    nodata_value (float, optional): NoData value for the output raster. Defaults to None.
    """
    minx, miny, maxx, maxy = bounds
    gdalwarp_cmd = [
        'gdalwarp',
        '-te', str(minx), str(miny), str(maxx), str(maxy),
        '-dstnodata', str(nodata_value) if nodata_value is not None else '-inf',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'TILED=YES',
        '-overwrite',
        input_path,
        output_path
    ]

    logging.info(f"Clipping raster to bounds {bounds} with command: {' '.join(gdalwarp_cmd)}")
    subprocess.run(gdalwarp_cmd, check=True)

def merge_and_clip_rasters(raster_paths, tile_bounds, output_raster_path, s3_bucket, s3_prefix, run_mode='default', nodata_value=None):
    """
    Merges multiple rasters and clips them to the tile bounds.

    Parameters:
    raster_paths (list): List of raster file paths to merge.
    tile_bounds (tuple): Bounding box coordinates for the tile (minx, miny, maxx, maxy).
    output_raster_path (str): Path to save the merged and clipped raster file.
    s3_bucket (str): S3 bucket name for uploading results.
    s3_prefix (str): S3 prefix for saving results.
    run_mode (str): The mode to run the script ('default' or 'test'). Defaults to 'default'.
    nodata_value (float, optional): NoData value for the output raster. Defaults to None.

    Returns:
    None
    """
    try:
        # Open and stack the rasters using rioxarray
        rasters = [rioxarray.open_rasterio(raster_path, masked=True) for raster_path in raster_paths]
        merged_raster = xr.concat(rasters, dim='band').sum(dim='band')

        # Clip to the tile bounds
        merged_raster_clipped = merged_raster.rio.clip_box(*tile_bounds)

        # Set nodata value if specified
        if nodata_value is not None:
            merged_raster_clipped = merged_raster_clipped.rio.write_nodata(nodata_value, inplace=True)

        # Save the merged and clipped raster to a temporary file
        merged_raster_clipped.rio.to_raster(output_raster_path)

        # Upload to S3 if not in test mode
        if run_mode != 'test':
            s3_key = os.path.join(s3_prefix, os.path.basename(output_raster_path)).replace("\\", "/")
            logging.info(f"Uploading {output_raster_path} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(output_raster_path, s3_bucket, s3_key)
            logging.info(f"Uploaded {output_raster_path} to s3://{s3_bucket}/{s3_key}")

            # Delete local files if not in test mode
            os.remove(output_raster_path)

        # Clean up memory
        del merged_raster, merged_raster_clipped
        gc.collect()

    except Exception as e:
        logging.error(f"Error in merge_and_clip_rasters function: {e}")


def hansenize(
    input_paths,
    output_raster_path,
    bounds,
    s3_bucket,
    s3_prefix,
    run_mode='default',
    nodata_value=None
):
    """
    Processes input shapefiles or rasters to produce a 30-meter resolution raster
    clipped into homogeneous 10x10 degree tiles.

    Parameters:
    input_paths (list): List of paths to the input shapefile or raster files.
    output_raster_path (str): Path to the output raster file.
    bounds (tuple): Bounding box (minx, miny, maxx, maxy) to clip the raster.
    s3_bucket (str): S3 bucket name for uploading results.
    s3_prefix (str): S3 prefix for saving results.
    run_mode (str): Mode to run the script ('default' or 'test'). Defaults to 'default'.
    nodata_value (float, optional): NoData value for the output raster. Defaults to None.
    """
    try:
        # Check if the input is a list of rasters
        if isinstance(input_paths, list):
            # Merge and clip if multiple rasters
            merge_and_clip_rasters(input_paths, output_raster_path, bounds, nodata_value)
        else:
            # Clip directly if a single raster
            clip_raster_to_tile(input_paths, output_raster_path, bounds, nodata_value)

        # Log data statistics
        with rasterio.open(output_raster_path) as output:
            data = output.read(1)
            logging.info(f"Processed raster stats - Min: {data.min()}, Max: {data.max()}, NoData: {output.nodata}")

        # Upload to S3 if not in test mode
        if run_mode != 'test':
            s3_key = os.path.join(s3_prefix, os.path.basename(output_raster_path)).replace("\\", "/")
            logging.info(f"Uploading {output_raster_path} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(output_raster_path, s3_bucket, s3_key)
            logging.info(f"Uploaded {output_raster_path} to s3://{s3_bucket}/{s3_key}")

            # Delete local files if not in test mode
            os.remove(output_raster_path)

        gc.collect()

    except Exception as e:
        logging.error(f"Error in hansenize function: {e}")

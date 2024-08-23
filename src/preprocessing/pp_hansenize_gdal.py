import os
import logging
import gc
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def clip_raster_to_tile(input_path, output_path, bounds, nodata_value=None, dtype='Byte'):
    """
    Clips a raster to specified bounds using GDAL's gdalwarp command.

    Args:
        input_path (str): Path to the input raster.
        output_path (str): Path to save the clipped raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
        nodata_value (float): NoData value to set for the output raster.
        dtype (str): Data type for the output raster.
    """
    try:
        minx, miny, maxx, maxy = bounds
        gdalwarp_cmd = [
            'gdalwarp',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
            '-ot', dtype,  # Set the output data type
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite',
            input_path,
            output_path
        ]
        logging.info(f"Clipping raster with command: {' '.join(gdalwarp_cmd)}")
        subprocess.run(gdalwarp_cmd, check=True)
        logging.info(f"Raster clipped successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error during clipping: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during raster clipping: {e}")


def merge_and_clip_rasters_gdal(raster_paths, output_path, bounds, nodata_value=None, dtype='Byte'):
    """
    Merges multiple rasters and clips to specified bounds using GDAL.

    Args:
        raster_paths (list): List of raster paths to merge.
        output_path (str): Path to save the merged and clipped raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
        nodata_value (float): NoData value to set for the output raster.
        dtype (str): Data type for the output raster.
    """
    try:
        minx, miny, maxx, maxy = bounds
        temp_merged_path = output_path.replace('.tif', '_merged.tif')
        gdal_merge_cmd = [
            'gdalwarp',
            '-te', str(minx), str(miny), str(maxx), str(maxy),
            '-tr', '0.00025', '0.00025',  # Set the output resolution explicitly
            '-dstnodata', str(nodata_value) if nodata_value is not None else '0',
            '-ot', dtype,  # Set the output data type
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'TILED=YES',
            '-overwrite'
        ] + raster_paths + [temp_merged_path]

        logging.info(f"Merging and clipping rasters with command: {' '.join(gdal_merge_cmd)}")
        subprocess.run(gdal_merge_cmd, check=True)

        # Move the merged output to final output path
        os.rename(temp_merged_path, output_path)
        logging.info(f"Rasters merged and clipped successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"GDAL error during merge and clip: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during raster merging and clipping: {e}")
    finally:
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)


def hansenize_gdal(input_paths, output_path, bounds, nodata_value=None, dtype='Byte'):
    """
    Main function for processing using GDAL.

    Args:
        input_paths (str or list): Input raster path or list of paths to process.
        output_path (str): Path to save the processed raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for processing.
        nodata_value (float): NoData value to set for the output raster.
        dtype (str): Data type for the output raster.
    """
    if isinstance(input_paths, list):
        merge_and_clip_rasters_gdal(input_paths, output_path, bounds, nodata_value, dtype)
    else:
        clip_raster_to_tile(input_paths, output_path, bounds, nodata_value, dtype)

    gc.collect()

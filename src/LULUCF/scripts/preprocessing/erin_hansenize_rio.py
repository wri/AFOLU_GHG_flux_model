#pp_hansenize_rio.py

import os
import logging
import gc
import rasterio
import rioxarray as rxr
import xarray as xr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clip_raster_to_tile_rio(input_path, output_path, bounds, nodata_value=None):
    """
    Clips a raster to specified bounds using rioxarray.

    Args:
        input_path (str): Path to the input raster.
        output_path (str): Path to save the clipped raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
        nodata_value (float, optional): NoData value to set for the output raster.
    """
    try:
        minx, miny, maxx, maxy = bounds
        logging.info(f"Clipping raster {input_path} to bounds {bounds}")

        raster = rxr.open_rasterio(input_path, masked=True)
        clipped_raster = raster.rio.clip_box(minx, miny, maxx, maxy)

        if nodata_value is not None:
            clipped_raster = clipped_raster.rio.write_nodata(nodata_value, inplace=True)
            logging.info(f"Set NoData value to {nodata_value} for {output_path}")

        clipped_raster.rio.to_raster(output_path, compress='DEFLATE', tiled=True)
        logging.info(f"Raster clipped successfully: {output_path}")

    except Exception as e:
        logging.error(f"Error during raster clipping with rioxarray: {e}")
        raise


def merge_and_clip_rasters_rio(raster_paths, output_path, bounds, nodata_value=None):
    """
    Merges multiple rasters and clips to specified bounds using rioxarray.

    Args:
        raster_paths (list): List of raster paths to merge.
        output_path (str): Path to save the merged and clipped raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
        nodata_value (float, optional): NoData value to set for the output raster.
    """
    try:
        logging.info(f"Merging rasters: {raster_paths}")
        rasters = [rxr.open_rasterio(raster_path, masked=True) for raster_path in raster_paths]
        merged_raster = xr.concat(rasters, dim='band').sum(dim='band')

        minx, miny, maxx, maxy = bounds
        logging.info(f"Clipping merged raster to bounds {bounds}")
        clipped_raster = merged_raster.rio.clip_box(minx, miny, maxx, maxy)

        if nodata_value is not None:
            clipped_raster = clipped_raster.rio.write_nodata(nodata_value, inplace=True)
            logging.info(f"Set NoData value to {nodata_value} for {output_path}")

        clipped_raster.rio.to_raster(output_path, compress='DEFLATE', tiled=True)
        logging.info(f"Rasters merged and clipped successfully: {output_path}")

    except Exception as e:
        logging.error(f"Error during raster merging and clipping with rioxarray: {e}")
        raise


def hansenize_rio(input_paths, output_path, bounds, nodata_value=None):
    """
    Main function for processing using rioxarray.

    Args:
        input_paths (str or list): Input raster path or list of paths to process.
        output_path (str): Path to save the processed raster.
        bounds (tuple): Bounding box (minx, miny, maxx, maxy) for processing.
        nodata_value (float, optional): NoData value to set for the output raster.
    """
    logging.info(f"Starting hansenize processing for {input_paths} with output {output_path}")
    try:
        if isinstance(input_paths, list):
            merge_and_clip_rasters_rio(input_paths, output_path, bounds, nodata_value)
        else:
            clip_raster_to_tile_rio(input_paths, output_path, bounds, nodata_value)
    except Exception as e:
        logging.error(f"Error in hansenize_rio processing: {e}")
        raise
    finally:
        gc.collect()
        logging.info(f"Finished hansenize processing for {output_path}")


if __name__ == "__main__":
    # Example usage
    input_rasters = ["path/to/raster1.tif", "path/to/raster2.tif"]
    output_raster = "path/to/output_raster.tif"
    bounding_box = (minx, miny, maxx, maxy)  # Define your bounding box here
    hansenize_rio(input_rasters, output_raster, bounding_box)
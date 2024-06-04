import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

target_shapefile = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
source_shapefile = (r'C:\GIS\Data\Global\Wetlands\Raw\Global'
                    r'\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\index'
                    r'\descals_tile_index.shp')
source_raster_path = (r'C:\GIS\Data\Global\Wetlands\Raw\Global'
                      r'\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\oil_palm_map')
target_raster_path = "path_to_output_directory"  # Optional, can be set in process_tile
s3_tiles_prefix = 's3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/descals_plantation/'

def load_shapefiles(target_shapefile, source_shapefile):
    logging.info(f"Loading target shapefile: {target_shapefile}")
    target_index = gpd.read_file(target_shapefile)
    logging.info(f"Loading source shapefile: {source_shapefile}")
    source_index = gpd.read_file(source_shapefile)
    return target_index, source_index

def get_overlapping_tiles(target_tile, source_index):
    logging.info(f"Finding overlapping tiles for target tile with ID: {target_tile['tile_id'].values[0]}")
    overlaps = source_index[source_index.geometry.intersects(target_tile.geometry.iloc[0])]
    logging.info(f"Found {len(overlaps)} overlapping tiles")
    return overlaps

def read_and_reproject(src_path, dst_crs):
    logging.info(f"Reading and reprojecting raster: {src_path}")
    with rasterio.open(src_path) as src:
        data = src.read(1)
        mask = (data == 1) | (data == 2)
        data = data * mask

        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        dst_array = np.empty((height, width), dtype=src.dtypes[0])
        reproject(
            source=data,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    logging.info(f"Reprojection completed for raster: {src_path}")
    return dst_array, kwargs

def process_tile(target_tile, source_index, source_raster_path, target_raster_path):
    logging.info(f"Processing target tile with ID: {target_tile['tile_id'].values[0]}")
    overlapping_tiles = get_overlapping_tiles(target_tile, source_index)

    src_arrays = []
    src_transforms = []

    for _, src_tile in overlapping_tiles.iterrows():
        tile_id = src_tile['tile_id']
        src_path = os.path.join(source_raster_path, tile_id)  # Adjust path concatenation as necessary

        if not os.path.exists(src_path):
            logging.warning(f"Raster file {src_path} does not exist. Skipping...")
            continue

        dst_array, dst_meta = read_and_reproject(src_path, target_tile.crs)
        src_arrays.append(dst_array)
        src_transforms.append(dst_meta['transform'])

    if src_arrays:
        logging.info("Merging source arrays into a mosaic")
        mosaic, mosaic_transform = merge(src_arrays, transforms=src_transforms)

        target_bounds = target_tile.geometry.bounds
        target_window = rasterio.windows.from_bounds(*target_bounds, transform=mosaic_transform)
        clipped = mosaic[
            target_window.row_off:target_window.row_off + target_window.height,
            target_window.col_off:target_window.col_off + target_window.width
        ]

        dst_meta.update({
            "height": clipped.shape[0],
            "width": clipped.shape[1],
            "transform": rasterio.windows.transform(target_window, mosaic_transform)
        })

        tile_id = target_tile['tile_id']
        output_path = os.path.join(target_raster_path, f"{tile_id}_peat_mask_processed.tif")
        logging.info(f"Saving clipped raster to {output_path}")
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            dst.write(clipped, 1)
        logging.info(f"Tile processing completed for tile ID: {tile_id}")

def main(target_shapefile, source_shapefile, source_raster_path, target_raster_path='output_path', tile_id=None):
    logging.info("Starting main process")
    target_index, source_index = load_shapefiles(target_shapefile, source_shapefile)

    if tile_id:
        logging.info(f"Processing single tile with ID: {tile_id}")
        target_tile = target_index[target_index['tile_id'] == tile_id]
        process_tile(target_tile, source_index, source_raster_path, target_raster_path)
    else:
        logging.info("Processing all tiles")
        for _, target_tile in target_index.iterrows():
            process_tile(target_tile, source_index, source_raster_path, target_raster_path)

if __name__ == "__main__":
    main(target_shapefile, source_shapefile, source_raster_path, target_raster_path, tile_id='00N_110E')

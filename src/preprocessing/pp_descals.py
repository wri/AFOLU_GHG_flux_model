import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import numpy as np

target_shapefile = 'path_to_target_shapefile.shp'
source_shapefile = 'path_to_source_shapefile.shp'
source_raster_path = 'path_to_source_raster_directory'
target_raster_path = 'path_to_output_directory'  # Optional, can be set in process_tile
s3_tiles_prefix = 's3://your-bucket/tiles/'

def load_shapefiles(target_shapefile, source_shapefile):
    target_index = gpd.read_file(target_shapefile)
    source_index = gpd.read_file(source_shapefile)
    return target_index, source_index

def get_overlapping_tiles(target_tile, source_index):
    overlaps = source_index[source_index.geometry.intersects(target_tile.geometry.iloc[0])]
    return overlaps

def read_and_reproject(src_path, dst_crs):
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
        return dst_array, kwargs

def process_tile(target_tile, source_index, source_raster_path, target_raster_path):
    overlapping_tiles = get_overlapping_tiles(target_tile, source_index)

    src_arrays = []
    src_transforms = []

    for _, src_tile in overlapping_tiles.iterrows():
        src_path = f"{source_raster_path}/{src_tile['path']}"  # Adjust path concatenation as necessary
        dst_array, dst_meta = read_and_reproject(src_path, target_tile.crs)
        src_arrays.append(dst_array)
        src_transforms.append(dst_meta['transform'])

    if src_arrays:
        mosaic, mosaic_transform = merge(src_arrays, transforms=src_transforms)

        # Clip the mosaic to the target tile
        target_bounds = target_tile.geometry.bounds
        target_window = rasterio.windows.from_bounds(*target_bounds, transform=mosaic_transform)
        clipped = mosaic[target_window.row_off:target_window.row_off + target_window.height,
                         target_window.col_off:target_window.col_off + target_window.width]

        # Update metadata with the target transform and bounds
        dst_meta.update({
            "height": clipped.shape[0],
            "width": clipped.shape[1],
            "transform": rasterio.windows.transform(target_window, mosaic_transform)
        })

        # Save the clipped raster
        tile_id = target_tile['tile_id']
        output_path = f"{target_raster_path}/{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            dst.write(clipped, 1)

def main(target_shapefile, source_shapefile, source_raster_path, target_raster_path='output_path', tile_id=None):
    target_index, source_index = load_shapefiles(target_shapefile, source_shapefile)

    if tile_id:
        target_tile = target_index[target_index['tile_id'] == tile_id]
        process_tile(target_tile, source_index, source_raster_path, target_raster_path)
    else:
        for _, target_tile in target_index.iterrows():
            process_tile(target_tile, source_index, source_raster_path, target_raster_path)

if __name__ == "__main__":
    main(target_shapefile, source_shapefile, source_raster_path, target_raster_path, tile_id='00N_110E')

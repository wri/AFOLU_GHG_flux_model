import geopandas as gpd
import rioxarray
from shapely.geometry import box

#todo... make sure tile has data in it

def get_tile_ids_from_raster(raster_path, index_shapefile_path):
    """
    Get the tile IDs that intersect with the bounds of the input raster.

    Args:
        raster_path (str): Path to the input raster file.
        index_shapefile_path (str): Path to the global index shapefile containing tile IDs.

    Returns:
        list: List of tile IDs that intersect with the raster bounds.
    """
    # Load the raster and get its bounds
    raster = rioxarray.open_rasterio(raster_path)
    raster_bounds = raster.rio.bounds()

    # Create a bounding box from the raster bounds
    raster_bbox = box(*raster_bounds)

    # Load the global index shapefile
    index_gdf = gpd.read_file(index_shapefile_path)

    # Find the tiles that intersect with the raster bounding box
    intersecting_tiles = index_gdf[index_gdf.geometry.intersects(raster_bbox)]

    # Get the tile IDs from the intersecting tiles
    tile_ids = intersecting_tiles["tile_id"].tolist()

    return tile_ids

# # Example usage
# raster_path = r"C:\path\to\your\raster_file.tif"
# index_shapefile_path = r"C:\path\to\your\index_shapefile.shp"

# tile_ids = get_tile_ids_from_raster(raster_path, index_shapefile_path)
# print("Intersecting tile IDs:", tile_ids)

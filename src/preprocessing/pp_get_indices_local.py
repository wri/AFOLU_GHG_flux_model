import os
import rasterio
import geopandas as gpd
from shapely.geometry import box

def get_raster_files_from_local(directory):
    raster_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):
                raster_files.append(os.path.join(root, file))
    return raster_files

def create_tile_index(local_directories, output_dir):
    for dataset_name, local_directory in local_directories.items():
        tile_index = []
        raster_files = get_raster_files_from_local(local_directory)
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
    # Dictionary of dataset names and their corresponding local paths
    local_directories = {
        "descals": r"C:\GIS\Data\Global\Wetlands\Raw\Global"
                   r"\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\oil_palm_map"
    }
    output_dir = (r"C:\GIS\Data\Global\Wetlands\Raw\Global"
                  r"\High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019\index")  # Output
    # directory for the shapefiles
    create_tile_index(local_directories, output_dir)

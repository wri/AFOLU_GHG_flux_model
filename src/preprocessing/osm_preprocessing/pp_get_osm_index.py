import geopandas as gpd
from shapely.geometry import Polygon, box
import os
import logging
import pandas as pd

"""
This script performs the following tasks:
1. Converts a set of .poly files (actually .txt files) into shapefiles.
2. Extracts the bounding boxes from these shapefiles.
3. Overlays a tile index shapefile with these region shapefiles.
4. Updates the tile index shapefile with information on which regions each tile overlaps.
5. Saves the updated tile index shapefile, which can be used by a main processing script to efficiently determine which regions to process for each tile.

Usage:
1. Ensure the .poly files (saved as .txt) are located in a specified directory.
2. Provide the path to the tile index shapefile.
3. The script will create shapefiles for each region, extract their bounding boxes, overlay these with the tile index, and save the updated tile index shapefile with region overlap information.
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_poly(file_path):
    """
    Parse a .poly file and convert it into a list of Polygon objects.

    Args:
        file_path (str): Path to the .poly (actually .txt) file.

    Returns:
        list: A list of shapely.geometry.Polygon objects.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    polygons = []
    current_polygon = []
    inside_ring = False

    for line in lines:
        line = line.strip()
        if line.startswith("END"):
            if current_polygon:
                polygons.append(current_polygon)
                current_polygon = []
            inside_ring = False
        elif line and not line.startswith("!"):
            coords = line.split()
            if len(coords) == 2:
                current_polygon.append((float(coords[0]), float(coords[1])))
            inside_ring = True

    return [Polygon(polygon) for polygon in polygons]

def poly_to_shapefile(poly_file_path, shapefile_path):
    """
    Convert a .poly file to a shapefile.

    Args:
        poly_file_path (str): Path to the .poly (actually .txt) file.
        shapefile_path (str): Path to save the output shapefile.

    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame created from the .poly file.
    """
    polygons = parse_poly(poly_file_path)

    if not polygons:
        logging.error(f"No polygons found in {poly_file_path}")
        return None

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

    # Save to shapefile
    gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    logging.info(f"Shapefile saved to {shapefile_path}")
    return gdf

def get_bounds_from_shapefile(shapefile_path):
    """
    Extract the bounding box from a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        tuple: The bounding box of the shapefile (minx, miny, maxx, maxy).
    """
    gdf = gpd.read_file(shapefile_path)
    bounds = gdf.total_bounds  # Returns (minx, miny, maxx, maxy)
    return bounds

def process_poly_files(poly_folder, output_dir):
    """
    Process all .poly (actually .txt) files in a folder and extract their bounding boxes.

    Args:
        poly_folder (str): Path to the folder containing .poly files.
        output_dir (str): Path to save the output shapefiles and bounding box shapefile.

    Returns:
        dict: A dictionary of region names and their bounding boxes.
    """
    bounds_dict = {}
    bounds_gdf_list = []

    for poly_file in os.listdir(poly_folder):
        if poly_file.endswith('.txt'):  # Process only .txt files
            poly_file_path = os.path.join(poly_folder, poly_file)
            region_name = os.path.splitext(os.path.basename(poly_file))[0]
            shapefile_path = os.path.join(output_dir, f"{region_name}.shp")

            # Convert .poly to shapefile
            gdf = poly_to_shapefile(poly_file_path, shapefile_path)

            if gdf is not None:
                # Extract bounds
                bounds = get_bounds_from_shapefile(shapefile_path)
                bounds_dict[region_name] = bounds
                logging.info(f"Bounds for {region_name}: {bounds}")

                # Create a GeoDataFrame for the bounding box
                bounds_geom = box(*bounds)
                bounds_gdf = gpd.GeoDataFrame({'region': [region_name], 'geometry': [bounds_geom]}, crs="EPSG:4326")
                bounds_gdf_list.append(bounds_gdf)

    if bounds_gdf_list:
        # Concatenate all bounding box GeoDataFrames
        all_bounds_gdf = gpd.GeoDataFrame(pd.concat(bounds_gdf_list, ignore_index=True), crs="EPSG:4326")
        # Save to shapefile
        bounds_shapefile_path = os.path.join(output_dir, 'bounds.shp')
        all_bounds_gdf.to_file(bounds_shapefile_path, driver='ESRI Shapefile')
        logging.info(f"Bounds shapefile saved to {bounds_shapefile_path}")

    return bounds_dict

# Define the region name mappings
region_name_mappings = {
    "north-america": "north-america-latest.osm.pbf",
    "africa": "africa-latest.osm.pbf",
    "antarctica": "antarctica-latest.osm.pbf",
    "asia": "asia-latest.osm.pbf",
    "australia-oceania": "australia-oceania-latest.osm.pbf",
    "central-america": "central-america-latest.osm.pbf",
    "europe": "europe-latest.osm.pbf",
    "south-america": "south-america-latest.osm.pbf"
}

def overlay_tiles_with_regions(tile_index_path, region_shapefile_dir, output_tile_index_path):
    """
    Overlay the tile index shapefile with region shapefiles and update the tile index with region information.

    Args:
        tile_index_path (str): Path to the tile index shapefile.
        region_shapefile_dir (str): Path to the folder containing region shapefiles.
        output_tile_index_path (str): Path to save the updated tile index shapefile.
    """
    # Read the tile index shapefile
    tiles_gdf = gpd.read_file(tile_index_path)

    # Initialize a column for region names
    tiles_gdf['regions'] = None

    # Iterate through each region shapefile
    for region_file in os.listdir(region_shapefile_dir):
        if region_file.endswith('.shp'):
            region_name = os.path.splitext(region_file)[0]
            mapped_region_name = region_name_mappings.get(region_name, region_name)
            region_path = os.path.join(region_shapefile_dir, region_file)

            # Read the region shapefile
            region_gdf = gpd.read_file(region_path)

            # Check for intersection between tiles and the current region
            for idx, tile in tiles_gdf.iterrows():
                if region_gdf.intersects(tile.geometry).any():
                    if tiles_gdf.at[idx, 'regions'] is None:
                        tiles_gdf.at[idx, 'regions'] = mapped_region_name
                    else:
                        tiles_gdf.at[idx, 'regions'] += f",{mapped_region_name}"

    # Save the updated tile index shapefile
    tiles_gdf.to_file(output_tile_index_path, driver='ESRI Shapefile')
    logging.info(f"Updated tile index shapefile saved to {output_tile_index_path}")

def main():
    # Define paths
    poly_folder = r"C:\GIS\Data\Global\OSM\poly_files"
    output_dir = r"C:\GIS\Data\Global\OSM\poly_files"
    tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
    output_tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands_Indexed.shp"

    os.makedirs(output_dir, exist_ok=True)

    # Process all .poly (actually .txt) files and get their bounds
    bounds_dict = process_poly_files(poly_folder, output_dir)

    # Output the bounds dictionary
    for region, bounds in bounds_dict.items():
        print(f"{region}: {bounds}")

    # Overlay tiles with regions and save the updated tile index
    overlay_tiles_with_regions(tile_index_path, output_dir, output_tile_index_path)

if __name__ == '__main__':
    main()

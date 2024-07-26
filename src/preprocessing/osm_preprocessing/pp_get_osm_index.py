"""
This script processes .poly files to create shapefiles, extracts bounding boxes, and overlays them with a tile index to
determine intersections. The primary steps are:
1. Parsing .poly files to create shapefiles.
2. Extracting bounding boxes from these shapefiles.
3. Overlaying the tile index with the region shapefiles to determine intersections and update the tile index.

The script includes detailed logging for troubleshooting and debugging purposes.

Modules used:
- os: For file and directory operations.
- logging: For logging messages and debugging information.
- geopandas: For handling geospatial data.
- shapely: For geometric operations.
- pandas: For data manipulation.

Functions:
- parse_poly(file_path): Parses a .poly file and returns a list of shapely Polygons.
- poly_to_shapefile(poly_file_path, shapefile_path): Converts a .poly file to a shapefile and returns a GeoDataFrame.
- get_bounds_from_shapefile(shapefile_path): Extracts the bounding box of the geometries in a shapefile.
- process_poly_files(poly_folder, output_dir): Processes all .poly files in a directory, converts them to shapefiles, and extracts their bounding boxes.
- overlay_tiles_with_regions(tile_index_path, region_shapefile_dir, output_tile_index_path): Overlays tile geometries with region geometries to determine intersections.
- main(): Main function to process .poly files and overlay them with the tile index.

Usage:
Run the script directly to process .poly files and update the tile index with region intersections.
"""

import os
import logging
import geopandas as gpd
from shapely.geometry import Polygon, box
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a custom logging filter to ignore specific warnings
class WarningFilter(logging.Filter):
    def filter(self, record):
        if "Value" in record.getMessage() and "Shape__Are" in record.getMessage():
            return False
        return True

# Add the filter to the logger
logging.getLogger().addFilter(WarningFilter())

def parse_poly(file_path):
    """
    Parses a .poly file and returns a list of shapely Polygons.

    Args:
        file_path (str): Path to the .poly file.

    Returns:
        List[Polygon]: List of polygons extracted from the file.
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
        elif line and not line.startswith("!") and not line.isdigit():
            coords = line.split()
            if len(coords) == 2:
                try:
                    current_polygon.append((float(coords[0]), float(coords[1])))
                except ValueError:
                    logging.warning(f"Skipping line in {file_path}: {line}")
                    continue
            inside_ring = True

    return [Polygon(polygon) for polygon in polygons]

def poly_to_shapefile(poly_file_path, shapefile_path):
    """
    Converts a .poly file to a shapefile and returns a GeoDataFrame.

    Args:
        poly_file_path (str): Path to the .poly file.
        shapefile_path (str): Path where the shapefile will be saved.

    Returns:
        GeoDataFrame: GeoDataFrame of the polygons.
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
    Extracts the bounding box of the geometries in a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        tuple: Bounding box of the shapefile (minx, miny, maxx, maxy).
    """
    gdf = gpd.read_file(shapefile_path)
    bounds = gdf.total_bounds  # Returns (minx, miny, maxx, maxy)
    return bounds

def process_poly_files(poly_folder, output_dir):
    """
    Processes all .poly files in a directory, converts them to shapefiles,
    and extracts their bounding boxes.

    Args:
        poly_folder (str): Directory containing .poly files.
        output_dir (str): Directory where the shapefiles and bounds will be saved.

    Returns:
        dict: Dictionary mapping region names to their bounding boxes.
    """
    bounds_dict = {}
    bounds_gdf_list = []

    for poly_file in os.listdir(poly_folder):
        if poly_file.endswith('.txt'):  # Process only .txt files
            poly_file_path = os.path.join(poly_folder, poly_file)
            region_name = os.path.splitext(os.path.basename(poly_file))[0]
            shapefile_path = os.path.join(output_dir, f"{region_name}.shp")

            logging.info(f"Processing .poly file: {poly_file_path}")
            print(f"Processing .poly file: {poly_file_path}")

            # Convert .poly to shapefile
            gdf = poly_to_shapefile(poly_file_path, shapefile_path)

            if gdf is not None:
                # Extract bounds
                bounds = get_bounds_from_shapefile(shapefile_path)
                bounds_dict[region_name] = bounds
                logging.info(f"Bounds for {region_name}: {bounds}")
                print(f"Bounds for {region_name}: {bounds}")

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
        print(f"Bounds shapefile saved to {bounds_shapefile_path}")

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
    Overlays tile geometries with region geometries to determine intersections.

    Args:
        tile_index_path (str): Path to the tile index shapefile.
        region_shapefile_dir (str): Directory containing region shapefiles.
        output_tile_index_path (str): Path where the updated tile index shapefile will be saved.
    """
    # Read the tile index shapefile
    tiles_gdf = gpd.read_file(tile_index_path)
    # Initialize the 'regions' field
    tiles_gdf['regions'] = None

    # Process each region shapefile
    for region_file in os.listdir(region_shapefile_dir):
        if region_file.endswith('.shp') and region_file != 'bounds.shp':  # Exclude bounds.shp
            region_name = os.path.splitext(region_file)[0]
            mapped_region_name = region_name_mappings.get(region_name, region_name)
            region_path = os.path.join(region_shapefile_dir, region_file)

            logging.info(f"Processing region shapefile: {region_path}")
            print(f"Processing region shapefile: {region_path}")

            # Read the region shapefile
            region_gdf = gpd.read_file(region_path)
            region_gdf = region_gdf.rename(columns=lambda x: f"{x}_region" if x in tiles_gdf.columns else x)

            # Add 'region' column to region_gdf
            region_gdf['region'] = mapped_region_name

            # Set geometry column back to the correct one
            region_gdf.set_geometry('geometry_region', inplace=True)

            # Print the bounds of the region shapefile
            region_bounds = region_gdf.total_bounds
            logging.info(f"Bounds for {region_name}: {region_bounds}")
            print(f"Bounds for {region_name}: {region_bounds}")

            # Iterate over each tile
            for idx, tile in tiles_gdf.iterrows():
                if tile.geometry.is_valid:
                    # Find intersecting regions
                    intersecting_regions = region_gdf[region_gdf.intersects(tile.geometry)]
                    if not intersecting_regions.empty:
                        # Concatenate intersecting region names
                        intersecting_region_names = ",".join(intersecting_regions['region'].unique())
                        logging.info(f"Tile {idx} intersects with regions: {intersecting_region_names}")
                        print(f"Tile {idx} intersects with regions: {intersecting_region_names}")
                        if tiles_gdf.at[idx, 'regions'] is None:
                            tiles_gdf.at[idx, 'regions'] = intersecting_region_names
                        else:
                            tiles_gdf.at[idx, 'regions'] += f",{intersecting_region_names}"
                    else:
                        logging.info(f"Tile {idx} does not intersect with any regions.")
                        print(f"Tile {idx} does not intersect with any regions.")
                else:
                    logging.warning(f"Tile {idx} has an invalid geometry: {tile.geometry}")

    # Save the updated tile index shapefile
    tiles_gdf.to_file(output_tile_index_path, driver='ESRI Shapefile')
    logging.info(f"Updated tile index shapefile saved to {output_tile_index_path}")
    print(f"Updated tile index shapefile saved to {output_tile_index_path}")

def main():
    """
    Main function to process .poly files and overlay them with the tile index.
    """
    poly_folder = r"C:\GIS\Data\Global\OSM\poly_files\OSM_bounds"
    output_dir = r"C:\GIS\Data\Global\OSM\poly_files\OSM_bounds"
    tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
    output_tile_index_path = r"C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands_Indexed.shp"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process .poly files to create shapefiles and extract bounding boxes
    bounds_dict = process_poly_files(poly_folder, output_dir)

    # Print bounds for debugging
    for region, bounds in bounds_dict.items():
        print(f"{region}: {bounds}")

    # Overlay tiles with regions to update the tile index
    overlay_tiles_with_regions(tile_index_path, output_dir, output_tile_index_path)

if __name__ == '__main__':
    main()

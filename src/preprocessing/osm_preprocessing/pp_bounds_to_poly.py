import geopandas as gpd
from shapely.geometry import Polygon


def get_chunk_bounds(chunk_params):
    """
    Returns list of all chunk boundaries within a bounding box for chunks of a given size.

    Parameters:
    chunk_params (list): A list containing [min_x, min_y, max_x, max_y, chunk_size]

    Returns:
    list: A list of chunk boundaries
    """
    min_x = chunk_params[0]
    min_y = chunk_params[1]
    max_x = chunk_params[2]
    max_y = chunk_params[3]
    chunk_size = chunk_params[4]

    x, y = (min_x, min_y)
    chunks = []

    # Polygon Size
    while y < max_y:
        while x < max_x:
            bounds = [
                x,
                y,
                x + chunk_size,
                y + chunk_size,
            ]
            chunks.append(bounds)
            x += chunk_size
        x = min_x
        y += chunk_size

    return chunks


def export_chunks_to_shapefile(chunk_params, filename):
    """
    Exports chunk bounds as polygons to a Shapefile with bounds as an attribute.

    Parameters:
    chunk_params (list): A list containing [min_x, min_y, max_x, max_y, chunk_size]
    filename (str): The name of the output Shapefile
    """
    chunks = get_chunk_bounds(chunk_params)

    polygons = []
    attributes = []

    for bounds in chunks:
        min_lon, min_lat, max_lon, max_lat = bounds
        polygon = Polygon(
            [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat)])
        polygons.append(polygon)
        attributes.append({'bounds': str(bounds)})

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, crs="EPSG:4326", geometry=polygons)

    # Export to Shapefile
    gdf.to_file(filename, driver='ESRI Shapefile')


def get_tile_bounds(global_index_shapefile, tile_id):
    """
    Filters the global index shapefile to find the bounds of the specified tile ID.

    Parameters:
    global_index_shapefile (str): Path to the global index shapefile
    tile_id (str): The tile ID to filter by

    Returns:
    list: A list containing [min_x, min_y, max_x, max_y] of the tile bounds
    """
    gdf = gpd.read_file(global_index_shapefile)
    tile = gdf[gdf['tile_id'] == tile_id]

    if tile.empty:
        raise ValueError(f"Tile ID {tile_id} not found in the global index shapefile.")

    bounds = tile.total_bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


# Main script to run the analysis
if __name__ == "__main__":
    global_index_shapefile = "C:\GIS\Data\Global\Wetlands\Raw\Global\gfw_peatlands\Global_Peatlands_Index\Global_Peatlands.shp"
    tile_id = "00N_080W"
    chunk_size = 2  # Set the desired chunk size

    # Get the bounds of the specified tile
    tile_bounds = get_tile_bounds(global_index_shapefile, tile_id)
    print(f'tile bounds are: {tile_bounds}')
    chunk_params = tile_bounds + [chunk_size]

    # Export the chunk bounds to a shapefile
    output_filename = f"C:/GIS/Data/Global/Wetlands/Raw/chunk_bounds/{tile_id}_chunks.shp"
    export_chunks_to_shapefile(chunk_params, output_filename)

    print(f"Chunk bounds exported to {output_filename}")

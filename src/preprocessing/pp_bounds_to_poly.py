import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from pp_utilities import get_tile_bounds  # Ensure this is correctly imported


def get_chunk_bounds(minx, miny, maxx, maxy, chunk_size):
    """
    Divide a bounding box into smaller chunks of the specified size.

    Args:
        minx (float): Minimum x-coordinate of the bounding box.
        miny (float): Minimum y-coordinate of the bounding box.
        maxx (float): Maximum x-coordinate of the bounding box.
        maxy (float): Maximum y-coordinate of the bounding box.
        chunk_size (float): Size of each chunk.

    Returns:
        list: List of tuples where each tuple contains a Polygon and its bounds.
    """
    chunks = []
    x_coords = np.arange(minx, maxx, chunk_size)
    y_coords = np.arange(miny, maxy, chunk_size)
    for x in x_coords:
        for y in y_coords:
            chunk_polygon = Polygon(
                [(x, y), (x + chunk_size, y), (x + chunk_size, y + chunk_size), (x, y + chunk_size)])
            chunk_bounds = (x, y, x + chunk_size, y + chunk_size)
            chunks.append((chunk_polygon, chunk_bounds))
    return chunks


def export_chunks_to_shapefile(chunk_params, output_filename):
    """
    Export chunk bounds to a shapefile, including chunk bounds as attributes.

    Args:
        chunk_params (list): List of parameters including bounding box and chunk size.
        output_filename (str): Path to the output shapefile.
    """
    try:
        minx, miny, maxx, maxy, chunk_size = chunk_params
        chunks = get_chunk_bounds(minx, miny, maxx, maxy, chunk_size)

        # Create attribute data
        chunk_data = []
        for chunk_polygon, chunk_bounds in chunks:
            chunk_data.append({
                'minx': chunk_bounds[0],
                'miny': chunk_bounds[1],
                'maxx': chunk_bounds[2],
                'maxy': chunk_bounds[3],
                'chunk_size': chunk_size
            })

        # Create GeoDataFrame with geometry and attributes
        gdf = gpd.GeoDataFrame(chunk_data, geometry=[chunk[0] for chunk in chunks])
        gdf.to_file(output_filename)
        print(f"Chunk bounds exported to {output_filename}")
    except Exception as e:
        print(f"Error exporting chunks to shapefile: {e}")


def get_tile_bounds(global_index_shapefile, tile_id):
    """
    Get the bounds of a specified tile from the global index shapefile.

    Args:
        global_index_shapefile (str): Path to the global index shapefile.
        tile_id (str): ID of the tile to get bounds for.

    Returns:
        tuple: A tuple of (minx, miny, maxx, maxy) representing the bounds of the tile.
    """
    gdf = gpd.read_file(global_index_shapefile)
    tile = gdf[gdf['tile_id'] == tile_id]
    if tile.empty:
        raise ValueError(f"Tile ID {tile_id} not found in the global index shapefile.")
    bounds = tile.total_bounds
    return bounds


# Main script to run the analysis
if __name__ == "__main__":
    global_index_shapefile = "C:/GIS/Data/Global/Wetlands/Raw/Global/gfw_peatlands/Global_Peatlands_Index/Global_Peatlands.shp"
    tile_id = "00N_110E"
    chunk_size = 0.25  # Set the desired chunk size

    # Get the bounds of the specified tile
    tile_bounds = get_tile_bounds(global_index_shapefile, tile_id)
    print(f'tile bounds are: {tile_bounds}')

    # Create chunk parameters
    chunk_params = list(tile_bounds) + [chunk_size]
    print(f'chunk_params are: {chunk_params}')

    # Export the chunk bounds to a shapefile
    output_filename = f"C:/GIS/Data/Global/Wetlands/Raw/chunk_bounds/{tile_id}_chunks.shp"
    export_chunks_to_shapefile(chunk_params, output_filename)

    print(f"Chunk bounds exported to {output_filename}")

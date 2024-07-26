import geopandas as gpd
from shapely.geometry import Polygon
from pp_utilities import get_chunk_bounds, export_chunks_to_shapefile, get_tile_bounds

"""
This script creates chunk bounds within a specified tile and exports them as a Shapefile.
"""

# Main script to run the analysis
if __name__ == "__main__":
    global_index_shapefile = "C:/GIS/Data/Global/Wetlands/Raw/Global/gfw_peatlands/Global_Peatlands_Index/Global_Peatlands.shp"
    tile_id = "00N_080W"
    chunk_size = 2  # Set the desired chunk size

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

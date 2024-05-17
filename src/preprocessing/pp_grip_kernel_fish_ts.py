import geopandas as gpd
from shapely.geometry import box
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tile_id = "00N_110E"

# Paths to the input data
roads_shapefile_path = r'C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region6_vector_shp\GRIP4_region6.shp'
template_raster_path = f'C:/GIS/Data/Global/Wetlands/Raw/Global/gfw_peatlands/{tile_id}_peat_mask_processed.tif'
output_dir = r'C:\GIS\Data\Global\Wetlands\Processed\grip'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

logging.info("Directories and paths set up")

def get_raster_bounds(raster_path):
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds

def resample_raster(input_raster_path, output_raster_path, scale_factor):
    logging.info(f"Resampling raster from {input_raster_path} to {output_raster_path} with scale factor {scale_factor}")
    with rasterio.open(input_raster_path) as src:
        transform = src.transform * src.transform.scale(
            (src.width / (src.width * scale_factor)),
            (src.height / (src.height * scale_factor))
        )
        width = int(src.width * scale_factor)
        height = int(src.height * scale_factor)

        profile = src.profile
        profile.update(transform=transform, width=width, height=height)

        data = src.read(
            out_shape=(src.count, height, width),
            resampling=Resampling.nearest
        )

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(data)

    logging.info("Resampling complete")

def mask_raster(raster_path, output_raster_path):
    logging.info(f"Masking raster at {raster_path} for values equal to 1")
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        mask = data == 1
        profile = src.profile

        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.uint8), 1)

    logging.info("Masking complete")

def create_fishnet_from_raster(masked_raster_path):
    logging.info(f"Creating fishnet from masked raster at {masked_raster_path}")
    with rasterio.open(masked_raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        rows, cols = mask.shape
        polygons = []

        for row in range(rows):
            for col in range(cols):
                if mask[row, col]:
                    x, y = transform * (col, row)
                    polygons.append(box(x, y, x + transform[0], y + transform[4]))

    fishnet_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")
    logging.info(f"Fishnet grid generated with {len(polygons)} cells")
    return fishnet_gdf

def reproject_gdf(gdf, epsg):
    logging.info(f"Reprojecting GeoDataFrame to EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)

def read_roads_within_bounds(roads_shapefile_path, bounds, epsg):
    logging.info(f"Reading roads shapefile within bounds: {bounds}")
    roads_gdf = gpd.read_file(roads_shapefile_path, bbox=bounds)
    logging.info(f"Read {len(roads_gdf)} road features within bounds")
    roads_gdf = reproject_gdf(roads_gdf, epsg)
    return roads_gdf

def assign_road_segments_to_cells(fishnet_gdf, roads_gdf):
    logging.info("Assigning road segments to fishnet cells and calculating lengths")
    road_lengths = []

    for idx, cell in fishnet_gdf.iterrows():
        logging.info(f"Processing cell {idx + 1}/{len(fishnet_gdf)}")
        roads_in_cell = gpd.clip(roads_gdf, cell.geometry)
        total_length = roads_in_cell.geometry.length.sum()
        road_lengths.append(total_length)

    fishnet_gdf['length'] = road_lengths
    logging.info(f"Fishnet with road lengths: {fishnet_gdf.head()}")
    return fishnet_gdf

def fishnet_to_raster(fishnet_gdf, template_raster_path, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    with rasterio.open(template_raster_path) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, compress='lzw')

        transform = src.transform
        out_shape = (src.height, src.width)
        fishnet_gdf = fishnet_gdf.to_crs(src.crs)

        rasterized = rasterize(
            [(geom, value) for geom, value in zip(fishnet_gdf.geometry, fishnet_gdf['length'])],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=rasterio.float32
        )

        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(rasterized, 1)

    logging.info("Fishnet converted to raster and saved")

def process_tile():
    logging.info("Starting processing of the tile")

    # Calculate bounds of the template raster
    template_bounds = get_raster_bounds(template_raster_path)

    # Define the cell size in degrees (for 1km x 1km grid cells)
    cell_size = 1 / 111.32  # 1km x 1km grid cell size in degrees (approx)

    # Resample the template raster to 1km resolution
    resampled_raster_path = os.path.join(output_dir, "resampled_template.tif")
    resample_raster(template_raster_path, resampled_raster_path, scale_factor=cell_size)

    # Mask the resampled raster
    masked_raster_path = os.path.join(output_dir, "masked_template.tif")
    mask_raster(resampled_raster_path, masked_raster_path)

    # Create the fishnet grid from the masked raster
    fishnet_gdf = create_fishnet_from_raster(masked_raster_path)
    fishnet_export_path_before = os.path.join(output_dir, "fishnet_before_lengths.shp")
    fishnet_gdf.to_file(fishnet_export_path_before)
    logging.info(f"Fishnet shapefile (before road lengths) saved to {fishnet_export_path_before}")

    # Reproject the fishnet to Albers Equal Area Conic (EPSG:5070)
    fishnet_gdf = reproject_gdf(fishnet_gdf, 5070)

    # Read the roads shapefile within the valid bounds and reproject to EPSG:5070
    roads_gdf = read_roads_within_bounds(roads_shapefile_path, template_bounds, 5070)

    # Assign road segments to cells and calculate lengths
    fishnet_with_lengths = assign_road_segments_to_cells(fishnet_gdf, roads_gdf)
    fishnet_export_path_after = os.path.join(output_dir, "fishnet_with_lengths.shp")
    fishnet_with_lengths.to_file(fishnet_export_path_after)
    logging.info(f"Fishnet shapefile (after road lengths) saved to {fishnet_export_path_after}")

    # Convert the fishnet to a raster and save it locally
    output_raster_path = os.path.join(output_dir, "road_density_tile.tif")
    fishnet_to_raster(fishnet_with_lengths, template_raster_path, output_raster_path)

    logging.info(f"Road density raster saved to {output_raster_path}")

# Execute the main processing function
process_tile()

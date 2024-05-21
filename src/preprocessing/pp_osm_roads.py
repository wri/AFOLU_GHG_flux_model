import geopandas as gpd
from shapely.geometry import box
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import boto3
import logging
import os
import fiona

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
s3_base_dir = 'gfw2-data/climate/AFOLU_flux_model/organic_soils/'
osm_roads_density_pattern = 'osm_roads_density'

# Local paths
output_dir = r"C:\GIS\Data\Global\Wetlands\Processed\grip"
os.makedirs(output_dir, exist_ok=True)

# Path to the roads shapefile on S3
roads_shapefile_path = f'/vsis3/{s3_base_dir}/inputs/raw/GRIP4_global_vector_shp/GRIP4_global_vector.shp'

logging.info("Directories and paths set up")

def get_raster_bounds(raster_path):
    logging.info(f"Reading raster bounds from {raster_path}")
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    logging.info(f"Bounds of the raster: {bounds}")
    return bounds

def resample_raster(src, scale_factor):
    logging.info(f"Resampling raster in memory with scale factor {scale_factor}")
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

    return data, profile

def mask_raster(data, profile):
    logging.info("Masking raster in memory for values equal to 1")
    mask = data == 1
    profile.update(dtype=rasterio.uint8)
    return mask.astype(rasterio.uint8), profile

def create_fishnet_from_raster(data, transform):
    logging.info("Creating fishnet from raster data in memory")
    rows, cols = data.shape
    polygons = []

    for row in range(rows):
        for col in range(cols):
            if data[row, col]:
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
    with fiona.Env():
        with fiona.open(roads_shapefile_path, bbox=bounds) as roads:
            roads_gdf = gpd.GeoDataFrame.from_features(roads, crs=roads.crs)
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

def convert_length_to_density(fishnet_gdf, crs):
    logging.info("Converting length to density (km/km2)")
    if crs.axis_info[0].unit_name == 'metre':
        fishnet_gdf['density'] = fishnet_gdf['length'] / (1 * 1)  # lengths are in meters, cell area in km2
    elif crs.axis_info[0].unit_name == 'kilometre':
        fishnet_gdf['density'] = fishnet_gdf['length']  # lengths are already in km, cell area in km2
    else:
        raise ValueError("Unsupported CRS units")
    return fishnet_gdf

def fishnet_to_raster(fishnet_gdf, profile, output_raster_path):
    logging.info(f"Converting fishnet to raster and saving to {output_raster_path}")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    transform = profile['transform']
    out_shape = (profile['height'], profile['width'])
    fishnet_gdf = fishnet_gdf.to_crs(profile['crs'])

    if fishnet_gdf.empty:
        logging.info(f"No valid geometries found for {output_raster_path}. Skipping rasterization.")
        return

    rasterized = rasterize(
        [(geom, value) for geom, value in zip(fishnet_gdf.geometry, fishnet_gdf['density'])],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.float32
    )

    if np.all(rasterized == 0) or np.all(np.isnan(rasterized)):
        logging.info(f"Skipping export of {output_raster_path} as all values are 0 or nodata")
        return

    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(rasterized, 1)

    logging.info("Fishnet converted to raster and saved")

def process_tile(tile_key):
    tile_id = os.path.splitext(os.path.basename(tile_key))[0]
    local_output_path = os.path.join(output_dir, f"{osm_roads_density_pattern}_{tile_id}.tif")

    if os.path.exists(local_output_path):
        logging.info(f"{local_output_path} already exists. Skipping processing.")
        return

    logging.info(f"Starting processing of the tile {tile_id}")

    s3_input_path = f'/vsis3/{s3_bucket_name}/{tile_key}'

    with rasterio.Env(AWS_SESSION=boto3.Session()):
        with rasterio.open(s3_input_path) as src:
            # Calculate bounds of the template raster
            template_bounds = src.bounds

            # Define the cell size in degrees (for 1km x 1km grid cells)
            cell_size = 1 / 111.32  # 1km x 1km grid cell size in degrees (approx)

            # Resample the template raster to 1km resolution
            resampled_data, resampled_profile = resample_raster(src, scale_factor=cell_size)

            # Mask the resampled raster
            masked_data, masked_profile = mask_raster(resampled_data[0], resampled_profile)

            # Create the fishnet grid from the masked raster
            fishnet_gdf = create_fishnet_from_raster(masked_data, resampled_profile['transform'])

            # Reproject the fishnet to Albers Equal Area Conic (EPSG:5070)
            fishnet_gdf = reproject_gdf(fishnet_gdf, 5070)

            # Read the roads shapefile within the valid bounds and reproject to EPSG:5070
            roads_gdf = read_roads_within_bounds(roads_shapefile_path, template_bounds, 5070)

            # Assign road segments to cells and calculate lengths
            fishnet_with_lengths = assign_road_segments_to_cells(fishnet_gdf, roads_gdf)

            # Convert length to density
            fishnet_with_density = convert_length_to_density(fishnet_with_lengths, fishnet_gdf.crs)

            # Convert the fishnet to a raster and save it locally
            fishnet_to_raster(fishnet_with_density, masked_profile, local_output_path)

            logging.info(f"Saved {local_output_path}")

def process_all_tiles():
    # Get the list of rasters from S3
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=s3_bucket_name, Prefix=s3_tiles_prefix)

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                tile_key = obj['Key']
                if tile_key.endswith('_peat_mask_processed.tif'):
                    process_tile(tile_key)

def main(tile_id=None):
    if tile_id:
        tile_key = f"{s3_tiles_prefix}{tile_id}_peat_mask_processed.tif"
        process_tile(tile_key)
    else:
        process_all_tiles()


# # Example usage
if __name__ == "__main__":
#     # Replace '00N_110E' with the tile ID you want to test
#     main(tile_id='00N_110E')

    # To process all tiles, comment out the above line and uncomment the line below
    main()
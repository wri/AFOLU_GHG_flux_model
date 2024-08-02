import os

# S3 bucket and prefixes
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
index_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/index/Global_Peatlands'

# Raw raster paths
raw_rasters = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/engert_roads/engert_asiapac_ghrdens_1km_resample_30m.tif',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/canals/Dadap_SEA_Drainage/canal_length_data/canal_length_1km_resample_30m.tif'
}

# Output prefixes for processed data
output_prefixes = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m'
}

# Regional shapefiles paths
s3_regional_shapefiles = [
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region1_vector_shp/GRIP4_region1.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region2_vector_shp/GRIP4_region2.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region3_vector_shp/GRIP4_region3.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region4_vector_shp/GRIP4_region4.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region5_vector_shp/GRIP4_region5.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region6_vector_shp/GRIP4_region6.shp",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region7_vector_shp/GRIP4_region7.shp"
]

# Local paths
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
output_dir = r"C:\GIS\Data\Global\GRIP\roads_by_tile"

os.makedirs(local_temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# S3 Output prefix
s3_output_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/roads_by_tile/'

# Tile suffix pattern
peat_pattern = "_peat_mask_processed.tif"

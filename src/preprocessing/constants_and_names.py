# Paths
index_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/index/Global_Peatlands'
s3_regional_shapefiles = [
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region1_vector_shp/GRIP4_region1",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region2_vector_shp/GRIP4_region2",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region3_vector_shp/GRIP4_region3",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region4_vector_shp/GRIP4_region4",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region5_vector_shp/GRIP4_region5",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region6_vector_shp/GRIP4_region6",
    "climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/grip_roads/regional_shapefiles/GRIP4_Region7_vector_shp/GRIP4_region7"
]
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
output_dir = r"C:\GIS\Data\Global\GRIP\roads_by_tile"


# AWS S3 setup
s3_client = boto3.client('s3')
s3_bucket_name = 'gfw2-data'
s3_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
index_shapefile_prefix = 'climate/AFOLU_flux_model/organic_soils/inputs/raw/index/Global_Peatlands'

# Local paths
local_temp_dir = "C:/GIS/Data/Global/Wetlands/Processed/30_m_temp"
os.makedirs(local_temp_dir, exist_ok=True)
raw_rasters = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/roads/engert_roads/engert_asiapac_ghrdens_1km_resample_30m.tif',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/raw/canals/Dadap_SEA_Drainage/canal_length_data/canal_length_1km_resample_30m.tif'
}
output_prefixes = {
    'engert': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/engert_density/30m',
    'dadap': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/dadap_density/30m'
}

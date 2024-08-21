import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# S3 bucket and prefixes
s3_bucket_name = 'gfw2-data'
local_root = 'C:/GIS/Data/Global'
# local_root = 'mnt/c/GIS/Data/Global'
# local_temp_dir = f"{local_root}/tmp"
local_temp_dir = '/tmp'
project_dir = 'climate/AFOLU_flux_model/organic_soils'
processed_dir = 'inputs/processed'
raw_dir = 'inputs/raw'

# Today's date for versioning in S3 and local processed paths
today_date = datetime.today().strftime('%Y%m%d')

# Peat tiles paths
# peat_tiles_prefix = f'{project_dir}/{processed_dir}/peatlands/processed/{today_date}/' #to be used when we have OGH
peat_tiles_prefix = 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'
peat_tiles_prefix_1km = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/1km/'
index_shapefile_prefix = f'{project_dir}/{raw_dir}/index/Global_Peatlands'
peat_pattern = '_peat_mask_processed.tif'

# Regional shapefiles paths
grip_regional_shapefiles = [
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region1_vector_shp/GRIP4_region1.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region2_vector_shp/GRIP4_region2.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region3_vector_shp/GRIP4_region3.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region4_vector_shp/GRIP4_region4.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region5_vector_shp/GRIP4_region5.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region6_vector_shp/GRIP4_region6.shp",
    f"{project_dir}/{raw_dir}/roads/grip_roads/regional_shapefiles/GRIP4_Region7_vector_shp/GRIP4_region7.shp"
]

osm_pbf_files = {
    'raw': f'{project_dir}/{raw_dir}/', #need to transfer the data and update this
    'filtered': {
        'roads': f'{project_dir}/{raw_dir}/roads/osm_roads/osm_filtered/filtered_roads/',
        'canals': f'{project_dir}/{raw_dir}/roads/osm_roads/osm_filtered/filtered_canals/'
    }
}

# Datasets
datasets = {
    'osm': {
        'roads': {
            's3_raw': f'{project_dir}/{raw_dir}/roads/osm_roads/roads_by_tile/',
            's3_processed_base': f'{project_dir}/{processed_dir}/osm_roads_density/',
            's3_processed': f'{project_dir}/{processed_dir}/osm_roads_density/{today_date}',
            'local_processed': f'{local_temp_dir}/osm_roads_density/{today_date}'
        },
        'canals': {
            's3_raw': f'{project_dir}/{raw_dir}/roads/osm_roads/canals_by_tile/',
            's3_processed_base': f'{project_dir}/{processed_dir}/osm_canals_density/',
            's3_processed': f'{project_dir}/{processed_dir}/osm_canals_density/{today_date}',
            'local_processed': f'{local_temp_dir}/osm_canals_density/{today_date}/'
        }
    },
    'grip': {
        'roads': {
            's3_raw': f'{project_dir}/{raw_dir}/roads/grip_roads/roads_by_tile/',
            's3_processed_base': f'{project_dir}/{processed_dir}/grip_density/',
            's3_processed': f'{project_dir}/{processed_dir}/grip_density/{today_date}',
            'local_processed': f'{local_temp_dir}/grip_density/{today_date}'
        }
    },
    'engert': {
        's3_raw': f'{project_dir}/{raw_dir}/roads/engert_roads/engert_asiapac_ghrdens_1km_resample_30m.tif',
        's3_processed_base': f'{project_dir}/{processed_dir}/engert_density/30m/',
        's3_processed': f'{project_dir}/{processed_dir}/engert_density/30m/{today_date}/',
        'local_processed': f'{local_temp_dir}/engert_density/{today_date}/'
    },
    'dadap': {
        's3_raw': f'{project_dir}/{raw_dir}/canals/Dadap_SEA_Drainage/canal_length_data/canal_length_1km_resample_30m.tif',
        's3_processed_base': f'{project_dir}/{processed_dir}/dadap_density/30m/',
        's3_processed': f'{project_dir}/{processed_dir}/dadap_density/30m/{today_date}/',
        'local_processed': f'{local_temp_dir}/dadap_density/{today_date}/'
    }
}

# Function to check if an S3 path exists
def check_s3_path_exists(s3_client, bucket, path):
    try:
        s3_client.head_object(Bucket=bucket, Key=path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"ClientError 404: {path} not found.")
        else:
            print(f"ClientError {e.response['Error']['Code']}: {path} - {e.response['Message']}")
        return False

# # Initialize S3 client
# s3_client = boto3.client('s3')
#
# # QA check for paths
# print("\n--- Paths for QA ---\n")
# print(f"Local Temp Dir: {local_temp_dir}")
# print(f"Peat Tiles Prefix: {peat_tiles_prefix}")
# print(f"Index Shapefile Prefix: {index_shapefile_prefix}")
# print(f"Peat Pattern: {peat_pattern}\n")
#
# print("Regional Shapefiles:")
# for path in grip_regional_shapefiles:
#     exists = check_s3_path_exists(s3_client, s3_bucket_name, path) if path.startswith(project_dir) else os.path.exists(path)
#     print(f"{path}: {'Exists' if exists else 'Does not exist'}")
#
# print("\nDatasets:")
# for dataset, features in datasets.items():
#     if 's3_raw' in features:
#         # Single-level dataset like 'engert' or 'dadap'
#         print(f"{dataset}:")
#         raw_exists = check_s3_path_exists(s3_client, s3_bucket_name, features['s3_raw']) if features['s3_raw'].startswith(project_dir) else os.path.exists(features['s3_raw'])
#         print(f"  S3 Raw: {features['s3_raw']} - {'Exists' if raw_exists else 'Does not exist'}")
#
#         base_processed_exists = check_s3_path_exists(s3_client, s3_bucket_name, features['s3_processed_base']) if features['s3_processed_base'].startswith(project_dir) else os.path.exists(features['s3_processed_base'])
#         print(f"  S3 Processed Base: {features['s3_processed_base']} - {'Exists' if base_processed_exists else 'Does not exist'}")
#
#         local_processed_exists = os.path.exists(features['local_processed'])
#         print(f"  Local Processed: {features['local_processed']} - {'Exists' if local_processed_exists else 'Does not exist'}")
#     else:
#         # Multi-level datasets like 'osm' or 'grip'
#         for feature, feature_paths in features.items():
#             print(f"{dataset} - {feature}:")
#             raw_exists = check_s3_path_exists(s3_client, s3_bucket_name, feature_paths['s3_raw']) if feature_paths['s3_raw'].startswith(project_dir) else os.path.exists(feature_paths['s3_raw'])
#             print(f"  S3 Raw: {feature_paths['s3_raw']} - {'Exists' if raw_exists else 'Does not exist'}")
#
#             base_processed_exists = check_s3_path_exists(s3_client, s3_bucket_name, feature_paths['s3_processed_base']) if feature_paths['s3_processed_base'].startswith(project_dir) else os.path.exists(feature_paths['s3_processed_base'])
#             print(f"  S3 Processed Base: {feature_paths['s3_processed_base']} - {'Exists' if base_processed_exists else 'Does not exist'}")
#
#             local_processed_exists = os.path.exists(feature_paths['local_processed'])
#             print(f"  Local Processed: {feature_paths['local_processed']} - {'Exists' if local_processed_exists else 'Does not exist'}")

# new paths for osm pre to be used after utilities refactoring
# filtered_canals_path = cn.osm_pbf_files['filtered']['canals'] #this path was formerly local and now needs to be read from s3
# filtered_highways_path = cn.osm_pbf_files['filtered']['roads'] #this path was formerly local and now needs to be read from s3
# output_dir_roads = cn.datasets['osm']['roads']['s3_raw'] #this path was formerly local and now needs to be uploaded to s3
# output_dir_canals = cn.datasets['osm']['roads']['s3_raw']#this path was formerly local and now needs to be uploaded to s3
# local_temp_dir = cn.local_temp_dir
# s3_bucket_name = cn.s3_bucket_name
# index_shapefile_prefix = cn.index_shapefile_prefix
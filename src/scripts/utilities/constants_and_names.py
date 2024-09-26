# config/constants_and_names.py

import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# S3 Configuration
s3_bucket_name = 'gfw2-data'
s3_region_name = 'us-east-1'

# Project Directories
project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = 'inputs/raw'
processed_dir = 'inputs/processed'

# Local Directories
local_root = 'C:/GIS/Data/Global'  # Adjust as needed for your local environment
local_temp_dir = '/tmp'  # Adjust based on your environment

# Date Configuration
today_date = datetime.today().strftime('%Y%m%d')

# File Patterns
peat_pattern = '_peat_mask_processed.tif'
peat_tiles_prefix_1km = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/1km/'

# Sample Tile ID Placeholder
sample_tile_id = '{tile_id}'

# ---------------------------------------------------
# 2. Dataset Configurations
# ---------------------------------------------------

# TODO update working version paths

datasets = {
    'osm': {
        'roads': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'osm_roads', 'roads_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'osm_roads_density'),
            's3_processed_small': f'{project_dir}/{processed_dir}/osm_roads_density/4000_pixels/{today_date}',
            's3_processed': os.path.join(project_dir, processed_dir, 'osm_roads_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'osm_roads_density', today_date),
            'working_version': os.path.join(project_dir, processed_dir, 'osm_roads_density', 'working_version') # note this needs to be updated
        },
        'canals': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'osm_roads', 'canals_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'osm_canals_density'),
            's3_processed_small': f'{project_dir}/{processed_dir}/osm_canals_density/4000_pixels/{today_date}',
            's3_processed': os.path.join(project_dir, processed_dir, 'osm_canals_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'osm_canals_density', today_date),
            'working_version': os.path.join(project_dir, processed_dir, 'osm_canals_density', '20240822')
        }
    },
    'grip': {
        'roads': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'grip_roads', 'roads_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'grip_density'),
            's3_processed_small': f'{project_dir}/{processed_dir}/grip_density/4000_pixels/{today_date}',
            's3_processed': os.path.join(project_dir, processed_dir, 'grip_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'grip_density', today_date),
            'working_version': os.path.join(project_dir, processed_dir, 'grip_density', '20240822') #note this needs to be updated
        }
    },
    'engert': {
        's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'engert_roads', 'engert_asiapac_ghrdens_1km_resample_30m.tif'),
        's3_processed_base': os.path.join(project_dir, processed_dir, 'engert_density', '30m'),
        's3_processed': os.path.join(project_dir, processed_dir, 'engert_density', '30m', today_date),
        'local_processed': os.path.join(local_temp_dir, 'engert_density', today_date),
        'working_version': os.path.join(project_dir, processed_dir, 'engert_density', '30m', '20240925')
    },
    'dadap': {
        's3_raw': os.path.join(project_dir, raw_dir, 'canals', 'Dadap_SEA_Drainage', 'canal_length_data', 'canal_length_1km_resample_30m.tif'),
        's3_processed_base': os.path.join(project_dir, processed_dir, 'dadap_density', '30m'),
        's3_processed': os.path.join(project_dir, processed_dir, 'dadap_density', '30m', today_date),
        'local_processed': os.path.join(local_temp_dir, 'dadap_density', today_date),
        'working_version': os.path.join(project_dir, processed_dir, 'dadap_density', '30m', '20240925')
    },
    'planted_forest_type': {
        's3_processed_base': os.path.join('climate', 'carbon_model', 'other_emissions_inputs', 'plantation_type', 'SDPTv2', '20230911'),
        'working_version': os.path.join('climate', 'carbon_model', 'other_emissions_inputs', 'plantation_type', 'SDPTv2', 'working_version')
    }
    # Add other datasets as needed
}

# ---------------------------------------------------
# 3. General Paths and Constants
# ---------------------------------------------------

# Land Cover URI
lc_uri = 's3://gfw2-data/climate/AFOLU_flux_model/LULUCF/inputs/LC'

# IPCC Codes
ipcc_codes = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

# File Name Patterns
file_patterns = {
    'land_cover': "IPCC_basic_classes",
    'vegetation_height': "vegetation_height",
    'planted_forest_type_layer': "planted_forest_type",
    'planted_forest_tree_crop_layer': "planted_forest_tree_crop",
    'peat': "peat",
    'dadap': "dadap",
    'engert': "engert",
    'grip': "grip",
    'osm_roads': "osm_roads",
    'osm_canals': "osm_canals"
}

# ---------------------------------------------------
# 4. Download Dictionary
# ---------------------------------------------------

# Prepare download dictionary using 'working_version' paths
download_dict = {
    f"{file_patterns['land_cover']}_2020": f"{lc_uri}/composite/2020/raw/{sample_tile_id}.tif",
    file_patterns['planted_forest_type_layer']: f"s3://{s3_bucket_name}/{datasets['planted_forest_type']['working_version']}/{sample_tile_id}_plantation_type_oilpalm_woodfiber_other.tif",
    file_patterns['peat']: f"s3://{s3_bucket_name}/{peat_tiles_prefix_1km}{sample_tile_id}{peat_pattern}",
    file_patterns['dadap']: f"s3://{s3_bucket_name}/{datasets['dadap']['working_version']}/dadap_{sample_tile_id}.tif",
    file_patterns['engert']: f"s3://{s3_bucket_name}/{datasets['engert']['working_version']}/engert_{sample_tile_id}.tif",
    file_patterns['grip']: f"s3://{s3_bucket_name}/{datasets['grip']['working_version']}/grip_density_{sample_tile_id}.tif",
    file_patterns['osm_roads']: f"s3://{s3_bucket_name}/{datasets['osm']['roads']['working_version']}/roads_density_{sample_tile_id}.tif",
    file_patterns['osm_canals']: f"s3://{s3_bucket_name}/{datasets['osm']['canals']['working_version']}/canals_density_{sample_tile_id}.tif"
}

# ---------------------------------------------------
# 5. Helper Functions
# ---------------------------------------------------

def check_s3_path_exists(s3_client, bucket, path):
    """
    Check if a specific path exists in an S3 bucket.

    Args:
        s3_client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        path (str): The S3 object key/path.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"ClientError 404: {path} not found.")
        else:
            print(f"ClientError {e.response['Error']['Code']}: {path} - {e.response['Message']}")
        return False

# ---------------------------------------------------
# 6. AWS S3 Client Initialization
# ---------------------------------------------------

# Initialize S3 Resource and Client
s3 = boto3.resource('s3', region_name=s3_region_name)
s3_client = boto3.client('s3', region_name=s3_region_name)
my_bucket = s3.Bucket(s3_bucket_name)

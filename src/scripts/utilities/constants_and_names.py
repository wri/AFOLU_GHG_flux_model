# config/constants_and_names.py

import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# S3 Configuration
s3_bucket_name = 'gfw2-data'  # Replace with your actual S3 bucket name
s3_region_name = 'us-east-1'  # Replace with your S3 bucket's region if different

# Project Directories
project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = 'inputs/raw'
processed_dir = 'inputs/processed'

# Local Directories
local_root = 'C:/GIS/Data/Global'  # Adjust as needed for your local environment
# local_root = '/mnt/c/GIS/Data/Global'  # Uncomment if using WSL
local_temp_dir = '/tmp'  # Adjust based on your environment

# Date Configuration
today_date = datetime.today().strftime('%Y%m%d')

# File Patterns
peat_pattern = '_peat_mask_processed.tif'
peat_tiles_prefix_1km = 'climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/1km/'  # Ensure this ends with a slash

# ---------------------------------------------------
# 2. Dataset Configurations
# ---------------------------------------------------

datasets = {
    'osm': {
        'roads': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'osm_roads', 'roads_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'osm_roads_density'),
            's3_processed': os.path.join(project_dir, processed_dir, 'osm_roads_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'osm_roads_density', today_date)
        },
        'canals': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'osm_roads', 'canals_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'osm_canals_density'),
            's3_processed': os.path.join(project_dir, processed_dir, 'osm_canals_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'osm_canals_density', today_date)
        }
    },
    'grip': {
        'roads': {
            's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'grip_roads', 'roads_by_tile'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'grip_density'),
            's3_processed': os.path.join(project_dir, processed_dir, 'grip_density', today_date),
            'local_processed': os.path.join(local_temp_dir, 'grip_density', today_date)
        }
    },
    'engert': {
        's3_raw': os.path.join(project_dir, raw_dir, 'roads', 'engert_roads', 'engert_asiapac_ghrdens_1km_resample_30m.tif'),
        's3_processed_base': os.path.join(project_dir, processed_dir, 'engert_density', '30m'),
        's3_processed': os.path.join(project_dir, processed_dir, 'engert_density', '30m', today_date),
        'local_processed': os.path.join(local_temp_dir, 'engert_density', today_date)
    },
    'dadap': {
        's3_raw': os.path.join(project_dir, raw_dir, 'canals', 'Dadap_SEA_Drainage', 'canal_length_data', 'canal_length_1km_resample_30m.tif'),
        's3_processed_base': os.path.join(project_dir, processed_dir, 'dadap_density', '30m'),
        's3_processed': os.path.join(project_dir, processed_dir, 'dadap_density', '30m', today_date),
        'local_processed': os.path.join(local_temp_dir, 'dadap_density', today_date)
    },
    'descals': {
        'descals_extent': {
            's3_raw': os.path.join(project_dir, raw_dir, 'plantations', 'plantation_extent'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'descals_plantation', 'extent'),
            's3_processed': os.path.join(project_dir, processed_dir, 'descals_plantation', 'extent', today_date),
            'local_processed': os.path.join(local_temp_dir, 'descals_plantation', 'extent', today_date)
        },
        'descals_year': {
            's3_raw': os.path.join(project_dir, raw_dir, 'plantations', 'plantation_year'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'descals_plantation', 'year'),
            's3_processed': os.path.join(project_dir, processed_dir, 'descals_plantation', 'year', today_date),
            'local_processed': os.path.join(local_temp_dir, 'descals_plantation', 'year', today_date)
        }
    },
    'extraction': {
        'finland': {
            's3_raw': os.path.join(project_dir, raw_dir, 'extraction', 'Finland', 'Finland_turvetuotantoalueet', 'turvetuotantoalueet_jalkikaytto'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'extraction'),
            's3_processed': os.path.join(project_dir, processed_dir, 'extraction', today_date),
            'local_processed': os.path.join(local_temp_dir, 'extraction', 'finland', today_date)
        },
        'ireland': {
            's3_raw': os.path.join(project_dir, raw_dir, 'extraction', 'Ireland', 'Ireland_Habibetal', 'RF_S2_LU_5_11_23.tif'),
            's3_processed_base': os.path.join(project_dir, processed_dir, 'extraction'),
            's3_processed': os.path.join(project_dir, processed_dir, 'extraction', today_date),
            'local_processed': os.path.join(local_temp_dir, 'extraction', 'ireland', today_date)
        },
        'russia': {
            's3_raw': [
                os.path.join(project_dir, raw_dir, 'extraction', 'Russia', 'allocated_without_licenses', 'allocated_mineral_reserve'),
                os.path.join(project_dir, raw_dir, 'extraction', 'Russia', 'allocated_with_licenses', 'peat_extraction_dates')
            ],
            's3_processed_base': os.path.join(project_dir, processed_dir, 'extraction'),
            's3_processed': os.path.join(project_dir, processed_dir, 'extraction', today_date),
            'local_processed': os.path.join(local_temp_dir, 'extraction', 'russia', today_date)
        }
    }
}

# ---------------------------------------------------
# 3. General Paths and Constants
# ---------------------------------------------------

# Land Cover URI
lc_uri = 's3://gfw2-data/landcover'

# S3 Output Directory
s3_out_dir = os.path.join(project_dir, processed_dir, 'outputs')

# IPCC Configuration
ipcc_class_max_val = 6

# IPCC Codes
ipcc_codes = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

# GLCLU Codes
glclu_codes = {
    'cropland': 244,  # Renamed to prevent conflict with IPCC 'cropland'
    'builtup': 250
}

# Vegetation Height Codes
vegetation_height_codes = {
    'tree_wet_min': 27,
    'tree_wet_max': 48,
    'tree_dry_min': 127,
    'tree_dry_max': 148
}

# Other Constants
first_year = 2000
last_year = 2020
full_raster_dims = 40000
interval_years = 5  # Number of years in interval
sig_height_loss_threshold = 5  # meters
biomass_to_carbon = 0.47  # Conversion of biomass to carbon
tree_threshold = 5  # Height minimum for trees (meters)

# File Name Patterns
file_patterns = {
    'ipcc_class_path': "IPCC_basic_classes",
    'ipcc_class_pattern': "IPCC_classes",
    'ipcc_change_path': "IPCC_basic_change",
    'ipcc_change_pattern': "IPCC_change",
    'land_state_pattern': "land_state_node",
    'agc_dens_pattern': "AGC_density_MgC_ha",
    'bgc_dens_pattern': "BGC_density_MgC_ha",
    'agc_flux_pattern': "AGC_flux_MgC_ha",
    'bgc_flux_pattern': "BGC_flux_MgC_ha",
    'land_cover': "land_cover",
    'vegetation_height': "vegetation_height",
    'agb_2000': "agb_2000",
    'agc_2000': "agc_2000",
    'bgc_2000': "bgc_2000",
    'deadwood_c_2000': "deadwood_c_2000",
    'litter_c_2000': "litter_c_2000",
    'soil_c_2000': "soil_c_2000",
    'r_s_ratio': "r_s_ratio",
    'burned_area': "burned_area",
    'forest_disturbance': "forest_disturbance",
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
# 4. Helper Functions
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
# 5. AWS S3 Client Initialization
# ---------------------------------------------------

# Initialize S3 Resource and Client
s3 = boto3.resource('s3', region_name=s3_region_name)
s3_client = boto3.client('s3', region_name=s3_region_name)
my_bucket = s3.Bucket(s3_bucket_name)
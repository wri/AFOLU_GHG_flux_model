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
    },
    'descals': {
        'descals_extent': {
            's3_raw': f'{project_dir}/{raw_dir}/plantations/plantation_extent/',
            's3_processed_base': f'{project_dir}/{processed_dir}/descals_plantation/extent',
            's3_processed': f'{project_dir}/{processed_dir}/descals_plantation/extent/{today_date}',
            'local_processed': f'{local_temp_dir}/descals_plantation/extent/{today_date}'
        },
        'descals_year': {
            's3_raw': f'{project_dir}/{raw_dir}/plantations/plantation_year/',
            's3_processed_base': f'{project_dir}/{processed_dir}/descals_plantation/year',
            's3_processed': f'{project_dir}/{processed_dir}/descals_plantation/year/{today_date}',
            'local_processed': f'{local_temp_dir}/descals_plantation/year/{today_date}'
        }
    },
    'extraction': {
        'finland': {
            's3_raw': f'{project_dir}/{raw_dir}/extraction/Finland_turvetuotantoalueet/turvetuotantoalueet_jalkikaytto',
            's3_processed_base': f'{project_dir}/{processed_dir}/extraction/',
            's3_processed': f'{project_dir}/{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/finland/{today_date}/'
        },
        'ireland': {
            's3_raw': f'{project_dir}/{raw_dir}/extraction/Ireland_Habibetal/RF_S2_LU_5_11_23.tif',
            's3_processed_base': f'{project_dir}/{processed_dir}/extraction/',
            's3_processed': f'{project_dir}/{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/ireland/{today_date}/'
        },
        'russia': {
            's3_raw': f'{project_dir}/{raw_dir}/peat_extraction/russia/your_raw_data_file',  # Placeholder
            's3_processed_base': f'{project_dir}/{processed_dir}/extraction/',
            's3_processed': f'{project_dir}/{processed_dir}/extraction/{today_date}/',
            'local_processed': f'{local_temp_dir}/extraction/russia/{today_date}/'
        }
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

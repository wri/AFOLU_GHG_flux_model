"""
Converts global geotifs into global COGs
From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/685ee215-b624-800a-9ab7-06d1c27a1697

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

python -m src.Agriculture.scripts.postprocessing.create_global_COGs
"""

import os
import boto3
import subprocess
from urllib.parse import urlparse
from tqdm import tqdm

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu

# Input and output root folders
input_s3_prefix = f'{cn.cropland_dir}raw__from_Cornell/20250828/year_2020/'
output_s3_prefix = f'{cn.cropland_dir}processed/Cornell_v20250828/year_2020/global_COG/'

# Local temp directories
local_input_dir = '/tmp/input_tifs'
local_output_dir = '/tmp/output_cogs'
os.makedirs(local_input_dir, exist_ok=True)
os.makedirs(local_output_dir, exist_ok=True)

# Parse S3 input
parsed_input = urlparse(input_s3_prefix)
input_bucket = parsed_input.netloc
input_prefix = parsed_input.path.lstrip('/')

# Parse output
parsed_output = urlparse(output_s3_prefix)
output_bucket = parsed_output.netloc
output_base_prefix = parsed_output.path.lstrip('/')

# Initialize S3 client
s3 = boto3.client('s3')

# Recursive list of all TIFF files
tif_keys = []
continuation_token = None

while True:
    list_kwargs = {
        'Bucket': input_bucket,
        'Prefix': input_prefix,
    }
    if continuation_token:
        list_kwargs['ContinuationToken'] = continuation_token

    response = s3.list_objects_v2(**list_kwargs)
    tif_keys += [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.tif')]

    if response.get('IsTruncated'):
        continuation_token = response['NextContinuationToken']
    else:
        break

print(f"Found {len(tif_keys)} TIFF files under {input_s3_prefix}")


# Process each GeoTIFF
for key in tqdm(tif_keys, desc="Converting to COGs"):
    filename = os.path.basename(key)
    name, ext = os.path.splitext(filename)
    cog_filename = f"{name}_COG{ext}"

    print(f"Processing {filename}...")

    # Paths for local I/O
    local_input_path = os.path.join(local_input_dir, filename)
    local_output_path = os.path.join(local_output_dir, cog_filename)

    # Download
    s3.download_file(input_bucket, key, local_input_path)

    # Convert to COG
    cmd = [
        'gdal_translate', local_input_path, local_output_path,
        '-of', 'COG',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'PREDICTOR=2',
        '-co', 'BIGTIFF=IF_SAFER',
        '-co', 'OVERVIEW_RESAMPLING=average'
    ]
    subprocess.run(cmd, check=True)

    # Construct output key with structure preserved
    relative_key = os.path.relpath(key, input_prefix)  # path under input root
    relative_folder = os.path.dirname(relative_key)
    output_key = os.path.join(output_base_prefix, relative_folder, cog_filename)

    # Upload to output S3
    s3.upload_file(local_output_path, output_bucket, output_key)

    # Cleanup
    os.remove(local_input_path)
    os.remove(local_output_path)

print("All nested GeoTIFFs converted to COGs and uploaded.")


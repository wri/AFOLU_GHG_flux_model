"""
Downloads matching files in the specified LULUCF output folder in S3 locally.
From https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67f3f252-8624-800a-a4d8-b0e29d05104e

Usage (run from /mnt/c/GIS/git/AFOLU_GHG_flux_model):

    python src/utilities/download_outputs_local.py <subfolder> [filename_filter]
    python src/utilities/download_outputs_local.py v32_COD_exploration 23_-4_24_-3

Arguments:
1. <subfolder>: Local subfolder to save outputs to (inside /mnt/c/GIS/AFOLU_flux_model/test_data/output/v0_4_1/)
2. [filename_filter]: Optional string that must be in the filename to be downloaded
"""

import boto3
import os
import sys
from botocore.exceptions import ClientError

import constants_and_names as cn


# Constants
BUCKET = "gfw2-data"
BASE_DEST = f"/mnt/c/GIS/AFOLU_flux_model/test_data/output/v{cn.veg_model_version_underscore}__standard__global/"
BASE_DEST = f"/mnt/c/GIS/AFOLU_flux_model/test_data/output/v1_0_4/"

# Speeds up accessing the input geotifs from s3 when they are in a folder with lots of files.
# The more files in an s3 folder, the longer it takes to access them without this environment variable.
# It takes about 9 minutes to access the inputs for a 1x1 deg summative output without this and <1 minute with it.
# Per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68bb4948-c75c-8331-bdf7-1d892029dc0f
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_outputs_local.py <subfolder> [filename_filter]")
        sys.exit(1)

    subfolder = sys.argv[1]
    filter_str = sys.argv[2] if len(sys.argv) >= 3 else None
    dest_folder = os.path.join(BASE_DEST, subfolder)

    os.makedirs(dest_folder, exist_ok=True)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    downloaded = 0
    skipped = 0
    filtered_out = 0

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = os.path.basename(key)

            if not filename:
                continue  # Skip keys that are directories or empty

            # Filter by substring if provided
            if filter_str and filter_str not in filename:
                filtered_out += 1
                continue

            local_path = os.path.join(dest_folder, filename)

            if os.path.exists(local_path):
                print(f"  SKIPPING {filename} (already exists)")
                skipped += 1
                continue

            print(f"  Downloading {key} → {filename}")
            try:
                s3.download_file(BUCKET, key, local_path)
                downloaded += 1
            except ClientError as e:
                print(f"  ERROR downloading {key}: {e}")

    print(f"\nDone. {downloaded} files downloaded, {skipped} skipped, {filtered_out} filtered out.")

if __name__ == "__main__":
    main()

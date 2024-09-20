# utils/s3_utils.py

import boto3
import subprocess
import os
from typing import List
from utils.logging_utils import timestr, print_and_log

def list_rasters_in_folder(full_in_folder: str) -> List[str]:
    """
    Lists rasters in an S3 folder and returns their names as a list.

    Args:
        full_in_folder (str): Full S3 folder path (e.g., 's3://bucket/folder/').

    Returns:
        List[str]: List of raster filenames ending with '.tif'.
    """
    cmd = ['aws', 's3', 'ls', full_in_folder]
    try:
        s3_contents_bytes = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print(f"flm: Error listing rasters in folder {full_in_folder}: {e}")
        return []

    # Converts subprocess results to useful string
    s3_contents_str = s3_contents_bytes.decode('utf-8')
    s3_contents_list = s3_contents_str.splitlines()
    rasters = [line.split()[-1] for line in s3_contents_list]
    rasters = [i for i in rasters if "tif" in i]

    return rasters

def upload_shp(full_in_folder: str, in_folder: str, shp: str):
    """
    Uploads a shapefile and its associated files to S3.

    Args:
        full_in_folder (str): Full S3 folder path (e.g., 's3://bucket/folder/').
        in_folder (str): Input folder path without 's3://' (e.g., 'bucket/folder/').
        shp (str): Shapefile name.

    """
    print(f"flm: Uploading to {full_in_folder}{shp}: {timestr()}")

    shp_pattern = shp[:-4]

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    bucket_name = "gfw2-data"

    # List of shapefile components
    shp_components = [shp, f"{shp_pattern}.dbf", f"{shp_pattern}.prj", f"{shp_pattern}.shx"]

    for component in shp_components:
        local_path = f"/tmp/{component}"
        s3_key = f"{in_folder[10:]}{component}"  # [10:] to remove 'gfw2-data/' prefix
        try:
            s3_client.upload_file(local_path, bucket_name, Key=s3_key)
            print(f"flm: Successfully uploaded {component} to s3://{bucket_name}/{s3_key}")
        except boto3.exceptions.S3UploadFailedError as e:
            print(f"flm: Error uploading {component} to s3: {e}")
            continue

    # Deletes the local shapefile components
    for component in shp_components:
        try:
            os.remove(f"/tmp/{component}")
        except OSError as e:
            print(f"flm: Error deleting local file /tmp/{component}: {e}")
            continue

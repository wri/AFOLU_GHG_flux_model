import boto3
import os
import subprocess
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
 """
 used this to compress and upload the grip and canals density tiles. Need to make sure to apply this in those actual
 scripts
 """

def compress_and_upload_directory_to_s3(local_directory, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
    existing_files = get_existing_s3_files(s3_bucket, s3_prefix)

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            compressed_file_path = os.path.join(root, f"compressed_{file}")
            s3_file_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, s3_file_path).replace("\\", "/")

            if s3_key in existing_files:
                print(f"File {s3_key} already exists in S3. Skipping upload.")
            else:
                try:
                    print(f"Compressing {local_file_path}")
                    compress_file(local_file_path, compressed_file_path)

                    print(f"Uploading {compressed_file_path} to s3://{s3_bucket}/{s3_key}")
                    s3_client.upload_file(compressed_file_path, s3_bucket, s3_key)

                    print(f"Successfully uploaded {compressed_file_path} to s3://{s3_bucket}/{s3_key}")

                    # Remove the compressed file after upload
                    os.remove(compressed_file_path)
                except (NoCredentialsError, PartialCredentialsError) as e:
                    print(f"Credentials error: {e}")
                    return
                except Exception as e:
                    print(f"Failed to upload {local_file_path} to s3://{s3_bucket}/{s3_key}: {e}")


def compress_file(input_file, output_file):
    try:
        subprocess.run(
            ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', input_file, output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error compressing file {input_file}: {e}")


def get_existing_s3_files(s3_bucket, s3_prefix):
    s3_client = boto3.client('s3')
    existing_files = set()

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                existing_files.add(obj['Key'])

    return existing_files


if __name__ == "__main__":
    tasks = [
        {
            "local_directory": r"C:\GIS\Data\Global\Wetlands\Processed\30_m_temp\grip",
            "s3_bucket": "gfw2-data",
            "s3_prefix": "climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/30m/"
        },
        {
            "local_directory": r"C:\GIS\Data\Global\Wetlands\Processed\30_m_temp\osm_canals",
            "s3_bucket": "gfw2-data",
            "s3_prefix": "climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/30m/"
        }
    ]

    for task in tasks:
        compress_and_upload_directory_to_s3(task["local_directory"], task["s3_bucket"], task["s3_prefix"])

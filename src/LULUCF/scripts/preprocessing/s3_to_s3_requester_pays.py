"""
Copy rasters from gfw-data-lake (requester-pays source bucket) to gfw2-data using Coiled.

Run:
python -m src.utilities.create_cluster -cn GPW_copy -n 5 -m 16
python -m src.LULUCF.scripts.preprocessing.s3_to_s3_requester_pays -cn GPW_copy --src-profile gfw-data-lake --dst-profile default
"""

import argparse
import os
import tempfile
from urllib.parse import urlparse

import boto3
from dask.distributed import as_completed

from src.utilities import universal_utilities as uu
import threading


class ProgressPercentage:
    def __init__(self, filename, filesize):
        self._filename = filename
        self._filesize = filesize
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._filesize) * 100

            print(
                f"{self._filename}: "
                f"{self._seen_so_far / 1024**3:.2f} / "
                f"{self._filesize / 1024**3:.2f} GB "
                f"({percentage:.1f}%)"
            )


def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Not an S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def creds_from_profile(profile_name):
    session = boto3.Session(profile_name=profile_name)
    creds = session.get_credentials().get_frozen_credentials()

    return {
        "aws_access_key_id": creds.access_key,
        "aws_secret_access_key": creds.secret_key,
        "aws_session_token": creds.token,
        "region_name": session.region_name or "us-east-1",
    }


def s3_client(creds):
    return boto3.client(
        "s3",
        aws_access_key_id=creds["aws_access_key_id"],
        aws_secret_access_key=creds["aws_secret_access_key"],
        aws_session_token=creds.get("aws_session_token"),
        region_name=creds.get("region_name") or "us-east-1",
    )


def copy_raster(pattern, year, src_dir, dst_dir, src_creds, dst_creds):
    filename = f"{pattern}_{year}.tif"

    src_bucket, src_prefix = parse_s3_uri(src_dir)
    dst_bucket, dst_prefix = parse_s3_uri(dst_dir)

    src_key = f"{src_prefix.rstrip('/')}/{filename}"
    dst_key = f"{dst_prefix.rstrip('/')}/{filename}"

    src_client = s3_client(src_creds)
    dst_client = s3_client(dst_creds)

    with tempfile.NamedTemporaryFile(suffix=".tif", dir="/tmp") as tmp:
        obj = src_client.head_object(Bucket=src_bucket, Key=src_key, RequestPayer="requester")
        filesize = obj["ContentLength"]

        src_client.download_file( Bucket=src_bucket, Key=src_key, Filename=tmp.name, ExtraArgs={"RequestPayer": "requester"}, Callback=ProgressPercentage(filename, filesize))
        dst_client.upload_file(Filename=tmp.name, Bucket=dst_bucket, Key=dst_key, Callback=ProgressPercentage(f"UPLOAD {filename}", filesize))

    return f"s3://{dst_bucket}/{dst_key}"


def main(cluster_name, src_profile, dst_profile):
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False, False)

    src_creds = creds_from_profile(src_profile)
    dst_creds = creds_from_profile(dst_profile)

    # YYYY = 2000-2024
    # Source: s3://gfw-data-lake/gfw_grasslands/v1.1/geotiff/grasslands_YYYY.tif
    # Destination: s3://gfw2-data/lcl/gpw/grasslands/v1.1/raw/grasslands_YYYY.tif
    src_dir = "s3://gfw-data-lake/gfw_grasslands/v1.1/geotiff/"
    dst_dir = "s3://gfw2-data/lcl/gpw/grasslands/v1.1/raw/"
    pattern = "grasslands"
    years = range(2000, 2025)

    futures = [client.submit(copy_raster, pattern, year, src_dir, dst_dir, src_creds, dst_creds) for year in years]

    copied = []
    for future in as_completed(futures):
        result = future.result()
        copied.append(result)
        print(f"Copied: {result}")

    print(f"Done. Copied {len(copied)} rasters.")
    
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cn", "--cluster_name", required=True)
    parser.add_argument("--src-profile", required=True)
    parser.add_argument("--dst-profile", required=True)

    args = parser.parse_args()

    main(cluster_name=args.cluster_name, src_profile=args.src_profile, dst_profile=args.dst_profile)
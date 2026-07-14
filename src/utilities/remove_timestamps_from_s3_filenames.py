"""
Remove timestamps from LULUCF output geotiff filenames in S3.

From Claude session 'Remove timestamps from geotif filenames'

Filename pattern:
  <tile_info>__<YYYYMMDD>_<HH>_<MM>_<SS>.tif
  ->
  <tile_info>.tif

Usage:
  # Dry run on one folder (default):
  python remove_timestamps_from_s3_filenames.py

  # Actually rename one folder:
  python remove_timestamps_from_s3_filenames.py --execute

  # Dry run on entire parent folder:
  python remove_timestamps_from_s3_filenames.py --all-folders

  # Actually rename entire parent folder:
  python remove_timestamps_from_s3_filenames.py --all-folders --execute
"""

import re
import argparse
import boto3

BUCKET = "gfw2-data"

# SINGLE_PREFIX = (
#     "climate/AFOLU_flux_model/LULUCF/outputs_LULUCF_totals/"
#     "LULUCF_version_1_0_0_standard__global__veg_v1_0_5__org_soil_v1_0_1__min_soil_v1_0_1/"
#     "LULUCF_gross_removals__all_C_pools__MgCO2/annual_intervals/avg_2016_2024/_ha_yr/4000_pixels/20260614/"
# )
SINGLE_PREFIX = (
    "climate/AFOLU_flux_model/LULUCF/outputs_LULUCF_totals/"
    "LULUCF_version_1_0_0_standard__global__veg_v1_0_5__org_soil_v1_0_1__min_soil_v1_0_1/"
    "LULUCF_gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/avg_2016_2024/_ha_yr/4000_pixels/20260614/"
)

PARENT_PREFIX = (
    "climate/AFOLU_flux_model/LULUCF/outputs_LULUCF_totals/"
    "LULUCF_version_1_0_0_standard__global__veg_v1_0_5__org_soil_v1_0_1__min_soil_v1_0_1/"
)

# Matches the timestamp suffix: __YYYYMMDD_HH_MM_SS before .tif
TIMESTAMP_RE = re.compile(r"__\d{8}_\d{2}_\d{2}_\d{2}(\.tif)$")


def list_objects(s3_client, bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def rename_objects(bucket, prefix, execute):
    s3 = boto3.client("s3")

    renamed = 0
    skipped = 0
    errors = 0

    for key in list_objects(s3, bucket, prefix):
        filename = key.split("/")[-1]
        match = TIMESTAMP_RE.search(filename)
        if not match:
            skipped += 1
            continue

        new_filename = TIMESTAMP_RE.sub(r"\1", filename)
        new_key = key[: key.rfind("/") + 1] + new_filename

        if key == new_key:
            skipped += 1
            continue

        print(f"  {'RENAME' if execute else 'WOULD RENAME'}:")
        print(f"    FROM: {key}")
        print(f"      TO: {new_key}")

        if execute:
            try:
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={"Bucket": bucket, "Key": key},
                    Key=new_key,
                )
                s3.delete_object(Bucket=bucket, Key=key)
                renamed += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                errors += 1
        else:
            renamed += 1

    label = "Renamed" if execute else "Would rename"
    print(f"\n{label}: {renamed} | Skipped (no timestamp): {skipped} | Errors: {errors}")
    return renamed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Remove timestamps from S3 geotiff filenames.")
    parser.add_argument(
        "--all-folders",
        action="store_true",
        help="Run on entire parent folder instead of the single test folder.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files. Without this flag, runs as a dry run.",
    )
    args = parser.parse_args()

    prefix = PARENT_PREFIX if args.all_folders else SINGLE_PREFIX
    mode = "EXECUTE" if args.execute else "DRY RUN"
    scope = "PARENT folder (all subfolders)" if args.all_folders else "SINGLE folder"

    print(f"Mode: {mode}")
    print(f"Scope: {scope}")
    print(f"Prefix: s3://{BUCKET}/{prefix}\n")

    if args.execute and args.all_folders:
        confirm = input("About to rename files across the entire parent folder. Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    rename_objects(BUCKET, prefix, args.execute)


if __name__ == "__main__":
    main()

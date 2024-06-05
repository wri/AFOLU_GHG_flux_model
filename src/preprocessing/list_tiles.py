import boto3
import re

# AWS S3 setup
s3_bucket = "gfw2-data"
s3_prefix = "climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/"


def list_tile_ids(bucket, prefix):
    s3 = boto3.client('s3')
    keys = []
    tile_ids = set()

    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])

        # Extract tile IDs from filenames
        for key in keys:
            match = re.match(r"(\d{2}[NS]_\d{3}[EW])_peat_mask_processed\.tif", key.split('/')[-1])
            if match:
                tile_ids.add(match.group(1))

    except Exception as e:
        print(f"Error listing files in s3://{bucket}/{prefix}: {e}")

    return list(tile_ids)


if __name__ == "__main__":
    tile_ids = list_tile_ids(s3_bucket, s3_prefix)
    print(f"Found {len(tile_ids)} tile IDs:")
    print(tile_ids)
    # for tile_id in tile_ids:
    #     print(tile_id)

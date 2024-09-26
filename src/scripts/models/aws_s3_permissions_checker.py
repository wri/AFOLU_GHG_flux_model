#!/usr/bin/env python3

"""
AWS S3 Permissions Checker for PyCharm

This script verifies AWS S3 permissions for a specified bucket, prefix, and object key.
It checks for the following permissions:
- ListBucket: Ability to list objects in a bucket/prefix.
- GetObject: Ability to retrieve a specific object from a bucket.
- PutObject: Ability to upload objects to a bucket (optional).

Usage:
    - Run the script in PyCharm.
    - Review the console output for permission statuses.

Configuration:
    - Set the bucket, prefix, object key, and other parameters directly in the CONFIG dictionary below.
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import logging
import sys
import os

# ----------------------------- Configuration -----------------------------

CONFIG = {
    'bucket': 'gfw2-data',
    'prefix': 'climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/4000_pixels/20240822/',
    'object_key': '00N_010E_18_-4_20_-2_osm_canals_density.tif',
    'put_test': False,  # Set to True to test PutObject permission
    'profile': None  # Replace with your AWS profile name if needed, e.g., 'my-profile'
}


# ----------------------------- Logging Setup -----------------------------

def setup_logging():
    """
    Sets up the logging configuration.
    Logs will be printed to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------- AWS Client Setup -----------------------------

def get_s3_client(profile=None):
    """
    Initializes a boto3 S3 client.

    Args:
        profile (str, optional): AWS CLI profile name.

    Returns:
        boto3.client: Configured S3 client.
    """
    try:
        if profile:
            session = boto3.Session(profile_name=profile)
            s3_client = session.client('s3')
            logging.info(f"Using AWS profile: {profile}")
        else:
            s3_client = boto3.client('s3')
            logging.info("Using default AWS profile.")
        return s3_client
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"AWS credentials error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error initializing S3 client: {e}")
        sys.exit(1)


# ----------------------------- Permission Checks -----------------------------

def check_list_bucket(s3_client, bucket, prefix=None):
    """
    Checks ListBucket permission by attempting to list objects in a bucket/prefix.

    Args:
        s3_client (boto3.client): Boto3 S3 client.
        bucket (str): S3 bucket name.
        prefix (str, optional): S3 prefix.

    Returns:
        bool: True if ListBucket is permitted, False otherwise.
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        operation_parameters = {'Bucket': bucket}
        if prefix:
            operation_parameters['Prefix'] = prefix
        page_iterator = paginator.paginate(**operation_parameters)
        # Attempt to retrieve the first page
        for page in page_iterator:
            logging.info(f"ListBucket permission: SUCCESS for bucket '{bucket}' with prefix '{prefix}'.")
            return True
        # If no contents, still indicates permission is granted
        logging.info(f"ListBucket permission: SUCCESS for bucket '{bucket}' with prefix '{prefix}' (no objects found).")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['AccessDenied', 'UnauthorizedOperation']:
            logging.error(f"ListBucket permission: DENIED for bucket '{bucket}' with prefix '{prefix}'.")
            return False
        else:
            logging.error(f"ListBucket permission: ERROR for bucket '{bucket}' with prefix '{prefix}': {e}")
            return False
    except Exception as e:
        logging.error(f"ListBucket permission: ERROR for bucket '{bucket}' with prefix '{prefix}': {e}")
        return False


def check_get_object(s3_client, bucket, prefix, key):
    """
    Checks GetObject permission by attempting to retrieve an object.

    Args:
        s3_client (boto3.client): Boto3 S3 client.
        bucket (str): S3 bucket name.
        prefix (str): S3 prefix.
        key (str): S3 object key relative to the prefix.

    Returns:
        bool: True if GetObject is permitted, False otherwise.
    """
    if prefix:
        # Concatenate prefix and key to form the full object key
        full_key = os.path.join(prefix, key)
        # Ensure that S3 uses forward slashes
        full_key = full_key.replace("\\", "/")
    else:
        full_key = key

    try:
        s3_client.head_object(Bucket=bucket, Key=full_key)
        logging.info(f"GetObject permission: SUCCESS for object '{full_key}' in bucket '{bucket}'.")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['AccessDenied', 'NoSuchKey', 'UnauthorizedOperation']:
            logging.error(
                f"GetObject permission: DENIED or object does not exist for key '{full_key}' in bucket '{bucket}'.")
            return False
        else:
            logging.error(f"GetObject permission: ERROR for key '{full_key}' in bucket '{bucket}': {e}")
            return False
    except Exception as e:
        logging.error(f"GetObject permission: ERROR for key '{full_key}' in bucket '{bucket}': {e}")
        return False


def check_put_object(s3_client, bucket, prefix=None):
    """
    Checks PutObject permission by attempting to upload and then delete a test object.

    Args:
        s3_client (boto3.client): Boto3 S3 client.
        bucket (str): S3 bucket name.
        prefix (str, optional): S3 prefix where the test object will be uploaded.

    Returns:
        bool: True if PutObject is permitted, False otherwise.
    """
    test_key = "permissions_checker_test_object.txt"
    if prefix:
        test_key = os.path.join(prefix, test_key)
        # Ensure that S3 uses forward slashes
        test_key = test_key.replace("\\", "/")

    try:
        # Create a small test file in memory
        test_content = b"This is a test file for PutObject permission check."
        s3_client.put_object(Bucket=bucket, Key=test_key, Body=test_content)
        logging.info(f"PutObject permission: SUCCESS for uploading object '{test_key}' to bucket '{bucket}'.")

        # Clean up by deleting the test object
        s3_client.delete_object(Bucket=bucket, Key=test_key)
        logging.info(f"Cleanup: Deleted test object '{test_key}' from bucket '{bucket}'.")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['AccessDenied', 'UnauthorizedOperation']:
            logging.error(f"PutObject permission: DENIED for uploading to bucket '{bucket}' with prefix '{prefix}'.")
            return False
        else:
            logging.error(f"PutObject permission: ERROR for uploading to bucket '{bucket}' with prefix '{prefix}': {e}")
            return False
    except Exception as e:
        logging.error(f"PutObject permission: ERROR for uploading to bucket '{bucket}' with prefix '{prefix}': {e}")
        return False


# ----------------------------- Main Execution -----------------------------

def main():
    setup_logging()

    # Extract configuration parameters
    bucket = CONFIG['bucket']
    prefix = CONFIG['prefix']
    key = CONFIG['object_key']
    put_test = CONFIG['put_test']
    profile = CONFIG['profile']

    if prefix is None and key is not None:
        logging.warning("No prefix provided. GetObject permission will be checked for the entire bucket.")

    s3_client = get_s3_client(profile=profile)

    logging.info(f"Starting AWS S3 Permissions Check for bucket '{bucket}'.")

    # 1. Check ListBucket permission
    list_bucket = check_list_bucket(s3_client, bucket, prefix)

    # 2. Check GetObject permission if key is provided
    get_object = True  # Default to True if no key is provided
    if key:
        get_object = check_get_object(s3_client, bucket, prefix, key)

    # 3. Check PutObject permission if put_test is specified
    put_object = True  # Default to True if not testing
    if put_test:
        put_object = check_put_object(s3_client, bucket, prefix)

    # Summary
    logging.info("=== AWS S3 Permissions Check Summary ===")
    logging.info(f"ListBucket permission: {'GRANTED' if list_bucket else 'DENIED'}")
    if key:
        full_key = os.path.join(prefix, key).replace("\\", "/") if prefix else key
        logging.info(f"GetObject permission for '{full_key}': {'GRANTED' if get_object else 'DENIED'}")
    if put_test:
        logging.info(f"PutObject permission: {'GRANTED' if put_object else 'DENIED'}")
    logging.info("=== End of Permissions Check ===\n")


if __name__ == "__main__":
    main()

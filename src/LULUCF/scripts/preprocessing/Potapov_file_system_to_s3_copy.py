"""

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/
python -m src.utilities.create_cluster -n 11 -cn AFOLU_flux_model_scripts
python -m scripts.preprocessing.Potapov_file_system_to_s3_copy -cn AFOLU_flux_model_scripts

For 2015-2023, there are 2826 files. Takes <10 minutes to transfer them.

# Based on https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/6794049f-2ba0-800a-b6ef-9445cfdd94a8

"""

import dask.bag as db
import boto3
import argparse
from bs4 import BeautifulSoup
import requests
from dask.distributed import print

# Project imports
from src.utilities import constants_and_names as cn, universal_utilities as uu


def download_and_upload_file(file_url, s3_key):
    """
    Download a file from the given URL and upload it to the specified S3 location.
    """

    s3_client = boto3.client("s3")

    try:
        # Download the file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Upload to S3
        s3_client.upload_fileobj(response.raw, cn.short_bucket_prefix, s3_key)
        print(f"Uploaded {file_url} to s3://{cn.short_bucket_prefix}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error processing {file_url}: {e}")
        return False


def process_year(year):
    """
    Process all files for a specific year by extracting file names from the HTML page
    and generating download URLs and S3 keys.
    """
    year_url = f"{cn.vegetation_height_annual_GLAD_path}/TCH_filter_{year}/"
    response = requests.get(year_url)

    if response.status_code != 200:
        raise ValueError(f"Unable to access {year_url}")

    # Parse the HTML to extract file names
    soup = BeautifulSoup(response.text, "html.parser")
    file_names = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".tif")]

    if not file_names:
        print(f"No .tifs found in {year_url}")
        return []

    # Create full file URLs and corresponding S3 keys
    tasks = [
        (f"{year_url}{file_name}", f"{cn.vegetation_height_annual_path}{year}/{file_name}"[cn.full_bucket_prefix_length:]) for
        file_name in file_names]

    return tasks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transfer geotifs from GLAD web folders to s3 bucket by year")
    parser.add_argument('-cn', '--cluster_name', type=str, help='Coiled cluster name')
    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    run_local = args.run_local

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)


    # Create tasks for all years
    all_tasks = []
    for year in cn.years_annual:
    # for year in [2015]:   # To test one year
        all_tasks.extend(process_year(year))

    # Create a Dask Bag for parallel processing
    bag = db.from_sequence(all_tasks, partition_size=10)
    results = bag.map(lambda x: download_and_upload_file(x[0], x[1])).compute()

    print("Transfer complete. Success:", sum(results), "Failures:", len(results) - sum(results))

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()

import os
import logging
import boto3
import rioxarray
import multiprocessing
"""
This was used to post process grip and osm canals output back into 30 meter. 
In the future, it should be integrated directly with those preprocessing scripts
and also the s3 upload piece
"""
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# AWS S3 setup
s3_bucket = "gfw2-data"
local_temp_dir = r"C:\GIS\Data\Global\Wetlands\Processed\30_m_temp"


def s3_file_exists(bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=key)
        logging.info(f"File exists: s3://{bucket}/{key}")
        return True
    except:
        logging.info(f"File does not exist: s3://{bucket}/{key}")
        return False


def list_s3_files(bucket, prefix):
    s3 = boto3.client('s3')
    keys = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
    except Exception as e:
        logging.error(f"Error listing files in s3://{bucket}/{prefix}: {e}")
    return keys


def log_raster_properties(raster, name):
    logging.info(f"{name} properties:")
    logging.info(f"  CRS: {raster.rio.crs}")
    logging.info(f"  Transform: {raster.rio.transform()}")
    logging.info(f"  Width: {raster.rio.width}")
    logging.info(f"  Height: {raster.rio.height}")
    logging.info(f"  Count: {raster.rio.count}")
    logging.info(f"  Dtype: {raster.dtype}")


def resample_raster(input_path, output_path, reference_path):
    input_raster = rioxarray.open_rasterio(input_path, masked=True)
    reference_raster = rioxarray.open_rasterio(reference_path, masked=True)

    # Log properties of input and reference rasters
    log_raster_properties(input_raster, "Input Raster")
    log_raster_properties(reference_raster, "Reference Raster")

    clipped_resampled_raster = input_raster.rio.clip_box(*reference_raster.rio.bounds())
    clipped_resampled_raster = clipped_resampled_raster.rio.reproject_match(reference_raster)

    clipped_resampled_raster.rio.to_raster(output_path)

    # Verify if the output file was created successfully
    if os.path.exists(output_path):
        logging.info(f"Successfully saved resampled raster to {output_path}")
    else:
        logging.error(f"Failed to save resampled raster to {output_path}")


def upload_to_s3(file_path, bucket, s3_path):
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, bucket, s3_path)
        logging.info(f"Successfully uploaded {file_path} to s3://{bucket}/{s3_path}")
    except Exception as e:
        logging.error(f"Failed to upload {file_path} to s3://{bucket}/{s3_path}: {e}")


def process_tile(tile_id):
    target_tiles = f'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/{tile_id}_peat_mask_processed.tif'
    grip_density = f"climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/1km/grip_density_{tile_id}.tif"
    osm_canals_density = f"climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/1km/canals_density_{tile_id}.tif"

    grip_output = f"climate/AFOLU_flux_model/organic_soils/inputs/processed/grip_density/grip_density_{tile_id}.tif"
    osm_canals_output = f"climate/AFOLU_flux_model/organic_soils/inputs/processed/osm_canals_density/canals_density_{tile_id}.tif"

    paths = {
        'grip': (grip_density, grip_output),
        'osm_canals': (osm_canals_density, osm_canals_output)
    }

    reference_path = f'/vsis3/{s3_bucket}/{target_tiles}'
    logging.info(f"Checking reference file: {reference_path}")
    if not s3_file_exists(s3_bucket, target_tiles):
        logging.error(f"Reference file {target_tiles} does not exist.")
        return

    for key, (input_path, output_path) in paths.items():
        input_s3_path = f'/vsis3/{s3_bucket}/{input_path}'
        local_output_dir = os.path.join(local_temp_dir, key)
        local_output_path = os.path.join(local_output_dir, os.path.basename(output_path))

        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)

        logging.info(f"Checking local output file for {key}: {local_output_path}")
        if os.path.exists(local_output_path):
            logging.info(f"Local output file {local_output_path} already exists. Skipping resampling for {key}.")
            continue

        logging.info(f"Checking input file for {key}: {input_path}")
        if not s3_file_exists(s3_bucket, input_path):
            logging.error(f"Input file {input_path} does not exist.")
            continue

        logging.info(f"Checking if output file exists for {key}: {output_path}")
        if s3_file_exists(s3_bucket, output_path):
            logging.info(f"Output file {output_path} already exists in S3. Skipping resampling for {key}.")
            continue

        try:
            resample_raster(input_s3_path, local_output_path, reference_path)
            logging.info(f"Resampled {key} for tile {tile_id}")

            # logging.info(f"Uploading {local_output_path} to s3://{s3_bucket}/{output_path}")
            # upload_to_s3(local_output_path, s3_bucket, output_path)
        except Exception as e:
            logging.error(f"Error processing {key} for tile {tile_id}: {e}")


def main(tile_id=None):
    try:
        if tile_id:
            logging.info(
                f"Listing files in S3 bucket '{s3_bucket}' with prefix 'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/'")
            files = list_s3_files(s3_bucket,
                                  'climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/')
            logging.info(f"Files in S3: {files}")

            process_tile(tile_id)
        else:
            s3_client = boto3.client('s3')
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=s3_bucket,
                                               Prefix='climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/')

            tile_ids = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        tile_key = obj['Key']
                        if tile_key.endswith('_peat_mask_processed.tif'):
                            tile_id = tile_key.split('/')[-1].replace('_peat_mask_processed.tif', '')
                            tile_ids.append(tile_id)

            for tile_id in tile_ids:
                process_tile(tile_id)
    finally:
        pass


# Example usage
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # main(tile_id='00N_110E')  # Replace with the tile ID you want to test
    main()  # Uncomment to process all tiles sequentially

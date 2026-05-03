"""
Script to create global COGS:
1) builds global VRT per dataset per year from all tiles in an s3 folder,
2) builds a global COG per dataset per year

run from /mnt/c/GIS/git/AFOLU_GHG_flux_model
python -m src.utilities.create_cluster -cn WWF_2016_emissions_cog -n 1 -m 32 --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs -cn WWF_2016_emissions_cog -d emissions -y 2016 -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_10x10_tile_ids.txt

python -m src.utilities.create_cluster -cn tclf_2025_cog -n 1 -m 32 --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs -cn tclf_2025_cog -d tclf

python -m src.utilities.create_cluster -cn hansen_pixel_area_cog -n 1 -m 32 --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs -cn hansen_pixel_area_cog -d pixel_area

python -m src.utilities.create_cluster -cn WWF_2016_removals_cog -t 1 -n 1 -m 64 -d 300 --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs -cn WWF_2016_removals_cog -d removals -y 2016 -t /mnt/c/GIS/rasters/AFOLU_cogs/operational_landscapes_10x10_tile_ids.txt --skip_existing

Notes:
For WWF Operational Landscapes:
    - Took 1.5 minutes to build VRT from 191 10x10 degree, per pixel tiles
    - For emissions, it took 5 hours to create a cog (with no overviews) from 191 10x10 degree, per pixel tiles (28% of global extent).
      I used a x2gd.xlarge wokrer (memory = 64 GB, CPU = 4, disk = 100 GB) which used 40 credits in total.
      GDAL_CACHEMAX = 70% and BIG_TIFF=IF_SAFER.
      It looks like memory peaked at 8.5 GB, disk peaked at about 25 GB, and required 4 CPUs.
      So for global run, try:
            - disk = 200 GB. Consider increasing CPU?
      The final COG only ended up being 12 GB.
    - For removals, it failed after 6.5 hours the first time I tried to create a cog (with no overviews) from 191 10x10 degree, per pixel tiles.
      Got the following error:
        No disk space left on workers (first happened at 2026-02-10 23:27:40) Some of your workers have run out of
        available disk space. This typically happens when you shuffle large amounts of data via the P2P shuffling
        mechanism, or you have to spill large amounts of data to disk because of memory pressure. To avoid your workers
        running out of disk space, you can manually request more disk space via coiled.Cluster(..., worker_disk_size="XYZ GB") or
        coiled.function(..., disk_size="XYZ GB").
      I used one 64 GB worker (type = x2gd.xlarge) with 100 GB of disk space which used 52 credits in total.
      It looks like memory peaked at 9 GB, disk peaked at 152 GB before crashing.

      Ran removals again and it finished successfully this time in 7 hours. The final COG was 85 GB.
      I used a r7i.2xlarge worker (memory = 64 GB, CPU = 8, disk = 300 GB) which used 110 credits.
      Memory peaked a little over 9 GB. The majority of the time it used 4 CPUs but spiked two-third of the way though
      and at the end where it used all CPUs. Disk peaked at around 151 GB.
      GDAL_CACHEMAX = 50% and BIG_TIFF=YES.



TODO:
- Decide on worker type, disk space, and GDAL_CACHEMAX, and BIG_TIFF for future runs
-Pass in creation option based on GDAL type (i.e. resampling algorithm for overviews, etc)
-Add progress bars to VRT and COG creation step
-Split by continent or quadrant to reduce metadata file size for COG-backed GEE assets?
-Add step to upload COGs to GCS storage?

"""
import os
import argparse
import time
from osgeo import gdal
import boto3

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

# TODO: move to UU
# Checks that file exists in s3 before deleting local copy
def check_s3_upload_and_clean_local(s3_path, local_path):
    logger_worker = lu.setup_logging_worker()
    if uu.exists_in_s3(s3_path):
        lu.print_and_log(f"File uploaded to S3: {s3_path}", False, logger_worker)
        try:
            os.remove(local_path)
        except Exception as e:
            lu.print_and_log(f"Warning: could not delete {local_path} - {e}", False, logger_worker)
#TODO: update usage in other places

# Read tile_ids from .txt file
def read_tile_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")}

# List s3 files in a path
# Option to filter the list to only files that match a list of tile_ids
def list_s3_tiles_from_tile_ids(s3_path, tile_ids):
    s3 = boto3.client("s3")
    bucket_name, prefix = uu.split_s3_path(s3_path)

    tile_ids = set(tile_ids) if tile_ids else None
    matching_tiles = []
    token = None

    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)

        for object in response.get("Contents", []):
            key = object["Key"]
            if not (key.lower().endswith(".tif") or key.lower().endswith(".tiff")):
                continue
            if tile_ids and not any(tile_id in key for tile_id in tile_ids):
                continue
            matching_tiles.append(key)

        if response.get("IsTruncated"):
            token = response["NextContinuationToken"]
        else:
            break

    return matching_tiles

# From Engineering: gdal_translate -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE="${BLOCK_SIZE}" -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS -co OVERVIEWS=AUTO -r "${RESAMPLE}" --config COMPRESS_OVERVIEW DEFLATE -co SPARSE_OK=TRUE --config GDAL_CACHEMAX 70% --config GDAL_NUM_THREADS ALL_CPUS
# From Michelle: tile_size=2048 for global 30m datasets
# From GEE documentation: COPY_SRC_OVERVIEWS=YES, TILED=YES, BLOCKXSIZE=512, BLOCKYSIZE=512, COMPRESS=ZSTD, ZSTD_LEVEL=22, INTERLEAVE=BAND, NUM_THREADS=ALL_CPUS
# From OpenGeoHub GPW: GDAL_CACHEMAX 10240, BLOCKSIZE=2048, BIGTIFF=YES, COMPRESS=DEFLATE, PREDICTOR=2,
#TODO: Resample should be mode for categorical
def gdal_translate_cog(vrt, cog, build_overviews=True, resample=None, nodata=None):

    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Translating COG: {vrt} -> {cog}", False, logger_worker)

    # Creation options
    co = [
            "COMPRESS=DEFLATE",
            "BLOCKSIZE=2048",
            "PREDICTOR=2",
            "BIGTIFF=IF_SAFER",
            "NUM_THREADS=ALL_CPUS",
            "SPARSE_OK=TRUE",
        ]

    # Internal overview generation when requested (only for final data)
    if build_overviews:
        co.append("OVERVIEWS=AUTO")

    # GDAL translate call
    opts = gdal.TranslateOptions(
        format="COG",
        creationOptions=co,
        resampleAlg=resample,
        noData=nodata,
    )

    # Set config options for GDAL translate
    with gdal.config_options({"GDAL_CACHEMAX": "60%", "GDAL_NUM_THREADS": "ALL_CPUS", "COMPRESS_OVERVIEW": "DEFLATE"}):
        ds = gdal.Translate(cog, vrt, options=opts)
        if ds is None:
            raise RuntimeError(f"GDAL Translate failed: {cog}: {gdal.GetLastErrorMsg()}")
        ds = None   #close


def create_cog_from_vrt(vrt_s3_path, tmp_cog_path, output_cog_s3_path):
    # Recommended from ChatGPT for vsis3 performance
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff,.vrt,.ovr,.aux.xml"
    os.environ["VSI_CACHE"] = "TRUE"
    os.environ["VSI_CACHE_SIZE"] = str(1 * 1024 * 1024 * 1024)  # 1GB
    os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

    logger_worker = lu.setup_logging_worker()

    # Check if the COG already exists in S3
    if uu.exists_in_s3(output_cog_s3_path):
        return lu.print_and_log(f"COG file already exists in S3: {output_cog_s3_path}. Skipping creation.", False, logger_worker)

    # Build COG via GDAL Python
    vrt = vrt_s3_path.replace("s3://", "/vsis3/")
    try:
        gdal_translate_cog(vrt, tmp_cog_path, build_overviews=True)
    except Exception as e:
        lu.print_and_log(f"COG build failed for {vrt}: {e}", False, logger_worker)
        raise
    lu.print_and_log(f"COG created locally at: {tmp_cog_path}", False, logger_worker)

    # Upload COG to S3
    uu.upload_s3_file(output_cog_s3_path, tmp_cog_path)

    # If successfully uploaded to s3, delete local COG
    check_s3_upload_and_clean_local(output_cog_s3_path, tmp_cog_path)



def main(cluster_name, datasets, years, tile_ids, skip_existing):

    # Connects to Coiled cluster if the named cluster exists
    if cluster_name:
        run_local = False
    else:
        run_local = True

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers= lu.populate_main_log_header(client, cluster, "Global COG creation", run_local, 'standard', 'Global COG creation')

    #TODO: This branch is behind the current model version (1.0.5) Change to cn paths after merging to updated model branch.
    emissions_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_pixel_yr/40000_pixels/20260130/"
    removals_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_removals__all_C_pools__MgCO2/annual_intervals/YYYY/_pixel_yr/40000_pixels/20260130/"
    net_flux_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_pixel_yr/40000_pixels/20260130/"
    mineral_soil_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/YYYY/_ha_yr/4000_pixels/20251224/"
    pixel_area_path = "s3://gfw2-data/analyses/area_28m/"
    tclf_path = "s3://gfw2-data/climate/carbon_model/other_emissions_inputs/tree_cover_loss_fires/20260316/processed/"

    emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e_pixel_yr"
    removals_pattern = "gross_removals__all_C_pools__MgCO2_pixel_yr"
    net_flux_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_pixel_yr"
    mineral_soil_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC_ha_yr"
    pixel_area_pattern = "hansen_pixel_area"
    tclf_pattern = "tree_cover_loss_fire_processed"

    # ------------------------------------------------------------------------------------------------------------------

    # Step 1: Create download/ upload dictionary for the datasets we want to create global COGs for.
    tile_ids = read_tile_ids(tile_ids) if tile_ids else None

    # Default to all available years if years are not provided by user
    if years is None:
        years = {
            "emissions": cn.interval_end_years_annual,
            "removals": cn.interval_end_years_annual,
            "net_flux": cn.interval_end_years_annual,
            "mineral_soil": cn.SOC_change_intervals,
            "pixel_area": [2013],
            "tclf": [2025]
        }

    # Datasets to pass in arguments for tile upload + GEE asset creation
    download_upload_dictionary = {}

    for dataset in datasets:
        ds_years = years if isinstance(years, list) else years[dataset]
        if not ds_years:
            raise ValueError(f"No years found for dataset={dataset}. Pass --years or add defaults.")

        for year in ds_years:
            if dataset == "emissions":
                s3_dir = emissions_path.replace("YYYY", str(year))
                pattern = emissions_pattern
            elif dataset == "removals":
                s3_dir = removals_path.replace("YYYY", str(year))
                pattern = removals_pattern
            elif dataset == "net_flux":
                s3_dir = net_flux_path.replace("YYYY", str(year))
                pattern = net_flux_pattern
            elif dataset == "mineral_soil":
                s3_dir = mineral_soil_path.replace("YYYY", str(year))
                pattern = mineral_soil_pattern
            elif dataset == "pixel_area":
                s3_dir = pixel_area_path
                pattern = pixel_area_pattern
            elif dataset == "tclf":
                s3_dir = tclf_path
                pattern = tclf_pattern
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            if dataset not in ("pixel_area", "tclf"):
                download_upload_dictionary[f"{dataset}_{year}"] = {
                    "dataset": dataset,
                    "year": year,
                    "s3_dir": s3_dir.rstrip("/") + "/",
                    "vrt_dir" : s3_dir.replace("40000_pixels", "global").rstrip("/") + "/",
                    'vrt': f"/tmp/{pattern}_{year}.vrt",
                    'cog': f"/tmp/{pattern}_{year}.tif"
                }
            elif dataset == "pixel_area":
                download_upload_dictionary[f"{dataset}"] = {
                    "dataset": dataset,
                    "s3_dir": s3_dir.rstrip("/") + "/",
                    "vrt_dir": s3_dir.replace("area_28m", "area_28m_global_cog").rstrip("/") + "/",
                    'vrt': f"/tmp/{pattern}.vrt",
                    'cog': f"/tmp/{pattern}.tif"
                }
            elif dataset == "tclf":
                download_upload_dictionary[f"{dataset}"] = {
                    "dataset": dataset,
                    "s3_dir": s3_dir.rstrip("/") + "/",
                    "vrt_dir": s3_dir.replace("processed", "cog").rstrip("/") + "/",
                    'vrt': f"/tmp/{pattern}.vrt",
                    'cog': f"/tmp/{pattern}.tif"
                }
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            #TODO: delete pixel area and tclf logic after running globally

    # -------------------------------------------------------------------------------------------------------------------

    # Step 2: Create a VRT per year for each dataset (i.e. 9 VRTs per dataset)
    start_time = time.time()
    main_logger.info(f"STEP 2 - Building VRTs")

    keys_to_remove = []
    for key, items in download_upload_dictionary.items():

        # Add output VRT s3 path to the dictionary
        vrt_s3_path = f"{items['vrt_dir']}{os.path.basename(items['vrt'])}"
        download_upload_dictionary[key]['vrt_s3_path'] = vrt_s3_path

        # If skip_existing flag is passed and VRT already exists in S3, skip VRT creation step
        download_upload_dictionary[key]["skip_vrt"] = False
        if skip_existing and uu.exists_in_s3(vrt_s3_path):
            main_logger.info(f"{key} - VRT exists in S3, skipping VRT build: {vrt_s3_path}")
            download_upload_dictionary[key]["skip_vrt"] = True
            continue

        # Otherwise, find all tiles in s3 directory (filtered if tile_ids if provided)
        bucket, prefix = uu.split_s3_path(items["s3_dir"])
        keys = list_s3_tiles_from_tile_ids(items["s3_dir"], tile_ids)
        s3_raster_list = [f"s3://{bucket}/{k}" for k in keys]

        # Remove datasetS with no tiles in s3 from the VRT + COG pipeline
        if not s3_raster_list:
            main_logger.warning(f"{key} - There were no rasters found in s3. Skipping VRT/COG creation.")
            keys_to_remove.append(key)
            continue

        download_upload_dictionary[key]['s3_raster_list'] = s3_raster_list
        main_logger.info( f" {key} - There are {len(s3_raster_list)} rasters in s3 to include in the vrt")

    # Remove datasets from the download_upload dictionary that don't have any rasters in their s3_dir
    for key in keys_to_remove:
        download_upload_dictionary.pop(key, None)

    # Create VRTs for each year x dataset in parallel using futures (Coiled) or locally
    if not run_local:
        vrt_futures = []
        for key, items in download_upload_dictionary.items():
            if items["skip_vrt"]:
                continue
            main_logger.info(f" Submitting VRT build for {key}: {uu.timestr('time')}")
            vrt_future = client.submit(uu.build_vrt_gdal_coiled, items['s3_raster_list'], items['vrt_s3_path'], items['vrt'])
            vrt_futures.append((key, vrt_future))

        # Wait for all VRTs to finish before moving on to Step 3
        for key, future in vrt_futures:
            future.result()
    else:
        for key, items in download_upload_dictionary.items():
            if items["skip_vrt"]:
                continue
            main_logger.info(f" Submitting VRT build for {key}: {uu.timestr('time')}")
            uu.build_vrt_gdal_coiled(items['s3_raster_list'], items['vrt_s3_path'], items['vrt'])

    end_time = time.time()
    main_logger.info(f"STEP 2 Complete - All VRTs built in {round(end_time-start_time)} seconds\n")

    #-------------------------------------------------------------------------------------------------------------------

    # Step 3: Get GDAL datatype of each dataset using the first tile in that dataset
    start_time = time.time()
    main_logger.info(f"STEP 3 - Getting GDAL datatypes")

    for key, items in download_upload_dictionary.items():
        simple_dict = {}
        simple_dict[key] = items["s3_dir"]

        # Path of first tile in the dataset
        first_tile = uu.first_file_name_in_s3_folder(simple_dict)

        # Gets datatype of first tile in input dataset and converts it to GDAL format
        download_dict_with_data_types = uu.add_file_type_to_dict(first_tile)
        dtype = download_dict_with_data_types[key][1]
        gdal_dtype = uu.string_to_gdal_dtype_mapping.get(dtype)
        gdal_dtype_str = uu.gdal_to_string_dtype_mapping.get(gdal_dtype)

        # Adds the dtype of the dataset to the processing dictionary
        download_upload_dictionary[key]["dt"] = gdal_dtype_str
        main_logger.info(f" Data type for {key} is {gdal_dtype_str}")

    end_time = time.time()
    main_logger.info(f"STEP 3 Complete - All GDAL datatypes added to dictionary in {round(end_time-start_time)} seconds\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Step 4: Building global COGs for each dataset x year
    start_time = time.time()
    main_logger.info(f"STEP 4 - Building global COGs")

    for key, items in download_upload_dictionary.items():
        cog_s3_path = f"{items['vrt_dir']}{os.path.basename(items['cog'])}"
        download_upload_dictionary[key]['cog_s3_path'] = cog_s3_path

        # If skip_existing flag is passed and COG exists in S3, skip COG creation step
        download_upload_dictionary[key]["skip_cog"] = False
        if skip_existing and uu.exists_in_s3(cog_s3_path):
            main_logger.info(f"{key} - COG exists in S3, skipping COG creation: {cog_s3_path}")
            download_upload_dictionary[key]["skip_cog"] = True
            continue

    if not run_local:
        cog_futures = []
        for key, items in download_upload_dictionary.items():
            if items["skip_cog"]:
                continue
            main_logger.info(f" Submitting global COG build for {key}")
            cog_future = client.submit(create_cog_from_vrt, items["vrt_s3_path"], items["cog"], items["cog_s3_path"])
            cog_futures.append((key, cog_future))

        for key, future in cog_futures:
            future.result()

    else:
        for key, items in download_upload_dictionary.items():
            if items["skip_cog"]:
                continue
            main_logger.info(f" Submitting global COG build for {key}")
            create_cog_from_vrt(items["vrt_s3_path"], items["cog"], items["cog_s3_path"])

    end_time = time.time()
    main_logger.info(f"STEP 4 Complete - All global COGs built in {round(end_time - start_time)} seconds\n")

    # -------------------------------------------------------------------------------------------------------------------

    # Closes the client if not running locally
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global COGs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--datasets', required=True, nargs='+', help='What datasets do you want to convert to global COGs? Current options: emissions, removals, net flux, mineral soil soc change')
    parser.add_argument('-y', '--years', nargs='+', help="Which year(s) to run? Defaults to use all available years if not specified.")
    parser.add_argument('-t', "--tile_ids", help="Optional text file with tile ids to filter to (one per line)")
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()
    cluster_name = args.cluster_name
    datasets = args.datasets
    years = args.years
    tile_ids = args.tile_ids
    skip_existing = args.skip_existing

    main(cluster_name, datasets, years, tile_ids, skip_existing)
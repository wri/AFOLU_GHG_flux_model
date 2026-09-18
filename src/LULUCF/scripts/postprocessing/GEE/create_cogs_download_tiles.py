"""
Create global COGs by staging all source rasters locally on the worker.

Workflow for each dataset/year:
1) list source tiles in S3
2) download all source tiles to /tmp/cog_staging/<dataset_year>/tiles/
3) read datatype and NoData from first tile
4) build a VRT from downloaded tiles
5) build the COG from the local VRT/ tiles
6) upload the final COG to S3
7) remove the local staging directory after successful upload
This avoids /vsis3/ reads during gdal.Translate(), which makes the long COG build independent of transient S3 range-read failures.

For global:
gcloud auth application-default login

python -m src.utilities.create_cluster -cn 2016_emissions_cog -t 1 -n 1 -m 64 -d 100 --cog --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2016_emissions_cog -d vegetation -f emissions -y 2016

python -m src.utilities.create_cluster -cn 2016_removals_cog -t 1 -n 1 -m 64 -d 500 --cog --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2016_removals_cog -d vegetation -f removals -y 2016

python -m src.utilities.create_cluster -cn 2016_2020_mineral_soil_net_cog -t 1 -n 1 -m 64 -d 1000 --cog --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2016_2020_mineral_soil_net_cog -d mineral_soil -f netflux -y 2016

python -m src.utilities.create_cluster -cn 2016_2020_organic_soil -t 1 -n 2 -m 64 -d 100 --cog --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2016_2020_organic_soil -d organic_soil -f emissions -y 2016

Cautions:
- Currently all int datasets are set to use mode (categorical) resampling algorithm and all float are set to use mean.
  If creating a COG for a non-categorical int dataset, update the code accordingly.


Notes:
    - For emissions could try 50 GB of disk. Keep or increase CPU.
    - For removals, could try 300 GB of disk. Keep or increase CPU.
    - Using 32 GB workers was WAAAAAY slower. Use 64 instead.

TODO:
- The COG translation step takes a very long time. How to make this faster?
- Ask Chris about attaching S3 directory so tiles don't have to be downloaded to save time. Then get rid of download_workers input argument.
- Get rid of --tile_ids input argument (option to filter VRT/ COG to only certain tiles)?
"""

import argparse
import concurrent.futures
import os
import shutil
import time
from pathlib import Path

import boto3
from botocore.config import Config
from osgeo import gdal
from google.cloud import storage
import google.auth
from google.auth.transport.requests import Request

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

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


# Download one source tile with an outer retry around boto3 download_file
def download_s3_tile(s3_client, bucket, key, local_path, max_attempts=3):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            s3_client.download_file(bucket, key, str(local_path))
            return str(local_path)
        except Exception:
            if local_path.exists():
                local_path.unlink()
            if attempt == max_attempts:
                raise
            time.sleep(10 * attempt)

# Download all source tiles to a local staging directory
def download_s3_tiles(s3_paths, local_tile_dir, max_workers=8):
    logger_worker = lu.setup_logging_worker()

    local_tile_dir = Path(local_tile_dir)
    local_tile_dir.mkdir(parents=True, exist_ok=True)

    # Increase connection pool because downloads are intentionally concurrent.
    s3_client = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=max_workers))

    download_args = []
    for s3_path in s3_paths:
        bucket, key = uu.split_s3_path(s3_path)
        local_path = local_tile_dir / os.path.basename(key)
        download_args.append((bucket, key, local_path))

    lu.print_and_log(f"Downloading {len(download_args)} source tiles to {local_tile_dir}",False, logger_worker)

    local_tiles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_s3_tile, s3_client, bucket, key, local_path): local_path
            for bucket, key, local_path in download_args
        }

        for future in concurrent.futures.as_completed(futures):
            local_tiles.append(future.result())

    # Deterministic ordering makes VRT contents reproducible.
    local_tiles.sort()
    return local_tiles

# Read GDAL datatype, NoData, and choose overview resampling from a local tile
def get_local_raster_info(first_tile):
    logger_worker = lu.setup_logging_worker()

    ds = gdal.Open(first_tile, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open local raster: {first_tile}")

    band = ds.GetRasterBand(1)
    gdal_dtype = band.DataType
    gdal_dtype_str = gdal.GetDataTypeName(gdal_dtype)
    nodata = band.GetNoDataValue()
    ds = None

    if gdal_dtype in (gdal.GDT_Float32, gdal.GDT_Float64):
        resample = "average"
    elif gdal_dtype in (
        gdal.GDT_Byte,
        gdal.GDT_Int8,
        gdal.GDT_UInt16,
        gdal.GDT_Int16,
        gdal.GDT_UInt32,
        gdal.GDT_Int32,
        gdal.GDT_UInt64,
        gdal.GDT_Int64,
    ):
        resample = "mode"
    else:
        raise ValueError(f"Unsupported GDAL datatype: {gdal_dtype_str}")

    lu.print_and_log(f"Local source datatype={gdal_dtype_str}; NoData={nodata}; resample={resample}", False, logger_worker)

    return gdal_dtype_str, nodata, resample

# Build a VRT that references local source files
def build_local_vrt(local_tiles, local_vrt):
    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Building VRT: {local_vrt}", False, logger_worker)

    vrt_ds = gdal.BuildVRT(local_vrt, local_tiles)
    if vrt_ds is None:
        raise RuntimeError(f"GDAL BuildVRT failed: {local_vrt}: {gdal.GetLastErrorMsg()}")
    vrt_ds.FlushCache()
    vrt_ds = None

# From Engineering: gdal_translate -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE="${BLOCK_SIZE}" -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS -co OVERVIEWS=AUTO -r "${RESAMPLE}" --config COMPRESS_OVERVIEW DEFLATE -co SPARSE_OK=TRUE --config GDAL_CACHEMAX 70% --config GDAL_NUM_THREADS ALL_CPUS
# From Michelle: tile_size=2048 for global 30m datasets
# From GEE documentation: COPY_SRC_OVERVIEWS=YES, TILED=YES, BLOCKXSIZE=512, BLOCKYSIZE=512, COMPRESS=ZSTD, ZSTD_LEVEL=22, INTERLEAVE=BAND, NUM_THREADS=ALL_CPUS
# From OpenGeoHub GPW: GDAL_CACHEMAX 10240, BLOCKSIZE=2048, BIGTIFF=YES, COMPRESS=DEFLATE, PREDICTOR=2,
def gdal_translate_cog(vrt, cog, nodata, resample=None, build_overviews=True):
    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"Translating COG: {cog}", False, logger_worker)

    # Creation options
    co = [
            "COMPRESS=ZSTD",
            "LEVEL=22",
            "INTERLEAVE=BAND",
            "TILED=YES",
            "BLOCKSIZE=2048",
            "PREDICTOR=YES",
            "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS",
            "SPARSE_OK=TRUE",
        ]

    # Internal overview generation when requested (only for final data)
    if build_overviews:
        co.extend([
            "OVERVIEWS=IGNORE_EXISTING",
            "OVERVIEW_COMPRESS=ZSTD",
            "OVERVIEW_PREDICTOR=YES",
        ])
    else:
        co.append("OVERVIEWS=NONE")

    # Print progress every 5%
    last_percent = {"value": -5}

    def progress_callback(complete, message, unknown):
        percent = int(complete * 100)

        if percent >= last_percent["value"] + 5 or percent == 100:
            bar_width = 20
            filled = int(bar_width * percent / 100)
            bar = "#" * filled + "-" * (bar_width - filled)

            lu.print_and_log( f"COG progress: [{bar}] {percent}%", False, logger_worker)

            last_percent["value"] = percent

        return 1

    # GDAL translate call
    opts = gdal.TranslateOptions(
        format="COG",
        creationOptions=co,
        resampleAlg=resample,
        noData=nodata,
        callback=progress_callback
    )

    # Set config options for GDAL translate
    with gdal.config_options({
        "GDAL_CACHEMAX": "12288",        # 12 GB
        "GDAL_NUM_THREADS": "ALL_CPUS"
      #"ZSTD_LEVEL_OVERVIEW": "22"
      }):
        ds = gdal.Translate(cog, vrt, options=opts)
        if ds is None:
            raise RuntimeError(f"GDAL Translate failed: {cog}: {gdal.GetLastErrorMsg()}")
        ds = None   #close

def refresh_gcp_credentials():
    logger_worker = lu.setup_logging_worker()

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    lu.print_and_log(f"Refreshing GCP credentials for project: {project}", False, logger_worker)
    credentials.refresh(Request())
    lu.print_and_log( "GCP credentials refreshed successfully", False, logger_worker)

    return credentials, project

# Upload local file to Google Cloud Storage
def upload_file_to_gcs(local_path, gcs_path):
    logger_worker = lu.setup_logging_worker()

    if not gcs_path.startswith("gs://"):
        raise ValueError(f"GCS path must start with gs://: {gcs_path}")

    path_without_prefix = gcs_path.replace("gs://", "", 1)
    bucket_name, blob_name = path_without_prefix.split("/", 1)

    credentials, project = refresh_gcp_credentials()
    client = storage.Client(project=project, credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(str(local_path))

    if not blob.exists(client):
        raise RuntimeError(f"GCS upload finished but file could not be verified: {gcs_path}")

    lu.print_and_log(f"File uploaded successfully: {gcs_path}",False, logger_worker)

# Download tiles -> build local VRT -> create local COG -> S3 and GCS upload -> tmp cleanup
def stage_build_upload_cog(job, download_workers=8):
    logger_worker = lu.setup_logging_worker()

    tmp_root = Path(job["tmp_root"])
    tile_dir = tmp_root / "tiles"
    tmp_vrt_path = Path(job["tmp_vrt_path"])
    tmp_cog_path = Path(job["tmp_cog_path"])

    s3_raster_list = job["s3_raster_list"]

    output_cog_s3_path = job["output_cog_s3_path"]
    output_cog_gcs_path = job["output_cog_gcs_path"]

    # Start clean so a failed prior attempt cannot contaminate this run.
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    tmp_root.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tmp_vrt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cog_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # -------------------------------------------------------------------------------------------------------------------
        # Step 1: Download tiles to a local tmp dir
        start_time = time.time()
        lu.print_and_log(f"STEP 1 - Downloading tiles", False, logger_worker)

        local_tiles = download_s3_tiles(s3_raster_list, tile_dir, max_workers=download_workers)
        if not local_tiles:
            raise RuntimeError("No local source tiles were downloaded")
        else:
            end_time = time.time()
            lu.print_and_log(f"STEP 1 Complete - Downloaded {len(local_tiles)} tiles in {round(end_time - start_time)} seconds\n", False, logger_worker)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 2: Get GDAL datatype of each dataset using the first tile in that dataset
        lu.print_and_log(f"STEP 2 - Getting GDAL datatype", False, logger_worker)
        gdal_dtype_str, nodata, resample = get_local_raster_info(local_tiles[0])
        lu.print_and_log(f"STEP 2 Complete\n", False, logger_worker)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 3: Build a VRT from local tiles
        start_time = time.time()
        lu.print_and_log(f"STEP 3 - Building VRT", False, logger_worker)

        build_local_vrt(local_tiles, str(tmp_vrt_path))

        end_time = time.time()
        lu.print_and_log(f"STEP 3 Complete - Built VRT in {round(end_time - start_time)} seconds\n", False, logger_worker)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 4: Translate COG from local VRT/ tiles
        start_time = time.time()
        lu.print_and_log(f"STEP 4 - Translating COG", False, logger_worker)

        gdal_translate_cog(str(tmp_vrt_path), str(tmp_cog_path), nodata, resample, build_overviews=True)

        end_time = time.time()
        lu.print_and_log(f"STEP 4 Complete - COG translated in {round(end_time - start_time)} seconds\n", False,  logger_worker)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 5: Upload COG to s3
        start_time = time.time()
        lu.print_and_log(f"STEP 5 - Upload COG to s3", False, logger_worker)

        uu.upload_s3_file(output_cog_s3_path, str(tmp_cog_path))

        if not uu.exists_in_s3(output_cog_s3_path):
            raise RuntimeError(f"Upload finished but output could not be verified in S3: {output_cog_s3_path}")
        else:
            end_time = time.time()
            lu.print_and_log(f"STEP 5 Complete - COG uploaded to s3 successfully in {round(end_time - start_time)} seconds", False, logger_worker)


        # -------------------------------------------------------------------------------------------------------------------
        # Step 6: Upload COG to GCS
        start_time = time.time()
        lu.print_and_log("STEP 6 - Upload COG to GCS", False, logger_worker)

        upload_file_to_gcs(tmp_cog_path, output_cog_gcs_path)

        end_time = time.time()
        lu.print_and_log(f"STEP 6 Complete - COG uploaded to GCS successfully in {round(end_time - start_time)} seconds",False, logger_worker)

        # -------------------------------------------------------------------------------------------------------------------

    except Exception:
        # Keep staged inputs after a failure so the worker can be inspected/retried without immediately losing the downloaded data.
        lu.print_and_log(f"COG workflow failed. Leaving staging directory in place for inspection: {tmp_root}", False, logger_worker)
        raise

    else:
        shutil.rmtree(tmp_root)
        lu.print_and_log(f"Removed local staging directory: {tmp_root}", False, logger_worker)


def main(cluster_name, datasets, fluxes, years, tile_ids, overwrite_existing_cog, download_workers):

    # Connects to Coiled cluster if the named cluster exists
    if cluster_name:
        run_local = False
    else:
        run_local = True

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers= lu.populate_main_log_header(client, cluster, "Global COG creation", run_local, 'standard', 'Global COG creation')

    #TODO: This script is behind the current model version (1.0.6) Change to cn paths after merging to updated model branch.
    #TODO: Add in other variations (by gas, land state node, land use, SOC gain, SOC loss, organic soil state, combined LULUCF, etc)
    #TODO: Require which gas to run from command line argument, if none supplied default to all_gases. Update in both s3 and gcs dirs.
    #TODO: Add in flag for per hectare or per pixel versions? 1x1 vs 10x10 tiles?
    veg_model_version_underscore = "1_0_5"

    veg_output_dir = f"{cn.LULUCF_dir}outputs_vegetation/version_{veg_model_version_underscore}__standard__global/"
    min_soil_output_dir = f"{cn.LULUCF_dir}outputs_soil_organic_carbon/version_{cn.SOC_model_version_underscore}__standard__global/"
    org_soil_output_dir = f"{cn.AFOLU_dir}organic_soils/outputs/version_{cn.organic_soil_model_version_underscore}/"

    veg_rundate = "20260130"
    min_soil_rundate = "20260611"
    org_soil_rundate = "20260525"

    emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e"
    removals_pattern  = "gross_removals__all_C_pools__MgCO2"
    netflux_pattern  = "net_flux__all_C_pools__all_gases__MgCO2e"
    # Note: the patterns include veg_ at the beginning starting with v1.0.6

    # Input tile s3 directories
    emissions_path = f"{veg_output_dir}{emissions_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    removals_path  = f"{veg_output_dir}{removals_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    netflux_path   = f"{veg_output_dir}{netflux_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"

    min_soil_loss_path = f"{min_soil_output_dir}{cn.SOC_loss_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_gain_path = f"{min_soil_output_dir}{cn.SOC_gain_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_net_path  = f"{min_soil_output_dir}{cn.SOC_net_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"

    org_soil_burned_path  = f"{org_soil_output_dir}{cn.burned_organic_soils_total_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    org_soil_drained_path = f"{org_soil_output_dir}{cn.drained_organic_soils_total_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"


    # GCS output directories
    gcs_bucket = "wri-lcl-lgms"
    veg_asset_folder = f"vegetation/v{veg_model_version_underscore}"
    min_soil_asset_folder = f"soil/mineral/v{cn.SOC_model_version_underscore}"
    org_soil_asset_folder = f"soil/organic/v{cn.organic_soil_model_version_underscore}"

    emissions_gcs_dir = f"gs://{gcs_bucket}/{veg_asset_folder}/emissions/{emissions_pattern}/"
    removals_gcs_dir  = f"gs://{gcs_bucket}/{veg_asset_folder}/removals/{removals_pattern}/"
    netflux_gcs_dir   = f"gs://{gcs_bucket}/{veg_asset_folder}/net_flux/{netflux_pattern}/"

    min_soil_loss_gcs_dir = f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_loss_min_soil_extent_pattern}/"
    min_soil_gain_gcs_dir = f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_gain_min_soil_extent_pattern}/"
    min_soil_net_gcs_dir  = f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_net_min_soil_extent_pattern}/"

    org_soil_burned_gcs_dir  = f"gs://{gcs_bucket}/{org_soil_asset_folder}/{cn.burned_organic_soils_total_pattern}/"
    org_soil_drained_gcs_dir = f"gs://{gcs_bucket}/{org_soil_asset_folder}/{cn.drained_organic_soils_total_pattern}/"


    # ------------------------------------------------------------------------------------------------------------------

    # Create download/ upload dictionary for the datasets we want to create global COGs for.

    tile_ids = read_tile_ids(tile_ids) if tile_ids else None

    # Resolve soil intervals once
    if years is None:
        years = cn.veg_outputs_years
    mineral_soil_years = sorted({cn.mineral_soil_year_map[year] for year in years})
    organic_soil_years = sorted({cn.organic_soil_year_map[year] for year in years})
    if "vegetation" in datasets:
        main_logger.info(f"Vegetation years required: {years}")
    if "mineral_soil" in datasets:
        main_logger.info(f"Mineral soil output years required: {mineral_soil_years}")
    if "organic_soil" in datasets:
        main_logger.info(f"Organic soil intervals required: {organic_soil_years}")

    # ------------------------------------------------------------------------------------------------------------------
    # Define available dataset / flux combinations

    dataset_config = {
        "vegetation": {
            "emissions": {"s3_path": emissions_path,
                          "pattern": f"{emissions_pattern}_ha_yr",
                          "gcs_dir": emissions_gcs_dir,
                          "years":   years},
            "removals":  {"s3_path": removals_path,
                          "pattern": f"{removals_pattern}_ha_yr",
                          "gcs_dir": removals_gcs_dir,
                          "years":   years},
            "netflux":   {"s3_path": netflux_path,
                          "pattern": f"{netflux_pattern}_ha_yr",
                          "gcs_dir": netflux_gcs_dir,
                          "years":   years},
        },
        "mineral_soil": {
            "emissions": {"s3_path": min_soil_loss_path,
                          "pattern": f"{cn.SOC_loss_min_soil_extent_pattern}_ha_yr",
                          "gcs_dir": min_soil_loss_gcs_dir,
                          "years":   mineral_soil_years},
            "removals":  {"s3_path": min_soil_gain_path,
                          "pattern": f"{cn.SOC_gain_min_soil_extent_pattern}_ha_yr",
                          "gcs_dir": min_soil_gain_gcs_dir,
                          "years":   mineral_soil_years},
            "netflux":   {"s3_path": min_soil_net_path,
                          "pattern": f"{cn.SOC_net_min_soil_extent_pattern}_ha_yr",
                          "gcs_dir": min_soil_net_gcs_dir,
                          "years":   mineral_soil_years},
        },
        "organic_soil": {
            "emissions": [
                {"component": "burned",
                 "s3_path": org_soil_burned_path,
                 "pattern": cn.burned_organic_soils_total_pattern,
                 "gcs_dir": org_soil_burned_gcs_dir,
                 "years": organic_soil_years},
                {"component": "drained",
                 "s3_path": org_soil_drained_path,
                 "pattern": cn.drained_organic_soils_total_pattern,
                 "gcs_dir": org_soil_drained_gcs_dir,
                 "years": organic_soil_years},
            ],
        },
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Build jobs dictionary

    jobs = {}

    for dataset in datasets:
        for flux in fluxes:

            if flux not in dataset_config[dataset]:
                main_logger.warning(f"{dataset} does not have a {flux} output. Skipping.")
                continue

            configs = dataset_config[dataset][flux]
            if not isinstance(configs, list):
                configs = [configs]

            for config in configs:
                for year in config["years"]:

                    s3_dir = config["s3_path"].replace("YYYY", str(year))
                    pattern = config["pattern"]
                    gcs_dir = config["gcs_dir"]
                    component = config.get("component")

                    output_year = cn.organic_soil_year_map[year] if dataset == "mineral_soil" else year

                    key = f"{dataset}_{flux}_{component}_{output_year}" if component else f"{dataset}_{flux}_{output_year}"
                    output_dir = s3_dir.replace("40000_pixels", "global").rstrip("/") + "/"
                    output_cog_s3_path = f"{output_dir}{pattern}_{output_year}.tif"
                    output_cog_gcs_path = f"{gcs_dir.rstrip('/')}/{pattern}_{output_year}.tif"

                    if not overwrite_existing_cog and uu.exists_in_s3(output_cog_s3_path):
                        main_logger.info(f"{key} - COG already exists in S3: {output_cog_s3_path}. Skipping creation.")
                        continue

                    bucket, _ = uu.split_s3_path(s3_dir)
                    s3_keys = list_s3_tiles_from_tile_ids(s3_dir, tile_ids)
                    s3_raster_list = [f"s3://{bucket}/{s3_key}" for s3_key in s3_keys]

                    if not s3_raster_list:
                        main_logger.warning(f"{key} - No source rasters found in {s3_dir}. Skipping.")
                        continue

                    jobs[key] = {
                        "s3_raster_list": s3_raster_list,
                        "tmp_root": f"/tmp/cog_staging/{key}",
                        "tmp_vrt_path": f"/tmp/cog_staging/{key}/vrt/{pattern}_{output_year}.vrt",
                        "tmp_cog_path": f"/tmp/cog_staging/{key}/cog/{pattern}_{output_year}.tif",
                        "output_cog_s3_path": output_cog_s3_path,
                        "output_cog_gcs_path": output_cog_gcs_path,
                    }

    # -------------------------------------------------------------------------------------------------------------------

    # Download tiles, build VRT, create COG and upload to s3
    # With one Coiled worker / one Dask thread, jobs execute one at a time.
    # This is intentional because each job may consume hundreds of GB of local disk.
    if not run_local:
        futures = []
        for key, job in jobs.items():
            main_logger.info(f"Submitting COG build for {key}")
            future = client.submit(stage_build_upload_cog, job, download_workers)
            futures.append((key, future))

        for key, future in futures:
            future.result()
    else:
        for key, job in jobs.items():
            main_logger.info(f"Staging COG build for {key}")
            stage_build_upload_cog(job, download_workers)

    if client is not None:
        client.close()

    # -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global COGs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--datasets', required=True, nargs='+', choices=["vegetation", "mineral_soil", "organic_soil"], help=("Dataset(s) to process. Options: vegetation, mineral_soil, organic_soil"))
    parser.add_argument('-f', '--flux', required=True, nargs='+', choices=["emissions", "removals", "netflux"], help=("Flux type(s) to process. Options: emissions, removals, netflux"))
    parser.add_argument('-y', '--years', nargs='+', type=int, choices=range(2016, 2025), help="Which year(s) to run? Defaults to use all available years if not specified.")
    parser.add_argument('-t', "--tile_ids", help="Optional text file with tile ids to filter to (one per line)")
    parser.add_argument("--overwrite_existing_cog", action="store_true", default=False, help="Option to overwrite existing COGs in s3. Default: False")
    parser.add_argument( '-dw', "--download_workers", type=int, default=8, help="Number of concurrent S3 tile downloads. Default: 8")

    args = parser.parse_args()
    cluster_name = args.cluster_name
    datasets = args.datasets
    fluxes = args.flux
    years = args.years
    tile_ids = args.tile_ids
    overwrite_existing_cog = args.overwrite_existing_cog
    download_workers = args.download_workers

    main(cluster_name, datasets, fluxes, years, tile_ids, overwrite_existing_cog, download_workers)

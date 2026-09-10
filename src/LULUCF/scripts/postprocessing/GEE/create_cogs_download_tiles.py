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
python -m src.utilities.create_cluster -cn 2017_emissions_cog -t 1 -n 1 -m 64 -d 300 -c --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2017_emissions_cog -d emissions -y 2017

python -m src.utilities.create_cluster -cn 2017_removals_cog -t 1 -n 1 -m 64 -d 1000 -c --on_demand
python -m src.LULUCF.scripts.postprocessing.GEE.create_cogs_download_tiles -cn 2017_removals_cog -d removals -y 2017

Cautions:
- Currently all int datasets are set to use mode (categorical) resampling algorithm and all float are set to use mean.
  If creating a COG for a non-categorical int dataset, update thr code accordingly.

TODO:
- Ask Chris about attaching S3 directory so tiles don't have to be downloaded to save time. Then get rid of download_workers input argument.
- Add final step to upload COGs to GCS storage?
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

    lu.print_and_log(f"     Downloading {len(download_args)} source tiles to {local_tile_dir}",False, logger_worker)

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
        raise RuntimeError(f"     Could not open local raster: {first_tile}")

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
        raise ValueError(f"     Unsupported GDAL datatype: {gdal_dtype_str}")

    lu.print_and_log(f"     Local source datatype={gdal_dtype_str}; NoData={nodata}; resample={resample}", False, logger_worker)

    return gdal_dtype_str, nodata, resample

# Build a VRT that references local source files
def build_local_vrt(local_tiles, local_vrt):
    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"     Building VRT: {local_vrt}", False, logger_worker)

    vrt_ds = gdal.BuildVRT(local_vrt, local_tiles)
    if vrt_ds is None:
        raise RuntimeError(f"     GDAL BuildVRT failed: {local_vrt}: {gdal.GetLastErrorMsg()}")
    vrt_ds.FlushCache()
    vrt_ds = None

# From Engineering: gdal_translate -of COG -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKSIZE="${BLOCK_SIZE}" -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS -co OVERVIEWS=AUTO -r "${RESAMPLE}" --config COMPRESS_OVERVIEW DEFLATE -co SPARSE_OK=TRUE --config GDAL_CACHEMAX 70% --config GDAL_NUM_THREADS ALL_CPUS
# From Michelle: tile_size=2048 for global 30m datasets
# From GEE documentation: COPY_SRC_OVERVIEWS=YES, TILED=YES, BLOCKXSIZE=512, BLOCKYSIZE=512, COMPRESS=ZSTD, ZSTD_LEVEL=22, INTERLEAVE=BAND, NUM_THREADS=ALL_CPUS
# From OpenGeoHub GPW: GDAL_CACHEMAX 10240, BLOCKSIZE=2048, BIGTIFF=YES, COMPRESS=DEFLATE, PREDICTOR=2,
def gdal_translate_cog(vrt, cog, nodata, resample=None, build_overviews=True):
    logger_worker = lu.setup_logging_worker()
    lu.print_and_log(f"     Translating COG: {cog}", False, logger_worker)

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

            lu.print_and_log( f"     COG progress: [{bar}] {percent}%", False, logger_worker)

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
            raise RuntimeError(f"     GDAL Translate failed: {cog}: {gdal.GetLastErrorMsg()}")
        ds = None   #close

# Download tiles -> build local VRT -> create local COG -> S3 upload -> tmp cleanup."""
def stage_build_upload_cog(s3_raster_list, tmp_root, tmp_vrt_path, tmp_cog_path, output_cog_s3_path, download_workers=8):
    logger_worker = lu.setup_logging_worker()

    tmp_root = Path(tmp_root)
    tile_dir = tmp_root / "tiles"
    tmp_vrt_path = Path(tmp_vrt_path)
    tmp_cog_path = Path(tmp_cog_path)

    # Start clean so a failed prior attempt cannot contaminate this run.
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    tmp_root.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(parents=True, exist_ok=True)
    tmp_vrt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_cog_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        start = time.time()

        # -------------------------------------------------------------------------------------------------------------------
        # Step 1: Download tiles to a local tmp dir

        lu.print_and_log(f"STEP 1: Downloading tiles", False, logger_worker)
        local_tiles = download_s3_tiles(s3_raster_list, tile_dir, max_workers=download_workers)
        if not local_tiles:
            raise RuntimeError("     No local source tiles were downloaded")
        else:
            lu.print_and_log(f"     Downloaded {len(local_tiles)} tiles in {round(time.time() - start)} seconds", False, logger_worker)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 2: Get GDAL datatype of each dataset using the first tile in that dataset

        lu.print_and_log(f"STEP 2 - Getting GDAL datatype", False, logger_worker)
        gdal_dtype_str, nodata, resample = get_local_raster_info(local_tiles[0])

        # -------------------------------------------------------------------------------------------------------------------
        # Step 3: Create a VRT from local tiles

        lu.print_and_log(f"STEP 3 - Building VRT", False, logger_worker)
        build_local_vrt(local_tiles, str(tmp_vrt_path))

        # -------------------------------------------------------------------------------------------------------------------
        # Step 4: Build COG from local VRT/ tiles

        lu.print_and_log(f"STEP 4 - Translating COG", False, logger_worker)
        gdal_translate_cog(str(tmp_vrt_path), str(tmp_cog_path), nodata, resample, build_overviews=True)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 5: Upload COG to s3
        lu.print_and_log(f"STEP 5 - Upload COG to s3", False, logger_worker)
        uu.upload_s3_file(output_cog_s3_path, str(tmp_cog_path))
        if not uu.exists_in_s3(output_cog_s3_path):
            raise RuntimeError(f"     Upload finished but output could not be verified in S3: {output_cog_s3_path}")
        lu.print_and_log(f"     COG uploaded successfully: {output_cog_s3_path}", False, logger_worker)

    except Exception:
        # Keep staged inputs after a failure so the worker can be inspected/retried
        # without immediately losing the downloaded data.
        lu.print_and_log(f"     COG workflow failed. Leaving staging directory in place for inspection: {tmp_root}", False, logger_worker)
        raise

    else:
        shutil.rmtree(tmp_root)
        lu.print_and_log(f"     Removed local staging directory: {tmp_root}", False, logger_worker)


def main(cluster_name, datasets, years, tile_ids, overwrite_existing_cog, download_workers):

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
    emissions_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_ha_yr/40000_pixels/20260130/"
    removals_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_removals__all_C_pools__MgCO2/annual_intervals/YYYY/_ha_yr/40000_pixels/20260130/"
    net_flux_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/YYYY/_ha_yr/40000_pixels/20260130/"
    mineral_soil_path = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_net__mineral_soil_extent__0-30cm_MgCO2/YYYY/_ha_yr/40000_pixels/20260611/"

    emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e_ha_yr"
    removals_pattern = "gross_removals__all_C_pools__MgCO2_ha_yr"
    net_flux_pattern = "net_flux__all_C_pools__all_gases__MgCO2e_ha_yr"
    mineral_soil_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC_ha_yr"

    # ------------------------------------------------------------------------------------------------------------------

    # Create download/ upload dictionary for the datasets we want to create global COGs for.

    tile_ids = read_tile_ids(tile_ids) if tile_ids else None

    # Default to all available years if years are not provided by user
    if years is None:
        years = {
            "emissions": cn.veg_outputs_years,
            "removals": cn.veg_outputs_years,
            "net_flux": cn.veg_outputs_years,
            "mineral_soil": cn.SOC_change_intervals,
        }

    jobs = {}

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
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            key = f"{dataset}_{year}"
            output_dir = s3_dir.replace("40000_pixels", "global").rstrip("/") + "/"
            output_cog_s3_path = f"{output_dir}{pattern}_{year}.tif"

            if not overwrite_existing_cog and uu.exists_in_s3(output_cog_s3_path):
                main_logger.info(f"{key} - COG already exists in S3: {output_cog_s3_path}. Skipping creation.")
                continue

            bucket, prefix = uu.split_s3_path(s3_dir)
            s3_keys = list_s3_tiles_from_tile_ids(s3_dir, tile_ids)
            s3_raster_list = [f"s3://{bucket}/{s3_key}" for s3_key in s3_keys]

            if not s3_raster_list:
                main_logger.warning(f"{key} - No source rasters found in {s3_dir}. Skipping.")
                continue

            jobs[key] = {
                "s3_raster_list": s3_raster_list,
                "tmp_root": f"/tmp/cog_staging/{key}",
                "tmp_vrt_path": f"/tmp/cog_staging/{key}/vrt/{pattern}_{year}.vrt",
                "tmp_cog_path": f"/tmp/cog_staging/{key}/cog/{pattern}_{year}.tif",
                "output_cog_s3_path": output_cog_s3_path,
            }

    # -------------------------------------------------------------------------------------------------------------------

    # Step 2: Download tiles, build VRT, create COG and upload to s3
    start_time = time.time()

    # With one Coiled worker / one Dask thread, jobs execute one at a time.
    # This is intentional because each job may consume hundreds of GB of local disk.
    if not run_local:
        futures = []
        for key, job in jobs.items():
            main_logger.info(f"Submitting COG build for {key}")
            future = client.submit(
                stage_build_upload_cog,
                job["s3_raster_list"],
                job["tmp_root"],
                job["tmp_vrt_path"],
                job["tmp_cog_path"],
                job["output_cog_s3_path"],
                download_workers,
            )
            futures.append((key, future))

        for key, future in futures:
            future.result()
    else:
        for key, job in jobs.items():
            main_logger.info(f"Staging COG build for {key}")
            stage_build_upload_cog(
                job["s3_raster_list"],
                job["tmp_root"],
                job["tmp_vrt_path"],
                job["tmp_cog_path"],
                job["output_cog_s3_path"],
                download_workers,
            )

    main_logger.info(f"COG build Complete in {round(time.time() - start_time)} seconds")

    if client is not None:
        client.close()

    # -------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create global COGs")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--datasets', required=True, nargs='+', help='What datasets do you want to convert to global COGs? Current options: emissions, removals, net flux, mineral soil soc change')
    parser.add_argument('-y', '--years', nargs='+', help="Which year(s) to run? Defaults to use all available years if not specified.")
    parser.add_argument('-t', "--tile_ids", help="Optional text file with tile ids to filter to (one per line)")
    parser.add_argument("--overwrite_existing_cog", action="store_true", default=False, help="Option to overwrite existing COGs in s3. Default: False")
    parser.add_argument( '-dw', "--download_workers", type=int, default=8, help="Number of concurrent S3 tile downloads. Default: 8")

    args = parser.parse_args()
    cluster_name = args.cluster_name
    datasets = args.datasets
    years = args.years
    tile_ids = args.tile_ids
    overwrite_existing_cog = args.overwrite_existing_cog
    download_workers = args.download_workers

    main(cluster_name, datasets, years, tile_ids, overwrite_existing_cog, download_workers)

"""
Purpose: Recreates IPCC_summary in existing IPCC land use zarr as uint16 and populates it from already-computed 1x1 IPCC_summary GeoTIFFs on S3.
In the initial global run, the GLCLU classification code used uint16 for LU summary but the zarr code code initialized it as uint8 so none of the LU summary data was written to zarr during the global run. This deleted the empty uint8 LU summary dataset in the zarr, created a new empty LU summary uint16 dataset, and wrote all tiles to zarr. 

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Coiled test:
python -m src.utilities.create_cluster -n 10 -m 8 -cn IPCC_summary_zarr
python -m src.LULUCF.scripts.postprocessing.LUC.repopulate_ipcc_summary_zarr -cn IPCC_summary_zarr -id 20260617 -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -bb 110 -10 120 0

Global run:
python -m src.utilities.create_cluster -n 50 -m 8 -cn IPCC_summary_zarr
python -m src.LULUCF.scripts.postprocessing.LUC.repopulate_ipcc_summary_zarr -cn IPCC_summary_zarr -id 20260617 -mpd global -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "Repopulating IPCC_summary zarr from existing 1x1 (uint16) rasters."

"""

import argparse
import dask
from dask.distributed import print
import fsspec
import zarr
import rasterio
import gc

import src.utilities.constants_and_names as cn
import src.utilities.universal_utilities as uu
import src.utilities.log_utilities as lu
import src.utilities.zarr_utilities as zu


def populate_one_ipcc_summary_zarr_chunk(chunk, summary_dir, zarr_path, main_logger):
    logger_worker = lu.setup_logging_worker()

    bounds_str = uu.boundstr(chunk)
    tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])

    raster_path = f"{summary_dir.rstrip('/')}/{tile_id}__{bounds_str}__{cn.IPCC_summary_pattern}_2015_2024.tif"

    try:
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype("uint16")

        fs = fsspec.filesystem("s3", anon=False)
        mapper = fs.get_mapper(zarr_path)
        z = zarr.open_group(mapper, mode="r+")

        lat_start, lon_start = zu.latlon_to_global_zarr_indices(chunk[3], chunk[0], cn.resolution)
        lat_end, lon_end = zu.latlon_to_global_zarr_indices(chunk[1], chunk[2], cn.resolution)

        z[cn.IPCC_summary_pattern][0, lat_start:lat_end, lon_start:lon_end] = arr

        chunk_stats = [{
            "tile_id": tile_id,
            "chunk_id": bounds_str,
            "dataset": cn.IPCC_summary_pattern,
            "year": "2015_2024",
            "min": int(arr.min()),
            "max": int(arr.max()),
            "count": int((arr != 0).sum()),
        }]

        del arr
        gc.collect()

        return {
            "status": "success",
            "chunk": bounds_str,
            "tile_id": tile_id,
            "chunk_stats": chunk_stats,
        }

    except Exception as e:
        return {
            "status": "failed",
            "chunk": bounds_str,
            "tile_id": tile_id,
            "raster_path": raster_path,
            "error": str(e),
            "chunk_stats": [],
        }


def main(cluster_name, input_date, model_type, no_log=False, chunk_shapefile_uri=False,
         bounding_box=None, first_chunks=None, model_path_description=None, log_note=None, recreate_array=False):

    ### Step 1: Preparation

    stage = f"repopulate_{cn.IPCC_summary_pattern}_zarr"

    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(
        client, cluster, log_note, run_local, model_type, stage
    )

    start_time = uu.timestr()
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"IPCC land use model version: {cn.IPCC_LU_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Run date: {input_date}")

    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(
        cn.LC_first_year,
        cn.LC_last_year,
        main_logger
    )

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    chunk_size_deg = 1
    # Always create chunks from the fishnet shapefile so chunk IDs match existing 1x1 outputs.
    chunk_list, chunk_size_pixels = uu.create_chunk_list(
        None,
        chunk_shapefile_uri,
        chunk_size_deg,
        first_chunks,
        fishnet_iso_df,
        main_logger
    )

    # If bounding box is supplied, filter shapefile chunks to chunks within/intersecting that bbox.
    if bounding_box:
        west, south, east, north = bounding_box

        original_count = len(chunk_list)

        chunk_list = [
            chunk for chunk in chunk_list
            if (
                    chunk[0] < east and
                    chunk[2] > west and
                    chunk[1] < north and
                    chunk[3] > south
            )
        ]

        main_logger.info(
            f"Filtered shapefile chunk list from {original_count} to {len(chunk_list)} "
            f"chunks using bounding box {bounding_box}"
        )

    zarr_path = zu.create_zarr_path(
        cn.IPCC_outputs_path_mega_zarr,
        cn.chunk_dims,
        interval_type,
        model_type,
        cn.IPCC_LU_version_underscore,
        model_path_description,
        input_date,
        main_logger
    )

    summary_dir = (
        cn.IPCC_summary_dir
        .replace("RUN_DATE", input_date)
        .replace("CHUNK_SIZE", str(chunk_size_pixels))
    )

    main_logger.info(f"IPCC summary GeoTIFF folder to ingest into zarr: {summary_dir}")
    main_logger.info(f"IPCC zarr path: {zarr_path}")


    ### Step 2: Recreate IPCC_summary array as uint16

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_path)

    z = zarr.open_group(mapper, mode="r+")

    main_logger.info(f"Datasets currently in zarr: {list(z.array_keys())}")

    lat_size = int(180 / cn.resolution)
    lon_size = int(360 / cn.resolution)

    array_exists = cn.IPCC_summary_pattern in z

    if recreate_array:
        if array_exists:
            main_logger.info(
                f"Array '{cn.IPCC_summary_pattern}' already exists. "
                f"Deleting and recreating as uint16: {uu.timestr()}"
            )
            del z[cn.IPCC_summary_pattern]

        new_arr = z.create_array(
            cn.IPCC_summary_pattern,
            shape=(10, lat_size, lon_size),
            chunks=(1, cn.chunk_dims, cn.chunk_dims),
            dtype="uint16",
            fill_value=0,
            compressors={"name": "zstd", "configuration": {"level": 3}},
            dimension_names=["year", "y", "x"],
        )
        new_arr.attrs["grid_mapping"] = "spatial_ref"
        zarr.consolidate_metadata(mapper)

        main_logger.info(
            f"Created '{cn.IPCC_summary_pattern}': "
            f"shape={new_arr.shape}, dtype={new_arr.dtype}: {uu.timestr()}"
        )

    elif not array_exists:
        raise RuntimeError(
            f"{cn.IPCC_summary_pattern} does not exist in zarr. "
            f"Rerun with --recreate_array."
        )

    else:
        main_logger.info(
            f"Using existing '{cn.IPCC_summary_pattern}' array without deleting it: {uu.timestr()}"
        )

    ### Step 3: Submit tasks through Coiled


    raster_paths, raster_count = uu.list_raster_full_paths_in_s3_folder_and_count(summary_dir)
    existing_raster_names = {p.split("/")[-1] for p in raster_paths}

    main_logger.info(f"Found {raster_count} IPCC_summary rasters in {summary_dir}")

    original_count = len(chunk_list)

    chunk_list = [
        chunk for chunk in chunk_list
        if f"{uu.xy_to_tile_id(chunk[0], chunk[3])}__{uu.boundstr(chunk)}__{cn.IPCC_summary_pattern}_2015_2024.tif"
           in existing_raster_names
    ]

    main_logger.info(
        f"Filtered chunk list from {original_count} to {len(chunk_list)} chunks "
        f"based on existing IPCC_summary rasters."
    )


    main_logger.info(f"Submitting {len(chunk_list)} tasks: {uu.timestr()}")

    delayed_results = [
        dask.delayed(populate_one_ipcc_summary_zarr_chunk)(
            chunk,
            summary_dir,
            zarr_path,
            main_logger
        )
        for chunk in chunk_list
    ]

    results = dask.compute(*delayed_results)

    total_success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    total_failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")

    chunk_stats_variable_year_zarr = [
        [stat for r in results for stat in r.get("chunk_stats", [])]
    ]

    main_logger.info(
        f"Ingestion to zarr complete: {total_success} succeeded, "
        f"{total_failed} failed: {uu.timestr()}"
    )



    if total_failed:
        failures = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]
        main_logger.warning(f"First 10 failures: {failures[:10]}")

    if chunk_stats_variable_year_zarr:
        main_logger.info(f"First 10 zarr chunk stats: {chunk_stats_variable_year_zarr[0][:10]}")

    ### Step 4: Gather worker logs and merge with main log

    if not run_local:
        worker_log_local_path = lu.compile_worker_logs(no_log, cluster, stage, start_time, main_logger)
        lu.merge_main_and_worker_upload_logs(no_log, main_log_local_path, worker_log_local_path, stage)

    if not run_local and client is not None:
        client.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Repopulate IPCC_summary in existing IPCC zarr.")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")
    parser.add_argument("-id", "--input_date", required=True, help="Date of run, in YYYYMMDD")
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float, help="W, S, E, N (degrees)")
    parser.add_argument("-cshp", "--chunk_shapefile_uri", help="s3 location for shapefile of 1x1 deg chunk footprints")
    parser.add_argument("-f", "--first_chunks", type=int, help="Number of chunks to process from shapefile")
    parser.add_argument("-mt", "--model_type", default="standard_model", help="Type of model run.")
    parser.add_argument("-mpd", "--model_path_description", default="global", help="Description of model run.")
    parser.add_argument("-ln", "--log_note", help="Note to include in the log.")
    parser.add_argument("--no_log", action="store_true", help="Do not create the combined log")
    parser.add_argument("--recreate_array", action="store_true", help="Delete and recreate IPCC_summary as uint16 before populating.")

    args = parser.parse_args()

    main(
        args.cluster_name,
        args.input_date,
        args.model_type,
        args.no_log,
        args.chunk_shapefile_uri,
        bounding_box=args.bounding_box,
        first_chunks=args.first_chunks,
        model_path_description=args.model_path_description,
        log_note=args.log_note,
        recreate_array=args.recreate_array,
    )

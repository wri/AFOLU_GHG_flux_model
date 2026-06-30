"""
Backfills the global starting carbon pool mega-zarr from previously generated
1x1 degree GeoTIFF outputs.

This script is intended for recovery if the global zarr was initialized
correctly but was not populated during the original starting carbon pool run.
It reads the existing 1x1 degree carbon density GeoTIFFs (AGC, BGC, deadwood,
litter, non-soil, land-cover-masked versions, and source flag), computes their
location in the global grid, and writes them directly into the corresponding
datasets in the mega-zarr.

The resulting zarr can then be used by
2_starting_C_outputs_to_10x10deg.py to regenerate 10x10 degree outputs without
rerunning the computationally expensive starting carbon pool calculations.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Test run:
python -m src.utilities.create_cluster -n 1 -t 1 -m 8 -cn fill_starting_carbon_pools_zarr__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.fill_ctrees_starting_C_zarr_from_1x1 -cn fill_starting_carbon_pools_zarr__Ctrees -bb -80 30 -70 40

Global run:
python -m src.utilities.create_cluster -n 100 -t 1 -m 8 -cn fill_starting_carbon_pools_zarr__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.fill_ctrees_starting_C_zarr_from_1x1 -cn fill_starting_carbon_pools_zarr__Ctrees -mpd Ctrees

Global rerun for failed 520 chunks:
python -m src.utilities.create_cluster -n 100 -t 1 -m 8 -cn fill_starting_carbon_pools_zarr__Ctrees
python -m src.LULUCF.scripts.preprocessing.starting_carbon_pools.fill_ctrees_starting_C_zarr_from_1x1 -cn fill_starting_carbon_pools_zarr__Ctrees  -mpd Ctrees -fct /mnt/c/GIS/AFOLU_flux_model/ctrees_failed_1x1_chunks.txt

"""


import argparse
from dask.distributed import as_completed
import os

import math
import random
import time
import fsspec
import numpy as np
import rasterio
import zarr

from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu
from src.utilities import zarr_utilities as zu
from src.utilities import log_utilities as lu


YEAR = 2015


def read_1x1_tif(s3_uri, max_retries=6, base_sleep=2):
    """Read one GeoTIFF band with retry/backoff for transient S3/rasterio throttling."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="TRUE"):
                with rasterio.open(s3_uri) as src:
                    return src.read(1)

        except (rasterio.errors.RasterioIOError, OSError) as e:
            last_error = e

            if attempt == max_retries:
                break

            sleep_seconds = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 1)
            time.sleep(sleep_seconds)

    raise last_error


def read_chunk_ids_from_txt(chunk_txt_path):
    """Read chunk IDs from a local or S3 text file; ignore blanks and comment lines."""
    chunk_ids = []

    with fsspec.open(chunk_txt_path, "rt") as src:
        for line in src:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Accept either a one-column txt or a CSV-like first column.
            chunk_ids.append(line.split(",")[0].strip())

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(chunk_ids))


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]

def write_tile_to_zarr(bounds, zarr_path, output_dirs, output_patterns, year, no_upload=False):
    bounds_str = uu.boundstr(bounds)
    tile_id = uu.xy_to_tile_id(bounds[0], bounds[3])
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    fs = fsspec.filesystem("s3", anon=False)
    z = zarr.open_group(fs.get_mapper(zarr_path), mode="r+")

    lat_start, lon_start = zu.latlon_to_global_zarr_indices(bounds[3], bounds[0], cn.resolution)
    lat_end, lon_end = zu.latlon_to_global_zarr_indices(bounds[1], bounds[2], cn.resolution)

    summary = {
        "bounds_str": bounds_str,
        "tile_id": tile_id,
        "written": [],
        "skipped": [],
        "failed": [],
    }

    if (lat_end - lat_start) != chunk_length_pixels or (lon_end - lon_start) != chunk_length_pixels:
        summary["failed"].append({
            "reason": "bad_zarr_slice_shape",
            "detail": f"{lat_end - lat_start}x{lon_end - lon_start}; expected {chunk_length_pixels}x{chunk_length_pixels}",
        })
        return summary

    for out_dir, core_pattern in zip(output_dirs, output_patterns):
        source_dir = out_dir.replace("CHUNK_SIZE", str(chunk_length_pixels))
        source_dir = source_dir.replace("PER_HA_OR_PIXEL", cn.C_density_pixel_meaning)

        if core_pattern == cn.starting_C_pools_LC_masked_source_flag_pattern:
            tif_name = f"{tile_id}__{bounds_str}__{core_pattern}_{year}.tif"
            zarr_key = f"{core_pattern}_{year}"
        else:
            tif_name = f"{tile_id}__{bounds_str}__{core_pattern}{cn.C_density_pixel_meaning}_{year}.tif"
            zarr_key = f"{core_pattern}{cn.C_density_pixel_meaning}_{year}"

        tif_uri = f"{source_dir}{tif_name}"

        if zarr_key not in z:
            summary["failed"].append({
                "zarr_key": zarr_key,
                "tif_uri": tif_uri,
                "reason": "missing_zarr_key",
            })
            continue

        if not fs.exists(tif_uri):
            summary["skipped"].append({
                "zarr_key": zarr_key,
                "tif_uri": tif_uri,
                "reason": "missing_source_tif",
            })
            continue

        try:
            data = read_1x1_tif(tif_uri)

            if data.shape != (chunk_length_pixels, chunk_length_pixels):
                summary["failed"].append({
                    "zarr_key": zarr_key,
                    "tif_uri": tif_uri,
                    "reason": "bad_tif_shape",
                    "detail": str(data.shape),
                })
                continue

            if not no_upload:
                z[zarr_key][0, lat_start:lat_end, lon_start:lon_end] = data

            summary["written"].append({
                "zarr_key": zarr_key,
                "tif_uri": tif_uri,
                "max_val": float(np.nanmax(data)) if data.size else np.nan,
            })

        except Exception as e:
            summary["failed"].append({
                "zarr_key": zarr_key,
                "tif_uri": tif_uri,
                "reason": "exception",
                "detail": repr(e),
            })

    return summary


def main(cluster_name, model_type, model_path_description, bounding_box=None,
         chunk_shapefile_uri=None, first_chunks=None, failed_chunks_txt=None, no_upload=False):

    stage = "backfill_ctrees_starting_C_zarr_from_1x1"
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    input_date = cn.ctrees_run_date

    main_logger, _, n_workers = lu.populate_main_log_header(client, cluster, None, run_local, model_type, stage)

    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    if failed_chunks_txt:
        failed_chunk_ids = read_chunk_ids_from_txt(failed_chunks_txt)
        valid_chunk_ids = set(fishnet_iso_df["chunk_id"])

        missing_from_fishnet = [
            chunk_id for chunk_id in failed_chunk_ids
            if chunk_id not in valid_chunk_ids
        ]

        if missing_from_fishnet:
            main_logger.warning(
                f"{len(missing_from_fishnet)} chunk IDs from {failed_chunks_txt} "
                f"were not found in the 1x1 fishnet and will be skipped. "
                f"First few missing: {missing_from_fishnet[:10]}"
            )

        failed_chunk_ids = [
            chunk_id for chunk_id in failed_chunk_ids
            if chunk_id in valid_chunk_ids
        ]

        if first_chunks:
            failed_chunk_ids = failed_chunk_ids[:first_chunks]

        chunk_list = [uu.process_chunk_id(chunk_id) for chunk_id in failed_chunk_ids]
        chunk_size_pixels = cn.chunk_dims

        main_logger.info(
            f"Using failed 1x1 chunk text file {failed_chunks_txt}; "
            f"{len(chunk_list)} chunks selected"
        )

    elif bounding_box:
        xmin, ymin, xmax, ymax = bounding_box

        def chunk_intersects_bb(chunk_id):
            west, south, east, north = map(float, chunk_id.split("_"))
            return not (
                    east <= xmin or
                    west >= xmax or
                    north <= ymin or
                    south >= ymax
            )

        fishnet_iso_df = fishnet_iso_df[
            fishnet_iso_df["chunk_id"].apply(chunk_intersects_bb)
        ].copy()

        if first_chunks:
            fishnet_iso_df = fishnet_iso_df.head(first_chunks)

        chunk_list = fishnet_iso_df["chunk_id"].apply(uu.process_chunk_id).tolist()
        chunk_size_pixels = cn.chunk_dims

        main_logger.info(
            f"Using 1x1 fishnet chunks filtered to bounding box {bounding_box}; "
            f"{len(chunk_list)} chunks selected"
        )

    else:
        chunk_list, chunk_size_pixels = uu.create_chunk_list(
            None,
            chunk_shapefile_uri,
            1,
            first_chunks,
            fishnet_iso_df,
            main_logger,
        )

    if run_local:
        batch_size = 1
    else:
        batch_size = max(1, int(n_workers) * 2)
        #batch_size = max(1, int(n_workers) * 10)

    total_batches = math.ceil(len(chunk_list) / batch_size)

    main_logger.info(f"Batch size: {batch_size} chunks")
    main_logger.info(f"Total batches: {total_batches}")

    zarr_path = zu.create_zarr_path(
        cn.starting_C_densities_2015_ctrees_path_mega_zarr,
        chunk_size_pixels,
        str(YEAR),
        model_type,
        cn.veg_model_version_underscore,
        model_path_description,
        input_date,
        main_logger,
    )

    output_patterns = [
        cn.agc_raw_dens_pattern,
        cn.bgc_raw_dens_pattern,
        cn.deadwood_c_raw_dens_pattern,
        cn.litter_c_raw_dens_pattern,
        cn.non_soil_c_raw_dens_pattern,
        cn.agc_LC_masked_dens_pattern,
        cn.bgc_LC_masked_dens_pattern,
        cn.deadwood_c_LC_masked_dens_pattern,
        cn.litter_c_LC_masked_dens_pattern,
        cn.non_soil_c_LC_masked_dens_pattern,
        cn.starting_C_pools_LC_masked_source_flag_pattern,
    ]

    output_dirs = [
        cn.agc_2015_ctrees_raw_dir,
        cn.bgc_2015_ctrees_raw_dir,
        cn.deadwood_c_2015_ctrees_raw_dir,
        cn.litter_c_2015_ctrees_raw_dir,
        cn.non_soil_c_2015_ctrees_raw_dir,
        cn.agc_2015_ctrees_LC_masked_dir,
        cn.bgc_2015_ctrees_LC_masked_dir,
        cn.deadwood_c_2015_ctrees_LC_masked_dir,
        cn.litter_c_2015_ctrees_LC_masked_dir,
        cn.non_soil_c_2015_ctrees_LC_masked_dir,
        cn.starting_C_pools_ctrees_LC_masked_state_dir,
    ]

    main_logger.info(f"Backfilling {len(chunk_list)} chunks into {zarr_path}")
    main_logger.info(f"no_upload: {no_upload}")

    main_logger.info(
        f"Submitting {len(chunk_list)} chunks for Ctrees zarr backfill "
        f"in batches of {batch_size}."
    )

    total_written = 0
    total_skipped = 0
    total_failed = 0
    success_chunk_count = 0
    partial_chunk_count = 0
    failed_chunk_count = 0
    completed_chunk_count = 0

    for batch_start_idx, batch in batched(chunk_list, batch_size):
        batch_num = (batch_start_idx // batch_size) + 1
        batch_end_idx = batch_start_idx + len(batch)

        main_logger.info(
            f"Starting batch {batch_num}/{total_batches}: "
            f"chunks {batch_start_idx + 1}-{batch_end_idx} of {len(chunk_list)}"
        )

        future_to_bounds = {
            client.submit(
                write_tile_to_zarr,
                bounds,
                zarr_path,
                output_dirs,
                output_patterns,
                YEAR,
                no_upload,
                retries=0,
            ): bounds
            for bounds in batch
        }

        batch_written = 0
        batch_skipped = 0
        batch_failed = 0

        for future in as_completed(list(future_to_bounds.keys())):
            try:
                result = future.result()
            except Exception as e:
                failed_chunk_count += 1
                total_failed += len(output_patterns)
                batch_failed += len(output_patterns)

                bounds = future_to_bounds.get(future)
                bounds_str = uu.boundstr(bounds) if bounds is not None else "unknown_chunk"
                tile_id = uu.xy_to_tile_id(bounds[0], bounds[3]) if bounds is not None else "unknown_tile"

                main_logger.warning(
                    f"FAILED chunk task before returning summary for {bounds_str} in {tile_id}: {repr(e)}"
                )
                continue

            completed_chunk_count += 1

            written_count = len(result["written"])
            skipped_count = len(result["skipped"])
            failed_count = len(result["failed"])

            total_written += written_count
            total_skipped += skipped_count
            total_failed += failed_count

            batch_written += written_count
            batch_skipped += skipped_count
            batch_failed += failed_count

            if failed_count == 0 and written_count > 0:
                success_chunk_count += 1
            elif written_count > 0 or skipped_count > 0:
                partial_chunk_count += 1
            else:
                failed_chunk_count += 1

            main_logger.info(
                f"{result['bounds_str']} in {result['tile_id']}: "
                f"written={written_count}, skipped={skipped_count}, failed={failed_count}"
            )

            for item in result["written"]:
                filename = os.path.basename(item["tif_uri"])
                # main_logger.info(
                #     f"  Wrote {item['zarr_key']} from {filename}"
                # )

            for item in result["skipped"]:
                filename = os.path.basename(item["tif_uri"])
                # main_logger.info(
                #     f"  Skipped {item['zarr_key']} from {filename}; "
                #     f"reason={item['reason']}"
                # )

            for item in result["failed"]:
                tif_uri = item.get("tif_uri", "N/A")
                filename = os.path.basename(tif_uri) if tif_uri != "N/A" else "N/A"
                # main_logger.warning(
                #     f"  Failed {item.get('zarr_key', 'N/A')} from {filename}; "
                #     f"reason={item['reason']}; detail={item.get('detail', '')}"
                # )

        main_logger.info(
            f"Finished batch {batch_num}/{total_batches}: "
            f"files_written={batch_written}, files_skipped={batch_skipped}, files_failed={batch_failed}; "
            f"completed_chunks_so_far={completed_chunk_count}/{len(chunk_list)}"
        )

    main_logger.info(
        f"Backfill complete. chunks_attempted={len(chunk_list)}, "
        f"success_chunks={success_chunk_count}, partial_chunks={partial_chunk_count}, "
        f"failed_chunks={failed_chunk_count}, files_written={total_written}, "
        f"files_skipped={total_skipped}, files_failed={total_failed}"
    )

    if client:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cn", "--cluster_name", required=True)
    parser.add_argument("-bb", "--bounding_box", nargs=4, type=float)
    parser.add_argument("-cshp", "--chunk_shapefile_uri")
    parser.add_argument("-f", "--first_chunks", type=int)
    parser.add_argument("-fct", "--failed_chunks_txt", help="Local or S3 text file of 1x1 chunk IDs to rerun, one chunk_id per line")
    parser.add_argument("-mt", "--model_type", default="standard")
    parser.add_argument("-mpd", "--model_path_description", default="global")
    parser.add_argument("--no_upload", action="store_true")
    args = parser.parse_args()

    main(args.cluster_name, args.model_type, args.model_path_description, bounding_box=args.bounding_box,
         chunk_shapefile_uri=args.chunk_shapefile_uri, first_chunks=args.first_chunks, failed_chunks_txt=args.failed_chunks_txt, no_upload=args.no_upload)
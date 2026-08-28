"""
Purpose: Download IPCC land use outputs for a 10 x 10 tile. Useful for QCing land use assignment. 
Run:
python -m src.LULUCF.scripts.postprocessing.LUC.download_land_use_outputs -t 00N_110E --run_date 20268888
"""

import argparse
import os
import sys
from pathlib import Path
from botocore.exceptions import ClientError
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu

root = Path("/mnt/c/GIS/git/AFOLU_GHG_flux_model")
sys.path.insert(0, str(root))

local_file_dir = Path("/mnt/c/GIS/AFOLU_flux_model/land_use")


def download_if_needed(s3_uri, local_path, overwrite=False):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and not overwrite:
        print(f"Exists, skipping: {local_path}")
        return

    print(f"Downloading: {s3_uri}")
    print(f"        to: {local_path}")

    try:
        uu.download_s3_file(s3_uri, str(local_path))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            print(f"WARNING: Missing output in S3, skipping: {s3_uri}")
            return
        raise


def build_land_use_output_paths(tile_id, run_date, chunk_size_pixels=40000):
    paths = {}

    for year in cn.LC_years:
        paths.setdefault("IPCC_class", {})[
            f"{cn.IPCC_class_pattern}_{year}"
        ] = (
            cn.IPCC_class_dir
            .replace("YEAR", str(year))
            .replace("CHUNK_SIZE", str(chunk_size_pixels))
            .replace("RUN_DATE", run_date)
            + f"{tile_id}_{cn.IPCC_class_pattern}_{year}.tif"
        )

        paths.setdefault("IPCC_node", {})[
            f"{cn.IPCC_node_pattern}_{year}"
        ] = (
            cn.IPCC_node_dir
            .replace("YEAR", str(year))
            .replace("CHUNK_SIZE", str(chunk_size_pixels))
            .replace("RUN_DATE", run_date)
            + f"{tile_id}_{cn.IPCC_node_pattern}_{year}.tif"
        )

    for start_year, end_year in zip(cn.LC_years[:-1], cn.LC_years[1:]):
        year_range = f"{start_year}_{end_year}"
        paths.setdefault("IPCC_change", {})[
            f"{cn.IPCC_change_pattern}_{year_range}"
        ] = (
            cn.IPCC_change_dir
            .replace("START_END", year_range)
            .replace("CHUNK_SIZE", str(chunk_size_pixels))
            .replace("RUN_DATE", run_date)
            + f"{tile_id}_{cn.IPCC_change_pattern}_{year_range}.tif"
        )

    summary_year_range = f"{cn.LC_years[0]}_{cn.LC_years[-1]}"
    paths["IPCC_summary"] = {
        cn.IPCC_summary_pattern: (
            cn.IPCC_summary_dir
            .replace("START_END", summary_year_range)
            .replace("CHUNK_SIZE", str(chunk_size_pixels))
            .replace("RUN_DATE", run_date)
            + f"{tile_id}_{cn.IPCC_summary_pattern}_2015_2024.tif"
        )
    }

    return paths


def main(tile_id, run_date, overwrite=False, chunk_size_pixels=40000):
    print(f"Downloading land-use outputs for {tile_id}")
    print(f"Run date: {run_date}")
    print(f"Chunk size pixels: {chunk_size_pixels}")

    output_paths = build_land_use_output_paths( tile_id=tile_id, run_date=run_date, chunk_size_pixels=chunk_size_pixels)

    for group_name, group_paths in output_paths.items():
        for layer_name, s3_uri in group_paths.items():
            filename = os.path.basename(s3_uri)
            local_path = local_file_dir / group_name / filename
            download_if_needed(s3_uri, local_path, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all IPCC land-use output rasters for a 10x10 degree tile.")
    parser.add_argument("-t", "--tile_id", required=True, help="10x10 tile ID, e.g. 00N_110E")
    parser.add_argument("-rd", "--run_date", required=True, help="Run date used in output S3 folders")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument( "--chunk_size_pixels", type=int, default=40000, help="Output chunk size in pixels. Use 40000 for 10x10 outputs or 4000 for 1x1 outputs.")

    args = parser.parse_args()

    main( tile_id=args.tile_id, run_date=args.run_date, overwrite=args.overwrite, chunk_size_pixels=args.chunk_size_pixels)

"""
Run:
python -m src.LULUCF.scripts.postprocessing.LUC.download_land_use_inputs -t 00N_110E
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

    print(f"Downloading {s3_uri}")
    print(f"        to: {local_path}")

    try:
        uu.download_s3_file(s3_uri, str(local_path))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            print(f"WARNING: Missing input in S3, skipping: {s3_uri}")
            return
        raise


def build_land_use_input_paths(tile_id):
    paths = {}

    paths["TCL"] = {cn.tree_cover_loss_pattern: f"{cn.tree_cover_loss_dir}{cn.tree_cover_loss_pattern}_{tile_id}.tif"}
    paths["driver"] = {cn.drivers_pattern: f"{cn.drivers_path}{tile_id}_{cn.drivers_pattern}.tif"}
    paths["oil_palm"] = {
        cn.oil_palm_2000_extent_pattern: f"{cn.oil_palm_2000_extent_dir}{tile_id}_{cn.oil_palm_2000_extent_pattern}.tif",
        cn.oil_palm_first_year_pattern: f"{cn.oil_palm_first_year_dir}{cn.oil_palm_first_year_pattern}_{tile_id}.tif",
    }
    paths["SDPT"] = {
        cn.planted_forest_tree_crop_pattern: f"{cn.planted_forest_tree_crop_dir}{tile_id}.tif",
        cn.planted_forest_type_pattern: f"{cn.planted_forest_type_dir}{tile_id}_{cn.planted_forest_type_pattern}.tif",
    }
    paths["GLCLU"] = {
        f"{cn.land_cover_pattern}_{year}": f"{cn.land_cover_annual_path}{year}/{tile_id}.tif"
        for year in cn.years_annual
    }
    paths["GPW"] = {
        f"{cn.GPW_extent_processed_pattern}_{year}": f"{cn.GPW_extent_processed_dir}{year}/{tile_id}_{cn.GPW_extent_processed_pattern}_{year}.tif"
        for year in cn.years_annual
    }
    paths["GMW"] = {
        f"{cn.mangrove_extent_processed_pattern}_{year}": f"{cn.mangrove_extent_processed_dir}{year}/{tile_id}__{cn.mangrove_extent_processed_pattern}_{year}.tif"
        for year in [2015, 2016, 2017, 2018, 2019, 2020]
    }

    return paths


def main(tile_id, overwrite=False):
    print(f"Downloading land-use inputs for {tile_id}")
    input_paths = build_land_use_input_paths(tile_id)

    for group_name, group_paths in input_paths.items():
        for layer_name, s3_uri in group_paths.items():
            filename = os.path.basename(s3_uri)
            local_path = local_file_dir / group_name / filename
            download_if_needed(s3_uri, local_path, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tile_id", required=True, help="10x10 tile ID, e.g. 00N_110E")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.tile_id, overwrite=args.overwrite)
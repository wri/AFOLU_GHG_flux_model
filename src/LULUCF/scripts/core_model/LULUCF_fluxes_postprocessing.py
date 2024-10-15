"""
Run from src/LULUCF

Test:
python -m scripts.utilities.create_cluster -n 1 -m 16 -c 2
python -m scripts.core_model.LULUCF_fluxes_postprocessing -cn AFOLU_flux_model_scripts -d 20240930

"""


import argparse
import concurrent.futures
import coiled
import dask
import numpy as np

from dask.distributed import Client
from dask.distributed import print
from numba import jit

import re
import subprocess
import boto3
import os
from osgeo import gdal

# Project imports
from ..utilities import constants_and_names as cn
from ..utilities import universal_utilities as uu
from ..utilities import log_utilities as lu
from ..utilities import numba_utilities as nu
from ..utilities import resize_cluster

def main(cluster_name, date, run_local=False, no_upload=False):

    # Connects to Coiled cluster if not running locally
    cluster, client = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__tile_index'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Folders to process
    s3_in_folders = [
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_density_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_density_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_density_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_density_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.AGC_flux_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_flux_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_flux_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.AGC_flux_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.BGC_flux_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_flux_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_flux_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.BGC_flux_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.deadwood_c_flux_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_flux_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_flux_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.deadwood_c_flux_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.litter_c_flux_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_flux_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_flux_path_part}/2010_2015/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.litter_c_flux_path_part}/2015_2020/4000_pixels/{date}/",
        #
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2000_2005/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2005_2010/4000_pixels/{date}/",
        # f"{cn.outputs_path}{cn.land_state_node_path_part}/2010_2015/4000_pixels/{date}/",
        f"{cn.outputs_path}{cn.land_state_node_path_part}/2015_2020/4000_pixels/{date}/"
    ]

    # # Creates dictionary of s3 tile set paths with corresponding tile index shapefile names
    # s3_in_folders_list_of_dicts = []
    #
    # for path in s3_in_folders:
    #     # Extracts the portion after 'cn.outputs_path'
    #     path_suffix = path.replace(cn.outputs_path, "")
    #
    #     # Replaces '/' with '__'
    #     value = path_suffix.rstrip('/').replace("/", "__")
    #
    #     s3_in_folders_list_of_dicts.append({path: value})
    #
    # # Make raster footprint shapefiles from output rasters
    # # Takes over 1 hour on global LULUCF output 1x1 tile set
    # delayed_result = [dask.delayed(uu.make_tile_footprint_shp)(input_dict) for input_dict in s3_in_folders_list_of_dicts]
    #
    # # Actually runs analysis
    # results = dask.compute(*delayed_result)
    # print(results)
    #
    # # Ending time for stage
    # end_time = uu.timestr()
    # print(f"Stage {stage} ended at: {end_time}")
    # uu.stage_duration(start_time, end_time, stage)


    # resize_cluster.resize_coiled_cluster("AFOLU_flux_model_scripts", 50)

    # Model stage being running
    stage = 'LULUCF_flux_postprocessing__merge_tiles'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Creates the list of aggregated 10x10 rasters that will be created (list of dictionaries of input s3 folder and output aggregated raster name.
    # These are the basis for the tasks.
    list_of_s3_name_dicts_total = uu.create_list_for_aggregation(s3_in_folders)

    # For testing. Limits the number of output rasters
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:3]  # First 3 tiles
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[40:41] # 10N_130E; Internal chunks missing and padding needed on right; FID40
    list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[0:2]  # 00N_000E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[16:17] # 00N_110E
    # list_of_s3_name_dicts_total = list_of_s3_name_dicts_total[41:42]  # 10S_010E; No padding needed; FID41
    # print(list_of_s3_name_dicts_total)

    delayed_result = [dask.delayed(uu.merge_small_tiles_gdal)(s3_name_dict, no_upload) for s3_name_dict in list_of_s3_name_dicts_total]

    results = dask.compute(*delayed_result)
    print(results)

    # Ending time for stage
    end_time = uu.timestr()
    print(f"Stage {stage} ended at: {end_time}")
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        # Closes the Dask client if not running locally
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LULUCF fluxes.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-d', '--date', help='Date in YYYYMMDD to process')

    parser.add_argument('--run_local', action='store_true', help='Run locally without Dask/Coiled')
    parser.add_argument('--no_stats', action='store_true', help='Do not create the chunk stats spreadsheet')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name, args.date, args.run_local, args.no_upload)
"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_cluster -n 20 -m 32 -c 4 -t 1 -i 15 -cn global_4km_raster
python -m scripts.postprocessing.create_global_4km_maps -cn global_4km_raster

It took 1 hour, 5 minutes (192 coiled credits) to create 10x10 per-pixel tiles (0.04 resolution) and
global maps (0.00025 resolution) for cropland emissions and net flux (2000-2005, 2005-2010, 2010-2015, 2015-2020)
using 40 workers with 32 GiB mem, 4 cpus, and 1 thread per worker.
NOTES: Low CPU utilization, and high memory pressure.
- Also, many workers idle while final tiles were processing so should lower number of workers next time
#TODO See note below about combining all 10x10s to be processed into a single list to reduce the number of times
# there are laggard tasks

"""
import numpy as np
import argparse
import dask
from dask.distributed import print
from src.utilities import constants_and_names as cn, log_utilities as lu, universal_utilities as uu


########################################################################################################################

def agg_4x4(tile_id, bounds, chunk_length_pixels, pixel_area_tile, mg_ha_yr_tile, per_pixel_output_tile, per_pixel_output_path):

    is_final = False
    logger = lu.setup_logging()



    # Gets numpy arrays of the model output being analyzed and the area (m^2) per pixel
    print(f"Getting rasters for {tile_id}: \n {pixel_area_tile} \n {mg_ha_yr_tile}")
    pixel_area_tile_chunk = uu.get_tile_dataset_rio(pixel_area_tile, bounds, chunk_length_pixels, 'Float32')
    pixel_area_tile_chunk = pixel_area_tile_chunk[0]  # Converts downloaded tuple (array, status) to just the array
    mg_ha_yr_tile_chunk = uu.get_tile_dataset_rio(mg_ha_yr_tile, bounds, chunk_length_pixels, 'Float32')
    mg_ha_yr_tile_chunk = mg_ha_yr_tile_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Converts per hectare values to per pixel values in the numpy array
    mg_yr_per_pixel_tile_chunk = mg_ha_yr_tile_chunk * pixel_area_tile_chunk * cn.m2_to_ha

    # Upload per-pixel raster to s3
    data_type = mg_yr_per_pixel_tile_chunk.dtype.name
    uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id, mg_yr_per_pixel_tile_chunk, data_type,
                                     per_pixel_output_tile, per_pixel_output_path, is_final, logger)
    #TODO Discuss where to create per-pixel tile and upload to s3 in a different script

    # Reaggregate into 0.04x0.04 degree resolution
    mg_yr_per_pixel_agg_4x4_tile_chunk = uu.reaggregate_resolution(mg_yr_per_pixel_tile_chunk, 0.00025, 0.04)

    return mg_yr_per_pixel_agg_4x4_tile_chunk


def combine_global_raster(tiles, bounds_list, tile_id, global_4km_outfile, global_4km_output_path):
    #Courtest of chatGPT
    #TODO Include the ChatGPT conversation link
    """
    Combines multiple 0.04x0.04 degree tiles into a single global raster.

    Parameters:
        tiles (list): List of numpy arrays for all tiles.
        bounds_list (list): List of bounds corresponding to each tile (W, S, E, N).
        tile_id (str): Identifier for the global raster.
        global_4km_outfile (str): Name of the output global raster file.
        global_4km_output_path (str): S3 output path for the global raster.
    """

    is_final = False
    logger = lu.setup_logging()

    # Define global raster size (360x180 degrees at 0.04 resolution)
    global_shape = (int(180 / 0.04), int(360 / 0.04))  # Rows, Columns
    global_raster = np.zeros(global_shape, dtype=np.float32)

    for tile, bounds in zip(tiles, bounds_list):
        min_x, min_y, max_x, max_y = bounds

        # Calculate pixel indices for placement in the global raster
        x_start = int((min_x + 180) / 0.04)
        x_end = int((max_x + 180) / 0.04)
        y_start = int((90 - max_y) / 0.04)
        y_end = int((90 - min_y) / 0.04)

        # Validate bounds alignment
        tile_height, tile_width = tile.shape
        assert (y_end - y_start) == tile_height, "Tile height does not match the calculated bounds"
        assert (x_end - x_start) == tile_width, "Tile width does not match the calculated bounds"

        # Insert the tile into the global raster
        global_raster[y_start:y_end, x_start:x_end] += tile

    # Save the global raster
    #TODO Maybe this should have NoData set to 0 since 0s aren't likely to be actual pixel values
    #TODO Discuss whether the aggregated maps should be converted to Mt (instead of Mg) because the pixel values are large in Mg
    global_bounds = (-180, -90, 180, 90)
    uu.save_and_upload_single_raster(global_bounds, global_raster.shape[1], tile_id, global_raster,
                                     np.float32, global_4km_outfile, global_4km_output_path, is_final,
                                     logger)

    return "Success"
    #TODO update with checking to see if the file exists in s3


def main(cluster_name):
    # -------------------------------------------------------------------------------------------------------------------
    # Step 1: Connects to Coiled cluster if not running locally and the named cluster exists
    run_local = False
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # -------------------------------------------------------------------------------------------------------------------
    # Step 2: Create download/ upload dictionary from list of processes to run
    #TODO: Pass in which fluxes and which years you want to process as command line arguments and add to download_upload_dictionary accordingly.
    # For now hardcoding the download_upload_dictionary. Update flux path/patterns from cn.
    #TODO Discuss where per-pixel outputs should go in s3: their own outer folders or as a variant inside outer folders
    download_upload_dictionary = {
        "cropland_emissions_mg_ha_yr" : {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/mean_rate/including_peatland/2019/physical_area/",
            'mg_ha_yr_pattern': "_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2019_Mg_ha_CO2.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/per_pixel/including_peatland/2019/physical_area/",
            'mg_per_pixel_pattern': "_all_GHGs_cropland_per_pixel_physical_area_CO2eq_all_crops_2019_Mg_CO2.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/0_04deg_output_aggregation/cropland_emissions_all_crops_all_gases__MgCO2e_yr/2019/",
            '4km_pattern': '0_04deg_global__all_GHGs_cropland_physical_area_CO2eq_all_crops_2019__MgCO2e_yr.tif'
        },
        "net_flux_2000_2005_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2000_2005/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2000_2005.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_px_yr/2000_2005/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_px_yr_2000_2005.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2000_2005/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2000_2005.tif'
        },
        "net_flux_2005_2010_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2005_2010/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2005_2010.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_px_yr/2005_2010/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_px_yr_2005_2010.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2005_2010/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2005_2010.tif'
        },
        "net_flux_2010_2015_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2010_2015/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2010_2015.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_px_yr/2010_2015/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_px_yr_2010_2015.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2010_2015/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2010_2015.tif'
        },
        "net_flux_2015_2020_mg_ha_yr": {
            'mg_ha_yr_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_ha_yr/2015_2020/40000_pixels/20241203/",
            'mg_ha_yr_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_ha_yr_2015_2020.tif",
            'mg_per_pixel_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/net_flux_all_C_pools_all_gases__MgCO2e_px_yr/2015_2020/40000_pixels/20241203/",
            'mg_per_pixel_pattern': "__net_flux_all_C_pools_all_gases__MgCO2e_px_yr_2015_2020.tif",
            '4km_dir': "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/0_04deg_output_aggregation/net_flux_all_C_pools_all_gases__MgCO2e_yr/2015_2020/",
            '4km_pattern': '0_04deg_global__net_flux_all_C_pools_all_gases__MgCO2e_yr_2015_2020.tif'
        }
    }
    #TODO Generally discuss post-processing pipelines and which steps should go where, particularly the per-pixel raster creation
    #TODO REMOVE 4KM PATTERN
    #TODO: After net flux and cropland per pixel tiles are created, update paths in constants and names, remove per-pixel creation step here,
    #  after per-pixel creation step is its own script, remove mg_ha_yr_dir/pattern from download_upload dictionary
    #TODO For efficiency, rather than having this process one set of outputs, wait for laggards, and then move to the next,
    # consider instead having it reprocess all 10x10s into the aggregated pixels, then making the global maps
    # at the very end, outside the loop. Theoretically, this would be more efficient because there won't be
    # multiple rounds of laggard tasks.
    #-------------------------------------------------------------------------------------------------------------------
    # Step 3: Create per-pixel rasters and aggregate into 0.04x0.04 degrees for each tile


    # Creating per-pixel rasters
    for key, items in download_upload_dictionary.items():
        bounds_list = []
        delayed_results = []
        for tile_id in cn.tile_id_list:
            # Model stage being run
            stage = f'create 0.04x0.04 deg tile rasters for {key}'
            start_time = uu.timestr()
            print(f"Stage {stage} started at: {start_time}")

            mg_ha_yr_tile = f"{items['mg_ha_yr_dir']}{tile_id}{items['mg_ha_yr_pattern']}"
            pixel_area_tile = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
            per_pixel_tile_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
            per_pixel_output_path = items["mg_per_pixel_dir"]

            # Get bounds and chunk_length_pixels to read in input data
            bounds = uu.get_10x10_tile_bounds(tile_id)
            bounds_list.append(bounds)
            chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

            #Submit dask task to create per-pixel raster and aggregate into 0.04x0.04 degrees for each tile
            delayed_results.append(dask.delayed(agg_4x4)(tile_id, bounds, chunk_length_pixels, pixel_area_tile, mg_ha_yr_tile, per_pixel_tile_outfile, per_pixel_output_path))

        #Check
        # print(bounds_list)
        # print(delayed_results)

        # -------------------------------------------------------------------------------------------------------------------
        # Step 4: Combine 0.04x0.04 degrees tiles into global raster
        # Model stage being run
        stage = f'create 0.04x0.04 degree global raster for {key}'
        start_time = uu.timestr()
        print(f"Stage {stage} started at: {start_time}")

        # Compute results
        tiles = dask.compute(*delayed_results)

        # Combine results into global raster
        tile_id = "0_04deg_global"
        global_4km_outfile = f"{tile_id}{items['mg_per_pixel_pattern']}"
        global_4km_output_path = items['4km_dir']

        global_raster = combine_global_raster(tiles, bounds_list, tile_id, global_4km_outfile, global_4km_output_path)
        print(global_raster)
        #TODO: Have it check that file was created in s3. Use the existing file counter uu function.

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregating AFOLU model output into global ~4km rasters.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name)
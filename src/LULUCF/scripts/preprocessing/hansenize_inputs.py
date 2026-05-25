"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

Coiled test area without land (i.e. no data):
python -m src.utilities.create_cluster -cn Hansenize -n 2 -m 4
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p secondary_natural_forest -bb -120 30 -110 40 -cs 10
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p secondary_natural_forest -bb 100 -10 110 10 -cs 10 --no_upload
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p mangroves -bb -120 30 -110 40 -cs 10
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p mangroves -bb 100 -10 110 10 -cs 10 --no_upload
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -cn Hansenize -bb 60 -11 61 -10 -cs 1
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.0_smooth_mangrove_extent_1x1_degree -cn Hansenize -bb 116 -3 117 -2 -cs 1 --no_upload
python -m src.LULUCF.scripts.preprocessing.gmw_smooth_mangrove_extent_timeseries.1_aggregate_smoothed_mangrove_extent_1x1_degree -cn Hansenize --first_10x10s_to_process 1 --no_upload

Coiled test area with data:
python -m src.utilities.create_cluster -cn Hansenize -n 2 -m 4
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p mangroves -bb 100 -10 110 10 -cs 10

Coiled full run (running with 20 clusters causes an error  of not finding the vrt, perhaps because it's being accessed too quickly-- better to run with fewer workers for now):
python -m src.utilities.create_cluster -cn Hansenize -n 10 -t 1 -m 4
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p mangroves -bb -180 -60 180 80 -cs 10
python -m src.LULUCF.scripts.preprocessing.hansenize_inputs -cn Hansenize -p AGB2015 AGB2015_stdev -bb -180 -60 180 80 -cs 10
# Note: Tried this with -n 20 -t 12 -m 16 (for mangrove extent from 1996 to 2016) and then again with -n 20 -t 12 -m 8 (for mangrove extent from 2017 to 2020).
    # After reducing the memory to 8 and using a smaller vm type, gdal_warp per dataset finshed 2x a fast (10 minutes in stead of 20).
    # Could be that it is stored closer on memory or that we just got a faster cluster the second time around.
Note: Try with 1 thread per worker

todo:
- delete .keep in each processed directory
- delete local tmp shp after they have been uploaded to s3 (double check that all files from vrt and gdal_warp are also deleted because memory is growing with each dataset x step)
- step 3: make a function and parallelize with dask using client.submit()
- step 4: parallelize all datasets (not just tiles) so it doesnt sit idle while last remaining tiles finish.
          Also, check that a single dataset (but not all datasets) finishes before moving onto step 5
- Supress GDAL warning in step 3?
- Add band1 logic to gdal_warp step for drivers
- Update path and pattern for drivers and robinson rates
"""
import os
import sys
import argparse
import dask
from dask.distributed import print

from src.utilities import constants_and_names as cn, log_utilities as lu, universal_utilities as uu

########################################################################################################################

def main(cluster_name, process, bounding_box, chunk_size, run_local, no_upload):

    # Step 1: Create download/ upload dictionary from list of processes to run
    # Create empty dictionary
    download_upload_dictionary = {}

    start_time = uu.timestr()

    # Add drivers data
    if 'drivers' in process:
        download_upload_dictionary["drivers"] = {
            'raw_dir': cn.drivers_raw_dir,
            'raw_pattern': cn.drivers_pattern,
            'vrt': f"/tmp/drivers.vrt",
            'processed_dir': cn.drivers_processed_dir,
            'processed_pattern': cn.drivers_pattern
        }

    # Add Robinson et al. secondary natural forest growth rates
    if 'secondary_natural_forest' in process:
        download_upload_dictionary[f"secondary_natural_forest_0_5"] = {
            'raw_dir': cn.secondary_natural_forest_5_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_0_5_pattern}_{cn.Robinson_5_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_0_5.vrt",
            'processed_dir': cn.secondary_natural_forest_0_5_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_0_5_pattern
        }
        download_upload_dictionary["secondary_natural_forest_6_10"] = {
            'raw_dir': cn.secondary_natural_forest_5_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_6_10_pattern}_{cn.Robinson_5_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_6_10.vrt",
            'processed_dir': cn.secondary_natural_forest_6_10_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_6_10_pattern
        }
        download_upload_dictionary["secondary_natural_forest_11_15"] = {
            'raw_dir': cn.secondary_natural_forest_5_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_11_15_pattern}_{cn.Robinson_5_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_11_15.vrt",
            'processed_dir': cn.secondary_natural_forest_11_15_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_11_15_pattern
        }
        download_upload_dictionary["secondary_natural_forest_16_20"] = {
            'raw_dir': cn.secondary_natural_forest_5_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_16_20_pattern}_{cn.Robinson_5_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_16_20.vrt",
            'processed_dir': cn.secondary_natural_forest_16_20_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_16_20_pattern
        }
        download_upload_dictionary["secondary_natural_forest_21_100"] = {
            'raw_dir': cn.secondary_natural_forest_20_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_21_100_pattern}_{cn.Robinson_20_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_21_100.vrt",
            'processed_dir': cn.secondary_natural_forest_21_100_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_21_100_pattern
        }
        download_upload_dictionary["secondary_natural_forest_21_40"] = {
            'raw_dir': cn.secondary_natural_forest_20_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_21_40_pattern}_{cn.Robinson_20_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_21_40.vrt",
            'processed_dir': cn.secondary_natural_forest_21_40_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_21_40_pattern
        }
        download_upload_dictionary["secondary_natural_forest_41_60"] = {
            'raw_dir': cn.secondary_natural_forest_20_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_41_60_pattern}_{cn.Robinson_20_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_41_60.vrt",
            'processed_dir': cn.secondary_natural_forest_41_60_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_41_60_pattern
        }
        download_upload_dictionary["secondary_natural_forest_61_80"] = {
            'raw_dir': cn.secondary_natural_forest_20_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_61_80_pattern}_{cn.Robinson_20_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_61_80.vrt",
            'processed_dir': cn.secondary_natural_forest_61_80_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_61_80_pattern
        }
        download_upload_dictionary["secondary_natural_forest_81_100"] = {
            'raw_dir': cn.secondary_natural_forest_20_year_raw_dir,
            'raw_pattern': f"{cn.secondary_natural_forest_81_100_pattern}_{cn.Robinson_20_year_raw_date}",
            'vrt': f"/tmp/secondary_natural_forest_81_100.vrt",
            'processed_dir': cn.secondary_natural_forest_81_100_processed_dir,
            'processed_pattern': cn.secondary_natural_forest_81_100_pattern
        }

    if 'AGB2015' in process:
        download_upload_dictionary["AGB2015"] = {
            'raw_dir': cn.agb_2015_dir_raw,
            'raw_pattern': cn.agb_2015_pattern_raw,
            'vrt': f"/tmp/agb2015.vrt",
            'processed_dir': cn.agb_2015_dir_processed,
            'processed_pattern': cn.agb_2015_pattern
        }

    if 'AGB2015_stdev' in process:
        download_upload_dictionary["AGB2015_stdev"] = {
            'raw_dir': cn.agb_stdev_2015_dir_raw,
            'raw_pattern': cn.agb_stdev_2015_pattern_raw,
            'vrt': f"/tmp/agb2015_stdev.vrt",
            'processed_dir': cn.agb_stdev_2015_dir_processed,
            'processed_pattern': cn.agb_stdev_2015_pattern
        }

    if 'climate_zone' in process:
        download_upload_dictionary["climate_zone"] = {
            'raw_dir': cn.climate_zone_raw_dir,
            'raw_pattern': cn.climate_zone_raw_pattern,
            'vrt': f"/tmp/climate_zone.vrt",
            'processed_dir': cn.climate_zone_processed_dir,
            'processed_pattern': cn.climate_zone_pattern
        }

    if 'mangroves' in process:
        for year in cn.mangrove_extent_years:
            download_upload_dictionary[f"mangrove_extent_{year}"] = {
                'raw_dir': f"{cn.mangrove_extent_raw_dir}{year}/",
                'raw_pattern': cn.mangrove_extent_raw_pattern,
                'vrt': f"/tmp/mangrove_extent_{year}_{cn.GMW_version}.vrt",
                'processed_dir': f"{cn.mangrove_extent_hansenized_dir}{year}/",
                'processed_pattern': f"{cn.mangrove_extent_hansenized_pattern}_{year}"
            }

    if 'cropland_fertilizer' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_manure' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_peatland' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_residues' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_residues_burnt' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    if 'cropland_rice' in process:
        download_upload_dictionary[""] = {
            'raw_dir': cn.x_raw_dir,
            'raw_pattern': cn.x_pattern,
            'vrt': "x.vrt",
            'processed_dir': cn.x_processed_dir,
            'processed_pattern': cn.x_pattern
        }

    #-------------------------------------------------------------------------------------------------------------------

    # Connects to Coiled cluster if the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False, False)
    client

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, f"Preprocessing: {process}", run_local,
                                                                   'standard', f'Hansenize: {process}')


    ####################################################################################################################
    # Step 1: Create chunk list
    # Makes list of chunks to analyze from the bounding box and chunk size (deg)
    # Output list form is [[110, -10, 120, 0], [...], [...], ...]  (W, S, E, N)
    main_logger.info("STEP 1: Using bounding box and chunk size to determine number of chunks")
    chunk_list = uu.get_chunk_bounds_from_bounding_box(bounding_box, chunk_size)
    main_logger.info(f"Chunks identified: {len(chunk_list)}\n")


    ####################################################################################################################
    # Step 2: Create a VRT for each dataset
    # creates VRTs in parallel for each dataset using futures
    vrt_futures = []
    for key, items in download_upload_dictionary.items():
        main_logger.info(f"STEP 2: Creating a VRT for {key}")

        # Add output_vrt_s3 to dictionary
        output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
        main_logger.info(f"output_vrt_s3: {output_vrt_s3}") #todo
        main_logger.info(f"Adding output_vrt_s3 path to dictionary: {output_vrt_s3}")
        download_upload_dictionary[key]['output_vrt_s3'] = output_vrt_s3

        # Find all files in s3 that match the raw pattern (w/ '*.tif') and add s3 paths to download_upload_dictionary
        if "mangrove" in key:    #uses a regular expression so has to be compiled in uu.list_s3_files_with_pattern
            input_raster_list_s3 = uu.list_s3_files_with_pattern(items['raw_dir'], items['raw_pattern'], True)
        else:
            input_raster_list_s3 = uu.list_s3_files_with_pattern(items['raw_dir'], items['raw_pattern'])

        main_logger.info(f"Raw data folder: {items['raw_dir']}")
        main_logger.info(f"Raw data pattern: {items['raw_pattern']}")
        main_logger.info(f"There are {len(input_raster_list_s3)} rasters in the raw data folder to include in the vrt")
        if input_raster_list_s3:
            download_upload_dictionary[key]['raw_raster_list'] = input_raster_list_s3
            main_logger.info(f"Input raster list: {items['raw_raster_list']}")

        # Create a vrt from all raw input rasters
        main_logger.info(f"Submitting VRT build for {key}: {uu.timestr('time')}\n")
        vrt_future = client.submit(uu.build_vrt_gdal_coiled, input_raster_list_s3, output_vrt_s3, items['vrt'], main_logger)
        vrt_futures.append((key, vrt_future))

    # Wait for all VRTs to finish before moving on to step 3
    for key, future in vrt_futures:
        future.result()
    main_logger.info(f"STEP 2 Complete - All VRTs built: {uu.timestr('time')}\n")


    ###########################################################################################################
    # Step 3: Get GDAL datatype of each dataset using the first tile in that dataset
    for key, items in download_upload_dictionary.items():
        main_logger.info(f"STEP 3: Getting GDAL datatype for {key}")

        # Dictionary that matches format expected by function that gets name of first tile in an s3 folder
        simple_dict = {}
        simple_dict_key = key
        simple_dict_value = items["raw_dir"]
        simple_dict[simple_dict_key] = simple_dict_value

        # Path of first tile in the dataset
        first_tile = uu.first_file_name_in_s3_folder(simple_dict)

        # Gets datatype of first tile in input dataset and converts it to GDAL format
        download_dict_with_data_types = uu.add_file_type_to_dict(first_tile)
        dtype = download_dict_with_data_types[simple_dict_key][1]
        gdal_dtype = uu.string_to_gdal_dtype_mapping.get(dtype)

        # Adds the dtype of the dataset to the processing dictionary
        download_upload_dictionary[key]["dt"] = gdal_dtype
        main_logger.info(f"Data type for {key} is {uu.gdal_to_string_dtype_mapping.get(gdal_dtype)}\n")
    main_logger.info(f"STEP 3 Complete - All GDAL datatypes added to dictionary: {uu.timestr('time')}\n")


    ###########################################################################################################
    #Step 4: Use warp_to_hansen to preprocess each dataset into 10x10 degree tiles
    # Iterates through all input datasets and parallelizes tile creation
    for key, items in download_upload_dictionary.items():
        main_logger.info(f"STEP 4: Creating 10 x 10 degree tiles for {key}")

        # Ensure processed directory exists
        uu.check_and_make_s3_dir(items['processed_dir'], main_logger)

        # Get dataset information
        dt = items['dt']
        output_vrt_s3 = f"{items['raw_dir']}{os.path.basename(items['vrt'])}"
        main_logger.info(f"Using {output_vrt_s3} for Hansenization")

        # Separate tile_futures list for each dataset being processed
        tile_start_time = uu.timestr()
        tile_futures = []  #creates tiles in parallel for each dataset

        # Iterates through all tiles in a given dataset
        for chunk in chunk_list:
            tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])
            output_filename = f"{tile_id}_{items['processed_pattern']}.tif"  #TODO make sure all patterns don't have tif extension
            output_tile_s3 = f"{items['processed_dir']}{output_filename}"
            xmin, ymin, xmax, ymax = uu.get_10x10_tile_bounds(tile_id)

            # Create 10 x 10 degree hansenized tile for each dataset in dictionary
            tile_future = client.submit(uu.warp_to_hansen_coiled, output_vrt_s3, output_filename, output_tile_s3,
                                        xmin, ymin, xmax, ymax, dt, 0, True, 400, 400)
            tile_futures.append(tile_future)

        main_logger.info(f"Tiles to process: {len(tile_futures)}")

        # Collect the results once they are finished
        client.gather(tile_futures)

        main_logger.info(f"Completed Hansenizing {len(tile_futures)} tiles for {key}: {uu.timestr('time')}")
        uu.stage_duration(tile_start_time, uu.timestr(), f"Hansenize {key}", main_logger, "time")
    main_logger.info(f"STEP 4 Complete - All datasets hansenized: {uu.timestr('time')}\n")

    ####################################################################################################################
    # Step 5: Creates a tile index shapefile of the output rasters to check completeness of Hansenization
    for key, items in download_upload_dictionary.items():
        main_logger.info(f"STEP 5: Making index shapefile for {key}")
        index_shp_start_time = uu.timestr()

        # Creates a list of dictionaries of s3 tile set path with corresponding tile index shapefile names,
        # e.g., [{'s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/AGB/processed/20250217/': 'AGB_2015_ESA_CCI_Mg_AGB_ha'}]
        tile_index_dict = []

        # The key for the dictionary: the s3 path with a tile set that will be indexed
        path = items['processed_dir'].replace(cn.veg_outputs_path, "")
        print(path) #TODO

        # The value for the dictionary: the pattern to use for naming the output shapefile
        value = items['processed_pattern']

        # Creates the dictionary:
        # e.g., {'s3://gfw2-data/climate/ESA_CCI_biomass/v5_01/2015/AGB/processed/20250217/': 'AGB_2015_ESA_CCI_Mg_AGB_ha'}
        tile_index_dict.append({path: value})

        # Makes raster footprint shapefile from output raster set
        delayed_result = [dask.delayed(uu.make_tile_footprint_shp)(input_dict, no_upload)
                          for input_dict in tile_index_dict]

        # Actually runs analysis
        dask.compute(*delayed_result)

        main_logger.info(f"Finished making index shapefile for {path}: {uu.timestr('time')}")
        uu.stage_duration(index_shp_start_time, uu.timestr(), f"shapefile_index_for_Hansenized_{key}", main_logger, "time")
    main_logger.info(f"STEP 5 Complete - All index shapefiles created: {uu.timestr('time')}\n")

    # Closes the Dask client
    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hansenize AFOLU model raster inputs.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-p', '--processes', action='store', nargs='+', help='What datasets do you want to hansenize?')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cs', '--chunk_size', type=float, help='Chunk size (degrees)')
    parser.add_argument('--no_upload', action='store_true', help='Do not save and upload outputs to s3')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    processes = args.processes
    bounding_box = args.bounding_box
    chunk_size = args.chunk_size
    no_upload = args.no_upload

    # Create the cluster with command line arguments
    main(cluster_name, processes, bounding_box, chunk_size, False, no_upload)
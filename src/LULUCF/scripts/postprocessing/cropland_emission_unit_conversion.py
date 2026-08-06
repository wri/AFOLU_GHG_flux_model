"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_cluster -n 40 -m 32 -t 1 -cn cropland_emissions_test
python -m scripts.postprocessing.create_global_4km_maps -cn cropland_emissions_test

"""
import argparse
import dask
from dask.distributed import print
from src.utilities import log_utilities as lu, universal_utilities as uu


########################################################################################################################

def cropland_emissions_unit_conversion(chunk, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir):

    is_final = True
    logger = lu.setup_logging()

    input_tile = chunk
    input_tile_path = f"{cropland_emissions_kg_input_dir}{input_tile}"
    output_tile = input_tile.replace("kg", "Mg")

    # Get bounds and chunk_length_pixels to read in input data
    tile_id = uu.string_to_tile_id(input_tile_path)
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_length_pixels = uu.calc_chunk_length_pixels(bounds)

    # Read in the raster
    print("Getting cropland raster")
    kg_tile_chunk = uu.get_tile_dataset_rio(input_tile_path, bounds, chunk_length_pixels, 'Float32')
    kg_tile_chunk = kg_tile_chunk[0]  # Converts downloaded tuple (array, status) to just the array

    # Conversion value from kg to Mg
    kg_to_Mg = 1e-3

    # Multiply the input tile by the conversion factor to get the Mg values
    print("Performing kg to Mg conversion")
    Mg_tile_chunk = kg_tile_chunk * kg_to_Mg

    # Upload raster to s3
    data_type = Mg_tile_chunk.dtype.name
    no_data_val = float(0)
    uu.save_and_upload_single_raster(bounds, chunk_length_pixels, tile_id, Mg_tile_chunk, data_type, output_tile,
                                     cropland_emissions_Mg_output_dir, is_final, logger, no_data_val)

def main(cluster_name):
    # -------------------------------------------------------------------------------------------------------------------
    # Step 1: Connects to Coiled cluster if not running locally and the named cluster exists
    run_local = False
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    # -------------------------------------------------------------------------------------------------------------------
    # Step 2: Convert cropland emissions from kg per hectare per year to mg per hectare per year
    #TODO: Move this step to hansenize function for cropland emissions
    #Model stage being run
    stage = 'convert_cropland_emissions_units_from_kg_to_mg'

    # Starting time for stage
    start_time = uu.timestr()
    print(f"Stage {stage} started at: {start_time}")

    # Input/ output dirs
    cropland_emissions_kg_input_dir = "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/kg/including_peatland/2019/physical_area/"
    cropland_emissions_Mg_output_dir = "s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/20241204/year_2020/all_sources/mean_rate/including_peatland/2019/physical_area/"
    #TODO: Pass in as command line arguments and add to download_upload_dictionary accordingly. For now hardcoding.

    # Get list of all tiles in the cropland emissions kg s3 folder
    cropland_emissions_kg_tiles_list = uu.list_raster_names_in_s3_folder(cropland_emissions_kg_input_dir)
    print(cropland_emissions_kg_tiles_list)

    # Creates list of tasks to run (1 task = 1 chunk)
    print(f"Creating tasks and starting processing: {uu.timestr()}")
    delayed_results = [dask.delayed(cropland_emissions_unit_conversion)(chunk, cropland_emissions_kg_input_dir, cropland_emissions_Mg_output_dir) for chunk in cropland_emissions_kg_tiles_list]

    # Runs analysis and gathers results
    results = dask.compute(*delayed_results)

    print(results)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cropland emissions unit conversion.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')

    args = parser.parse_args()

    # Create the cluster with command line arguments
    main(args.cluster_name)
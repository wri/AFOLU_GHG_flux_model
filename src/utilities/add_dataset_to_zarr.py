"""
Standalone backfill script: adds removal_factor__AGC__MgC_ha_yr to an existing vegetation
mega-zarr and populates it from already-computed tile TIFFs on S3.

Background: the vegetation model computes and saves removal-factor tiles per interval but the
pattern was never added to core_veg_outputs_to_zarr, so the array is absent from the mega-zarr.
This script repairs that without rerunning the model.

Created by Claude Code desktop

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/

Coiled small test:
python -m src.utilities.create_cluster -n 1 -m 4 -cn add_dataset_to_zarr
python -m src.utilities.add_dataset_to_zarr -cn add_dataset_to_zarr -ds removal_factor__AGC__MgC -mpd global -id 20260130 -bb 10 49 11 50 -mcstn chunk_stats/parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5

Coiled shapefile test:
python -m src.utilities.create_cluster -n 25 -m 4 -cn add_dataset_to_zarr
python -m src.utilities.add_dataset_to_zarr -cn add_dataset_to_zarr -ds removal_factor__AGC__MgC -mpd global -id 20260130 -mcstn chunk_stats/parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -f 10

Global run:
python -m src.utilities.create_cluster -n 200 -m 4 -cn add_dataset_to_zarr
python -m src.utilities.add_dataset_to_zarr -cn add_dataset_to_zarr -ds removal_factor__AGC__MgC -mpd global -id 20260130 -mcstn chunk_stats/parquet_20260131_10_37_46__KEEP/vegetation_fluxes_20260131_10_37_28__v1_0_5 -cshp s3://gfw2-data/climate/AFOLU_flux_model/fishnet_1x1deg/20250429/fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp -ln "Adding removal factor dataset to the zarr."

"""

import argparse
import dask
from dask.distributed import print

import rasterio
import fsspec
import numpy as np
import zarr

import src.utilities.constants_and_names as cn
import src.utilities.universal_utilities as uu
import src.utilities.log_utilities as lu
import src.utilities.zarr_utilities as zu

def populate_one_zarr_chunk(chunk, var_dir_no_year, zarr_store_url, interval_end_years,
                            var_name_units, var_name_no_units, resolution, main_logger):
    """
    For one spatial chunk, constructs the expected tile path for each year directly,
    stacks them into a (n_years, height, width) block, writes it to the correct zarr
    slice, and returns chunk stats for verification.
    """

    main_logger.info(f"Adding {chunk} to zarr: {uu.timestr()}")

    bounds_str = uu.boundstr(chunk)
    tile_id = uu.xy_to_tile_id(chunk[0], chunk[3])

    # Read one existing tile to get spatial extent and raster dimensions
    sample_path = None
    for year in interval_end_years:
        candidate = f"{var_dir_no_year.replace('START_END', str(year))}{tile_id}__{bounds_str}__{var_name_units}_{year}.tif"
        fs = fsspec.filesystem("s3", anon=False)
        if fs.exists(candidate.replace("s3://", "")):
            sample_path = candidate
            break

    if sample_path is None:
        return {"status": "skipped", "bounds_str": bounds_str, "reason": "no tile files found", "chunk_stats": []}

    with rasterio.open(sample_path) as src:
        b = src.bounds
        tile_height = src.height
        tile_width  = src.width

    # Compute the slice into the global zarr
    lat_start = int(round((90.0 - b.top)  / resolution))
    lon_start = int(round((b.left + 180.0) / resolution))
    lat_end   = lat_start + tile_height
    lon_end   = lon_start + tile_width

    # Build an empty (n_years, height, width) numpy array
    n_years = len(interval_end_years)
    block = np.full((n_years, tile_height, tile_width), np.float32(np.nan), dtype="float32")

    # Populate the array year by year, constructing each tile path directly
    years_written = []
    for i, year in enumerate(interval_end_years):
        tile_path = f"{var_dir_no_year.replace('START_END', str(year))}{tile_id}__{bounds_str}__{var_name_units}_{year}.tif"
        try:
            with rasterio.open(tile_path) as src:
                data = src.read(1).astype("float32")
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan   # Writes NaN rather than 0 for NoData
                block[i] = data
            years_written.append(year)
        except Exception:
            pass  # Leave year slice as NaN if tile is missing

    # Write the full time block for this chunk to the zarr
    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_store_url)
    z = zarr.open_group(mapper, mode="r+", use_consolidated=False)
    z[var_name_units][0:n_years, lat_start:lat_end, lon_start:lon_end] = block

    main_logger.info(f"Added {chunk} to zarr: {uu.timestr()}")

    # Calculate chunk stats from the zarr for verification
    chunk_stats = zu.zarr_1x1_deg_stats(chunk, var_name_no_units, zarr_store_url, interval_end_years)

    return {"status": "success", "bounds_str": bounds_str, "years_written": years_written, "chunk_stats": chunk_stats}


def main(cluster_name, input_date, var_name_no_units, model_type, no_log=False, chunk_shapefile_uri=False,
         bounding_box=None, first_chunks=None,
         model_path_description=None, model_chunk_stats_table_name=None, log_note=None):

    ### Step 1: Preparation

    # Model stage being run
    stage = f'add_{var_name_no_units}_to_zarr'

    # Connects to Coiled cluster if not running locally and the named cluster exists
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, False)

    # Shapefile of chunk footprints to use if none is supplied on the command line
    if not chunk_shapefile_uri:
        chunk_shapefile_uri = cn.fishnet_1x1deg_uri

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(client, cluster, log_note, run_local, model_type, stage)

    start_time = uu.timestr() # Starting time for stage
    main_logger.info(f"Stage {stage} started at: {start_time}")
    main_logger.info(f"Model version: {cn.veg_model_version}")
    main_logger.info(f"Model path descriptor: {model_path_description}")
    main_logger.info(f"Run date: {input_date}")
    main_logger.info(f"Tolerance for comparison between model and zarr chunk stat metrics: {cn.zarr_difference_tolerance}")

    # Calculates the interval type, difference between start and end years of intervals, and the model output years
    # for the model run
    interval_type, interval_year_diff_list, interval_length_list, interval_end_years = uu.get_interval_info(cn.first_model_year_annual, cn.last_model_year_annual, main_logger)

    # Returns a dataframe of chunk_id and ISO for the GADM4.1 1x1 deg fishnet.
    # chunk_ids for making chunk list if shapefile is supplied in command line.
    # chunk_ids and iso code used for chunk stats.
    fishnet_iso_df = uu.fishnet_with_GADM_iso(chunk_shapefile_uri)

    # Creates the list of chunks to process, depending on the approach: shapefile attribute table or a bounding box
    chunk_size_deg = 1   # Chunk size for geotifs is set at 1x1 deg
    chunk_list, chunk_size_pixels = uu.create_chunk_list(bounding_box, chunk_shapefile_uri, chunk_size_deg, first_chunks, fishnet_iso_df, main_logger)

    # The zarr path that's being used
    zarr_path = zu.create_zarr_path(cn.veg_outputs_path_mega_zarr, cn.chunk_dims, 'annual',
                                         model_type, cn.veg_model_version_underscore, model_path_description,
                                         input_date, main_logger)

    # Dataset name to add to zarr, e.g., "removal_factor__AGC__MgC_ha_yr". Assumes that the unit is _ha_yr for now.
    unit = cn.flux_density_pixel_meaning
    var_name_units = f"{var_name_no_units}{unit}"

    # Chunk folder to ingest into zarr
    var_dir_no_year = f"{cn.veg_outputs_path}{var_name_no_units}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
    var_dir_no_year = var_dir_no_year.replace(cn.model_version_type_description_placeholder,f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
    var_dir_no_year = var_dir_no_year.replace("MODEL_INTERVAL_TYPE", interval_type)
    var_dir_no_year = var_dir_no_year.replace("RUN_DATE", input_date)
    var_dir_no_year = var_dir_no_year.replace("CHUNK_SIZE", str(chunk_size_pixels))
    var_dir_no_year = var_dir_no_year.replace("CHUNK_SIZE_pixels", f"{cn.full_raster_dims}_pixels")
    var_dir_no_year = var_dir_no_year.replace("PER_HA_OR_PIXEL", cn.flux_density_pixel_meaning)
    main_logger.info(f"Geotif folder to ingest into zarr: {var_dir_no_year}")


    ### Step 2: Add the new array to the zarr

    fs = fsspec.filesystem("s3", anon=False)
    mapper = fs.get_mapper(zarr_path)

    z = zarr.open_group(mapper, mode="r+")

    n_years  = len(interval_end_years)
    lat_size = int(180 / cn.resolution)
    lon_size = int(360 / cn.resolution)

    # Checks for the json of the new dataset, to see if it's already been created
    array_exists = fs.exists(f"{zarr_path}/{var_name_units}/zarr.json")

    if not array_exists:
        new_arr = z.create_array(
            var_name_units,
            shape=(n_years, lat_size, lon_size),
            chunks=(n_years, cn.chunk_dims, cn.chunk_dims),
            dtype="float32",
            fill_value=0.0,
            compressors={"name": "zstd", "configuration": {"level": 3}},
            dimension_names=["year", "y", "x"],
        )
        new_arr.attrs["grid_mapping"] = "spatial_ref"

        # Updates consolidated metadata (zarr.json in the parent zarr folder) to include the new dataset
        zarr.consolidate_metadata(mapper)

        main_logger.info(f"Created '{var_name_units}': shape={new_arr.shape}, dtype={new_arr.dtype}: {uu.timestr()}")
    else:
        main_logger.info(f"Array '{var_name_units}' already exists — skipping creation, will populate slices: {uu.timestr()}")


    ### Step 3: Submit tasks through Coiled

    main_logger.info(f"Submitting {len(chunk_list)} tasks: {uu.timestr()}")

    # Builds a tile_paths lookup from bounds_to_files for each chunk in chunk_list
    delayed_results = [dask.delayed(populate_one_zarr_chunk)(
        chunk, var_dir_no_year,
        zarr_path, interval_end_years, var_name_units, var_name_no_units, cn.resolution, main_logger
    ) for chunk in chunk_list]

    results = dask.compute(*delayed_results)

    total_success = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    total_skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")

    chunk_stats_variable_year_zarr = [[stat for r in results for stat in r["chunk_stats"]]] # Needs extract brackets to make it a list, which is what is expected for comparison below
    print(chunk_stats_variable_year_zarr)

    main_logger.info(f"Ingestion to zarr complete: {total_success} succeeded, {total_skipped} skipped: {uu.timestr()}")


    ### Step 4: Compare zarr chunk stats to geotif chunk stats for the new dataset

    main_logger.info(f"Starting zarr chunk stats for {var_name_units}: {uu.timestr()}")

    comparison_insert = "_zarr_comparison"

    tables_to_compare_dict, zarr_comparison_stats_name, zarr_comparison_stats_path = zu.get_table_names_for_zarr_stats_comparison(
        comparison_insert, main_logger, model_chunk_stats_table_name)

    all_merged_tables = []
    chunks_count_exceeding_total = 0
    chunks_without_zarr_stats_total = 0

    # chunk_stats_variable_year_zarr = zu.run_parallel_stats(
    #     client=client,
    #     chunk_list=chunk_list,
    #     var=var_name_no_units,
    #     zarr_path=zarr_path,
    #     interval_end_years=interval_end_years
    # )
    # print(chunk_stats_variable_year_zarr)

    chunks_count_exceeding, chunks_without_zarr_stats = zu.compare_dataset_year_chunk_stats(
        all_merged_tables,
        chunk_stats_variable_year_zarr,
        main_logger,
        tables_to_compare_dict,
        var_name_no_units,
        zarr_comparison_stats_path
    )

    chunks_count_exceeding_total += chunks_count_exceeding
    chunks_without_zarr_stats_total += chunks_without_zarr_stats

    zu.upload_zarr_chunk_stat_comparisons(chunks_count_exceeding_total, chunks_without_zarr_stats_total,
                                          main_logger, model_chunk_stats_table_name,
                                          stage, start_time, zarr_comparison_stats_name, zarr_comparison_stats_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add dataset to existing zarr.")
    parser.add_argument('-cn', '--cluster_name', help='Coiled cluster name')
    parser.add_argument('-id', '--input_date', required=True, help='Date of run, in YYYYMMDD')
    parser.add_argument('-bb', '--bounding_box', nargs=4, type=float, help='W, S, E, N (degrees)')
    parser.add_argument('-cshp', '--chunk_shapefile_uri', help='s3 location for shapefile of 1x1 deg chunk footprints')
    parser.add_argument('-f', '--first_chunks', type=int, help='Number of chunks to process from shapefile')
    parser.add_argument('-mcstn', '--model_chunk_stats_table_name', required=True, help='model chunk stats table that will be compared with zarr chunk stats')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')
    parser.add_argument('-ds', '--dataset_no_units', help='Dataset to add (using pattern in constants_and_names)')
    parser.add_argument('-ln', '--log_note', help='Note to include in the log.')

    parser.add_argument('--no_log', action='store_true', help='Do not create the combined log')

    args = parser.parse_args()

    cluster_name = args.cluster_name
    input_date = args.input_date
    bounding_box = args.bounding_box
    chunk_shapefile_uri = args.chunk_shapefile_uri
    first_chunks = args.first_chunks
    model_chunk_stats_table_name = args.model_chunk_stats_table_name
    model_type = args.model_type
    model_path_description = args.model_path_description
    dataset_no_units = args.dataset_no_units
    log_note = args.log_note

    no_log = args.no_log

    # Create the cluster with command line arguments
    main(cluster_name, input_date, dataset_no_units, model_type, no_log, chunk_shapefile_uri,
         bounding_box=bounding_box, first_chunks=first_chunks,
         model_path_description=model_path_description, model_chunk_stats_table_name=model_chunk_stats_table_name,
         log_note=log_note)

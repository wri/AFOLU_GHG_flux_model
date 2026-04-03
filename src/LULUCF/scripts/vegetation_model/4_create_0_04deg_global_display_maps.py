"""
Global:
python -m src.LULUCF.scripts.vegetation_model.4_create_0_04deg_global_display_maps -mt standard -mpd global --input_date YYYYMMDD

For Central Africa:
python -m src.LULUCF.scripts.vegetation_model.4_create_0_04deg_global_display_maps -mt standard -mpd global --input_date YYYYMMDD --center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

For Borneo/Sumatra:
python -m src.LULUCF.scripts.vegetation_model.4_create_0_04deg_global_display_maps -mt standard -mpd global --input_date YYYYMMDD --center_latitude 1 --center_longitude 108 --lat_height 12 -bbd Borneo_Sumatra


Run locally (not in Coiled)

Defaults to global coverage but a zoomed in map can be created by supplying central lat-long arguments,
as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
Annual average maps from https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69835153-d804-832e-a9bf-ecf01d221a11
"""

import argparse
from pathlib import Path

from src.utilities import constants_and_names as cn
from src.utilities import map_utilities as mu
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

def main(input_date, model_type, model_path_description=None,
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

    # Model stage being run
    stage = 'summative_4x4km_vegetation_maps'
    log_note = '4x4 km maps for vegetation'

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header("NA", "NA", log_note, True, "NA", stage)

    # Datasets that need to be expanded to all output years
    basic_dirs_to_expand = [
        f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_CO2_only_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_non_CO2_only_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.gross_emis_all_C_pools_all_gases_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.gross_removals_all_C_pools_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.net_flux_all_C_pools_CO2_only_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
        f"{cn.veg_outputs_path}{cn.net_flux_all_C_pools_all_gases_pattern}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
    ]

    # Creates a list of output directories for all outputs and intervals based on specifics of the model run
    inputs_by_interval_dir_list = uu.create_output_dir_name_list(basic_dirs_to_expand, "annual", cn.first_model_year_annual,"global",
                                                                 model_type, cn.veg_model_version_underscore, model_path_description,
                                                                 cn.interval_end_years_annual,
                                                                 [1, 1, 1, 1, 1, 1, 1, 1, 1], input_date,
                                                                 True, cn.flux_aggreg_pixel_meaning)

    # s3 folders to process
    gross_emis_CO2_only_input_folders_s3 = []
    gross_emis_non_CO2_input_folders_s3 = []
    gross_emis_all_gases_input_folders_s3 = []
    gross_removals_input_folders_s3 = []
    net_CO2_only_input_folders_s3 = []
    net_all_gases_input_folders_s3 = []
    other_input_folders_s3 = []

    for input in inputs_by_interval_dir_list:
        if cn.gross_emis_all_C_pools_CO2_only_pattern in input:
            gross_emis_CO2_only_input_folders_s3.append(input)
        elif cn.gross_emis_all_C_pools_non_CO2_only_pattern in input:
            gross_emis_non_CO2_input_folders_s3.append(input)
        elif cn.gross_emis_all_C_pools_all_gases_pattern in input:
            gross_emis_all_gases_input_folders_s3.append(input)
        elif cn.gross_removals_all_C_pools_pattern in input:
            gross_removals_input_folders_s3.append(input)
        elif cn.net_flux_all_C_pools_CO2_only_pattern in input:
            net_CO2_only_input_folders_s3.append(input)
        elif cn.net_flux_all_C_pools_all_gases_pattern in input:
            net_all_gases_input_folders_s3.append(input)
        else:
            other_input_folders_s3.append(input)

    # print(gross_emis_CO2_only_input_folders_s3)
    # print(gross_emis_non_CO2_input_folders_s3)
    # print(gross_emis_all_gases_input_folders_s3)
    # print(gross_removals_input_folders_s3)
    # print(net_CO2_only_input_folders_s3)
    # print(net_all_gases_input_folders_s3)

    # Folders for local outputs
    local_reproj_folder = Path(cn.local_jpeg_folder_vegetation)
    local_reproj_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_vegetation}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_non_pres")
    local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_vegetation}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_pres")
    local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    local_gif_folder = Path(f"{cn.local_jpeg_folder_vegetation}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/gifs")
    local_gif_folder.mkdir(parents=True, exist_ok=True)

    # Reprojects simplified country boundary shapefile, if needed
    country_shapefile = mu.check_and_reproject_shapefile(main_logger,
        shapefile_path=cn.original_shapefile_path,
        target_crs=cn.Robinson_crs,
        reprojected_shapefile_path=cn.reprojected_shapefile_path
    )

    # Creates bounding box in degrees from given map center and desired latitude range (optional)
    if center_latitude is not None and center_longitude is not None and lat_height is not None:
        bounding_box = mu.calculate_bbox_centered(main_logger,
            center_lat=center_latitude,
            center_lon=center_longitude,
            lat_height=lat_height,
            aspect_ratio=2.0  # panel_dims = (12, 6), same as global map for simplicity
        )
        main_logger.info(f"Using custom bounding box: {bounding_box}")
    else:
        bounding_box = None
        main_logger.info("No bounding box specified; using global extent.")

    # Generates jpegs for net flux, gross emissions, and gross removals
    mu.map_net_flux(net_all_gases_input_folders_s3, model_type, model_path_description, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 cn.net_colors_rgb, main_logger, country_shapefile, bounding_box, bounding_box_description)

    # mu.map_net_flux(net_CO2_only_input_folders_s3, model_type, model_path_description, local_reproj_folder,
    #              local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #              cn.net_colors_rgb, main_logger, country_shapefile, bounding_box, bounding_box_description)
    #
    # mu.map_gross(gross_emis_CO2_only_input_folders_s3, model_type, model_path_description, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  cn.emissions_colors_rgb, main_logger, cn.emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # mu.map_gross(gross_emis_non_CO2_input_folders_s3, model_type, model_path_description, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  cn.emissions_colors_rgb, main_logger, cn.emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    # mu.map_gross(gross_emis_all_gases_input_folders_s3, model_type, model_path_description, local_reproj_folder,
    #                  local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
    #                  cn.emissions_colors_rgb, main_logger, cn.emissions_percentiles, country_shapefile, bounding_box, bounding_box_description)
    #
    mu.map_gross(gross_removals_input_folders_s3, model_type, model_path_description, local_reproj_folder,
                 local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 cn.removals_colors_rgb, main_logger, cn.removals_percentiles, country_shapefile, bounding_box, bounding_box_description)

    # # Generates three-panel map
    # create_three_panel_map()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Latitude to show around lat center (value is total north/south) (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box (if used) to include in output names.')
    parser.add_argument('-id', '--input_date', help='Date of run, in YYYYMMDD')
    parser.add_argument('-mt', '--model_type', default='standard', help='Type of model run (e.g., standard).')
    parser.add_argument('-mpd', '--model_path_description', help='Description of model run (e.g., global, test, X_area).')

    args = parser.parse_args()
    center_latitude = args.center_latitude
    center_longitude = args.center_longitude
    lat_height = args.lat_height
    bounding_box_description = args.bounding_box_description
    input_date = args.input_date
    model_type = args.model_type
    model_path_description = args.model_path_description

    main(input_date, model_type, model_path_description=model_path_description,
         center_latitude=center_latitude, center_longitude=center_longitude, lat_height=lat_height, bounding_box_description=bounding_box_description)


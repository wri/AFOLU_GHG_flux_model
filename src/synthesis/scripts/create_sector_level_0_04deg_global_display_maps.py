"""
Creates 4x4km maps of LULUCF and AFOLU, if cropland and livestock maps provided.
User supplies locations for vegetation, organic soil, mineral soil, cropland, and livestock global geotifs.
Average annual vegetation (average of 2016-2024), organic soil (average of 2016-2020 and 2021-2024 emissions)
and mineral soil (average of 2011-2015 vs. 2016-2020 change and 2016-2020 vs. 2021-2022 change) are added together for LULUCF.
All five are added together for AFOLU.
Also, the four non-vegetation maps are added to vegetation pairwise for completeness.

Command line arguments include the most recent year for organic and mineral soil.
Those paths are used to find the second-most recent year in s3, which is averaged with the supplied most recent year
to get the average over 2016 onwards.

SOC is converted from Mg C/0.04 deg pixel/yr to Mg CO2/0.04deg pixel/yr
and the sign is flipped (negative is gain, positive is loss-- to match vegetation) for this synthesis.

Need to run the vegetation-only jpeg creation script before this to create global annual average jpegs for vegetation.

Defaults to global coverage but a zoomed in map can be created by supplying central lat-long arguments,
as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

A zoomed in map can be created by supplying central lat-long arguments, as well as a north-south extent for the map to include.
The aspect ratio used in the global map of 2:1 (width:height) is maintained, and the east-west extent is determined
from that information. That keeps all zoomed in maps in the same shape as the global map for simplicity.

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model
Runs locally, not in Coiled.

Global LULUCF:
python -m src.synthesis.scripts.create_sector_level_0_04deg_global_display_maps
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global_YYYYMMDD/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif

Global AFOLU:
python -m src.synthesis.scripts.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global_YYYYMMDD/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif

For Central Africa:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global_YYYYMMDD/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif
--center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

For Borneo/Sumatra:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global_YYYYMMDD/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif
--center_latitude 1 --center_longitude 108 --lat_height 12 -bbd Borneo_Sumatra

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
This specific code at https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69778c22-2538-8325-a70e-1a2b70312505

#TODO Average soil values for last two intervals instead of using just the most recent value.
For mineral soil, I could calculate the annual density change for 2015 vs. 2022, rather than averaging the two change values.
For organic soil, I'll have to average the two periods of emissions.
"""

import argparse
from pathlib import Path
import time
import os
import sys
import rasterio
import math
import numpy as np
import re
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from scipy.stats import percentileofscore
from pyproj import Transformer

from src.utilities import constants_and_names as cn
from src.utilities import map_utilities as mu
from src.utilities import universal_utilities as uu
from src.utilities import log_utilities as lu

# Reprojects global geotifs to the projection/extent/resolution that the vegetation geotifs use, if not already reprojected
def reproject_to_vegetation(geotif_to_reproj, local_reproj_folder, net_all_gases_geotif_local, main_logger):

    # Extracts the file name with extension, then removes extension
    filename_with_ext = os.path.basename(geotif_to_reproj)
    filename = os.path.splitext(filename_with_ext)[0]

    path_unproj = geotif_to_reproj
    path_reproj = f"{local_reproj_folder}/{filename}_reproj.tif"

    if not os.path.exists(path_reproj):

        main_logger.info("  Reprojected raster does not exist. Reprojecting now...")
        main_logger.info(f"   Unprojected raster: {path_unproj}")
        main_logger.info(f"   Reprojected raster: {path_reproj}")

        # Opens reference raster to extract desired CRS, transform, and shape
        with rasterio.open(net_all_gases_geotif_local) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height

        # Opens source raster to reproject
        with rasterio.open(path_unproj) as src:

            # Prepare output metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': 0,
                'compress': 'lzw'
            })

            # Reprojects and write to file
            with rasterio.open(path_reproj, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest  # or bilinear/cubic as needed
                    )

    else:
        main_logger.info("  Reprojected raster already exists")

    return path_reproj

# Converts geotif from kg to megagrams (tonnes)
def convert_kg_to_Mg(path_reproj, main_logger):

    # Unit-converted raster
    converted_path = path_reproj.replace("kg", "Mg")

    if not os.path.exists(converted_path):

        main_logger.info("  Unit-converted raster does not exist. Converting kg to Mg...")

        with rasterio.open(path_reproj) as src:
            data = src.read(1)
            meta = src.meta.copy()
            nodata = src.nodata

            # Masks nodata values (e.g., 0) to avoid dividing them
            data = np.where(data == nodata, nodata, data / 1000.0)

            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(data.astype('float32'), 1)

    else:
        main_logger.info("  Unit-converted raster already exists")

    return converted_path

# Sums the vegetation net flux and other dataset
def add_veg_and_other_data(output_sum_path, additional_data, net_all_gases_geotif_local, main_logger):

    with rasterio.open(net_all_gases_geotif_local) as veg_flux_src:
        veg_flux = veg_flux_src.read(1)

        # Copy metadata from one of the sources (assumed identical)
        meta = veg_flux_src.meta.copy()
        meta.update(dtype='float32')

    # Add rasters directly — no masking
    data_sum = veg_flux + additional_data

    with rasterio.open(output_sum_path, 'w', **meta) as dst:
        dst.write(data_sum.astype('float32'), 1)

    # All non-zero values (used for calculating legend values)
    non_zero_values = data_sum[data_sum != 0]

    return non_zero_values


def map_AFOLU_totals(veg_net_all_gases_geotif_local,
                     organic_soil_local,
                     mineral_soil_s3,
                     cropland_geotif_s3,
                     livestock_geotif_s3,
                     net_colors_rgb, country_shapefile, bounding_box, bounding_box_description, main_logger):

    start_time = time.time()

    # Folders for local outputs
    LULUCF_reproj_folder = Path(cn.local_jpeg_folder_LULUCF)
    LULUCF_reproj_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_non_pres")
    LULUCF_local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_pres")
    LULUCF_local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    LULUCF_local_gif_folder = Path(f"{cn.local_jpeg_folder_LULUCF}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/gifs")
    LULUCF_local_gif_folder.mkdir(parents=True, exist_ok=True)

    cropland_reproj_folder = Path(cn.local_jpeg_folder_cropland)
    cropland_reproj_folder.mkdir(parents=True, exist_ok=True)
    livestock_reproj_folder = Path(cn.local_jpeg_folder_livestock)
    livestock_reproj_folder.mkdir(parents=True, exist_ok=True)

    AFOLU_local_jpeg_non_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_non_pres")
    AFOLU_local_jpeg_non_pres_folder.mkdir(parents=True, exist_ok=True)
    AFOLU_local_jpeg_pres_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/jpegs_pres")
    AFOLU_local_jpeg_pres_folder.mkdir(parents=True, exist_ok=True)
    AFOLU_local_gif_folder = Path(f"{cn.local_jpeg_folder_AFOLU}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}/gifs")
    AFOLU_local_gif_folder.mkdir(parents=True, exist_ok=True)

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = mu.transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    data_to_add = {}

    if organic_soil_local:
        data_to_add['organic_soil'] = [organic_soil_local, LULUCF_reproj_folder, cn.organic_soil_pres_text, LULUCF_local_jpeg_non_pres_folder, LULUCF_local_jpeg_pres_folder]

    if mineral_soil_s3:
        data_to_add['mineral_soil'] = [mineral_soil_s3, LULUCF_reproj_folder, cn.mineral_soil_pres_text, LULUCF_local_jpeg_non_pres_folder, LULUCF_local_jpeg_pres_folder]

    if cropland_geotif_s3:
        data_to_add['cropland'] = [cropland_geotif_s3, cropland_reproj_folder, cn.cropland_pres_text, AFOLU_local_jpeg_non_pres_folder, AFOLU_local_jpeg_pres_folder]

    if livestock_geotif_s3:
        data_to_add['livestock'] = [livestock_geotif_s3, livestock_reproj_folder, cn.livestock_pres_text, AFOLU_local_jpeg_non_pres_folder, AFOLU_local_jpeg_pres_folder]

    veg_gross_emis_all_gases_local = veg_net_all_gases_geotif_local.replace(cn.net_flux_all_C_pools_all_gases_pattern, cn.gross_emis_all_C_pools_all_gases_pattern)
    veg_gross_remv_all_gases_local = veg_net_all_gases_geotif_local.replace(cn.net_flux_all_C_pools_all_gases_pattern, cn.gross_removals_all_C_pools_pattern)

    main_logger.info(f"Vegetation net flux: {veg_net_all_gases_geotif_local}")
    main_logger.info(f"Vegetation gross emissions: {veg_gross_emis_all_gases_local}")
    main_logger.info(f"Vegetation gross removals: {veg_gross_remv_all_gases_local}")
    main_logger.info(f"Inputs to add to vegetation: {data_to_add}")

    veg_analysis_years = f"{cn.interval_end_years_annual[0]}_{cn.last_model_year_annual}"

    # Version of the vegetation model being used
    veg_version = re.search(r'v\d+_\d+_\d+', veg_net_all_gases_geotif_local).group(0)

    # Loads base vegetation rasters once
    # Net vegetation flux
    with rasterio.open(veg_net_all_gases_geotif_local) as src_veg_net:
        veg_meta = src_veg_net.meta.copy()
        LULUCF_net = src_veg_net.read(1).astype('float32')  # base raster to accumulate into for LULUCF net
        AFOLU_net = src_veg_net.read(1).astype('float32')  # base raster to accumulate into for AFOLU net
        print("LULUCF_net.min:", LULUCF_net.min())
        print("LULUCF_net.max:", LULUCF_net.max())

    # Gross vegetation emissions
    with rasterio.open(veg_gross_emis_all_gases_local) as src_veg_emis:
        LULUCF_emis = src_veg_emis.read(1).astype('float32')  # base raster to accumulate into for LULUCF emis
        print("LULUCF_emis.min:", LULUCF_emis.min())
        print("LULUCF_emis.max:", LULUCF_emis.max())

    # Gross vegetation removals
    with rasterio.open(veg_gross_remv_all_gases_local) as src_veg_remv:
        LULUCF_remv = src_veg_remv.read(1).astype('float32')  # base raster to accumulate into for LULUCF emis
        print("LULUCF_remv.min:", LULUCF_remv.min())
        print("LULUCF_remv.max:", LULUCF_remv.max())


    # ### Part 1: Maps average annual vegetation net flux by itself (for completeness).
    # ### This should be equivalent to the full model period annual average output from the vegetation model,
    # ### but I'm recreating it here so that maps for all components are created here.
    # ### The vegetation jpeg/gif script must have already been run (to create local reprojected vegetation net flux geotif).
    # ### NOTE: I can't get this averge annual net flux map to match the one in 4_create_0_04deg_global_display_maps.
    # ### The legend here has very different min and max values and the map colors are different.
    # ### I assume this has to do with masking or removing NoData pixels in some way.
    #
    # ### TODO be able to create the same average annual net flux map here as in the vegetation jpeg script
    # ### so that the vegetation map is created alongside vegetation+[other], LULUCF, and AFOLU.
    #
    # main_logger.info(f"  Plotting average annual vegetation net flux map")
    #
    # percentile_0 = mu.percentile_for_0(mean_veg_net)
    # main_logger.info(f"  0 is at the {percentile_0}th percentile of the average annual net flux vegetation raster.")
    # percentiles = [percentile_0 * cn.net_percentiles[0], percentile_0 * cn.net_percentiles[1],
    #                percentile_0 * cn.net_percentiles[2],
    #                percentile_0 * cn.net_percentiles[3], percentile_0 * cn.net_percentiles[4],
    #                percentile_0 * cn.net_percentiles[5], percentile_0 * cn.net_percentiles[6],
    #                percentile_0 * cn.net_percentiles[7],
    #                percentile_0 * cn.net_percentiles[8], percentile_0 * cn.net_percentiles[9]]
    # # print("percentiles:", percentiles)
    #
    # main_logger.info(f"  Calculating percentiles and breaks for average annual net flux vegetation")
    #
    # # Converts RGB color palette to matplotlib color palette
    # colors_matplotlib = mu.rgb_to_mpl_palette(cn.net_colors_rgb)
    #
    # # Matches percentile breaks with colors for the map.
    # # Normalizes percentiles to a 0-1 scale.
    # percentiles_normalized = np.linspace(0, 1, len(percentiles))
    # # print("percentiles_normalized:", percentiles_normalized)
    # cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))
    #
    # main_logger.info(f"  Masking raster for average annual net flux vegetation to non-0 values")
    # masked_data = np.ma.masked_where(mean_veg_net == 0, mean_veg_net)
    #
    # percentile_for_saturation = 1
    # breaks_all_yrs = np.percentile(mean_veg_net, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate
    #
    # lower_lim_all_yrs = breaks_all_yrs[0]
    # global_neutral = 0
    # upper_lim_all_yrs = breaks_all_yrs[-1]
    #
    # main_logger.info("For average raster:")
    # main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_all_yrs}")
    # main_logger.info(f"  neutral: {global_neutral}")
    # main_logger.info(f"  upper limit ({(100-percentile_for_saturation)} percentile): {upper_lim_all_yrs}")
    #
    # # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # # Rounds data_min down and data_max up for legend.
    # rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
    # rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down
    # tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
    #                f"{0}        (neutral)",
    #                f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]
    # print("tick_labels:", tick_labels)
    #
    # # For map (not legend)
    # norm = TwoSlopeNorm(
    #     vmin=lower_lim_all_yrs,
    #     vcenter=global_neutral,
    #     vmax=upper_lim_all_yrs
    # )
    #
    # main_logger.info(f"  Plotting map for average annual net flux vegetation")
    # ax, fig = mu.create_plot()
    #
    # # Sets the ocean color
    # mu.set_ocean_color(ax)
    #
    # # Limits shapefile to focal extent (if requested)
    # if bounding_box_proj is not None:
    #     bbox_geom = box(*bounding_box_proj)
    #     country_shapefile = country_shapefile.clip(bbox_geom)
    #
    # # Plots the country polygons first
    # mu.plot_country_polygons(ax, country_shapefile)
    #
    # # Raster extent
    # extent = list(raster_extent)
    #
    # # Plots the raster next
    # img = mu.plot_raster(ax, cmap, extent, masked_data, norm)
    #
    # # Plots the country boundaries on top
    # mu.plot_country_boundaries(ax, country_shapefile)
    #
    # # Explicitly sets the bounding box for the plot image
    # if bounding_box_proj is not None:
    #     ax.set_xlim(extent[0], extent[1])
    #     ax.set_ylim(extent[2], extent[3])
    #
    # title_text = f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
    #
    # # Creates legend
    # mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
    #                                    title_text, tick_labels,
    #                                    "", cn.net_colors_rgb, percentiles, percentile_0, main_logger)
    #
    # # Removes axis ticks and labels
    # mu.remove_ticks(ax)
    #
    #
    # core_jpeg_name = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{veg_analysis_years}__kt_CO2e_yr__{uu.timestr()[0:8]}"
    # if bounding_box_description:  # Adds bounding box description to file name, if supplied
    #     core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
    # jpeg_path = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
    # jpeg_for_pres_path = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"
    #
    # # Saves two versions of the map: without and with a source note in the bottom right
    # out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", cn.veg_pres_text, main_logger)

    main_logger.info(f"\n---Combining individual datasets with vegetation net flux")
    ### Part 2: Maps average annual vegetation net flux + one other dataset at a time (pairwise)

    # Iterates through non-vegetation layers to combine them with vegetation individually
    for key, value in data_to_add.items():

        input_s3_path = value[0]
        local_reproj_folder = value[1]
        presentation_slide_text = value[2]
        non_pres_folder = value[3]
        pres_folder = value[4]

        main_logger.info(f"\n---Reprojecting {key} to vegetation projection")

        # Date/version of the other dataset being used (depends on specific dataset)
        if key == 'organic_soil':
            additional_data_date = cn.organic_soil_model_version_underscore
        elif key == 'mineral_soil':
            additional_data_date = cn.SOC_model_version_underscore
        elif key == 'cropland':
            additional_data_date = re.search(r'/(\d{8})/', input_s3_path).group(1)
        elif key == 'livestock':
            additional_data_date = re.search(r'/(\d{8})/', input_s3_path).group(1)
        else:
            additional_data_date = 'no_version_info'

        value.append(additional_data_date)

        # Reprojects to match vegetation net flux (if not already reprojected)
        path_reproj = reproject_to_vegetation(input_s3_path, local_reproj_folder, veg_net_all_gases_geotif_local, main_logger)
        # print("path_reproj:", path_reproj)

        # Converts from kg to Mg (if not already converted, like for cropland and livestock)
        unit_converted_path = convert_kg_to_Mg(path_reproj, main_logger)
        # print("unit_converted_path:", unit_converted_path)

        # Loads unit-converted raster
        with rasterio.open(unit_converted_path) as src:
            additional_data = src.read(1).astype('float32')

        # Need to convert SOC change from Mg C/yr to Mg CO2/yr and make its sign match vegetation (negative=removals, positive=emissions)
        if key == "mineral_soil":
            print("Mineral soil data")
            print("additional_data.min:", additional_data.min())
            print("additional_data.max:", additional_data.max())
            additional_data = additional_data * cn.C_to_CO2  # Converts mineral soil SOC change from Mg C/yr to Mg CO2/yr
            additional_data = additional_data * -1  # Converts mineral soil SOC change to positive for loss and negative for gain
            print("additional_data.min:", additional_data.min())
            print("additional_data.max:", additional_data.max())

            # Splits mineral soil into separate loss and gain arrays
            SOC_loss = np.where(additional_data > 0, additional_data, 0)
            SOC_gain = np.where(additional_data < 0, additional_data, 0)
            print("SOC_loss.min:", SOC_loss.min())
            print("SOC_loss.max:", SOC_loss.max())
            print("SOC_gain.min:", SOC_gain.min())
            print("SOC_gain.max:", SOC_gain.max())

            # Includes SOC loss with LULUCF emissions and SOC gain with LULUCF removals
            LULUCF_emis += SOC_loss
            LULUCF_remv += SOC_gain

        # Adds emissions from organic soil to LULUCF gross emissions total
        if key == "organic_soil":
            print("Adding organic soil to emissions")
            LULUCF_emis += additional_data

        # Only adds soil data to running LULUCF net total
        if "soil" in unit_converted_path:
            LULUCF_net += additional_data
            print("LULUCF_net.min:", LULUCF_net.min())
            print("LULUCF_net.max:", LULUCF_net.max())

        # Adds all datasets to running AFOLU total
        AFOLU_net += additional_data
        print("AFOLU_net.min:", AFOLU_net.min())
        print("AFOLU_net.max:", AFOLU_net.max())

        main_logger.info(f"Combining vegetation net flux and {key}")
        output_name = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{key}_{additional_data_date}__{veg_analysis_years}__kt_CO2e_yr"
        output_sum_path = f"{local_reproj_folder}/{output_name}.tif"
        main_logger.info(f"Combined vegetation and {key} at {output_sum_path}")

        # Sums the vegetation net flux and other data
        non_zero_values = add_veg_and_other_data(output_sum_path, additional_data, veg_net_all_gases_geotif_local, main_logger)


        main_logger.info(f"\n\n---Preparing legend")

        # Calculates min, center and max across all years
        percentile_for_saturation = 1
        breaks_all_yrs = np.percentile(non_zero_values, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate

        lower_lim_all_yrs = breaks_all_yrs[0]
        global_neutral = 0
        upper_lim_all_yrs = breaks_all_yrs[-1]

        main_logger.info(f"Across vegetation+{key}:")
        main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_all_yrs}")
        main_logger.info(f"  neutral: {global_neutral}")
        main_logger.info(f"  upper limit ({(100-percentile_for_saturation)} percentile): {upper_lim_all_yrs}")

        # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
        # Rounds data_min down and data_max up for legend.
        rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds up
        rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100  # Rounds down
        tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
                       "0        (neutral)",
                       f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]
        # print(tick_labels)


        main_logger.info(f"\n\n---Mapping vegetation + {key}")

        # Reads raster data
        with rasterio.open(output_sum_path) as src:

            if bounding_box_proj is not None:
                minx, miny, maxx, maxy = bounding_box_proj

                window = from_bounds(minx, miny, maxx, maxy, src.transform)

                data = src.read(1, window=window)

                # Update extent from the window
                left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
                raster_extent = (left, right, bottom, top)

            else:
                data = src.read(1)
                b = src.bounds
                raster_extent = (b.left, b.right, b.bottom, b.top)

        # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
        main_logger.info(f"  Calculating percentiles and breaks")
        percentile_0 = mu.percentile_for_0(data)
        main_logger.info(f"  0 is at the {percentile_0}th percentile of the raster.")
        percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
                       percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
        # print("percentiles:", percentiles)

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        main_logger.info(f"  Masking raster to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)

        # For map (not legend)
        norm = TwoSlopeNorm(
            vmin=lower_lim_all_yrs,
            vcenter=global_neutral,
            vmax=upper_lim_all_yrs
        )

        main_logger.info(f"  Plotting map")
        ax, fig = mu.create_plot()

        # Sets the ocean color
        mu.set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
        if bounding_box_proj is not None:
            bbox_geom = box(*bounding_box_proj)
            country_shapefile = country_shapefile.clip(bbox_geom)

        # Plots the country polygons first
        mu.plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = list(raster_extent)

        # Plots the raster next
        img = mu.plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        mu.plot_country_boundaries(ax, country_shapefile)

        # Explicitly sets the bounding box for the plot image
        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Title
        legend_title = key.replace("_", " ")
        title_text = f"Vegetation and {legend_title}\nkt CO$_2$e yr$^{{-1}}$"

        # Creates legend
        mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                              title_text, tick_labels,
                                              veg_analysis_years, net_colors_rgb, percentiles, percentile_0, main_logger)

        # Removes axis ticks and labels
        mu.remove_ticks(ax)

        core_jpeg_name = f"{output_name}__{uu.timestr()[0:8]}"
        if bounding_box_description:  # Adds bounding box description to file name, if supplied
            core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
        jpeg_path = f"{non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        full_slide_text_LULUCF = f"{cn.veg_pres_text}; {presentation_slide_text} \n {cn.legend_percentile_disclaimer}"
        veg_addtl_pres_text = full_slide_text_LULUCF.replace("YYYYMMDD", additional_data_date)  # For livestock and cropland, whose versions are dates
        out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", veg_addtl_pres_text, main_logger)

        end_time = time.time()
        main_logger.info(f"vegetation+{key} {bounding_box_description} took {round(end_time - start_time)} seconds: {uu.timestr()}")


    ### Part 3: Maps net LULUCF

    main_logger.info("\n\n\n---Mapping net LULUCF:")

    # Iteratively collects the names and versions of non-vegetation datasets, and text for bottom-right of maps
    non_veg_versions = ''
    full_slide_text_LULUCF = f'{cn.veg_pres_text}; '
    if organic_soil_local:
        non_veg_versions = f'{non_veg_versions}_organic_soil_v{cn.organic_soil_model_version_underscore}'
        full_slide_text_LULUCF = f'{full_slide_text_LULUCF}{cn.organic_soil_pres_text};'
    if mineral_soil_s3:
        non_veg_versions = f'{non_veg_versions}_mineral_soil_v{cn.SOC_model_version_underscore}'
        full_slide_text_LULUCF = f'{full_slide_text_LULUCF}{cn.mineral_soil_pres_text};'
    full_slide_text_LULUCF_with_disclaimer = f"{full_slide_text_LULUCF} \n {cn.legend_percentile_disclaimer}"

    # Final combined output
    output_name = f"LULUCF__veg_{veg_version}__{non_veg_versions}__kt_CO2e_yr"
    # print("output_name:", output_name)
    final_total_path = f"{cn.local_jpeg_folder_LULUCF}/{output_name}.tif"
    # print("final_total_path:", final_total_path)
    with rasterio.open(final_total_path, 'w', **veg_meta) as dst:
        dst.write(LULUCF_net.astype('float32'), 1)

    non_zero_values_LULUCF = LULUCF_net[LULUCF_net != 0]

    main_logger.info(f"\n\n---Preparing LULUCF legend")

    # Calculates min, center and max across all years
    percentile_for_saturation = 1
    breaks_LULUCF = np.percentile(non_zero_values_LULUCF, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_LULUCF = breaks_LULUCF[0]
    global_neutral_LULUCF = 0
    upper_lim_LULUCF = breaks_LULUCF[-1]

    main_logger.info(f"Across LULUCF:")
    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_LULUCF}")
    main_logger.info(f"  neutral: {global_neutral_LULUCF}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_LULUCF}")

    # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_LULUCF = math.ceil(lower_lim_LULUCF / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_LULUCF = math.floor(upper_lim_LULUCF / 10 ** 3 * 100) / 100  # Rounds down
    tick_labels = [f"< {rounded_lower_lim_LULUCF:.0f}  (sink)",
                   # Spaces are to horizontally align the text explanations
                   "0        (neutral)",
                   f"> {rounded_upper_lim_LULUCF:.0f}  (source)"]
    # print(tick_labels)

    main_logger.info(f"\n\n---Generating net LULUCF map:")

    # Reads raster data
    with rasterio.open(final_total_path) as src:

        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj

            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            data = src.read(1, window=window)

            # Update extent from the window
            left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
            raster_extent = (left, right, bottom, top)

        else:
            data = src.read(1)
            b = src.bounds
            raster_extent = (b.left, b.right, b.bottom, b.top)

    # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
    main_logger.info(f"  Calculating percentiles and breaks for LULUCF")
    percentile_0 = mu.percentile_for_0(data)
    main_logger.info(f"  0 is at the {percentile_0}th percentile of the raster.")
    percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
                   percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
    # print("percentiles:", percentiles)

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles to a 0-1 scale.
    percentiles_normalized = np.linspace(0, 1, len(percentiles))
    # print("percentiles_normalized:", percentiles_normalized)
    cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

    main_logger.info(f"  Masking raster to non-0 values for LULUCF")
    masked_data = np.ma.masked_where(data == 0, data)

    # For map (not legend)
    norm = TwoSlopeNorm(
        vmin=lower_lim_LULUCF,
        vcenter=global_neutral_LULUCF,
        vmax=upper_lim_LULUCF
    )

    main_logger.info(f"  Plotting net LULUCF map")
    ax, fig = mu.create_plot()

    # Sets the ocean color
    mu.set_ocean_color(ax)

    # Limits shapefile to focal extent (if requested)
    if bounding_box_proj is not None:
        bbox_geom = box(*bounding_box_proj)
        country_shapefile = country_shapefile.clip(bbox_geom)

    # Plots the country polygons first
    mu.plot_country_polygons(ax, country_shapefile)

    # Raster extent
    extent = list(raster_extent)

    # Plots the raster next
    img = mu.plot_raster(ax, cmap, extent, masked_data, norm)

    # Plots the country boundaries on top
    mu.plot_country_boundaries(ax, country_shapefile)

    # Explicitly sets the bounding box for the plot image
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Title
    title_text = f"LULUCF net GHG flux (vegetation+soil)\nkt CO$_2$e yr$^{{-1}}$"

    # Creates legend
    mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_LULUCF, rounded_upper_lim_LULUCF,
                                          title_text, tick_labels,
                                          veg_analysis_years, net_colors_rgb, percentiles, percentile_0, main_logger)

    # Removes axis ticks and labels
    mu.remove_ticks(ax)

    core_jpeg_name = f"{output_name}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
    jpeg_path = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
    jpeg_for_pres_path = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    end_time = time.time()
    main_logger.info(f"LULUCF for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


    ### Part 4: Maps LULUCF gross emissions

    main_logger.info("\n\n\n---Mapping LULUCF gross emissions and removals:")

    LULUCF_emis_name = "LULUCF_emis"
    LULUCF_emis_path = f"{LULUCF_reproj_folder}/{LULUCF_emis_name}.tif"
    with rasterio.open(LULUCF_emis_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_emis, 1)

    LULUCF_remv_name = "LULUCF_remv"
    LULUCF_remv_path = f"{LULUCF_reproj_folder}/{LULUCF_remv_name}.tif"
    with rasterio.open(LULUCF_remv_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_remv, 1)


    ### Part 4: Maps AFOLU

    main_logger.info("\n\n\n---Mapping AFOLU:")

    # Iteratively collects the names and versions of non-vegetation datasets, and text for bottom-right of maps (LULUCF versions already collected)
    full_slide_text_AFOLU = f"{full_slide_text_LULUCF} \n"
    if cropland_geotif_s3:
        cropland_date = re.search(r'/(\d{8})/', cropland_geotif_s3).group(1)
        non_veg_versions = f'{non_veg_versions}__cropland_v{cropland_date}'
        full_slide_text_AFOLU = f'{full_slide_text_AFOLU}{cn.cropland_pres_text};'
    if livestock_geotif_s3:
        livestock_date = re.search(r'/(\d{8})/', livestock_geotif_s3).group(1)
        non_veg_versions = f'{non_veg_versions}__livestock_v{livestock_date}'
        full_slide_text_AFOLU = f'{full_slide_text_AFOLU}{cn.livestock_pres_text};'
    if cropland_geotif_s3 == None and livestock_geotif_s3 == None:
        sys.exit("No Agriculture datasets supplied. Not creating total AFOLU maps.")
    full_slide_text_AFOLU_with_disclaimer = f"{full_slide_text_AFOLU} \n {cn.legend_percentile_disclaimer}"


    # Final combined output
    output_name = f"AFOLU__veg_{veg_version}_{non_veg_versions}__kt_CO2e_yr"
    final_total_path = f"{cn.local_jpeg_folder_AFOLU}/{output_name}.tif"
    with rasterio.open(final_total_path, 'w', **veg_meta) as dst:
        dst.write(AFOLU_net.astype('float32'), 1)

    non_zero_values_AFOLU = AFOLU_net[AFOLU_net != 0]

    main_logger.info(f"\n\n---Preparing legend for AFOLU")

    # Calculates min, center and max across all years
    percentile_for_saturation = 1
    breaks_AFOLU = np.percentile(non_zero_values_AFOLU, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_AFOLU = breaks_AFOLU[0]
    global_neutral_AFOLU = 0
    upper_lim_AFOLU = breaks_AFOLU[-1]

    main_logger.info(f"Across AFOLU:")
    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_AFOLU}")
    main_logger.info(f"  neutral: {global_neutral_AFOLU}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_AFOLU}")

    # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_AFOLU = math.ceil(lower_lim_AFOLU / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_AFOLU = math.floor(upper_lim_AFOLU / 10 ** 3 * 100) / 100  # Rounds down
    tick_labels = [f"< {rounded_lower_lim_AFOLU:.0f}  (sink)",
                   # Spaces are to horizontally align the text explanations
                   "0        (neutral)",
                   f"> {rounded_upper_lim_AFOLU:.0f}  (source)"]
    # print(tick_labels)

    main_logger.info(f"\n\n---Generating AFOLU map:")

    # Reads raster data
    with rasterio.open(final_total_path) as src:

        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj

            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            data = src.read(1, window=window)

            # Update extent from the window
            left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
            raster_extent = (left, right, bottom, top)

        else:
            data = src.read(1)
            b = src.bounds
            raster_extent = (b.left, b.right, b.bottom, b.top)

    # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
    main_logger.info(f"  Calculating percentiles and breaks")
    percentile_0 = mu.percentile_for_0(data)
    main_logger.info(f"  0 is at the {percentile_0}th percentile of the raster.")
    percentiles = [percentile_0 / 6, percentile_0 / 4, percentile_0 / 2, percentile_0 / 1.3, percentile_0 / 1.05,
                   percentile_0 * 1.05, percentile_0 * 1.1, percentile_0 * 1.2, percentile_0 * 1.3, percentile_0 * 1.5]
    # print("percentiles:", percentiles)

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles to a 0-1 scale.
    percentiles_normalized = np.linspace(0, 1, len(percentiles))
    # print("percentiles_normalized:", percentiles_normalized)
    cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

    main_logger.info(f"  Masking raster to non-0 values for AFOLU")
    masked_data = np.ma.masked_where(data == 0, data)

    # For map (not legend)
    norm = TwoSlopeNorm(
        vmin=lower_lim_AFOLU,
        vcenter=global_neutral_AFOLU,
        vmax=upper_lim_AFOLU
    )

    main_logger.info(f"  Plotting AFOLU map")
    ax, fig = mu.create_plot()

    # Sets the ocean color
    mu.set_ocean_color(ax)

    # Limits shapefile to focal extent (if requested)
    if bounding_box_proj is not None:
        bbox_geom = box(*bounding_box_proj)
        country_shapefile = country_shapefile.clip(bbox_geom)

    # Plots the country polygons first
    mu.plot_country_polygons(ax, country_shapefile)

    # Raster extent
    extent = list(raster_extent)

    # Plots the raster next
    img = mu.plot_raster(ax, cmap, extent, masked_data, norm)

    # Plots the country boundaries on top
    mu.plot_country_boundaries(ax, country_shapefile)

    # Explicitly sets the bounding box for the plot image
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Title
    title_text = f"AFOLU net GHG flux\nkt CO$_2$e yr$^{{-1}}$"

    # Creates legend
    mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_AFOLU, rounded_upper_lim_AFOLU,
                                          title_text, tick_labels,
                                          veg_analysis_years, net_colors_rgb, percentiles, percentile_0, main_logger)

    # Removes axis ticks and labels
    mu.remove_ticks(ax)

    core_jpeg_name = f"{output_name}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
    jpeg_path = f"{AFOLU_local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
    jpeg_for_pres_path = f"{AFOLU_local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

    full_slide_text_AFOLU_with_disclaimer = full_slide_text_AFOLU_with_disclaimer.replace("Cropland: vYYYYMMDD", f"Cropland: v{cropland_date}")
    full_slide_text_AFOLU_with_disclaimer = full_slide_text_AFOLU_with_disclaimer.replace("Livestock: vYYYYMMDD", f"Livestock: v{livestock_date}")

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, "", full_slide_text_AFOLU_with_disclaimer, main_logger)

    end_time = time.time()
    main_logger.info(f"AFOLU for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


def main(veg_net_all_gases_geotif_local,
         organic_soil_local=None,
         mineral_soil_s3=None,
         cropland_geotif_s3=None,
         livestock_geotif_s3=None,
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

    # Model stage being run
    stage = 'summative_4x4km_LULUCF_and_or_AFOLU_maps'
    log_note = '4x4 km maps for various AFOLU components'

    # Creates the log for the main function and populates it with basic run information
    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header("NA", "NA", log_note, True, "NA", stage)

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

    # Generates jpegs for LULUCF and AFOLU
    map_AFOLU_totals(veg_net_all_gases_geotif_local,
                     organic_soil_local,
                     mineral_soil_s3,
                     cropland_geotif_s3,
                     livestock_geotif_s3,
                     cn.net_colors_rgb, country_shapefile, bounding_box, bounding_box_description, main_logger)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create jpegs of 0.04x0.04 deg output maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Latitude to show around lat center (value is total north/south) (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box (if used) to include in output names.')

    parser.add_argument('-veg', '--veg_net_all_gases_geotif_local', help='Local vegetation net flux file to use')
    parser.add_argument('-os', '--organic_soil_local', help='Local organic soil emissions file (eventually should come from s3 but added drained and burned together locally')
    parser.add_argument('-ms', '--mineral_soil_s3', help='s3 path for mineral soil net flux')
    parser.add_argument('-cl', '--cropland_geotif_s3', help='s3 path for cropland management emissions')
    parser.add_argument('-ls', '--livestock_geotif_s3', help='s3 path for livestock emissions')


    args = parser.parse_args()
    center_latitude = args.center_latitude
    center_longitude = args.center_longitude
    lat_height = args.lat_height
    bounding_box_description = args.bounding_box_description

    veg_net_all_gases_geotif_local = args.veg_net_all_gases_geotif_local
    organic_soil_local = args.organic_soil_local
    mineral_soil_s3 = args.mineral_soil_s3
    cropland_geotif_s3 = args.cropland_geotif_s3
    livestock_geotif_s3 = args.livestock_geotif_s3

    main(veg_net_all_gases_geotif_local,
         organic_soil_local=organic_soil_local,  # Created by downloading and summing drainage and burning. Erin hasn't made a combined map yet.
         mineral_soil_s3=mineral_soil_s3,
         cropland_geotif_s3=cropland_geotif_s3,
         livestock_geotif_s3=livestock_geotif_s3,
         center_latitude=center_latitude, center_longitude=center_longitude, lat_height=lat_height, bounding_box_description=bounding_box_description)


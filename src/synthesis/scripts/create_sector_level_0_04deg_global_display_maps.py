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
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif

Global AFOLU:
python -m src.synthesis.scripts.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif

For Central Africa:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif
--center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

For Borneo/Sumatra:
python -m src.LULUCF.scripts.vegetation_model.create_sector_level_0_04deg_global_display_maps
-veg /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v1_0_5_standard_global/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2016_2024_mean_global_reproj.tif
-os /mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/veg_v1_0_5_standard_global__org_soil_v_0_9_7__min_soil_v_1_0_0/organic_soil_0_01deg_global__drained_burned_total_Mg_CO2e_pixel_yr_2021_2024.tif
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_0__standard__global/SOC_change__mineral_soil_extent__0-30cm_MgC/2022/_0_04deg_yr/global/20251224/SOC_change__mineral_soil_extent__0-30cm_MgC_0_04deg_yr_v1_0_0_2022_global.tif
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/raw__from_Cornell/20250828/year_2020/all_sources/Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif
--center_latitude 1 --center_longitude 108 --lat_height 12 -bbd Borneo_Sumatra

With https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67634e63-bbcc-800a-8267-004e88ced2e4
Continued at https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
This specific code at https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69778c22-2538-8325-a70e-1a2b70312505

#TODO Average mineral and organic soil values for last two intervals instead of using just the most recent value.
For mineral soil, I could calculate the annual density change for 2015 vs. 2022, rather than averaging the two change values.
For organic soil, I'll have to average the two periods of emissions.
#TODO I'm going to add the sign switching to the mineral soil processing step, so I won't need to do it in this script.
#TODO In the emissions fraction maps, mineral soil fraction is the residual of veg and organic soil for now but I want to calculate it on its own once I have corrected mineral soil
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
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, BoundaryNorm, ListedColormap
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
        meta.update(compress='LZW')

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

    # Gross vegetation emissions
    with rasterio.open(veg_gross_emis_all_gases_local) as src_veg_emis:
        LULUCF_emis = src_veg_emis.read(1).astype('float32')  # base raster to accumulate into for LULUCF emis

    # Gross vegetation removals
    with rasterio.open(veg_gross_remv_all_gases_local) as src_veg_remv:
        LULUCF_remv = src_veg_remv.read(1).astype('float32')  # base raster to accumulate into for LULUCF emis

    # Ensures compression for all outputs
    veg_meta.update({
        "compress": "LZW"
    })


    ### Part 1: Maps average annual vegetation net flux by itself.
    ### Matches the approach used by map_net_flux() in map_utilities.py:
    ### percentile limits are computed from non-zero pixels only, using cn.net_percentiles multipliers.

    main_logger.info("\n\n\n---Part 1: Mapping average annual vegetation net flux:")

    # Reads the mean annual vegetation net flux raster (already reprojected), applying bbox clipping if requested.
    # This also establishes raster_extent for use in subsequent parts.
    with rasterio.open(veg_net_all_gases_geotif_local) as src:
        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data_veg_net = src.read(1, window=window).astype('float32')
            left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
            raster_extent = (left, right, bottom, top)
        else:
            data_veg_net = src.read(1).astype('float32')
            b = src.bounds
            raster_extent = (b.left, b.right, b.bottom, b.top)

    # Percentile limits from non-zero pixels only (matching map_net_flux behavior)
    non_zero_values_veg_net = data_veg_net[data_veg_net != 0]
    percentile_for_saturation = 1
    breaks_veg_net = np.percentile(non_zero_values_veg_net, [percentile_for_saturation, (100 - percentile_for_saturation)])
    lower_lim_veg_net = breaks_veg_net[0]
    upper_lim_veg_net = breaks_veg_net[-1]

    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_veg_net}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_veg_net}")

    rounded_lower_lim_veg_net = math.ceil(lower_lim_veg_net / 10 ** 3 * 100) / 100
    rounded_upper_lim_veg_net = math.floor(upper_lim_veg_net / 10 ** 3 * 100) / 100
    tick_labels_veg_net = [f"< {rounded_lower_lim_veg_net:.0f}  (sink)",
                           "0        (neutral)",
                           f"> {rounded_upper_lim_veg_net:.0f}  (source)"]

    # Colormap percentile breaks using cn.net_percentiles multipliers (same as map_net_flux)
    percentile_0_veg_net = mu.percentile_for_0(data_veg_net)
    main_logger.info(f"  0 is at the {percentile_0_veg_net}th percentile of the vegetation net flux raster.")
    percentiles_veg_net = [percentile_0_veg_net * cn.net_percentiles[0], percentile_0_veg_net * cn.net_percentiles[1],
                           percentile_0_veg_net * cn.net_percentiles[2], percentile_0_veg_net * cn.net_percentiles[3],
                           percentile_0_veg_net * cn.net_percentiles[4], percentile_0_veg_net * cn.net_percentiles[5],
                           percentile_0_veg_net * cn.net_percentiles[6], percentile_0_veg_net * cn.net_percentiles[7],
                           percentile_0_veg_net * cn.net_percentiles[8], percentile_0_veg_net * cn.net_percentiles[9]]

    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)
    percentiles_normalized_veg_net = np.linspace(0, 1, len(percentiles_veg_net))
    cmap_veg_net = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_veg_net, colors_matplotlib)))

    masked_data_veg_net = np.ma.masked_where(data_veg_net == 0, data_veg_net)

    norm_veg_net = TwoSlopeNorm(
        vmin=lower_lim_veg_net,
        vcenter=0,
        vmax=upper_lim_veg_net
    )

    ax, fig_veg_net = mu.create_plot()
    mu.set_ocean_color(ax)

    if bounding_box_proj is not None:
        bbox_geom = box(*bounding_box_proj)
        country_shapefile = country_shapefile.clip(bbox_geom)

    mu.plot_country_polygons(ax, country_shapefile)

    extent = list(raster_extent)
    mu.plot_raster(ax, cmap_veg_net, extent, masked_data_veg_net, norm_veg_net)
    mu.plot_country_boundaries(ax, country_shapefile)

    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    title_text_veg_net = f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
    mu.create_divergent_legend_asymmetric(fig_veg_net, rounded_lower_lim_veg_net, rounded_upper_lim_veg_net,
                                          title_text_veg_net, tick_labels_veg_net,
                                          veg_analysis_years, net_colors_rgb, percentiles_veg_net, percentile_0_veg_net, main_logger)
    mu.remove_ticks(ax)

    core_jpeg_name_veg_net = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{veg_analysis_years}__ktCO2e_yr__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_veg_net = f"{core_jpeg_name_veg_net}_{bounding_box_description}"
    jpeg_path_veg_net = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_veg_net}.jpeg"
    jpeg_for_pres_path_veg_net = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_veg_net}__for_pres.jpeg"

    mu.save_pres_non_pres_jpegs(ax, jpeg_path_veg_net, jpeg_for_pres_path_veg_net, "", cn.veg_pres_text, main_logger)

    end_time = time.time()
    main_logger.info(f"Vegetation net flux for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


    main_logger.info(f"\n\n\n---Part 2: Combining individual datasets with vegetation net flux")
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
        #TODO I'm going to add the sign switching to the mineral soil processing step, so I won't need to do it here. Not adding unit conversion (C to CO2) to mineral soil script, though, so still need to do that there.
        if key == "mineral_soil":
            main_logger.info("Adding mineral soil data to gross emissions and removals")
            additional_data = additional_data * cn.C_to_CO2  # Converts mineral soil SOC change from Mg C/yr to Mg CO2/yr
            additional_data = additional_data * -1  # Converts mineral soil SOC change to positive for loss and negative for gain

            # Splits mineral soil into separate loss and gain arrays
            SOC_loss = np.where(additional_data > 0, additional_data, 0)
            SOC_gain = np.where(additional_data < 0, additional_data, 0)

            # Includes SOC loss with LULUCF emissions and SOC gain with LULUCF removals
            LULUCF_emis += SOC_loss
            LULUCF_remv += SOC_gain

        # Adds emissions from organic soil to LULUCF gross emissions total
        if key == "organic_soil":
            main_logger.info("Adding organic soil to emissions to LULUCF gross emissions")
            LULUCF_emis += additional_data

        # Only adds soil data to running LULUCF net total
        if "soil" in unit_converted_path:
            LULUCF_net += additional_data

        # Adds all datasets to running AFOLU total
        AFOLU_net += additional_data

        main_logger.info(f"Combining vegetation net flux and {key}")
        output_name_veg_pairwise = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{key}_{additional_data_date}__{veg_analysis_years}__MgCO2e_yr"
        output_sum_path_veg_pairwise = f"{local_reproj_folder}/{output_name_veg_pairwise}.tif"
        main_logger.info(f"Combined vegetation and {key} at {output_sum_path_veg_pairwise}")

        # Sums the vegetation net flux and other data
        non_zero_values_veg_pairwise = add_veg_and_other_data(output_sum_path_veg_pairwise, additional_data, veg_net_all_gases_geotif_local, main_logger)


        main_logger.info(f"\n\n---Preparing legend")

        # Calculates min, center and max across all years
        percentile_for_saturation = 1
        breaks_all_yrs_veg_pairwise = np.percentile(non_zero_values_veg_pairwise, [1, (100-percentile_for_saturation)])  # The min and max percentiles at which colors saturate

        lower_lim_all_yrs_veg_pairwise = breaks_all_yrs_veg_pairwise[0]
        global_neutral_veg_pairwise = 0
        upper_lim_all_yrs_veg_pairwise = breaks_all_yrs_veg_pairwise[-1]

        main_logger.info(f"Across vegetation+{key}:")
        main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_all_yrs_veg_pairwise}")
        main_logger.info(f"  neutral: {global_neutral_veg_pairwise}")
        main_logger.info(f"  upper limit ({(100-percentile_for_saturation)} percentile): {upper_lim_all_yrs_veg_pairwise}")

        # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
        # Rounds data_min down and data_max up for legend.
        rounded_lower_lim_all_yrs_veg_pairwise = math.ceil(lower_lim_all_yrs_veg_pairwise / 10 ** 3 * 100) / 100  # Rounds up
        rounded_upper_lim_all_yrs_veg_pairwise = math.floor(upper_lim_all_yrs_veg_pairwise / 10 ** 3 * 100) / 100  # Rounds down
        tick_labels_veg_pairwise = [f"< {rounded_lower_lim_all_yrs_veg_pairwise:.0f}  (sink)",  # Spaces are to horizontally align the text explanations
                       "0        (neutral)",
                       f"> {rounded_upper_lim_all_yrs_veg_pairwise:.0f}  (source)"]
        output_name_kt_veg_pairwise = output_name_veg_pairwise.replace("MgCO2", "ktCO2")
        # print(output_name_kt_veg_pairwise)


        main_logger.info(f"\n\n---Mapping vegetation + {key}")

        # Reads raster data
        with rasterio.open(output_sum_path_veg_pairwise) as src:

            if bounding_box_proj is not None:
                minx, miny, maxx, maxy = bounding_box_proj

                window = from_bounds(minx, miny, maxx, maxy, src.transform)

                data_veg_pairwise = src.read(1, window=window)

                # Update extent from the window
                left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
                raster_extent = (left, right, bottom, top)

            else:
                data_veg_pairwise = src.read(1)
                b = src.bounds
                raster_extent = (b.left, b.right, b.bottom, b.top)

        # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
        main_logger.info(f"  Calculating percentiles_veg_pairwise and breaks")
        percentile_0_veg_pairwise = mu.percentile_for_0(data_veg_pairwise)
        main_logger.info(f"  0 is at the {percentile_0_veg_pairwise}th percentile of the raster.")
        percentiles_veg_pairwise = [percentile_0_veg_pairwise / 6, percentile_0_veg_pairwise / 4, percentile_0_veg_pairwise / 2, percentile_0_veg_pairwise / 1.3, percentile_0_veg_pairwise / 1.05,
                       percentile_0_veg_pairwise * 1.05, percentile_0_veg_pairwise * 1.1, percentile_0_veg_pairwise * 1.2, percentile_0_veg_pairwise * 1.3, percentile_0_veg_pairwise * 1.5]
        # print("percentiles_veg_pairwise:", percentiles_veg_pairwise)

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles_veg_pairwise to a 0-1 scale.
        percentiles_normalized_veg_pairwise = np.linspace(0, 1, len(percentiles_veg_pairwise))
        # print("percentiles_normalized_veg_pairwise:", percentiles_normalized_veg_pairwise)
        cmap_veg_pairwise = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_veg_pairwise, colors_matplotlib)))

        main_logger.info(f"  Masking raster to non-0 values")
        masked_data_veg_pairwise = np.ma.masked_where(data_veg_pairwise == 0, data_veg_pairwise)

        # For map (not legend)
        norm_veg_pairwise = TwoSlopeNorm(
            vmin=lower_lim_all_yrs_veg_pairwise,
            vcenter=global_neutral_veg_pairwise,
            vmax=upper_lim_all_yrs_veg_pairwise
        )

        main_logger.info(f"  Plotting map")
        ax, fig_veg_pairwise = mu.create_plot()

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
        img_veg_pairwise = mu.plot_raster(ax, cmap_veg_pairwise, extent, masked_data_veg_pairwise, norm_veg_pairwise)

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
        mu.create_divergent_legend_asymmetric(fig_veg_pairwise, rounded_lower_lim_all_yrs_veg_pairwise, rounded_upper_lim_all_yrs_veg_pairwise,
                                              title_text, tick_labels_veg_pairwise,
                                              veg_analysis_years, net_colors_rgb, percentiles_veg_pairwise, percentile_0_veg_pairwise, main_logger)

        # Removes axis ticks and labels
        mu.remove_ticks(ax)

        core_jpeg_name_veg_pairwise = f"{output_name_kt_veg_pairwise}__{uu.timestr()[0:8]}"
        if bounding_box_description:  # Adds bounding box description to file name, if supplied
            core_jpeg_name_veg_pairwise = f"{core_jpeg_name_veg_pairwise}_{bounding_box_description}"
        jpeg_path_veg_pairwise = f"{non_pres_folder}/{core_jpeg_name_veg_pairwise}.jpeg"
        jpeg_for_pres_path_veg_pairwise = f"{pres_folder}/{core_jpeg_name_veg_pairwise}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        full_slide_text_LULUCF = f"{cn.veg_pres_text}; {presentation_slide_text} \n {cn.legend_percentile_disclaimer}"
        veg_addtl_pres_text = full_slide_text_LULUCF.replace("YYYYMMDD", additional_data_date)  # For livestock and cropland, whose versions are dates
        out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path_veg_pairwise, jpeg_for_pres_path_veg_pairwise, "", veg_addtl_pres_text, main_logger)

        end_time = time.time()
        main_logger.info(f"vegetation+{key} {bounding_box_description} took {round(end_time - start_time)} seconds: {uu.timestr()}")


    ### Part 3: Maps net LULUCF

    main_logger.info("\n\n\n---Part 3: Mapping net LULUCF:")

    # Iteratively collects the names and versions of non-vegetation datasets, and text for bottom-right of maps
    non_veg_versions = ''
    full_slide_text_LULUCF = f'{cn.veg_pres_text}'
    if organic_soil_local:
        non_veg_versions = f'{non_veg_versions}_organic_soil_v{cn.organic_soil_model_version_underscore}'
        full_slide_text_LULUCF = f'{full_slide_text_LULUCF}; {cn.organic_soil_pres_text}'
    if mineral_soil_s3:
        non_veg_versions = f'{non_veg_versions}_mineral_soil_v{cn.SOC_model_version_underscore}'
        full_slide_text_LULUCF = f'{full_slide_text_LULUCF}; {cn.mineral_soil_pres_text}'
    full_slide_text_LULUCF_with_disclaimer = f"{full_slide_text_LULUCF} \n {cn.legend_percentile_disclaimer}"

    # Final combined output
    LULUCF_net_output_name = f"LULUCF_net__veg_{veg_version}__{non_veg_versions}__MgCO2e_yr"
    # print("LULUCF_net_output_name:", LULUCF_net_output_name)
    LULUCF_net_final_total_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_net_output_name}.tif"
    # print("LULUCF_net_final_total_path:", LULUCF_net_final_total_path)
    with rasterio.open(LULUCF_net_final_total_path, 'w', **veg_meta) as dst:
        dst.write(LULUCF_net.astype('float32'), 1)

    non_zero_values_LULUCF_net = LULUCF_net[LULUCF_net != 0]

    main_logger.info(f"\n\n---Preparing net LULUCF legend")

    # Calculates min, center and max across all years
    percentile_for_saturation = 1
    breaks_LULUCF_net = np.percentile(non_zero_values_LULUCF_net, [1, (100 - percentile_for_saturation)])  # The min and max percentiles_LULUCF_net at which colors saturate

    lower_lim_LULUCF_net = breaks_LULUCF_net[0]
    global_neutral_LULUCF_net = 0
    upper_lim_LULUCF_net = breaks_LULUCF_net[-1]

    main_logger.info(f"Across net LULUCF:")
    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_LULUCF_net}")
    main_logger.info(f"  neutral: {global_neutral_LULUCF_net}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_LULUCF_net}")

    # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_LULUCF_net = math.ceil(lower_lim_LULUCF_net / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_LULUCF_net = math.floor(upper_lim_LULUCF_net / 10 ** 3 * 100) / 100  # Rounds down
    tick_labels_LULUCF_net = [f"< {rounded_lower_lim_LULUCF_net:.0f}  (sink)",
                   # Spaces are to horizontally align the text explanations
                   "0        (neutral)",
                   f"> {rounded_upper_lim_LULUCF_net:.0f}  (source)"]
    LULUCF_net_output_name_kt = LULUCF_net_output_name.replace("MgCO2", "ktCO2")
    # print(tick_labels_LULUCF_net)

    main_logger.info(f"\n\n---Generating net LULUCF map:")

    # Reads raster data
    with rasterio.open(LULUCF_net_final_total_path) as src:

        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj

            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            data_LULUCF_net = src.read(1, window=window)

            # Update extent from the window
            left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
            raster_extent = (left, right, bottom, top)

        else:
            data_LULUCF_net = src.read(1)
            b = src.bounds
            raster_extent = (b.left, b.right, b.bottom, b.top)

    # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
    main_logger.info(f"  Calculating percentiles and breaks for net LULUCF")
    percentile_0_LULUCF_net = mu.percentile_for_0(data_LULUCF_net)
    main_logger.info(f"  0 is at the {percentile_0_LULUCF_net}th percentile of the raster.")
    percentiles_LULUCF_net = [percentile_0_LULUCF_net / 6, percentile_0_LULUCF_net / 4, percentile_0_LULUCF_net / 2, percentile_0_LULUCF_net / 1.3, percentile_0_LULUCF_net / 1.05,
                   percentile_0_LULUCF_net * 1.05, percentile_0_LULUCF_net * 1.1, percentile_0_LULUCF_net * 1.2, percentile_0_LULUCF_net * 1.3, percentile_0_LULUCF_net * 1.5]
    # print("percentiles_LULUCF_net:", percentiles_LULUCF_net)

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles_LULUCF_net to a 0-1 scale.
    percentiles_normalized_LULUCF_net = np.linspace(0, 1, len(percentiles_LULUCF_net))
    # print("percentiles_normalized_LULUCF_net:", percentiles_normalized_LULUCF_net)
    cmap_LULUCF_net = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_LULUCF_net, colors_matplotlib)))

    main_logger.info(f"  Masking raster to non-0 values for net LULUCF")
    masked_data_LULUCF_net = np.ma.masked_where(data_LULUCF_net == 0, data_LULUCF_net)

    # For map (not legend)
    norm_LULUCF_net = TwoSlopeNorm(
        vmin=lower_lim_LULUCF_net,
        vcenter=global_neutral_LULUCF_net,
        vmax=upper_lim_LULUCF_net
    )

    main_logger.info(f"  Plotting net LULUCF map")
    ax, fig_LULUCF_net = mu.create_plot()

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
    img_LULUCF_net = mu.plot_raster(ax, cmap_LULUCF_net, extent, masked_data_LULUCF_net, norm_LULUCF_net)

    # Plots the country boundaries on top
    mu.plot_country_boundaries(ax, country_shapefile)

    # Explicitly sets the bounding box for the plot image
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Title
    title_text = f"Net LULUCF flux\nkt CO$_2$e yr$^{{-1}}$"

    # Creates legend
    mu.create_divergent_legend_asymmetric(fig_LULUCF_net, rounded_lower_lim_LULUCF_net, rounded_upper_lim_LULUCF_net,
                                          title_text, tick_labels_LULUCF_net,
                                          veg_analysis_years, net_colors_rgb, percentiles_LULUCF_net, percentile_0_LULUCF_net, main_logger)

    # Removes axis ticks and labels
    mu.remove_ticks(ax)

    core_jpeg_name_LULUCF_net = f"{LULUCF_net_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name_LULUCF_net = f"{core_jpeg_name_LULUCF_net}_{bounding_box_description}"
    jpeg_path_LULUCF_net = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_LULUCF_net}.jpeg"
    jpeg_for_pres_path_LULUCF_net = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_LULUCF_net}__for_pres.jpeg"

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path_LULUCF_net, jpeg_for_pres_path_LULUCF_net, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    end_time = time.time()
    main_logger.info(f"LULUCF net for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


    ### Part 4: Maps LULUCF gross emissions

    main_logger.info("\n\n\n---Part 4: Mapping LULUCF gross emissions and removals:")

    # Gross LULUCF emissions

    LULUCF_emis_output_name = f"LULUCF_gross_emis__veg_{veg_version}__{non_veg_versions}__MgCO2e_yr"
    # print("LULUCF_emis_output_name:", LULUCF_emis_output_name)
    LULUCF_gross_emis_final_total_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_emis_output_name}.tif"

    with rasterio.open(LULUCF_gross_emis_final_total_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_emis, 1)

    main_logger.info(f"  Removing 0s from LULUCF emissions for legend breakpoints")
    masked_data_for_legend_LULUCF_emis = LULUCF_emis[LULUCF_emis != 0]  # Removes 0s but doesn't actually mask-- creates 1D array for legend breakpoints

    main_logger.info(f"\n\n---Preparing LULUCF emissions legend")

    ax, fig_LULUCF_emis = mu.create_plot()
    mu.set_ocean_color(ax)

    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)  # Use the extent from last year (they should all match)

    # Matches percentile breaks with colors.
    # Normalizes percentiles to a 0-1 scale.
    main_logger.info(f"  Calculating percentiles and breaks for LULUCF gross emissions")

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(cn.emissions_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles to a 0-1 scale.
    percentiles_normalized_LULUCF_emis = np.linspace(0, 1, len(cn.emissions_percentiles))
    # print("percentiles_normalized_LULUCF_emis:", percentiles_normalized_LULUCF_emis)
    cmap_LULUCF_emis = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_LULUCF_emis, colors_matplotlib)))

    percentile_for_saturation = 1
    breaks_all_yrs_LULUCF_emis = np.percentile(masked_data_for_legend_LULUCF_emis, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_all_yrs_LULUCF_emis = breaks_all_yrs_LULUCF_emis[0]
    upper_lim_all_yrs_LULUCF_emis = breaks_all_yrs_LULUCF_emis[-1]

    # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_upper_lim_all_yrs_LULUCF_emis = math.floor(upper_lim_all_yrs_LULUCF_emis / 10 ** 3 * 100) / 100  # Rounds down

    # Legend labels depend on what exact input is displayed
    tick_labels_LULUCF_emis = [0, f"> {rounded_upper_lim_all_yrs_LULUCF_emis:.0f}"]
    title_text_LULUCF_emis = f"Gross LULUCF emissions\nkt CO$_2$e yr$^{{-1}}$"
    main_logger.info(f"tick labels {tick_labels_LULUCF_emis}")

    norm_LULUCF_emis = Normalize(vmin=lower_lim_all_yrs_LULUCF_emis, vmax=upper_lim_all_yrs_LULUCF_emis)

    # Masks data for emissions mapping (different from removing 0s for legend percentiles)
    masked_data_for_map_LULUCF_emis = np.ma.masked_where(LULUCF_emis <= 0, LULUCF_emis)

    img_LULUCF_emis = mu.plot_raster(ax, cmap_LULUCF_emis, extent, masked_data_for_map_LULUCF_emis, norm_LULUCF_emis)
    mu.plot_country_boundaries(ax, country_shapefile)

    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Legend for gross fluxes
    mu.create_unidirection_legend(fig_LULUCF_emis, img_LULUCF_emis, lower_lim_all_yrs_LULUCF_emis, upper_lim_all_yrs_LULUCF_emis,
                               title_text_LULUCF_emis, tick_labels_LULUCF_emis,
                               'avg', cn.emissions_colors_rgb, cn.emissions_percentiles, main_logger)

    mu.remove_ticks(ax)

    # Saves LULUCF gross emissions JPEG
    LULUCF_emis_output_name_kt = LULUCF_emis_output_name.replace("MgCO2", "ktCO2")
    core_jpeg_name_LULUCF_emis = f"{LULUCF_emis_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name_LULUCF_emis = f"{core_jpeg_name_LULUCF_emis}_{bounding_box_description}"
    jpeg_path_LULUCF_emis = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_LULUCF_emis}.jpeg"
    jpeg_for_pres_path_LULUCF_emis = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_LULUCF_emis}__for_pres.jpeg"

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path_LULUCF_emis, jpeg_for_pres_path_LULUCF_emis, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    end_time = time.time()
    main_logger.info(f"LULUCF emissions for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


    # Gross LULUCF removals

    LULUCF_remv_output_name = f"LULUCF_gross_remv__veg_{veg_version}__{non_veg_versions}__MgCO2e_yr"
    # print("LULUCF_remv_output_name:", LULUCF_remv_output_name)
    LULUCF_gross_remv_final_total_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_remv_output_name}.tif"

    with rasterio.open(LULUCF_gross_remv_final_total_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_remv, 1)

    main_logger.info(f"  Removing 0s from LULUCF removals for legend breakpoints")
    masked_data_for_legend_LULUCF_remv = LULUCF_remv[LULUCF_remv != 0]  # Removes 0s but doesn't actually mask-- creates 1D array for legend breakpoints

    main_logger.info(f"\n\n---Preparing LULUCF removals legend")

    ax, fig_LULUCF_remv = mu.create_plot()
    mu.set_ocean_color(ax)

    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)  # Use the extent from last year (they should all match)

    # Matches percentile breaks with colors.
    # Normalizes percentiles to a 0-1 scale.
    main_logger.info(f"  Calculating percentiles and breaks for LULUCF gross removals")

    # Converts RGB color palette to matplotlib color palette
    colors_matplotlib = mu.rgb_to_mpl_palette(cn.removals_colors_rgb)

    # Matches percentile breaks with colors for the map.
    # Normalizes percentiles to a 0-1 scale.
    percentiles_normalized_LULUCF_remv = np.linspace(0, 1, len(cn.removals_percentiles))
    # print("percentiles_normalized_LULUCF_remv:", percentiles_normalized_LULUCF_remv)
    cmap_LULUCF_remv = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_LULUCF_remv, colors_matplotlib)))

    percentile_for_saturation = 1
    breaks_all_yrs_LULUCF_remv = np.percentile(masked_data_for_legend_LULUCF_remv, [1, (100 - percentile_for_saturation)])  # The min and max percentiles at which colors saturate

    lower_lim_all_yrs_LULUCF_remv = breaks_all_yrs_LULUCF_remv[0]
    upper_lim_all_yrs_LULUCF_remv = breaks_all_yrs_LULUCF_remv[-1]

    # Creates the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # Rounds data_min down and data_max up for legend.
    rounded_lower_lim_all_yrs_LULUCF_remv = math.ceil(lower_lim_all_yrs_LULUCF_remv / 10 ** 3 * 100) / 100  # Rounds up
    rounded_upper_lim_all_yrs_LULUCF_remv = math.floor(upper_lim_all_yrs_LULUCF_remv / 10 ** 3 * 100) / 100  # Rounds down

    # Legend labels depend on what exact input is displayed
    tick_labels_LULUCF_remv = [f"< {rounded_lower_lim_all_yrs_LULUCF_remv:.0f}", 0]
    title_text_LULUCF_remv = f"Gross LULUCF removals\nkt CO$_2$ yr$^{{-1}}$"
    main_logger.info(f"tick labels {tick_labels_LULUCF_remv}")

    norm_LULUCF_remv = Normalize(vmin=lower_lim_all_yrs_LULUCF_remv, vmax=upper_lim_all_yrs_LULUCF_remv)

    # Masks data for removals mapping (different from removing 0s for legend percentiles)
    masked_data_for_map_LULUCF_remv = np.ma.masked_where(LULUCF_remv >= 0, LULUCF_remv)

    img_LULUCF_remv = mu.plot_raster(ax, cmap_LULUCF_remv, extent, masked_data_for_map_LULUCF_remv, norm_LULUCF_remv)
    mu.plot_country_boundaries(ax, country_shapefile)

    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Legend for gross fluxes
    mu.create_unidirection_legend(fig_LULUCF_remv, img_LULUCF_remv, lower_lim_all_yrs_LULUCF_remv, upper_lim_all_yrs_LULUCF_remv,
                               title_text_LULUCF_remv, tick_labels_LULUCF_remv,
                               'avg', cn.removals_colors_rgb, cn.removals_percentiles, main_logger)

    mu.remove_ticks(ax)

    # Saves LULUCF gross removals JPEG
    LULUCF_remv_output_name_kt = LULUCF_remv_output_name.replace("MgCO2", "ktCO2")
    core_jpeg_name_LULUCF_remv = f"{LULUCF_remv_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name_LULUCF_remv = f"{core_jpeg_name_LULUCF_remv}_{bounding_box_description}"
    jpeg_path_LULUCF_remv = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_LULUCF_remv}.jpeg"
    jpeg_for_pres_path_LULUCF_remv = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_LULUCF_remv}__for_pres.jpeg"

    # Saves two versions of the map: without and with a source note in the bottom right
    out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path_LULUCF_remv, jpeg_for_pres_path_LULUCF_remv, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    end_time = time.time()
    main_logger.info(f"LULUCF removals for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


    ### Part 5: Three-panel map of LULUCF (gross emissions, gross removals, net flux)

    main_logger.info("\n\n\n---Part 5: Making three-panel LULUCF map (gross emissions, gross removals, net flux")

    # Saves LULUCF three-panel map JPEG
    LULUCF_three_panel_output_name_kt = f"LULUCF_three_panel__emis_remv_net__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    core_jpeg_name_LULUCF_three_panel = f"{LULUCF_three_panel_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:  # Adds bounding box description to file name, if supplied
        core_jpeg_name_LULUCF_three_panel = f"{core_jpeg_name_LULUCF_three_panel}_{bounding_box_description}"
    jpeg_path_LULUCF_three_panel = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_LULUCF_three_panel}.jpeg"
    mu.create_three_panel_map(jpeg_path_LULUCF_three_panel, jpeg_path_LULUCF_emis, jpeg_path_LULUCF_remv, jpeg_path_LULUCF_net, "", main_logger)


    ### Part 6: Maps fraction of LULUCF gross emissions due to vegetation, mineral soil, and organic soil

    main_logger.info(f"\n\n\n---Part 6: Mapping fraction of gross emissions from organic soil: {uu.timestr()}")

    # Loads gross vegetation emissions raster
    with rasterio.open(veg_gross_emis_all_gases_local) as src_veg_emis:
        veg_emis = src_veg_emis.read(1).astype('float32')

    # Loads reprojected organic soil emissions raster
    organic_soil_local_reproj = organic_soil_local.replace(".tif", "_reproj.tif")
    print("organic_soil_local_reproj:", organic_soil_local_reproj)
    with rasterio.open(organic_soil_local_reproj) as src:
        org_soil_emis = src.read(1).astype('float32')

    # Fraction of gross emissions due to each component
    LULUCF_emis_fract_veg =  veg_emis / LULUCF_emis
    LULUCF_emis_fract_org_soil =  org_soil_emis / LULUCF_emis
    #TODO Mineral soil is the residual of veg and organic soil for now but I want to calculate it on its own once I have corrected mineral soil
    LULUCF_emis_fract_min_soil =  1 - LULUCF_emis_fract_veg - LULUCF_emis_fract_org_soil

    # Saves fraction rasters as GeoTIFFs
    LULUCF_emis_fract_veg_output_name = f"LULUCF_emis_fract_veg_{veg_version}__{non_veg_versions}"
    LULUCF_emis_fract_veg_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_emis_fract_veg_output_name}.tif"
    with rasterio.open(LULUCF_emis_fract_veg_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_emis_fract_veg, 1)

    LULUCF_emis_fract_org_soil_output_name = f"LULUCF_emis_fract_org_soil_{veg_version}__{non_veg_versions}"
    LULUCF_emis_fract_org_soil_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_emis_fract_org_soil_output_name}.tif"
    with rasterio.open(LULUCF_emis_fract_org_soil_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_emis_fract_org_soil, 1)

    LULUCF_emis_fract_min_soil_output_name = f"LULUCF_emis_fract_min_soil_{veg_version}__{non_veg_versions}"
    LULUCF_emis_fract_min_soil_path = f"{cn.local_jpeg_folder_LULUCF}/{LULUCF_emis_fract_min_soil_output_name}.tif"
    with rasterio.open(LULUCF_emis_fract_min_soil_path, "w", **veg_meta) as dst:
        dst.write(LULUCF_emis_fract_min_soil, 1)

    # Map creation with Claude ("Create LULUCF emissions fraction visualizations")
    # Categorical colormap: equal-interval classes (0–1), black for out-of-range (<0)
    fract_boundaries = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
    n_classes = len(fract_boundaries) - 1
    fract_cmap = cm.get_cmap(cn.fraction_base_cmap, n_classes).copy()  # Claude notes that .copy() is important
    fract_cmap.set_under('black')
    fract_norm = BoundaryNorm(fract_boundaries, fract_cmap.N)
    fract_class_labels = (
        [f">0–{int(fract_boundaries[1] * 100)}%"] +
        [f"{int(fract_boundaries[i] * 100)}–{int(fract_boundaries[i + 1] * 100)}%"
         for i in range(1, len(fract_boundaries) - 1)]
    )

    # Raster, output jpeg name, and legend title for each map
    fract_maps = [
        (LULUCF_emis_fract_veg, LULUCF_emis_fract_veg_output_name, "Fraction gross LULUCF emissions: \nVegetation"),
        (LULUCF_emis_fract_org_soil, LULUCF_emis_fract_org_soil_output_name, "Fraction gross LULUCF emissions: \nOrganic soil"),
        (LULUCF_emis_fract_min_soil, LULUCF_emis_fract_min_soil_output_name, "Fraction gross LULUCF emissions: \nMineral soil"),
    ]

    # Iterates through LULUCF components, collecting JPEG paths for the three-panel map
    jpeg_paths_fract_no_legend = []
    jpeg_paths_fract = []

    for i, (fract_data, output_name, legend_title) in enumerate(fract_maps):

        main_logger.info(f"\n  Creating fraction map: {output_name}")

        masked_fract = np.ma.masked_where(
            (LULUCF_emis <= 0) | ~np.isfinite(fract_data) | (fract_data == 0),
            fract_data
        )

        ax, fig_fract = mu.create_plot()
        mu.set_ocean_color(ax)
        mu.plot_country_polygons(ax, country_shapefile)

        img_fract = mu.plot_raster(ax, fract_cmap, list(raster_extent), masked_fract, fract_norm)
        mu.plot_country_boundaries(ax, country_shapefile)

        if bounding_box_proj is not None:
            ax.set_xlim(raster_extent[0], raster_extent[1])
            ax.set_ylim(raster_extent[2], raster_extent[3])

        mu.remove_ticks(ax)

        core_jpeg_name_fract = f"{output_name}__{uu.timestr()[0:8]}"
        if bounding_box_description:
            core_jpeg_name_fract = f"{core_jpeg_name_fract}_{bounding_box_description}"
        jpeg_path_fract_no_legend = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_fract}__no_legend.jpeg"
        jpeg_path_fract = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_fract}.jpeg"
        jpeg_for_pres_path_fract = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_fract}__for_pres.jpeg"

        # Save without legend as source for three-panel top/middle panels
        mu.save_jpeg(jpeg_path_fract_no_legend, "", main_logger)

        # Add legend and save individual map
        mu.create_categorical_fraction_legend(fig_fract, img_fract, legend_title, fract_boundaries, fract_class_labels, main_logger)
        mu.save_pres_non_pres_jpegs(ax, jpeg_path_fract, jpeg_for_pres_path_fract, "",
                                    full_slide_text_LULUCF, main_logger)

        jpeg_paths_fract_no_legend.append(jpeg_path_fract_no_legend)
        jpeg_paths_fract.append(jpeg_path_fract)

    # Three-panel map: no-legend versions for top two panels, with-legend for bottom
    LULUCF_fract_three_panel_output_name = f"LULUCF_emis_fract_three_panel_{veg_version}__{non_veg_versions}"
    core_jpeg_name_fract_three_panel = f"{LULUCF_fract_three_panel_output_name}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_fract_three_panel = f"{core_jpeg_name_fract_three_panel}_{bounding_box_description}"
    jpeg_path_fract_three_panel = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_fract_three_panel}.jpeg"

    mu.create_three_panel_map(
        jpeg_path_fract_three_panel,
        jpeg_paths_fract_no_legend[0], jpeg_paths_fract_no_legend[1], jpeg_paths_fract[2],
        "",
        main_logger,
        panel_labels=["a  Vegetation", "b  Organic soil", "c  Mineral soil"]
    )


    ### Part 7: Four-panel map of LULUCF component fluxes
    ### (a) net vegetation flux  (b) net mineral soil SOC change
    ### (c) gross organic soil emissions  (d) net LULUCF flux
    ### Panels a and d reuse JPEGs from Part 1 (jpeg_path_veg_net) and Part 3 (jpeg_path_LULUCF_net).
    ### Panels b and c are created here.

    main_logger.info("\n\n\n---Part 7: Mapping LULUCF component fluxes and net LULUCF (four-panel):")


    # --- Panel b: Net mineral soil SOC change (divergent colormap) ---

    main_logger.info("\n---Mapping net mineral soil SOC change:")

    # Derives the reprojected mineral soil path — same path produced by reproject_to_vegetation() in Part 2
    mineral_soil_reproj_path = f"{cn.local_jpeg_folder_LULUCF}/{os.path.splitext(os.path.basename(mineral_soil_s3))[0]}_reproj.tif"

    with rasterio.open(mineral_soil_reproj_path) as src:
        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data_min_soil = src.read(1, window=window).astype('float32')
        else:
            data_min_soil = src.read(1).astype('float32')

    # Applies same transformations as Part 2: Mg C/yr → Mg CO2/yr, then sign flip (loss positive, gain negative)
    data_min_soil = data_min_soil * cn.C_to_CO2 * -1

    non_zero_min_soil = data_min_soil[data_min_soil != 0]
    percentile_for_saturation = 1
    breaks_min_soil = np.percentile(non_zero_min_soil, [1, (100 - percentile_for_saturation)])
    lower_lim_min_soil = breaks_min_soil[0]
    upper_lim_min_soil = breaks_min_soil[-1]

    rounded_lower_lim_min_soil = math.ceil(lower_lim_min_soil / 10 ** 3 * 100) / 100
    rounded_upper_lim_min_soil = math.floor(upper_lim_min_soil / 10 ** 3 * 100) / 100
    tick_labels_min_soil = [f"< {rounded_lower_lim_min_soil:.0f}  (sink)",
                            "0        (neutral)",
                            f"> {rounded_upper_lim_min_soil:.0f}  (source)"]

    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_min_soil}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_min_soil}")

    percentile_0_min_soil = mu.percentile_for_0(data_min_soil)
    main_logger.info(f"  0 is at the {percentile_0_min_soil}th percentile of the mineral soil raster.")
    percentiles_min_soil = [percentile_0_min_soil / 6, percentile_0_min_soil / 4, percentile_0_min_soil / 2,
                            percentile_0_min_soil / 1.3, percentile_0_min_soil / 1.05,
                            percentile_0_min_soil * 1.05, percentile_0_min_soil * 1.1,
                            percentile_0_min_soil * 1.2, percentile_0_min_soil * 1.3, percentile_0_min_soil * 1.5]

    colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)
    percentiles_normalized_min_soil = np.linspace(0, 1, len(percentiles_min_soil))
    cmap_min_soil = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_min_soil, colors_matplotlib)))
    masked_data_min_soil = np.ma.masked_where(data_min_soil == 0, data_min_soil)
    norm_min_soil = TwoSlopeNorm(vmin=lower_lim_min_soil, vcenter=0, vmax=upper_lim_min_soil)

    ax, fig_min_soil = mu.create_plot()
    mu.set_ocean_color(ax)
    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)
    mu.plot_raster(ax, cmap_min_soil, extent, masked_data_min_soil, norm_min_soil)
    mu.plot_country_boundaries(ax, country_shapefile)
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    title_text_min_soil = f"Net mineral soil SOC change\nkt CO$_2$e yr$^{{-1}}$"
    mu.create_divergent_legend_asymmetric(fig_min_soil, rounded_lower_lim_min_soil, rounded_upper_lim_min_soil,
                                          title_text_min_soil, tick_labels_min_soil,
                                          veg_analysis_years, net_colors_rgb, percentiles_min_soil, percentile_0_min_soil, main_logger)
    mu.remove_ticks(ax)

    min_soil_output_name_kt = f"min_soil_net__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    core_jpeg_name_min_soil = f"{min_soil_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_min_soil = f"{core_jpeg_name_min_soil}_{bounding_box_description}"
    jpeg_path_min_soil = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_min_soil}.jpeg"
    jpeg_for_pres_path_min_soil = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_min_soil}__for_pres.jpeg"
    mu.save_pres_non_pres_jpegs(ax, jpeg_path_min_soil, jpeg_for_pres_path_min_soil, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    main_logger.info(f"Mineral soil net for {bounding_box_description} extent took {round(time.time() - start_time)} seconds: {uu.timestr()}")


    # --- Panel c: Gross organic soil emissions (unidirectional colormap) ---

    main_logger.info("\n---Mapping gross organic soil emissions:")

    # organic_soil_local_reproj is defined in Part 6; re-reading here to apply bbox clipping consistently
    with rasterio.open(organic_soil_local_reproj) as src:
        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data_org_soil = src.read(1, window=window).astype('float32')
        else:
            data_org_soil = src.read(1).astype('float32')

    non_zero_org_soil = data_org_soil[data_org_soil != 0]
    percentile_for_saturation = 1
    breaks_org_soil = np.percentile(non_zero_org_soil, [1, (100 - percentile_for_saturation)])
    lower_lim_org_soil = breaks_org_soil[0]
    upper_lim_org_soil = breaks_org_soil[-1]
    rounded_upper_lim_org_soil = math.floor(upper_lim_org_soil / 10 ** 3 * 100) / 100
    tick_labels_org_soil = [0, f"> {rounded_upper_lim_org_soil:.0f}"]
    title_text_org_soil = f"Gross organic soil emissions\nkt CO$_2$e yr$^{{-1}}$"

    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_org_soil}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_org_soil}")

    colors_matplotlib = mu.rgb_to_mpl_palette(cn.emissions_colors_rgb)
    percentiles_normalized_org_soil = np.linspace(0, 1, len(cn.emissions_percentiles))
    cmap_org_soil = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_org_soil, colors_matplotlib)))
    norm_org_soil = Normalize(vmin=lower_lim_org_soil, vmax=upper_lim_org_soil)
    masked_data_org_soil = np.ma.masked_where(data_org_soil <= 0, data_org_soil)

    ax, fig_org_soil = mu.create_plot()
    mu.set_ocean_color(ax)
    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)
    img_org_soil = mu.plot_raster(ax, cmap_org_soil, extent, masked_data_org_soil, norm_org_soil)
    mu.plot_country_boundaries(ax, country_shapefile)
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    mu.create_unidirection_legend(fig_org_soil, img_org_soil, lower_lim_org_soil, upper_lim_org_soil,
                                   title_text_org_soil, tick_labels_org_soil,
                                   'avg', cn.emissions_colors_rgb, cn.emissions_percentiles, main_logger)
    mu.remove_ticks(ax)

    org_soil_output_name_kt = f"org_soil_gross_emis__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    core_jpeg_name_org_soil = f"{org_soil_output_name_kt}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_org_soil = f"{core_jpeg_name_org_soil}_{bounding_box_description}"
    jpeg_path_org_soil = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_org_soil}.jpeg"
    jpeg_for_pres_path_org_soil = f"{LULUCF_local_jpeg_pres_folder}/{core_jpeg_name_org_soil}__for_pres.jpeg"
    mu.save_pres_non_pres_jpegs(ax, jpeg_path_org_soil, jpeg_for_pres_path_org_soil, "", full_slide_text_LULUCF_with_disclaimer, main_logger)

    main_logger.info(f"Organic soil gross emissions for {bounding_box_description} extent took {round(time.time() - start_time)} seconds: {uu.timestr()}")


    # --- Four-panel composite ---
    # Panel a: jpeg_path_veg_net — created by Part 1 (vegetation net flux)
    # Panel b: jpeg_path_min_soil — created above
    # Panel c: jpeg_path_org_soil — created above
    # Panel d: jpeg_path_LULUCF_net — created by Part 3

    four_panel_output_name = f"LULUCF_four_panel__component_fluxes__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    core_jpeg_name_four_panel = f"{four_panel_output_name}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_four_panel = f"{core_jpeg_name_four_panel}_{bounding_box_description}"
    jpeg_path_four_panel = f"{LULUCF_local_jpeg_non_pres_folder}/{core_jpeg_name_four_panel}.jpeg"

    mu.create_four_panel_map(
        jpeg_path_four_panel,
        jpeg_path_veg_net,
        jpeg_path_min_soil,
        jpeg_path_org_soil,
        jpeg_path_LULUCF_net,
        "",
        main_logger,
        panel_labels=["a  Net vegetation flux", "b  Net mineral soil SOC change",
                      "c  Gross organic soil emissions", "d  Net LULUCF flux"]
    )

    end_time = time.time()
    main_logger.info(f"Through Part 7 for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")




    # ### Part 8: Maps AFOLU
    #
    # main_logger.info("\n\n\n---Mapping AFOLU:")
    #
    # # Iteratively collects the names and versions of non-vegetation datasets, and text for bottom-right of maps (LULUCF versions already collected)
    # full_slide_text_AFOLU = f"{full_slide_text_LULUCF} \n"
    # if cropland_geotif_s3:
    #     cropland_date = re.search(r'/(\d{8})/', cropland_geotif_s3).group(1)
    #     non_veg_versions = f'{non_veg_versions}__cropland_v{cropland_date}'
    #     full_slide_text_AFOLU = f'{full_slide_text_AFOLU}; {cn.cropland_pres_text}'
    # if livestock_geotif_s3:
    #     livestock_date = re.search(r'/(\d{8})/', livestock_geotif_s3).group(1)
    #     non_veg_versions = f'{non_veg_versions}__livestock_v{livestock_date}'
    #     full_slide_text_AFOLU = f'{full_slide_text_AFOLU}; {cn.livestock_pres_text}'
    # if cropland_geotif_s3 == None and livestock_geotif_s3 == None:
    #     sys.exit("No Agriculture datasets supplied. Not creating total AFOLU maps.")
    # full_slide_text_AFOLU_with_disclaimer = f"{full_slide_text_AFOLU} \n {cn.legend_percentile_disclaimer}"
    #
    #
    # # Final combined output
    # AFOLU_output_name = f"AFOLU__veg_{veg_version}_{non_veg_versions}__MgCO2e_yr"
    # AFOLU_final_total_path = f"{cn.local_jpeg_folder_AFOLU}/{AFOLU_output_name}.tif"
    # with rasterio.open(AFOLU_final_total_path, 'w', **veg_meta) as dst:
    #     dst.write(AFOLU_net.astype('float32'), 1)
    #
    # non_zero_values_AFOLU = AFOLU_net[AFOLU_net != 0]
    #
    # main_logger.info(f"\n\n---Preparing legend for AFOLU")
    #
    # # Calculates min, center and max across all years
    # percentile_for_saturation = 1
    # breaks_AFOLU = np.percentile(non_zero_values_AFOLU, [1, (100 - percentile_for_saturation)])  # The min and max percentiles_AFOLU at which colors saturate
    #
    # lower_lim_AFOLU = breaks_AFOLU[0]
    # global_neutral_AFOLU = 0
    # upper_lim_AFOLU = breaks_AFOLU[-1]
    #
    # main_logger.info(f"Across AFOLU:")
    # main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_AFOLU}")
    # main_logger.info(f"  neutral: {global_neutral_AFOLU}")
    # main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_AFOLU}")
    #
    # # Creates the min and max values for the legend in kt CO2e (converts legend units from Mg (t) to kt with 10**3-- data doesn't change).
    # # Rounds data_min down and data_max up for legend.
    # rounded_lower_lim_AFOLU = math.ceil(lower_lim_AFOLU / 10 ** 3 * 100) / 100  # Rounds up
    # rounded_upper_lim_AFOLU = math.floor(upper_lim_AFOLU / 10 ** 3 * 100) / 100  # Rounds down
    # tick_labels_AFOLU = [f"< {rounded_lower_lim_AFOLU:.0f}  (sink)",
    #                # Spaces are to horizontally align the text explanations
    #                "0        (neutral)",
    #                f"> {rounded_upper_lim_AFOLU:.0f}  (source)"]
    # AFOLU_output_name_kt = AFOLU_output_name.replace("MgCO2", "ktCO2")
    # # print(tick_labels_AFOLU)
    #
    # main_logger.info(f"\n\n---Generating AFOLU map:")
    #
    # # Reads raster data
    # with rasterio.open(AFOLU_final_total_path) as src:
    #
    #     if bounding_box_proj is not None:
    #         minx, miny, maxx, maxy = bounding_box_proj
    #
    #         window = from_bounds(minx, miny, maxx, maxy, src.transform)
    #
    #         data_AFOLU = src.read(1, window=window)
    #
    #         # Update extent from the window
    #         left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
    #         raster_extent = (left, right, bottom, top)
    #
    #     else:
    #         data_AFOLU = src.read(1)
    #         b = src.bounds
    #         raster_extent = (b.left, b.right, b.bottom, b.top)
    #
    # # Calculates the percentile for 0 for the year (neutral, no flux) for mapping
    # main_logger.info(f"  Calculating percentiles_AFOLU and breaks")
    # percentile_0_AFOLU = mu.percentile_for_0(data_AFOLU)
    # main_logger.info(f"  0 is at the {percentile_0_AFOLU}th percentile of the raster.")
    # percentiles_AFOLU = [percentile_0_AFOLU / 6, percentile_0_AFOLU / 4, percentile_0_AFOLU / 2, percentile_0_AFOLU / 1.3, percentile_0_AFOLU / 1.05,
    #                percentile_0_AFOLU * 1.05, percentile_0_AFOLU * 1.1, percentile_0_AFOLU * 1.2, percentile_0_AFOLU * 1.3, percentile_0_AFOLU * 1.5]
    # # print("percentiles_AFOLU:", percentiles_AFOLU)
    #
    # # Converts RGB color palette to matplotlib color palette
    # colors_matplotlib = mu.rgb_to_mpl_palette(net_colors_rgb)
    #
    # # Matches percentile breaks with colors for the map.
    # # Normalizes percentiles_AFOLU to a 0-1 scale.
    # percentiles_normalized_AFOLU = np.linspace(0, 1, len(percentiles_AFOLU))
    # # print("percentiles_normalized_AFOLU:", percentiles_normalized_AFOLU)
    # cmap_AFOLU = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized_AFOLU, colors_matplotlib)))
    #
    # main_logger.info(f"  Masking raster to non-0 values for AFOLU")
    # masked_data_AFOLU = np.ma.masked_where(data_AFOLU == 0, data_AFOLU)
    #
    # # For map (not legend)
    # norm = TwoSlopeNorm(
    #     vmin=lower_lim_AFOLU,
    #     vcenter=global_neutral_AFOLU,
    #     vmax=upper_lim_AFOLU
    # )
    #
    # main_logger.info(f"  Plotting AFOLU map")
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
    # img = mu.plot_raster(ax, cmap_AFOLU, extent, masked_data_AFOLU, norm)
    #
    # # Plots the country boundaries on top
    # mu.plot_country_boundaries(ax, country_shapefile)
    #
    # # Explicitly sets the bounding box for the plot image
    # if bounding_box_proj is not None:
    #     ax.set_xlim(extent[0], extent[1])
    #     ax.set_ylim(extent[2], extent[3])
    #
    # # Title
    # title_text = f"AFOLU net GHG flux\nkt CO$_2$e yr$^{{-1}}$"
    #
    # # Creates legend
    # mu.create_divergent_legend_asymmetric(fig, rounded_lower_lim_AFOLU, rounded_upper_lim_AFOLU,
    #                                       title_text, tick_labels_AFOLU,
    #                                       veg_analysis_years, net_colors_rgb, percentiles_AFOLU, percentile_0_AFOLU, main_logger)
    #
    # # Removes axis ticks and labels
    # mu.remove_ticks(ax)
    #
    # core_jpeg_name_AFOLU = f"{AFOLU_output_name_kt}__{uu.timestr()[0:8]}"
    # if bounding_box_description:  # Adds bounding box description to file name, if supplied
    #     core_jpeg_name_AFOLU = f"{core_jpeg_name_AFOLU}_{bounding_box_description}"
    # jpeg_path_AFOLU = f"{AFOLU_local_jpeg_non_pres_folder}/{core_jpeg_name_AFOLU}.jpeg"
    # jpeg_for_pres_path_AFOLU = f"{AFOLU_local_jpeg_pres_folder}/{core_jpeg_name_AFOLU}__for_pres.jpeg"
    #
    # full_slide_text_AFOLU_with_disclaimer = full_slide_text_AFOLU_with_disclaimer.replace("Cropland: vYYYYMMDD", f"Cropland: v{cropland_date}")
    # full_slide_text_AFOLU_with_disclaimer = full_slide_text_AFOLU_with_disclaimer.replace("Livestock: vYYYYMMDD", f"Livestock: v{livestock_date}")
    #
    # # Saves two versions of the map: without and with a source note in the bottom right
    # out_jpeg_for_pres = mu.save_pres_non_pres_jpegs(ax, jpeg_path_AFOLU, jpeg_for_pres_path_AFOLU, "", full_slide_text_AFOLU_with_disclaimer, main_logger)
    #
    # end_time = time.time()
    # main_logger.info(f"AFOLU for {bounding_box_description} extent took {round(end_time - start_time)} seconds: {uu.timestr()}")


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


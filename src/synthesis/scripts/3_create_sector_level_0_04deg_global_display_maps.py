"""
Creates 4x4km (0.04-degree) display maps for LULUCF and (placeholder) AFOLU sectors.

Inputs:
- Vegetation average annual net flux geotif (S3 or local path, WGS84 — reprojected to Robinson here)
- An input_date (YYYYMMDD) for the summative LULUCF maps, used to construct S3 paths for
  the pre-made LULUCF annual-average global geotifs (net flux, gross emissions, gross removals)
- Organic soil drained S3 path (Mg CO2e/pixel/yr, WGS84 — reprojected to Robinson here)
- Organic soil burned S3 path (Mg CO2e/pixel/yr, WGS84 — reprojected to Robinson here)
- Mineral soil S3 path (Mg C/pixel/yr, WGS84 — reprojected to Robinson here)
- Optional cropland and livestock S3 paths (stub for future AFOLU maps)
- Optional parquet path for global average annual flux annotations on each map

Both organic soil inputs are summed into one organic soil emissions layer.

Maps produced:
  Part 1 — Net LULUCF flux (annual average from pre-made S3 geotif)
  Part 2 — Gross LULUCF emissions and removals (annual averages from pre-made S3 geotifs)
  Part 3 — Three-panel LULUCF: gross emissions | gross removals | net flux (from parts 1 and 2 above)
  Part 4 — Four-panel LULUCF components: veg net | mineral soil net | organic soil gross emis | LULUCF net

Legend min/max use the 1st and 99th percentiles of non-zero pixels.

Defaults to global coverage; supply --center_latitude, --center_longitude, and --lat_height for a
zoomed map. The global 2:1 (width:height) aspect ratio is maintained in all zoomed maps.

Made with Claude session 'Sector-level display maps refactor'

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

LULUCF global (all four parts):
python -m src.synthesis.scripts.3_create_sector_level_0_04deg_global_display_maps \
-veg s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/2024/_0_04deg_yr/global/20260130/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2024_global.tif \
-osd s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/drained_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__drained_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-osb s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/burned_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__burned_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-ms s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_net__mineral_soil_extent__0-30cm_MgCO2/2020/_0_04deg_yr/global/20260611/SOC_net__mineral_soil_extent__0-30cm_MgCO2_0_04deg_yr_v1_0_1_2020_global.tif \
-ld 20260614 \
-pq /mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/LULUCF_v1_0_0__veg_v1_0_5__minsoil_v1_0_1__orgsoil_v1_0_1/LULUCF__v1_0_0__for_figures__wide__20260617.parquet

Example — Central Africa zoom (Parts 1-3 only, no component data):
python -m src.synthesis.scripts.3_create_sector_level_0_04deg_global_display_maps
  -veg s3://... -ld 20260614
  --center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

#TODO Vegetation net flux is currently using just 2024, not averaging all years. Organic soil is using just 2021-2024, not averaging both intervals.
"""

import argparse
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds
from shapely.geometry import box

from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import map_utilities as mu
from src.utilities import universal_utilities as uu


# ── Raster helpers ──────────────────────────────────────────────────────────────

def reproject_to_robinson(path, local_folder, logger, reference_path=None):
    """Reproject a WGS84 geotif to Robinson projection. Skips if already done.

    Uses calculate_default_transform to derive the Robinson pixel grid from the
    source raster. Pass reference_path to force the output onto an existing
    raster's grid — necessary when the source resolution differs from the target
    (e.g. 0.01-degree organic soil inputs being matched to a 0.04-degree grid).
    Returns the reprojected file path.
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    path_reproj = f"{local_folder}/{filename}_reproj.tif"

    if not os.path.exists(path_reproj):
        logger.info(f"  Reprojecting to Robinson: {path}")
        logger.info(f"  → {path_reproj}")
        with rasterio.open(path) as src:
            src_nodata = src.nodata
            if reference_path is not None:
                with rasterio.open(reference_path) as ref:
                    dst_transform = ref.transform
                    dst_width = ref.width
                    dst_height = ref.height
            else:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, cn.Robinson_crs, src.width, src.height, *src.bounds
                )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': cn.Robinson_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': 0,
                'compress': 'lzw',
            })
            with rasterio.open(path_reproj, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=cn.Robinson_crs,
                        resampling=Resampling.nearest,
                        src_nodata=src_nodata,
                        dst_nodata=0,
                    )
    else:
        logger.info(f"  Reprojected raster already exists: {path_reproj}")

    return path_reproj


def convert_kg_to_Mg(path, logger):
    """Convert a geotif from kg to Mg (tonnes). Skips if already converted.

    Returns the converted file path. (Retained for future AFOLU use with cropland/livestock.)
    """
    converted_path = path.replace("kg", "Mg")

    if not os.path.exists(converted_path):
        logger.info(f"  Converting kg → Mg: {path}")
        with rasterio.open(path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            nodata = src.nodata
            data = np.where(data == nodata, nodata, data / 1000.0)
            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(data.astype('float32'), 1)
    else:
        logger.info(f"  Unit-converted raster already exists: {converted_path}")

    return converted_path


def build_lulucf_s3_path(pattern, lulucf_date, model_type='standard', model_path_description='global'):
    """Return the S3 path for a LULUCF annual-average global geotif produced by script 2.

    Mirrors the path construction in 2_create_LULUCF_global_0_04x0_04deg.py /
    universal_utilities.mosaic_tiles_to_global.
    """
    avg_year = f"avg_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}"
    lulucf_run = (
        cn.LULUCF_full_version_underscore
        .replace("MODEL_TYPE", model_type)
        .replace("MODEL_PATH_DESCRIPTION", model_path_description)
    )
    folder = (
        cn.LULUCF_outputs_path
        .replace(cn.model_version_type_description_placeholder, lulucf_run)
        + f"{pattern}/annual_intervals/{avg_year}/{cn.flux_aggreg_pixel_meaning}/global/{lulucf_date}/"
    )
    filename = (
        f"{pattern}{cn.flux_aggreg_pixel_meaning}"
        f"_v{cn.LULUCF_model_version_underscore}_{avg_year}_global.tif"
    )
    return folder + filename


# ── Parquet annotation helpers ──────────────────────────────────────────────────

def _global_avg_annual_Gt(df, col):
    """Average annual global total in Gt from a per-row Mg flux column."""
    return df.groupby('year')[col].sum().mean() / 1e9


def _flux_annotation(df, col, unit='Gt CO$_2$e yr$^{-1}$'):
    """Format a bottom-of-map annotation string. Returns None if df is None."""
    if df is None:
        return None
    val = _global_avg_annual_Gt(df, col)
    year_min = df['year'].min()
    year_max = df['year'].max()
    direction = 'net emissions' if val >= 0 else 'net removals'
    return f"Average {direction}, {year_min}\u2013{year_max}: \n{val:.2g} {unit}"


# ── Map rendering helpers ───────────────────────────────────────────────────────

# Default percentile multipliers for divergent maps (applied to percentile_0).
# Part 1 (vegetation) uses cn.net_percentiles instead.
_DIVERGENT_PERCENTILE_MULTIPLIERS = [
    1/6, 1/4, 1/2, 1/1.3, 1/1.05,
    1.05, 1.1, 1.2, 1.3, 1.5,
]


def read_raster_clipped(path, bounding_box_proj):
    """Read a raster as float32, optionally windowed to bounding_box_proj.

    Returns (data_array, (left, right, bottom, top)).
    """
    with rasterio.open(path) as src:
        if bounding_box_proj is not None:
            minx, miny, maxx, maxy = bounding_box_proj
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            data = src.read(1, window=window).astype('float32')
            left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
        else:
            data = src.read(1).astype('float32')
            b = src.bounds
            left, bottom, right, top = b.left, b.bottom, b.right, b.top
    return data, (left, right, bottom, top)


def compute_percentile_limits(data, saturation_pct=0.5):
    """Calculate the 1 and 99% of valid pixels, used to set the min and max for the legend."""
    valid = data[np.isfinite(data) & (data != 0)]
    if valid.size == 0:
        raise ValueError("No valid (non-zero, finite) pixels found in raster — check reprojection nodata handling.")
    lower, upper = np.percentile(valid, [saturation_pct, 100 - saturation_pct])
    return lower, upper


def _round_limits_to_kt(lower_lim, upper_lim):
    """Convert Mg limits to kt, rounding outward (lower up, upper down)."""
    rounded_lower = math.ceil(lower_lim / 1e3 * 100) / 100
    rounded_upper = math.floor(upper_lim / 1e3 * 100) / 100
    return rounded_lower, rounded_upper


def render_divergent_map(data, raster_extent, bounding_box_proj, country_shapefile,
                          net_colors_rgb, title_text, veg_analysis_years,
                          non_pres_folder, pres_folder, jpeg_name, slide_text, logger,
                          percentile_multipliers=None, bottom_annotation=None):
    """Full pipeline for a divergent (net flux) map: percentiles → colormap → figure → legend → save.

    Returns the non-presentation JPEG path.
    """
    if percentile_multipliers is None:
        percentile_multipliers = _DIVERGENT_PERCENTILE_MULTIPLIERS

    # 1 and 99 percentiles for legend
    lower_lim, upper_lim = compute_percentile_limits(data)
    rounded_lower, rounded_upper = _round_limits_to_kt(lower_lim, upper_lim)
    logger.info(f"  1-pct limit: {lower_lim:.2f}    99-pct limit: {upper_lim:.2f}")

    tick_labels = [
        f"< {rounded_lower:.0f}  (sink)",
        "0        (neutral)",
        f"> {rounded_upper:.0f}  (source)",
    ]

    # Percentile for 0 (neutral) flux
    percentile_0 = mu.percentile_for_0(data)
    logger.info(f"  0 is at the {percentile_0:.1f} percentile.")

    # Sets anchor points for the colors in the colormap, relative to the percentile for 0
    percentiles = [percentile_0 * m for m in percentile_multipliers]

    colors_mpl = mu.rgb_to_mpl_palette(net_colors_rgb)
    cmap = LinearSegmentedColormap.from_list(
        "custom_colormap",
        list(zip(np.linspace(0, 1, len(percentiles)), colors_mpl)),
    )
    masked = np.ma.masked_where(data == 0, data)
    norm = TwoSlopeNorm(vmin=lower_lim, vcenter=0, vmax=upper_lim)

    # Creates various graphics layers
    ax, fig = mu.create_plot()
    mu.set_ocean_color(ax)
    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)
    mu.plot_raster(ax, cmap, extent, masked, norm)
    mu.plot_country_boundaries(ax, country_shapefile)
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if bottom_annotation:
        ax.text(0.5, 0.03, bottom_annotation, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=cn.legend_fontsize * 1, color='black')

    mu.create_divergent_legend_asymmetric(
        fig, rounded_lower, rounded_upper, title_text, tick_labels,
        veg_analysis_years, net_colors_rgb, percentiles, percentile_0, logger,
        colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
        show_direction_arrows=True, colorbar_left_offset=0.02,
    )
    mu.remove_ticks(ax)

    jpeg_path = f"{non_pres_folder}/{jpeg_name}.jpeg"
    mu.save_pres_non_pres_jpegs(ax, jpeg_path, f"{pres_folder}/{jpeg_name}__for_pres.jpeg",
                                 "", slide_text, logger)
    return jpeg_path


def render_unidirectional_map(data, raster_extent, bounding_box_proj, country_shapefile,
                               colors_rgb, percentiles_cfg, title_text,
                               non_pres_folder, pres_folder, jpeg_name, slide_text, logger,
                               mask_positive=True, bottom_annotation=None):
    """Full pipeline for a unidirectional (emissions or removals) map.

    mask_positive=True  → show only positive values (emissions, masks <= 0).
    mask_positive=False → show only negative values (removals, masks >= 0).
    Returns the non-presentation JPEG path.
    """
    lower_lim, upper_lim = compute_percentile_limits(data)
    logger.info(f"  1-pct limit: {lower_lim:.2f}    99-pct limit: {upper_lim:.2f}")

    colors_mpl = mu.rgb_to_mpl_palette(colors_rgb)
    cmap = LinearSegmentedColormap.from_list(
        "custom_colormap",
        list(zip(np.linspace(0, 1, len(percentiles_cfg)), colors_mpl)),
    )
    norm = Normalize(vmin=lower_lim, vmax=upper_lim)

    if mask_positive:
        masked = np.ma.masked_where(data <= 0, data)
        rounded_upper = math.floor(upper_lim / 1e3 * 100) / 100
        tick_labels = [0, f"> {rounded_upper:.0f}"]
    else:
        masked = np.ma.masked_where(data >= 0, data)
        rounded_lower = math.ceil(lower_lim / 1e3 * 100) / 100
        tick_labels = [f"< {rounded_lower:.0f}", 0]

    ax, fig = mu.create_plot()
    mu.set_ocean_color(ax)
    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)
    img = mu.plot_raster(ax, cmap, extent, masked, norm)
    mu.plot_country_boundaries(ax, country_shapefile)
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    if bottom_annotation:
        ax.text(0.5, 0.03, bottom_annotation, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=cn.legend_fontsize * 1, color='black')

    mu.create_unidirection_legend(
        fig, img, lower_lim, upper_lim, title_text, tick_labels,
        'avg', colors_rgb, percentiles_cfg, logger,
        colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
        label_divisor=1e3, colorbar_left_offset=0.00,
    )
    mu.remove_ticks(ax)

    jpeg_path = f"{non_pres_folder}/{jpeg_name}.jpeg"
    mu.save_pres_non_pres_jpegs(ax, jpeg_path, f"{pres_folder}/{jpeg_name}__for_pres.jpeg",
                                 "", slide_text, logger)
    return jpeg_path


# Names the jpeg
def jpeg_name(core, bounding_box_description):
    ts = uu.timestr()[0:8]
    return f"{core}__{ts}_{bounding_box_description}" if bounding_box_description else f"{core}__{ts}"


# ── Main mapping function ───────────────────────────────────────────────────────

def map_LULUCF_maps(veg_net_geotif, lulucf_input_date,
                    model_type, model_path_description,
                    organic_soil_drained_s3, organic_soil_burned_s3, mineral_soil_s3,
                    cropland_geotif_s3, livestock_geotif_s3,
                    net_colors_rgb, country_shapefile, bounding_box, bounding_box_description,
                    main_logger, parquet_path=None):

    start_time = time.time()

    # Local working folder. Reprojected geotifs go in it.
    reproj_folder = cn.local_jpeg_folder_LULUCF.replace("MODEL_TYPE", model_type).replace("MODEL_PATH_DESCRIPTION", model_path_description)
    reproj_folder = reproj_folder.replace("MODEL_TYPE", model_type)
    reproj_folder = reproj_folder.replace("MODEL_PATH_DESCRIPTION", model_path_description)
    Path(reproj_folder).mkdir(parents=True, exist_ok=True)

    # Output folder for jpegs
    out_dir = f"{reproj_folder}output_jpegs_and_gifs_{bounding_box_description}_{uu.timestr()[0:8]}"
    non_pres_folder = f"{out_dir}/jpegs_non_pres"
    pres_folder = f"{out_dir}/jpegs_pres"
    Path(non_pres_folder).mkdir(parents=True, exist_ok=True)
    Path(pres_folder).mkdir(parents=True, exist_ok=True)

    # Load stats table for bottom-of-map annotations (global maps only)
    if parquet_path and bounding_box is None:
        df_stats = pd.read_parquet(parquet_path)
        main_logger.info(f"Loaded stats parquet: {parquet_path}  ({len(df_stats):,} rows, years {df_stats['year'].min()}–{df_stats['year'].max()})")
    else:
        df_stats = None
        if parquet_path and bounding_box is not None:
            main_logger.info("Skipping flux annotations: map is not global.")

    # Builds LULUCF S3 paths
    lulucf_net_s3 = build_lulucf_s3_path(
        cn.net_flux_all_C_pools_all_gases_LULUCF_pattern, lulucf_input_date, model_type, model_path_description,
    )
    lulucf_emis_s3 = build_lulucf_s3_path(
        cn.gross_emis_all_C_pools_all_gases_LULUCF_pattern, lulucf_input_date, model_type, model_path_description,
    )
    lulucf_remv_s3 = build_lulucf_s3_path(
        cn.gross_removals_all_C_pools_LULUCF_pattern, lulucf_input_date, model_type, model_path_description,
    )
    main_logger.info(f"LULUCF S3 paths:\n  net:  {lulucf_net_s3}\n  emis: {lulucf_emis_s3}\n  remv: {lulucf_remv_s3}")

    # Reprojects all inputs to Robinson (each independently)
    main_logger.info("\nReprojecting inputs to Robinson")
    veg_net_reproj = reproject_to_robinson(veg_net_geotif, reproj_folder, main_logger)
    lulucf_net_reproj = reproject_to_robinson(lulucf_net_s3, reproj_folder, main_logger)
    lulucf_emis_reproj = reproject_to_robinson(lulucf_emis_s3, reproj_folder, main_logger)
    lulucf_remv_reproj = reproject_to_robinson(lulucf_remv_s3, reproj_folder, main_logger)
    mineral_soil_reproj = reproject_to_robinson(mineral_soil_s3, reproj_folder, main_logger)
    org_soil_drained_reproj = reproject_to_robinson(organic_soil_drained_s3, reproj_folder, main_logger, reference_path=veg_net_reproj)
    org_soil_burned_reproj = reproject_to_robinson(organic_soil_burned_s3, reproj_folder, main_logger, reference_path=veg_net_reproj)

    # Robinson bounding box and shapefile clip
    bounding_box_proj = mu.transform_bbox_to_robinson(bounding_box) if bounding_box is not None else None
    if bounding_box_proj is not None:
        country_shapefile = country_shapefile.clip(box(*bounding_box_proj))

    # Shared naming and version metadata
    veg_version = re.search(r'v\d+_\d+_\d+', veg_net_geotif).group(0)

    non_veg_versions = ''
    lulucf_slide_text = cn.veg_pres_text
    non_veg_versions += f'_organic_soil_v{cn.organic_soil_model_version_underscore}'
    lulucf_slide_text += f'; {cn.organic_soil_pres_text}'
    non_veg_versions += f'_mineral_soil_v{cn.SOC_model_version_underscore}'
    lulucf_slide_text += f'; {cn.mineral_soil_pres_text}'
    lulucf_slide_text_with_disclaimer = f"{lulucf_slide_text} \n {cn.legend_percentile_disclaimer}"

    # Read rasters. raster_extent from veg is used as the display extent for all parts.
    data_veg_net,     raster_extent = read_raster_clipped(veg_net_reproj, bounding_box_proj)
    data_lulucf_net,  _             = read_raster_clipped(lulucf_net_reproj, bounding_box_proj)
    data_lulucf_emis, _             = read_raster_clipped(lulucf_emis_reproj, bounding_box_proj)
    data_lulucf_remv, _             = read_raster_clipped(lulucf_remv_reproj, bounding_box_proj)
    main_logger.info(f"Raster extent (from veg): {raster_extent}")


    ### Part 1: Net LULUCF flux

    main_logger.info("\n\n\n---Part 1: Mapping net LULUCF flux")

    lulucf_net_core = f"LULUCF_net_flux_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_lulucf_net = render_divergent_map(
        data_lulucf_net, raster_extent, bounding_box_proj, country_shapefile,
        net_colors_rgb,
        title_text=f"Net land-based flux\nkt CO$_2$e yr$^{{-1}}$",
        veg_analysis_years=cn.year_range_str,
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_net_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_net_flux__MgCO2e_yr'),
    )
    main_logger.info(f"Part 1 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    ### Part 2: LULUCF gross emissions and removals

    main_logger.info("\n\n\n---Part 2: Mapping LULUCF gross emissions and removals")

    lulucf_emis_core = f"LULUCF_gross_emis_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_lulucf_emis = render_unidirectional_map(
        data_lulucf_emis, raster_extent, bounding_box_proj, country_shapefile,
        cn.emissions_colors_rgb, cn.emissions_percentiles,
        title_text=f"Gross land-based emissions\nkt CO$_2$e yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_emis_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=True,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_gross_emissions__all_gases__MgCO2e_yr'),
    )

    lulucf_remv_core = f"LULUCF_gross_remv_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_lulucf_remv = render_unidirectional_map(
        data_lulucf_remv, raster_extent, bounding_box_proj, country_shapefile,
        cn.removals_colors_rgb, cn.removals_percentiles,
        title_text=f"Gross land-based removals\nkt CO$_2$ yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_remv_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=False,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_gross_removals__MgCO2_yr',
                                            unit='Gt CO$_2$ yr$^{-1}$'),
    )
    main_logger.info(f"Part 2 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    ### Part 3: Three-panel LULUCF map (gross emissions | gross removals | net flux)

    main_logger.info("\n\n\n---Part 3: Three-panel LULUCF map")

    three_panel_core = f"LULUCF_three_panel__emis_remv_net__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_three_panel = f"{non_pres_folder}/{jpeg_name(three_panel_core, bounding_box_description)}.jpeg"
    mu.create_three_panel_map(
        jpeg_path_three_panel,
        jpeg_path_lulucf_emis, jpeg_path_lulucf_remv, jpeg_path_lulucf_net,
        "", main_logger,
    )
    main_logger.info(f"Part 3 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    ### Part 4: Four-panel LULUCF component map
    ###   a: veg net flux
    ###   b: mineral soil net SOC change
    ###   c: organic soil gross emissions (drained + burned)
    ###   d: LULUCF net flux

    main_logger.info("\n\n\n---Part 4: Four-panel LULUCF component map")

    # Panel a: Vegetation net flux
    main_logger.info(f"  Creating vegetation net flux map")
    veg_net_core = f"vegetation_net_flux_all_pools_all_gases_{veg_version}__{cn.year_range_str}__ktCO2e_yr"
    jpeg_path_veg_net = render_divergent_map(
        data_veg_net, raster_extent, bounding_box_proj, country_shapefile,
        net_colors_rgb,
        title_text=f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$",
        veg_analysis_years=cn.year_range_str,
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(veg_net_core, bounding_box_description),
        slide_text=cn.veg_pres_text,
        logger=main_logger,
        percentile_multipliers=cn.net_percentiles,
        bottom_annotation=_flux_annotation(df_stats, 'veg__net_flux__all_C_pools__all_gases__MgCO2e_yr'),
    )

    # Panel b: Mineral soil net SOC change
    main_logger.info(f"  Creating mineral soil net change map")
    data_min_soil, _ = read_raster_clipped(mineral_soil_reproj, bounding_box_proj)

    min_soil_core = f"mineral_soil_net__veg_{veg_version}__{non_veg_versions}__ktCO2_yr"
    jpeg_path_min_soil = render_divergent_map(
        data_min_soil, raster_extent, bounding_box_proj, country_shapefile,
        net_colors_rgb,
        title_text=f"Net mineral soil SOC change\nkt CO$_2$e yr$^{{-1}}$",
        veg_analysis_years=cn.year_range_str,
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(min_soil_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        bottom_annotation=_flux_annotation(df_stats, 'SOC_net__mineral_soil_extent__0_30cm_MgCO2_yr',
                                            unit='Gt CO$_2$ yr$^{-1}$'),
    )

    # Panel c: Organic soil gross emissions (drained + burned)
    main_logger.info(f"  Creating organic soil emissions map")
    data_drained, _ = read_raster_clipped(org_soil_drained_reproj, bounding_box_proj)
    data_burned, _  = read_raster_clipped(org_soil_burned_reproj, bounding_box_proj)
    data_org_soil = data_drained + data_burned
    main_logger.info("  Organic soil: summing drained + burned")

    org_soil_core = f"org_soil_gross_emis__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_org_soil = render_unidirectional_map(
        data_org_soil, raster_extent, bounding_box_proj, country_shapefile,
        cn.emissions_colors_rgb, cn.emissions_percentiles,
        title_text=f"Gross organic soil emissions\nkt CO$_2$e yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(org_soil_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=True,
        bottom_annotation=_flux_annotation(df_stats, 'org_soil_emis__all_gases__MgCO2e_yr'),
    )

    # Four-panel composite
    main_logger.info(f"  Creating four-panel map")
    four_panel_core = f"LULUCF_four_panel__component_fluxes__veg_{veg_version}__{non_veg_versions}__ktCO2e_yr"
    jpeg_path_four_panel = f"{non_pres_folder}/{jpeg_name(four_panel_core, bounding_box_description)}.jpeg"
    mu.create_four_panel_map(
        jpeg_path_four_panel,
        jpeg_path_veg_net, jpeg_path_min_soil, jpeg_path_org_soil, jpeg_path_lulucf_net,
        "", main_logger,
        panel_labels=["a", "b", "c", "d"],
    )
    main_logger.info(f"Part 4 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    # ### Part 5 (stub): AFOLU total map — cropland + livestock + LULUCF
    # Implement when agriculture datasets are ready. Pixel-wise addition requires all inputs
    # to be on a common grid; use a shared reference raster when adding reproject_to_reference().
    #
    # if not cropland_geotif_s3 and not livestock_geotif_s3:
    #     main_logger.info("No agriculture inputs supplied — skipping AFOLU map.")
    #     return
    # cropland_reproj = reproject_to_robinson(cropland_geotif_s3, AFOLU_reproj_folder, main_logger)
    # cropland_mg = convert_kg_to_Mg(cropland_reproj, main_logger)
    # livestock_reproj = reproject_to_robinson(livestock_geotif_s3, AFOLU_reproj_folder, main_logger)
    # livestock_mg = convert_kg_to_Mg(livestock_reproj, main_logger)
    # data_afolu = data_lulucf_net + data_cropland + data_livestock  # ← needs aligned grids
    # ...


def main(veg_net_geotif,
         organic_soil_drained_s3,
         organic_soil_burned_s3,
         mineral_soil_s3,
         lulucf_input_date,
         lulucf_model_type='standard',
         lulucf_model_path_description='global',
         cropland_geotif_s3=None,
         livestock_geotif_s3=None,
         center_latitude=None, center_longitude=None, lat_height=None,
         bounding_box_description=None,
         parquet_path=None):

    stage = 'summative_4x4km_LULUCF_jpegs'
    log_note = '4x4 km jpegs for presenations/manuscript'

    main_logger, main_log_local_path, n_workers = lu.populate_main_log_header(
        "NA", "NA", log_note, True, "NA", stage,
    )

    # Reprojects country shapefile if not already reprojected
    country_shapefile = mu.check_and_reproject_shapefile(
        main_logger,
        shapefile_path=cn.original_shapefile_path,
        target_crs=cn.Robinson_crs,
        reprojected_shapefile_path=cn.reprojected_shapefile_path,
    )

    # Establishes bounding box if not global
    if center_latitude is not None and center_longitude is not None and lat_height is not None:
        bounding_box = mu.calculate_bbox_centered(
            main_logger,
            center_lat=center_latitude,
            center_lon=center_longitude,
            lat_height=lat_height,
            aspect_ratio=2.0,
        )
        main_logger.info(f"Using custom bounding box: {bounding_box}")
    else:
        bounding_box = None
        main_logger.info("No bounding box specified; using global extent.")

    # Creates jpegs
    map_LULUCF_maps(
        veg_net_geotif, lulucf_input_date,
        lulucf_model_type, lulucf_model_path_description,
        organic_soil_drained_s3, organic_soil_burned_s3, mineral_soil_s3,
        cropland_geotif_s3, livestock_geotif_s3,
        cn.net_colors_rgb, country_shapefile, bounding_box, bounding_box_description,
        main_logger, parquet_path=parquet_path,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create 0.04x0.04 deg LULUCF display maps.")
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Total latitude height around center (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box to include in output names')

    parser.add_argument('-veg', '--veg_net_geotif', required=True, help='S3 or local path to vegetation net-flux geotif (WGS84, will be reprojected)')
    parser.add_argument('-osd', '--organic_soil_drained_s3', required=True, help='S3 path for organic soil drained emissions (Mg CO2e/pixel/yr, WGS84)')
    parser.add_argument('-osb', '--organic_soil_burned_s3', required=True, help='S3 path for organic soil burned emissions (Mg CO2e/pixel/yr, WGS84)')
    parser.add_argument('-ms', '--mineral_soil_s3', required=True, help='S3 path for mineral soil net flux (Mg C/pixel/yr, WGS84)')
    parser.add_argument('-ld', '--lulucf_input_date', required=True, help='Run date (YYYYMMDD) of the script-2 LULUCF outputs to map')
    parser.add_argument('-mt', '--lulucf_model_type', default='standard', help='Model type used to create summative LULUCF outputs (default: standard)')
    parser.add_argument('-mpd', '--lulucf_model_path_description', default='global', help='Model path description used to create the LULUCF summative outputs (default: global)')
    parser.add_argument('-pq', '--parquet_path', help='Path to wide-format LULUCF parquet for bottom-of-map flux annotations (optional)')
    parser.add_argument('-cl', '--cropland_geotif_s3', help='S3 path for cropland emissions (AFOLU stub, optional)')
    parser.add_argument('-ls', '--livestock_geotif_s3', help='S3 path for livestock emissions (AFOLU stub, optional)')

    args = parser.parse_args()

    main(
        args.veg_net_geotif,
        organic_soil_drained_s3=args.organic_soil_drained_s3,
        organic_soil_burned_s3=args.organic_soil_burned_s3,
        mineral_soil_s3=args.mineral_soil_s3,
        lulucf_input_date=args.lulucf_input_date,
        lulucf_model_type=args.lulucf_model_type,
        lulucf_model_path_description=args.lulucf_model_path_description,
        parquet_path=args.parquet_path,
        cropland_geotif_s3=args.cropland_geotif_s3,
        livestock_geotif_s3=args.livestock_geotif_s3,
        center_latitude=args.center_latitude,
        center_longitude=args.center_longitude,
        lat_height=args.lat_height,
        bounding_box_description=args.bounding_box_description,
    )
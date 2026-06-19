"""
Creates 4x4km (0.04-degree) display maps for LULUCF and (placeholder) AFOLU sectors.

Inputs:
- An input_date (YYYYMMDD) for the summative LULUCF maps, used to construct S3 paths for
  the pre-made LULUCF annual-average global geotifs (net flux, gross emissions, gross removals)
- LULUCF model type
- LULUCF model path description (optional)
- Parquet path for global average annual flux annotations on each map (optional)

- Vegetation last year (2024) net flux geotif S3 path (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Drained organic soil last interval (2021-2024) S3 path (Mg CO2e/0.01x0.01 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Burned organic soil last interval (2021-2024) S3 path (Mg CO2e/0.01x0.01 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Mineral soil net change S3 path (2020 change) (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)

- Vegetation last year (2024) gross emissions geotif S3 path (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Mineral soil gross loss S3 path (2020 change) (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)

- Cropland emissions (optional)
- Livestock emissions (optional)

- Regional map arguments

Vegetation net flux and gross emissions: mean of all annual rasters in cn.interval_end_years_annual, inferred from the latest year path supplied on command line
Organic soil: weighted average across cn.organic_soil_year_intervals (weight = years per interval), inferred from the latest interval path supplied on command line

Both organic soil inputs for an interval are summed into one organic soil emissions layer for that interval.

Maps produced:
  Part 1 — Net and gross emis and removals LULUCF fluxes (annual average from pre-made S3 geotifs)
  Part 2 — Three-panel LULUCF: gross emissions | gross removals | net flux (from parts 1 and 2 above)
  Part 3 — Four-panel LULUCF components: average annual veg net | mineral soil net change | organic soil gross emis | LULUCF net
  Part 4 — Percentage contribution to average annual LULUCF gross emissions from vegetation, organic soil, and mineral soil

Legend min/max use the 0.5 and 99.5 percentiles of non-zero pixels.

Defaults to global coverage; supply --center_latitude, --center_longitude, and --lat_height for a
zoomed map. The global 2:1 (width:height) aspect ratio is maintained in all zoomed maps.

Made with Claude session 'Sector-level display maps refactor'

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

LULUCF global (all four parts):
python -m src.synthesis.scripts.3_create_sector_level_0_04deg_global_display_maps \
-ld 20260614 \
-pq /mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/LULUCF_v1_0_0__veg_v1_0_5__minsoil_v1_0_1__orgsoil_v1_0_1/LULUCF__v1_0_0__for_figures__wide__20260617.parquet \
-veg_net s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/2024/_0_04deg_yr/global/20260130/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2024_global.tif \
-osd s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/drained_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__drained_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-osb s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/burned_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__burned_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-ms_net s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_net__mineral_soil_extent__0-30cm_MgCO2/2020/_0_04deg_yr/global/20260611/SOC_net__mineral_soil_extent__0-30cm_MgCO2_0_04deg_yr_v1_0_1_2020_global.tif \
-veg_emis s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/2024/_0_04deg_yr/global/20260130/gross_emissions__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2024_global.tif \
-ms_loss s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_loss__mineral_soil_extent__0-30cm_MgCO2/2020/_0_04deg_yr/global/20260611/SOC_loss__mineral_soil_extent__0-30cm_MgCO2_0_04deg_yr_v1_0_1_2020_global.tif

Example — Central Africa zoom (Parts 1-3 only, no component data-- and no flux annotation):
python -m src.synthesis.scripts.3_create_sector_level_0_04deg_global_display_maps
  [all the above arguments] \
  --center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa
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

def reproject_to_robinson(path, local_folder, logger, reference_path=None, prefix='', out_label=None):
    """Reproject a WGS84 geotif to Robinson projection. Skips if already done.

    Uses calculate_default_transform to derive the Robinson pixel grid from the
    source raster. Pass reference_path to force the output onto an existing
    raster's grid — necessary when the source resolution differs from the target
    (e.g. 0.01-degree organic soil inputs being matched to a 0.04-degree grid).
    Pass prefix to prepend a string to the output filename (e.g. 'veg_').
    Pass out_label to use a custom stem instead of prefix+source_filename — useful
    when the source filename is long enough to push the output path over WSL's
    ~240-character write limit on /mnt/c/ paths.
    Returns the reprojected file path.
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    stem = out_label if out_label is not None else f"{prefix}{filename}"
    path_reproj = os.path.join(local_folder, f"{stem}_reproj.tif")

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


def save_array_as_geotif(data, reference_path, out_path, logger):
    """Write a float32 numpy array to a GeoTIF using spatial metadata from reference_path. Skips if already exists."""
    if os.path.exists(out_path):
        logger.info(f"  Average raster already exists: {out_path}")
        return
    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()
    meta.update({'dtype': 'float32', 'count': 1, 'nodata': 0, 'compress': 'lzw'})
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(data.astype('float32'), 1)
    logger.info(f"  Saved: {out_path}")


def convert_kg_to_Mg(path, logger):
    """Convert a geotif from kg to Mg (tonnes). Skips if already converted.
    Only used for cropland and livestock.

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
    """Return the S3 path for a LULUCF annual average global geotif.

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


# ── Multi-year / multi-interval path inference ─────────────────────────────────

def _infer_veg_year_paths(latest_path, years):
    """Infer all annual vegetation net-flux paths from the latest-year path.

    Replaces every occurrence of the latest year string in the path and filename,
    which covers both the annual_intervals/YEAR/ subfolder and the _YEAR_global.tif
    filename segment. Works for S3 and local paths.
    """
    latest_year = str(years[-1])
    return [latest_path.replace(latest_year, str(y)) for y in years]


def _infer_org_soil_interval_paths(latest_path, intervals):
    """Infer all organic soil interval paths from the latest-interval path.

    Replaces every occurrence of the YYYY_YYYY interval string (folder + filename).
    """
    latest_interval = re.search(r'\d{4}_\d{4}', latest_path).group(0)
    return [latest_path.replace(latest_interval, ivl) for ivl in intervals]


def _interval_weight(interval_str):
    """Number of years covered by an interval string like '2016_2020'."""
    start, end = interval_str.split('_')
    return int(end) - int(start) + 1


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


def compute_percentile_limits(data, saturation_pct):
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
    lower_lim, upper_lim = compute_percentile_limits(data, cn.saturation_percentile)
    rounded_lower, rounded_upper = _round_limits_to_kt(lower_lim, upper_lim)
    logger.info(f"  {cn.saturation_percentile}-pct limit: {lower_lim:.2f}    {100-cn.saturation_percentile}-pct limit: {upper_lim:.2f}")

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
    lower_lim, upper_lim = compute_percentile_limits(data, cn.saturation_percentile)
    logger.info(f"  {cn.saturation_percentile}-pct limit: {lower_lim:.2f}    {100-cn.saturation_percentile}-pct limit: {upper_lim:.2f}")

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


def render_percentage_map(data, raster_extent, bounding_box_proj, country_shapefile,
                           colors_rgb, percentiles_cfg, title_text,
                           non_pres_folder, pres_folder, jpeg_name, slide_text, logger):
    """Full pipeline for a percentage (0–100%) contribution map.

    Always anchors the colormap at 0% and masks non-positive pixels.
    Returns the non-presentation JPEG path.
    """
    _, upper_lim = compute_percentile_limits(data, cn.saturation_percentile)
    logger.info(f"  {100-cn.saturation_percentile}-pct limit: {upper_lim:.2f}%")

    colors_mpl = mu.rgb_to_mpl_palette(colors_rgb)
    cmap = LinearSegmentedColormap.from_list(
        "custom_colormap",
        list(zip(np.linspace(0, 1, len(percentiles_cfg)), colors_mpl)),
    )
    norm = Normalize(vmin=0, vmax=upper_lim)
    masked = np.ma.masked_where(data <= 0, data)
    rounded_upper = math.floor(upper_lim)
    tick_labels = ['0%', f'> {rounded_upper}%']

    ax, fig = mu.create_plot()
    mu.set_ocean_color(ax)
    mu.plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)
    img = mu.plot_raster(ax, cmap, extent, masked, norm)
    mu.plot_country_boundaries(ax, country_shapefile)
    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    mu.create_unidirection_legend(
        fig, img, 0, upper_lim, title_text, tick_labels,
        'avg', colors_rgb, percentiles_cfg, logger,
        colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
        label_divisor=1, colorbar_left_offset=0.00,
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

def map_LULUCF_maps(lulucf_input_date,
                    model_type, model_path_description,
                    net_colors_rgb, country_shapefile, bounding_box, bounding_box_description,
                    main_logger,
                    parquet_path=None,
                    veg_net_geotif=None,
                    organic_soil_drained_s3=None, organic_soil_burned_s3=None, mineral_soil_net_s3=None,
                    veg_emis_geotif=None, mineral_soil_loss_s3=None,
                    cropland_geotif_s3=None, livestock_geotif_s3=None):

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

    # Robinson bounding box and shapefile clip
    bounding_box_proj = mu.transform_bbox_to_robinson(bounding_box) if bounding_box is not None else None
    if bounding_box_proj is not None:
        country_shapefile = country_shapefile.clip(box(*bounding_box_proj))

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

    main_logger.info("\nReprojecting LULUCF summative maps to Robinson")
    lulucf_net_reproj = reproject_to_robinson(lulucf_net_s3, reproj_folder, main_logger)
    lulucf_emis_reproj = reproject_to_robinson(lulucf_emis_s3, reproj_folder, main_logger)
    lulucf_remv_reproj = reproject_to_robinson(lulucf_remv_s3, reproj_folder, main_logger)

    # Read LULUCF summative maps; derive raster_extent from LULUCF net
    main_logger.info(f"Reading average annual LULUCF gross and net maps")
    data_lulucf_net,  _ = read_raster_clipped(lulucf_net_reproj, bounding_box_proj)
    data_lulucf_emis, _ = read_raster_clipped(lulucf_emis_reproj, bounding_box_proj)
    data_lulucf_remv, _ = read_raster_clipped(lulucf_remv_reproj, bounding_box_proj)
    _, raster_extent = read_raster_clipped(lulucf_net_reproj, bounding_box_proj)
    main_logger.info(f"Raster extent (from LULUCF net): {raster_extent}")

    # Net flux component reprojection, reading, and Part 3 require all four Part 3 inputs (component net fluxes)
    has_net_component_inputs = all([veg_net_geotif, organic_soil_drained_s3, organic_soil_burned_s3, mineral_soil_net_s3])
    if has_net_component_inputs:

        # Vegetation net: reproject all years
        veg_net_year_paths = _infer_veg_year_paths(veg_net_geotif, cn.interval_end_years_annual)
        main_logger.info(f"\nReprojecting net vegetation ({len(veg_net_year_paths)} years) to Robinson")
        veg_net_reprojected = [reproject_to_robinson(p, reproj_folder, main_logger, prefix='veg_') for p in veg_net_year_paths]
        veg_net_reproj_ref_grid = veg_net_reprojected[-1]  # reference grid for organic soil reprojection

        main_logger.info("\nReprojecting net mineral soil change to Robinson")
        mineral_soil_net_reproj = reproject_to_robinson(mineral_soil_net_s3, reproj_folder, main_logger)

        # Organic soil: reproject all intervals since start of vegetation model
        drained_interval_paths = _infer_org_soil_interval_paths(organic_soil_drained_s3, cn.organic_soil_year_intervals)
        burned_interval_paths  = _infer_org_soil_interval_paths(organic_soil_burned_s3,  cn.organic_soil_year_intervals)
        main_logger.info(f"\nReprojecting organic soil ({len(cn.organic_soil_year_intervals)} intervals) to Robinson")
        drained_reprojected = [reproject_to_robinson(p, reproj_folder, main_logger, reference_path=veg_net_reproj_ref_grid) for p in drained_interval_paths]
        burned_reprojected  = [reproject_to_robinson(p, reproj_folder, main_logger, reference_path=veg_net_reproj_ref_grid) for p in burned_interval_paths]

        # Version strings for file naming and slide text
        file_version_str = (f"{cn.veg_model_version_underscore}__organic_soil_v{cn.organic_soil_model_version_underscore}"
                            f"__mineral_soil_v{cn.SOC_model_version_underscore}")
        lulucf_slide_text = f"{cn.veg_pres_text}; {cn.organic_soil_pres_text}; {cn.mineral_soil_pres_text}"

        # Read and average vegetation rasters
        veg_net_arrays = [read_raster_clipped(p, bounding_box_proj)[0] for p in veg_net_reprojected]
        data_veg_net_avg = np.mean(np.stack(veg_net_arrays), axis=0)
        main_logger.info(f"Vegetation net flux: averaged {len(veg_net_arrays)} annual rasters")
        veg_avg_path = f"{reproj_folder}veg_{cn.net_flux_all_C_pools_all_gases_pattern}_v{cn.flux_aggreg_pixel_meaning}{cn.veg_model_version_underscore}_{cn.year_range_str}_avg_global_reproj.tif"
        save_array_as_geotif(data_veg_net_avg, veg_net_reprojected[-1], veg_avg_path, main_logger)

        main_logger.info(f"Reading mineral soil net change map")
        data_min_soil, _ = read_raster_clipped(mineral_soil_net_reproj, bounding_box_proj)

        main_logger.info(f"Organic soil: averaging {len(cn.organic_soil_year_intervals)} intervals")
        org_weights    = [_interval_weight(ivl) for ivl in cn.organic_soil_year_intervals]
        drained_arrays = [read_raster_clipped(p, bounding_box_proj)[0] for p in drained_reprojected]
        burned_arrays  = [read_raster_clipped(p, bounding_box_proj)[0] for p in burned_reprojected]
        data_org_soil = np.average(
            np.stack([d + b for d, b in zip(drained_arrays, burned_arrays)]),
            axis=0,
            weights=org_weights,
        )
        main_logger.info(f"Organic soil: weighted average over intervals {dict(zip(cn.organic_soil_year_intervals, org_weights))}")
        org_start = cn.organic_soil_year_intervals[0].split('_')[0]
        org_end   = cn.organic_soil_year_intervals[-1].split('_')[1]
        org_soil_avg_path = f"{reproj_folder}org_soil_emis_MgCO2e_{cn.flux_aggreg_pixel_meaning}__{org_start}_{org_end}_wght_avg_global_reproj.tif"
        save_array_as_geotif(data_org_soil, drained_reprojected[-1], org_soil_avg_path, main_logger)

    else:
        main_logger.info("Part 3 inputs not supplied — Parts 3 and 4 will be skipped.")
        file_version_str = lulucf_input_date
        lulucf_slide_text = f"LULUCF model run {lulucf_input_date}"

    # Gross emis component reprojection, reading, and Part 4 require all three Part 4 inputs (component gross emissions fluxes)
    has_gross_component_inputs = all([veg_emis_geotif, organic_soil_drained_s3, organic_soil_burned_s3, mineral_soil_loss_s3])
    if has_gross_component_inputs:

        veg_emis_year_paths = _infer_veg_year_paths(veg_emis_geotif, cn.interval_end_years_annual)
        main_logger.info(f"\nReprojecting vegetation gross emissions ({len(veg_emis_year_paths)} years) to Robinson")
        veg_emis_reprojected = [reproject_to_robinson(p, reproj_folder, main_logger, prefix='veg_') for p in veg_emis_year_paths]

        main_logger.info("\nReprojecting gross mineral soil loss to Robinson")
        mineral_soil_loss_reproj = reproject_to_robinson(mineral_soil_loss_s3, reproj_folder, main_logger)

        veg_emis_arrays = [read_raster_clipped(p, bounding_box_proj)[0] for p in veg_emis_reprojected]
        data_veg_emis_avg = np.mean(np.stack(veg_emis_arrays), axis=0)
        main_logger.info(f"Vegetation gross emissions: averaged {len(veg_emis_arrays)} annual rasters")
        veg_emis_avg_path = f"{reproj_folder}veg_{cn.gross_emis_all_C_pools_all_gases_pattern}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.year_range_str}_avg_reproj.tif"
        save_array_as_geotif(data_veg_emis_avg, veg_emis_reprojected[-1], veg_emis_avg_path, main_logger)
        data_min_soil_loss, _ = read_raster_clipped(mineral_soil_loss_reproj, bounding_box_proj)

    lulucf_slide_text_with_disclaimer = f"{lulucf_slide_text} \n {cn.legend_percentile_disclaimer}"


    ### Part 1: Net and gross LULUCF flux maps

    main_logger.info("\n\n\n---Part 1: Mapping net and gross LULUCF flux")

    lulucf_net_core = f"LULUCF_net_flux__{file_version_str}__ktCO2e_yr"
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

    lulucf_emis_core = f"LULUCF_gross_emis__{file_version_str}__ktCO2e_yr"
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

    lulucf_remv_core = f"LULUCF_gross_remv__{file_version_str}__ktCO2e_yr"
    jpeg_path_lulucf_remv = render_unidirectional_map(
        data_lulucf_remv, raster_extent, bounding_box_proj, country_shapefile,
        cn.removals_colors_rgb, cn.removals_percentiles,
        title_text=f"Gross land-based removals\nkt CO$_2$ yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_remv_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=False,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_gross_removals__MgCO2_yr', unit='Gt CO$_2$ yr$^{-1}$'),
    )
    main_logger.info(f"Part 1 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    ### Part 2: Three-panel LULUCF map (gross emissions | gross removals | net flux)

    main_logger.info("\n\n\n---Part 2: Three-panel LULUCF map")

    three_panel_core = f"LULUCF_three_panel__emis_remv_net__{file_version_str}__ktCO2e_yr"
    jpeg_path_three_panel = f"{non_pres_folder}/{jpeg_name(three_panel_core, bounding_box_description)}.jpeg"
    mu.create_three_panel_map(
        jpeg_path_three_panel,
        jpeg_path_lulucf_emis, jpeg_path_lulucf_remv, jpeg_path_lulucf_net,
        "", main_logger,
    )
    main_logger.info(f"Part 2 done in {round(time.time() - start_time)}s: {uu.timestr()}")

    ### Part 3: Four-panel LULUCF component map
    ###   a: veg net flux
    ###   b: mineral soil net SOC change
    ###   c: organic soil gross emissions (drained + burned)
    ###   d: LULUCF net flux
    if has_net_component_inputs:
        main_logger.info("\n\n\n---Part 3: Four-panel LULUCF component map")

        # Panel a: Vegetation net flux (annual average)
        main_logger.info(f"  Creating annual average vegetation net flux map")
        veg_net_core = f"vegetation_net_flux_all_pools_all_gases_{cn.veg_model_version_underscore}__{cn.year_range_str}__ktCO2e_yr"
        jpeg_path_veg_net = render_divergent_map(
            data_veg_net_avg, raster_extent, bounding_box_proj, country_shapefile,
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
        min_soil_core = f"mineral_soil_net__{file_version_str}__ktCO2_yr"
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

        # Panel c: Organic soil gross emissions — weighted average across intervals, computed above
        main_logger.info(f"  Creating organic soil emissions map")

        org_soil_core = f"org_soil_gross_emis__{file_version_str}__ktCO2e_yr"
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
        four_panel_core = f"LULUCF_four_panel__component_fluxes__{file_version_str}__ktCO2e_yr"
        jpeg_path_four_panel = f"{non_pres_folder}/{jpeg_name(four_panel_core, bounding_box_description)}.jpeg"
        mu.create_four_panel_map(
            jpeg_path_four_panel,
            jpeg_path_veg_net, jpeg_path_min_soil, jpeg_path_org_soil, jpeg_path_lulucf_net,
            "", main_logger,
            panel_labels=["a", "b", "c", "d"],
        )
        main_logger.info(f"Part 3 done in {round(time.time() - start_time)}s: {uu.timestr()}")
    else:
        main_logger.infof("Skipping net component mapping")


    ### Part 4: Three-panel emissions source contribution map
    ###   Top:    vegetation gross emissions as % of LULUCF gross emissions
    ###   Middle: organic soil emissions as % of LULUCF gross emissions
    ###   Bottom: mineral soil gross loss as % of LULUCF gross emissions
    if has_gross_component_inputs:
        main_logger.info("\n\n\n---Part 4: Three-panel emissions source contribution map")

        pct_veg_emis = np.where(data_lulucf_emis > 0, data_veg_emis_avg / data_lulucf_emis * 100, 0).astype('float32')
        pct_org_soil = np.where(data_lulucf_emis > 0, data_org_soil    / data_lulucf_emis * 100, 0).astype('float32')
        pct_min_loss = np.where(data_lulucf_emis > 0, data_min_soil_loss / data_lulucf_emis * 100, 0).astype('float32')

        save_array_as_geotif(pct_veg_emis, lulucf_net_reproj, f"{reproj_folder}pct_veg_gross_emis_of_LULUCF_gross_emis__{file_version_str}_reproj.tif", main_logger)
        save_array_as_geotif(pct_org_soil, lulucf_net_reproj, f"{reproj_folder}pct_org_soil_emis_of_LULUCF_gross_emis__{file_version_str}_reproj.tif", main_logger)
        save_array_as_geotif(pct_min_loss, lulucf_net_reproj, f"{reproj_folder}pct_min_soil_loss_of_LULUCF_gross_emis__{file_version_str}_reproj.tif", main_logger)

        pct_veg_core = f"pct_veg_gross_emis_of_LULUCF_gross_emis__{file_version_str}"
        jpeg_path_pct_veg = render_percentage_map(
            pct_veg_emis, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Vegetation % of gross\nland-based emissions",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(pct_veg_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
        )

        pct_org_core = f"pct_org_soil_emis_of_LULUCF_gross_emis__{file_version_str}"
        jpeg_path_pct_org = render_percentage_map(
            pct_org_soil, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Organic soil % of gross\nland-based emissions",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(pct_org_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
        )

        pct_min_core = f"pct_min_soil_loss_of_LULUCF_gross_emis__{file_version_str}"
        jpeg_path_pct_min = render_percentage_map(
            pct_min_loss, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Mineral soil loss % of gross\nland-based emissions",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(pct_min_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
        )

        pct_three_panel_core = f"LULUCF_pct_gross_emis__veg_orgsoil_minsoil__{file_version_str}"
        jpeg_path_pct_three_panel = f"{non_pres_folder}/{jpeg_name(pct_three_panel_core, bounding_box_description)}.jpeg"
        mu.create_three_panel_map(
            jpeg_path_pct_three_panel,
            jpeg_path_pct_veg, jpeg_path_pct_org, jpeg_path_pct_min,
            "", main_logger,
        )
        main_logger.info(f"Part 4 done in {round(time.time() - start_time)}s: {uu.timestr()}")
    else:
        main_logger.infof("Skipping gross component percentage mapping")


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


def main(lulucf_input_date,
         lulucf_model_type='standard',
         lulucf_model_path_description='global',
         parquet_path=None,

         # For Part 3
         veg_net_geotif=None, organic_soil_drained_s3=None, organic_soil_burned_s3=None, mineral_soil_s3=None,

         # For Part 4
         veg_emis_geotif=None,  mineral_soil_loss_s3=None,

         # For AFOLU stub
         cropland_geotif_s3=None, livestock_geotif_s3=None,

         # For regional maps
         center_latitude=None, center_longitude=None, lat_height=None, bounding_box_description=None):

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
        lulucf_input_date,
        lulucf_model_type, lulucf_model_path_description,
        cn.net_colors_rgb, country_shapefile, bounding_box, bounding_box_description,
        main_logger,
        parquet_path=parquet_path,
        veg_net_geotif=veg_net_geotif,
        organic_soil_drained_s3=organic_soil_drained_s3,
        organic_soil_burned_s3=organic_soil_burned_s3,
        mineral_soil_net_s3=mineral_soil_s3,
        veg_emis_geotif=veg_emis_geotif,
        mineral_soil_loss_s3=mineral_soil_loss_s3,
        cropland_geotif_s3=cropland_geotif_s3,
        livestock_geotif_s3=livestock_geotif_s3,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create 0.04x0.04 deg LULUCF display maps.")

    # For parts 1 and 2
    parser.add_argument('-ld', '--lulucf_input_date', required=True, help='Run date (YYYYMMDD) of the LULUCF summative outputs')
    parser.add_argument('-mt', '--lulucf_model_type', default='standard', help='Model type used to create summative LULUCF outputs (default: standard)')
    parser.add_argument('-mpd', '--lulucf_model_path_description', default='global', help='Model path description used to create the LULUCF summative outputs (default: global)')
    parser.add_argument('-pq', '--parquet_path', help='Path to wide-format LULUCF parquet for bottom-of-map flux annotations (optional)')

    # For part 3
    parser.add_argument('-veg_net', '--veg_net_geotif', help='S3 or local path to vegetation net-flux geotif (WGS84, will be reprojected)')
    parser.add_argument('-osd', '--organic_soil_drained_s3',  help='S3 path for organic soil drained emissions (Mg CO2e/pixel/yr, WGS84)')
    parser.add_argument('-osb', '--organic_soil_burned_s3',  help='S3 path for organic soil burned emissions (Mg CO2e/pixel/yr, WGS84)')
    parser.add_argument('-ms_net', '--mineral_soil_s3',  help='S3 path for mineral soil net flux (Mg C/pixel/yr, WGS84)')

    # For part 4
    parser.add_argument('-veg_emis', '--veg_emis_geotif_s3', help='S3 path for latest-year vegetation gross emissions geotif (WGS84); all years inferred and averaged for Part 5 (optional)')
    parser.add_argument('-ms_loss', '--mineral_soil_loss_s3', help='S3 path for gross mineral soil carbon loss (Mg CO2/pixel/yr, WGS84); used for Part 5 (optional)')

    # For AFOLU stub
    parser.add_argument('-cl', '--cropland_geotif_s3', help='S3 path for cropland emissions (AFOLU stub, optional)')
    parser.add_argument('-ls', '--livestock_geotif_s3', help='S3 path for livestock emissions (AFOLU stub, optional)')

    # For regional map
    parser.add_argument('-clat', '--center_latitude', type=float, help='Latitude to center output maps (optional)')
    parser.add_argument('-clon', '--center_longitude', type=float, help='Longitude to center output maps (optional)')
    parser.add_argument('-lh', '--lat_height', type=float, help='Total latitude height around center (optional)')
    parser.add_argument('-bbd', '--bounding_box_description', default='global', help='Description of bounding box to include in output names')

    args = parser.parse_args()

    main(
        args.lulucf_input_date,
        lulucf_model_type=args.lulucf_model_type,
        lulucf_model_path_description=args.lulucf_model_path_description,
        parquet_path=args.parquet_path,

        veg_net_geotif=args.veg_net_geotif,
        organic_soil_drained_s3=args.organic_soil_drained_s3,
        organic_soil_burned_s3=args.organic_soil_burned_s3,
        mineral_soil_s3=args.mineral_soil_s3,

        veg_emis_geotif=args.veg_emis_geotif_s3,
        mineral_soil_loss_s3=args.mineral_soil_loss_s3,

        cropland_geotif_s3=args.cropland_geotif_s3,
        livestock_geotif_s3=args.livestock_geotif_s3,

        center_latitude=args.center_latitude,
        center_longitude=args.center_longitude,
        lat_height=args.lat_height,
        bounding_box_description=args.bounding_box_description,
    )
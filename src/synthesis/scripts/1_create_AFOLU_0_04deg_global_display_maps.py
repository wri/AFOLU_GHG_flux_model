"""
Creates 4x4km (0.04-degree) goetifs and display maps (jpegs) for LULUCF and (placeholder) AFOLU.

Inputs:
- An input_date (YYYYMMDD) for the summative LULUCF maps, used to construct S3 paths for
  the pre-made LULUCF annual-average global geotifs (net flux, gross emissions, gross removals)
- LULUCF model type
- LULUCF model path description (optional)
- Parquet path for global average annual flux annotations on each map (optional)

- Vegetation last year (2024) net flux geotif S3 path (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Drained organic soil last interval (2021-2024) S3 path (Mg CO2e/0.01x0.01 deg/yr, WGS84 — resampled to 0.04° then reprojected to Robinson here) (optional)
- Burned organic soil last interval (2021-2024) S3 path (Mg CO2e/0.01x0.01 deg/yr, WGS84 — resampled to 0.04° then reprojected to Robinson here) (optional)
- Mineral soil net change S3 path (2020 change) (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)

- Vegetation last year (2024) gross emissions geotif S3 path (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)
- Mineral soil gross loss S3 path (2020 change) (Mg CO2e/0.04x0.04 deg/yr, WGS84 — reprojected to Robinson here) (optional)

- Flux uncertainties: hardcoded as arguments to annotation rendering function.

- Cropland emissions (optional)
- Livestock emissions (optional)

- Regional map arguments

Vegetation net flux and gross emissions: mean of all annual rasters in cn.interval_end_years_annual, inferred from the latest year path supplied on command line
Organic soil: each 0.01°×0.01° interval raster is resampled (sum) to 0.04° WGS84 on the veg net grid, then drained+burned are summed per interval,
then a weighted average is taken across cn.organic_soil_year_intervals (weight = years per interval), then the result is reprojected to Robinson once.
Paths are inferred from the latest interval paths supplied on the command line.

Maps produced:
  Part 1 — Individual maps of net and gross emissions and removals for LULUCF (uses annual average from pre-made S3 geotifs)
  Part 2 — Three-panel LULUCF: gross emissions | gross removals | net flux (from part 1)
  Part 3 — Four-panel LULUCF components: average annual veg net | mineral soil net change | organic soil gross emis | LULUCF net
  Part 4 — Percentage contribution to average annual LULUCF gross emissions from vegetation, organic soil, and mineral soil

Legend min/max use the 0.5 and 99.5 percentiles of non-zero pixels.

Defaults to global coverage; supply --center_latitude, --center_longitude, and --lat_height for a
zoomed map. The global 2:1 (width:height) aspect ratio is maintained in all zoomed maps.

Made with Claude session 'Sector-level display maps refactor'

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

LULUCF global (all four parts):
python -m src.synthesis.scripts.1_create_AFOLU_0_04deg_global_display_maps \
-ld 20260614 \
-pq /mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/LULUCF_v1_0_0__veg_v1_0_5__minsoil_v1_0_1__orgsoil_v1_0_1/LULUCF__v1_0_0__for_figures__wide__20260617.parquet \
-veg_net s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/net_flux__all_C_pools__all_gases__MgCO2e/annual_intervals/2024/_0_04deg_yr/global/20260130/net_flux__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2024_global.tif \
-osd s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/drained_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__drained_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-osb s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_1_0_1/0_01deg_output_aggregation/burned_total_Mg_CO2e_pixel_yr/ogh_mixed_f1_f15_f2_20260513/2021_2024/0_01deg_global__burned_total_Mg_CO2e_pixel_yr_2021_2024.tif \
-ms_net s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_net__mineral_soil_extent__0-30cm_MgCO2/2020/_0_04deg_yr/global/20260611/SOC_net__mineral_soil_extent__0-30cm_MgCO2_0_04deg_yr_v1_0_1_2020_global.tif \
-veg_emis s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/version_1_0_5__standard__global/gross_emissions__all_C_pools__all_gases__MgCO2e/annual_intervals/2024/_0_04deg_yr/global/20260130/gross_emissions__all_C_pools__all_gases__MgCO2e_0_04deg_yr_v1_0_5_2024_global.tif \
-ms_loss s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/version_1_0_1__standard__global/SOC_loss__mineral_soil_extent__0-30cm_MgCO2/2020/_0_04deg_yr/global/20260611/SOC_loss__mineral_soil_extent__0-30cm_MgCO2_0_04deg_yr_v1_0_1_2020_global.tif \
-cl s3://gfw2-data/climate/AFOLU_flux_model/cropland_emissions/processed/Cornell_v20250828/year_2020/global_COG/all_sources/Global_grid_cropland_emissions_total_amount_CO2eq_all_crops_without_peat_burn_kg_CO2__20260803_COG.tif \
-ls s3://gfw2-data/climate/AFOLU_flux_model/livestock_emissions/raw__from_Cornell/20251223/Total_GHG_Emissions/Tot_CO2eq_kg_livestock_GHG_emissions.tif

Example — Central Africa zoom (Parts 1-3 only, no component data-- and no flux annotation):
python -m src.synthesis.scripts.1_create_AFOLU_0_04deg_global_display_maps
  [all the above arguments] \
  --center_latitude 0 --center_longitude 20 --lat_height 20 -bbd central_Africa

#TODO Sampling the cropland and livestock emissions form 0.083 deg to 0.04 deg is distorting the values-- output geotifs don't seem to match originals well. Need to explore and fix.
"""

import argparse
import math
import os
import re
import time
from pathlib import Path

import matplotlib as mpl
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

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})


# ── Raster helpers ──────────────────────────────────────────────────────────────

def _read_full(p):
    with rasterio.open(p) as src:
        return src.read(1).astype('float32')

def read_wgs84(path):
    """Read a raster (S3 or local), masking nodata to 0. Safe for files with any nodata value."""
    with rasterio.open(path) as src:
        data = src.read(1).astype('float32')
        return np.where(src.dataset_mask() == 0, 0.0, data)

def reproject_to_robinson(path, local_folder, logger, reference_path=None, prefix='', out_label=None, nodata=0):
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
                'nodata': nodata,
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
                        dst_nodata=nodata,
                    )
    else:
        logger.info(f"  Reprojected raster already exists: {path_reproj}")

    return path_reproj


def resample_to_0_04deg(path, reference_path, local_folder, logger, out_label=None, src_nodata=None):
    """Resample a WGS84 raster to 0.04-degree resolution using sum resampling.

    Uses the grid of reference_path as the output template — same CRS, transform,
    width, and height.  Designed for aggregating fine-resolution organic soil inputs
    (0.01°) onto the vegetation/mineral-soil grid (0.04°).
    Pass out_label to override the output filename stem (recommended to keep paths
    short enough for WSL's /mnt/c/ write limit).
    Pass src_nodata to override the source file's declared nodata value — needed when
    the file stores nodata as a data value (e.g. 0) without declaring it in metadata.
    Skips if output already exists.  Returns the output path.
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    stem = out_label if out_label is not None else f"{filename}_0_04deg"
    path_out = os.path.join(local_folder, f"{stem}.tif")

    if not os.path.exists(path_out):
        logger.info(f"  Resampling to 0.04°: {path}")
        logger.info(f"  → {path_out}")
        with rasterio.open(reference_path) as ref:
            dst_transform = ref.transform
            dst_width     = ref.width
            dst_height    = ref.height
            dst_crs       = ref.crs
        with rasterio.open(path) as src:
            effective_src_nodata = src_nodata if src_nodata is not None else src.nodata
            src_pixel_area = abs(src.transform.a * src.transform.e)
            dst_pixel_area = abs(dst_transform.a * dst_transform.e)
            # Resampling.sum assigns the full source value to each overlapping output
            # pixel (GDAL weights by output-pixel-area fraction, which is 1 for
            # disaggregation). Divide by the area ratio to restore flux conservation.
            disaggregation_scale = dst_pixel_area / src_pixel_area if dst_pixel_area < src_pixel_area else 1.0
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': 0,
                'compress': 'lzw',
            })
            with rasterio.open(path_out, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    dest_arr = np.zeros((dst_height, dst_width), dtype='float32')
                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest_arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.sum,
                        src_nodata=effective_src_nodata,
                        dst_nodata=0,
                    )
                    dest_arr *= disaggregation_scale
                    dst.write(dest_arr, i)
    else:
        logger.info(f"  0.04-degree raster already exists: {path_out}")

    return path_out


def save_array_as_geotif(data, reference_path, out_path, logger, nodata=0):
    """Write a float32 numpy array to a GeoTIF using spatial metadata from reference_path. Skips if already exists."""
    if os.path.exists(out_path):
        logger.info(f"  Average raster already exists: {out_path}")
        return
    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()
    meta.update({'dtype': 'float32', 'count': 1, 'nodata': nodata, 'compress': 'lzw'})
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
    avg_year = f"avg_{cn.veg_outputs_years[0]}_{cn.veg_outputs_years[-1]}"
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


def _flux_annotation(df, col, flux_description, unit='Gt CO$_2$e yr$^{-1}$', uncertainty=None):
    """Format a bottom-of-map annotation string. Returns None if df is None."""
    if df is None:
        return None
    val = _global_avg_annual_Gt(df, col)
    year_min = df['year'].min()
    year_max = df['year'].max()
    unc_str = f" \u00b1 {uncertainty}" if uncertainty is not None else ''
    return f"{flux_description}, {year_min}\u2013{year_max}: \n{val:.2g}{unc_str} {unit}"


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
    masked = np.ma.masked_where(~np.isfinite(data) | (data < 0), data)
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

    main_logger.info("\nSaving LULUCF summative maps locally (WGS84) and reprojecting to Robinson")
    lulucf_net_wgs84_path  = os.path.join(reproj_folder, os.path.basename(lulucf_net_s3))
    lulucf_emis_wgs84_path = os.path.join(reproj_folder, os.path.basename(lulucf_emis_s3))
    lulucf_remv_wgs84_path = os.path.join(reproj_folder, os.path.basename(lulucf_remv_s3))
    save_array_as_geotif(read_wgs84(lulucf_net_s3),  lulucf_net_s3,  lulucf_net_wgs84_path,  main_logger)
    save_array_as_geotif(read_wgs84(lulucf_emis_s3), lulucf_emis_s3, lulucf_emis_wgs84_path, main_logger)
    save_array_as_geotif(read_wgs84(lulucf_remv_s3), lulucf_remv_s3, lulucf_remv_wgs84_path, main_logger)
    lulucf_net_reproj  = reproject_to_robinson(lulucf_net_wgs84_path,  reproj_folder, main_logger)
    lulucf_emis_reproj = reproject_to_robinson(lulucf_emis_wgs84_path, reproj_folder, main_logger)
    lulucf_remv_reproj = reproject_to_robinson(lulucf_remv_wgs84_path, reproj_folder, main_logger)

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

        ### Vegetation
        # Vegetation net: average all years in WGS84, reproject the average once
        veg_net_avg_wgs84_path = os.path.join(
            reproj_folder,
            f"{cn.net_flux_all_C_pools_all_gases_pattern}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.veg_year_range_str}_avg_global.tif",
        )
        if not os.path.exists(veg_net_avg_wgs84_path):
            veg_net_year_paths = _infer_veg_year_paths(veg_net_geotif, cn.veg_outputs_years)
            main_logger.info(f"\nAveraging net vegetation ({len(veg_net_year_paths)} years) in WGS84")
            data_veg_net_avg_wgs84 = np.mean(
                np.stack([read_wgs84(p) for p in veg_net_year_paths]), axis=0
            ).astype('float32')
            main_logger.info(f"Vegetation net flux: averaged {len(veg_net_year_paths)} annual rasters in WGS84")
            save_array_as_geotif(data_veg_net_avg_wgs84, veg_net_geotif, veg_net_avg_wgs84_path, main_logger)
        else:
            main_logger.info(f"\nVegetation net WGS84 average already exists, skipping averaging: {veg_net_avg_wgs84_path}")

        main_logger.info("Reprojecting averaged vegetation net flux WGS84→Robinson")
        veg_net_avg_reproj_path = reproject_to_robinson(
            veg_net_avg_wgs84_path, reproj_folder, main_logger,
            out_label=f"{cn.net_flux_all_C_pools_all_gases_pattern}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.veg_year_range_str}_avg_global",
        )
        veg_net_reproj_ref_grid = veg_net_avg_reproj_path  # reference grid for organic soil reprojection
        data_veg_net_avg, _ = read_raster_clipped(veg_net_avg_reproj_path, bounding_box_proj)


        ### Mineral soil
        main_logger.info("\nReprojecting net mineral soil change to Robinson")
        mineral_soil_net_reproj = reproject_to_robinson(mineral_soil_net_s3, reproj_folder, main_logger)

        main_logger.info(f"Reading mineral soil net change map")
        data_min_soil, _ = read_raster_clipped(mineral_soil_net_reproj, bounding_box_proj)


        ### Organic soil
        # Organic soil: resample 0.01°→0.04° WGS84 (sum), average in WGS84, reproject once
        drained_interval_paths = _infer_org_soil_interval_paths(organic_soil_drained_s3, cn.organic_soil_year_intervals)
        burned_interval_paths  = _infer_org_soil_interval_paths(organic_soil_burned_s3,  cn.organic_soil_year_intervals)

        main_logger.info(f"\nResampling organic soil ({len(cn.organic_soil_year_intervals)} intervals) 0.01°→0.04° WGS84")
        # Takes ~5 minutes for each of the 4 geotifs, so ~20 minutes total
        drained_0_04deg = [
            resample_to_0_04deg(p, veg_net_geotif, reproj_folder, main_logger,
                                out_label=f"org_soil_drained_{ivl}_0_04deg")
            for p, ivl in zip(drained_interval_paths, cn.organic_soil_year_intervals)
        ]
        burned_0_04deg = [
            resample_to_0_04deg(p, veg_net_geotif, reproj_folder, main_logger,
                                out_label=f"org_soil_burned_{ivl}_0_04deg")
            for p, ivl in zip(burned_interval_paths, cn.organic_soil_year_intervals)
        ]

        main_logger.info(f"Organic soil: averaging {len(cn.organic_soil_year_intervals)} intervals in WGS84")
        org_weights = [_interval_weight(ivl) for ivl in cn.organic_soil_year_intervals]

        # Year-weighted annual average emissions from organic soil, 0.04x0.04 deg resolution WGS84 (drained + burned)
        drained_arrays = [_read_full(p) for p in drained_0_04deg]
        burned_arrays  = [_read_full(p) for p in burned_0_04deg]
        data_org_soil_wgs84 = np.average(
            np.stack([d + b for d, b in zip(drained_arrays, burned_arrays)]),
            axis=0,
            weights=org_weights,
        ).astype('float32')
        main_logger.info(f"Organic soil: weighted average over intervals {dict(zip(cn.organic_soil_year_intervals, org_weights))}")

        org_start = cn.organic_soil_year_intervals[0].split('_')[0]
        org_end   = cn.organic_soil_year_intervals[-1].split('_')[1]
        org_soil_avg_wgs84_path = os.path.join(
            reproj_folder,
            f"org_soil_emis_MgCO2e_{cn.flux_aggreg_pixel_meaning}__{org_start}_{org_end}_wght_avg_global_0_04deg.tif",
        )
        save_array_as_geotif(data_org_soil_wgs84, drained_0_04deg[-1], org_soil_avg_wgs84_path, main_logger)

        main_logger.info("\nReprojecting averaged total organic emissions (drained+burned) WGS84→Robinson")
        org_soil_reproj_path = reproject_to_robinson(
            org_soil_avg_wgs84_path, reproj_folder, main_logger,
            reference_path=veg_net_reproj_ref_grid,
        )
        data_org_soil, _ = read_raster_clipped(org_soil_reproj_path, bounding_box_proj)


        ### Combined soil (organic emissions + net mineral)
        main_logger.info("\nCreating combined soil (organic + net mineral) geotifs")

        with rasterio.open(mineral_soil_net_s3) as src:
            _raw = src.read(1).astype('float32')
            data_min_soil_wgs84 = np.where(src.dataset_mask() == 0, 0.0, _raw)   # Need to do this to handle NoData values; otherwise, I try summing NoData and values
        min_soil_net_wgs84_path = os.path.join(
            reproj_folder,
            f"{os.path.splitext(os.path.basename(mineral_soil_net_s3))[0]}_nodata_masked.tif",
        )
        save_array_as_geotif(data_min_soil_wgs84, mineral_soil_net_s3, min_soil_net_wgs84_path, main_logger)
        data_combined_soil_wgs84 = (data_org_soil_wgs84 + data_min_soil_wgs84).astype('float32')
        combined_soil_wgs84_path = os.path.join(
            reproj_folder,
            f"soil_combined__MgCO2e_{cn.flux_aggreg_pixel_meaning}__{org_start}_{org_end}_avg_global_0_04deg.tif",
        )
        save_array_as_geotif(data_combined_soil_wgs84, drained_0_04deg[-1], combined_soil_wgs84_path, main_logger)

        main_logger.info("Reprojecting combined soil WGS84→Robinson")
        combined_soil_reproj_path = reproject_to_robinson(
            combined_soil_wgs84_path, reproj_folder, main_logger,
            reference_path=veg_net_reproj_ref_grid,
        )
        data_combined_soil, _ = read_raster_clipped(combined_soil_reproj_path, bounding_box_proj)


        # Version strings for file naming and slide text
        file_version_str = (f"{cn.veg_model_version_underscore}__organic_soil_v{cn.organic_soil_model_version_underscore}"
                            f"__mineral_soil_v{cn.SOC_model_version_underscore}")
        lulucf_slide_text = f"{cn.veg_pres_text}; {cn.organic_soil_pres_text}; {cn.mineral_soil_pres_text}"

    else:
        main_logger.info("Part 3 inputs not supplied — Parts 3 and 4 will be skipped.")
        file_version_str = lulucf_input_date
        lulucf_slide_text = f"LULUCF model run {lulucf_input_date}"

    # Gross emis component reprojection, reading, and Part 4 require all three Part 4 inputs (component gross emissions fluxes)
    has_gross_component_inputs = all([veg_emis_geotif, organic_soil_drained_s3, organic_soil_burned_s3, mineral_soil_loss_s3])
    if has_gross_component_inputs:

        # Vegetation gross emissions: average all years in WGS84, reproject the average once
        veg_emis_avg_wgs84_path = os.path.join(
            reproj_folder,
            f"{cn.gross_emis_all_C_pools_all_gases_pattern}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.veg_year_range_str}_avg.tif",
        )
        if not os.path.exists(veg_emis_avg_wgs84_path):
            veg_emis_year_paths = _infer_veg_year_paths(veg_emis_geotif, cn.veg_outputs_years)
            main_logger.info(f"\nAveraging vegetation gross emissions ({len(veg_emis_year_paths)} years) in WGS84")
            data_veg_emis_avg_wgs84 = np.mean(
                np.stack([read_wgs84(p) for p in veg_emis_year_paths]), axis=0
            ).astype('float32')
            main_logger.info(f"Vegetation gross emissions: averaged {len(veg_emis_year_paths)} annual rasters in WGS84")
            save_array_as_geotif(data_veg_emis_avg_wgs84, veg_emis_geotif, veg_emis_avg_wgs84_path, main_logger)
        else:
            main_logger.info(f"\nVegetation gross emissions WGS84 average already exists, skipping averaging: {veg_emis_avg_wgs84_path}")
            data_veg_emis_avg_wgs84 = _read_full(veg_emis_avg_wgs84_path)

        main_logger.info("Reprojecting averaged vegetation gross emissions WGS84→Robinson")
        veg_emis_avg_reproj_path = reproject_to_robinson(
            veg_emis_avg_wgs84_path, reproj_folder, main_logger,
            out_label=f"{cn.gross_emis_all_C_pools_all_gases_pattern}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.veg_year_range_str}_avg",
        )
        data_veg_emis_avg, _ = read_raster_clipped(veg_emis_avg_reproj_path, bounding_box_proj)

        # Percentage contributions: compute in WGS84, save WGS84 geotifs, reproject each to Robinson
        # data_org_soil_wgs84 is computed in has_net_component_inputs (requires same organic soil args)
        main_logger.info("\nComputing gross emissions percentage contributions in WGS84")
        data_lulucf_emis_wgs84 = _read_full(lulucf_emis_wgs84_path)
        data_min_soil_loss_wgs84 = read_wgs84(mineral_soil_loss_s3)
        # Use NaN as nodata for pct files: 0% is a valid value (source has no emissions but LULUCF does),
        # so 0 cannot serve as the nodata sentinel. NaN marks pixels where LULUCF has no emissions.
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_veg_emis_wgs84 = np.where(data_lulucf_emis_wgs84 > 0, data_veg_emis_avg_wgs84 / data_lulucf_emis_wgs84 * 100, np.nan).astype('float32')
            pct_org_soil_wgs84 = np.where(data_lulucf_emis_wgs84 > 0, data_org_soil_wgs84      / data_lulucf_emis_wgs84 * 100, np.nan).astype('float32')
            pct_min_loss_wgs84 = np.where(data_lulucf_emis_wgs84 > 0, data_min_soil_loss_wgs84 / data_lulucf_emis_wgs84 * 100, np.nan).astype('float32')

        pct_veg_wgs84_path = os.path.join(reproj_folder, f"pct_veg_gross_emis_of_LULUCF_gross_emis__{file_version_str}_wgs84.tif")
        pct_org_wgs84_path = os.path.join(reproj_folder, f"pct_org_soil_emis_of_LULUCF_gross_emis__{file_version_str}_wgs84.tif")
        pct_min_wgs84_path = os.path.join(reproj_folder, f"pct_min_soil_loss_of_LULUCF_gross_emis__{file_version_str}_wgs84.tif")
        save_array_as_geotif(pct_veg_emis_wgs84, veg_emis_geotif, pct_veg_wgs84_path, main_logger, nodata=np.nan)
        save_array_as_geotif(pct_org_soil_wgs84, veg_emis_geotif, pct_org_wgs84_path, main_logger, nodata=np.nan)
        save_array_as_geotif(pct_min_loss_wgs84, veg_emis_geotif, pct_min_wgs84_path, main_logger, nodata=np.nan)

        min_soil_loss_wgs84_path = os.path.join(
            reproj_folder,
            f"{os.path.splitext(os.path.basename(mineral_soil_loss_s3))[0]}_nodata_masked.tif",
        )
        save_array_as_geotif(data_min_soil_loss_wgs84, mineral_soil_loss_s3, min_soil_loss_wgs84_path, main_logger)

        main_logger.info("\nReprojecting gross mineral soil loss to Robinson")
        reproject_to_robinson(mineral_soil_loss_s3, reproj_folder, main_logger)

        main_logger.info("Reprojecting percentage geotifs WGS84→Robinson")
        pct_veg_reproj_path = reproject_to_robinson(pct_veg_wgs84_path, reproj_folder, main_logger,
            reference_path=veg_net_reproj_ref_grid, out_label=f"pct_veg_gross_emis_of_LULUCF_gross_emis__{file_version_str}", nodata=np.nan)
        pct_org_reproj_path = reproject_to_robinson(pct_org_wgs84_path, reproj_folder, main_logger,
            reference_path=veg_net_reproj_ref_grid, out_label=f"pct_org_soil_emis_of_LULUCF_gross_emis__{file_version_str}", nodata=np.nan)
        pct_min_reproj_path = reproject_to_robinson(pct_min_wgs84_path, reproj_folder, main_logger,
            reference_path=veg_net_reproj_ref_grid, out_label=f"pct_min_soil_loss_of_LULUCF_gross_emis__{file_version_str}", nodata=np.nan)

    # Agriculture and AFOLU require both cropland and livestock
    has_agriculture_inputs = all([cropland_geotif_s3, livestock_geotif_s3])
    if has_agriculture_inputs:

        ### Agriculture: resample 0.083333°→0.04° WGS84, convert kg→Mg, sum
        main_logger.info("\nResampling cropland and livestock 0.083333°→0.04° WGS84")
        cropland_kg_path  = resample_to_0_04deg(cropland_geotif_s3,  lulucf_net_wgs84_path, reproj_folder, main_logger,
                                                 out_label='cropland_emissions_all_crops_without_peat_burn_kg_CO2e_0_04deg',
                                                 src_nodata=0)
        livestock_kg_path = resample_to_0_04deg(livestock_geotif_s3, lulucf_net_wgs84_path, reproj_folder, main_logger,
                                                 out_label='livestock_emissions_all_animals_kg_CO2e_0_04deg',
                                                 src_nodata=0)

        main_logger.info("Converting cropland and livestock kg→Mg")
        cropland_Mg_path  = convert_kg_to_Mg(cropland_kg_path,  main_logger)
        livestock_Mg_path = convert_kg_to_Mg(livestock_kg_path, main_logger)

        main_logger.info("Summing cropland + livestock → agriculture total in WGS84")
        data_cropland_wgs84    = _read_full(cropland_Mg_path)
        data_livestock_wgs84   = _read_full(livestock_Mg_path)
        data_agriculture_wgs84 = (data_cropland_wgs84 + data_livestock_wgs84).astype('float32')
        agriculture_wgs84_path = os.path.join(reproj_folder, 'agriculture_emis_MgCO2e_0_04deg.tif')
        save_array_as_geotif(data_agriculture_wgs84, lulucf_net_wgs84_path, agriculture_wgs84_path, main_logger)

        main_logger.info("Reprojecting agriculture total WGS84→Robinson")
        agriculture_reproj_path = reproject_to_robinson(agriculture_wgs84_path, reproj_folder, main_logger,
                                                         reference_path=lulucf_net_reproj)
        data_agriculture, _ = read_raster_clipped(agriculture_reproj_path, bounding_box_proj)

        ### AFOLU: LULUCF + agriculture, arithmetic in WGS84, reproject once each
        main_logger.info("\nComputing AFOLU totals in WGS84")
        data_afolu_net_wgs84  = (_read_full(lulucf_net_wgs84_path)  + data_agriculture_wgs84).astype('float32')
        data_afolu_emis_wgs84 = (_read_full(lulucf_emis_wgs84_path) + data_agriculture_wgs84).astype('float32')
        data_afolu_remv_wgs84 = _read_full(lulucf_remv_wgs84_path)

        afolu_net_wgs84_path  = os.path.join(reproj_folder, 'AFOLU_net_flux_MgCO2e_0_04deg.tif')
        afolu_emis_wgs84_path = os.path.join(reproj_folder, 'AFOLU_gross_emis_MgCO2e_0_04deg.tif')
        afolu_remv_wgs84_path = os.path.join(reproj_folder, 'AFOLU_gross_remv_MgCO2_0_04deg.tif')
        save_array_as_geotif(data_afolu_net_wgs84,  lulucf_net_wgs84_path, afolu_net_wgs84_path,  main_logger)
        save_array_as_geotif(data_afolu_emis_wgs84, lulucf_net_wgs84_path, afolu_emis_wgs84_path, main_logger)
        save_array_as_geotif(data_afolu_remv_wgs84, lulucf_net_wgs84_path, afolu_remv_wgs84_path, main_logger)

        main_logger.info("Reprojecting AFOLU totals WGS84→Robinson")
        afolu_net_reproj_path  = reproject_to_robinson(afolu_net_wgs84_path,  reproj_folder, main_logger, reference_path=lulucf_net_reproj)
        afolu_emis_reproj_path = reproject_to_robinson(afolu_emis_wgs84_path, reproj_folder, main_logger, reference_path=lulucf_net_reproj)
        afolu_remv_reproj_path = reproject_to_robinson(afolu_remv_wgs84_path, reproj_folder, main_logger, reference_path=lulucf_net_reproj)
        data_afolu_net,  _ = read_raster_clipped(afolu_net_reproj_path,  bounding_box_proj)
        data_afolu_emis, _ = read_raster_clipped(afolu_emis_reproj_path, bounding_box_proj)
        data_afolu_remv, _ = read_raster_clipped(afolu_remv_reproj_path, bounding_box_proj)

    lulucf_slide_text_with_disclaimer = f"{lulucf_slide_text} \n {cn.legend_percentile_disclaimer}"


    ### Part 1: Net and gross LULUCF flux maps

    main_logger.info("\n\n\n---Part 1: Mapping net and gross LULUCF flux")

    lulucf_emis_core = f"LULUCF_gross_emis__{file_version_str}__ktCO2e_yr"
    jpeg_path_lulucf_emis = render_unidirectional_map(
        data_lulucf_emis, raster_extent, bounding_box_proj, country_shapefile,
        cn.emissions_colors_rgb, cn.emissions_percentiles,
        title_text=f"Gross land-based emissions\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_emis_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=True,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_gross_emissions__all_gases__MgCO2e_yr', 'Average gross emissions', uncertainty="2.6"),
    )

    lulucf_remv_core = f"LULUCF_gross_remv__{file_version_str}__ktCO2e_yr"
    jpeg_path_lulucf_remv = render_unidirectional_map(
        data_lulucf_remv, raster_extent, bounding_box_proj, country_shapefile,
        cn.removals_colors_rgb, cn.removals_percentiles,
        title_text=f"Gross land-based removals\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$ yr$^{{-1}}$",
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_remv_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        mask_positive=False,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_gross_removals__MgCO2_yr', 'Average gross removals', unit='Gt CO$_2$ yr$^{-1}$', uncertainty="3.1"),
    )

    lulucf_net_core = f"LULUCF_net_flux__{file_version_str}__ktCO2e_yr"
    jpeg_path_lulucf_net = render_divergent_map(
        data_lulucf_net, raster_extent, bounding_box_proj, country_shapefile,
        net_colors_rgb,
        title_text=f"Net land-based flux\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
        veg_analysis_years=cn.veg_year_range_str,
        non_pres_folder=non_pres_folder, pres_folder=pres_folder,
        jpeg_name=jpeg_name(lulucf_net_core, bounding_box_description),
        slide_text=lulucf_slide_text_with_disclaimer,
        logger=main_logger,
        bottom_annotation=_flux_annotation(df_stats, 'LULUCF_net_flux__MgCO2e_yr', 'Average net flux', uncertainty="4.0"),
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
        veg_net_core = f"vegetation_net_flux_all_pools_all_gases_{cn.veg_model_version_underscore}__{cn.veg_year_range_str}__ktCO2e_yr"
        jpeg_path_veg_net = render_divergent_map(
            data_veg_net_avg, raster_extent, bounding_box_proj, country_shapefile,
            net_colors_rgb,
            title_text=f"Net greenhouse gas flux\nAll vegetation pools, all gases\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            veg_analysis_years=cn.veg_year_range_str,
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(veg_net_core, bounding_box_description),
            slide_text=cn.veg_pres_text,
            logger=main_logger,
            percentile_multipliers=cn.net_percentiles,
            bottom_annotation=_flux_annotation(df_stats, 'veg__net_flux__all_C_pools__all_gases__MgCO2e_yr', 'Average net flux'),
        )

        # Panel b: Mineral soil net SOC change
        main_logger.info(f"  Creating mineral soil net change map")
        min_soil_core = f"mineral_soil_net__{file_version_str}__ktCO2_yr"
        jpeg_path_min_soil = render_divergent_map(
            data_min_soil, raster_extent, bounding_box_proj, country_shapefile,
            net_colors_rgb,
            title_text=f"Net mineral soil SOC change\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            veg_analysis_years=cn.veg_year_range_str,
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(min_soil_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
            bottom_annotation=_flux_annotation(df_stats, 'SOC_net__mineral_soil_extent__0_30cm_MgCO2_yr', 'Average net flux',
                                                unit='Gt CO$_2$ yr$^{-1}$'),
        )

        # Panel c: Organic soil gross emissions — weighted average across intervals, computed above
        main_logger.info(f"  Creating organic soil emissions map")

        org_soil_core = f"org_soil_gross_emis__{file_version_str}__ktCO2e_yr"
        jpeg_path_org_soil = render_unidirectional_map(
            data_org_soil, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Gross organic soil emissions\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(org_soil_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
            mask_positive=True,
            bottom_annotation=_flux_annotation(df_stats, 'org_soil_emis__all_gases__MgCO2e_yr', 'Average gross emissions'),
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

        pct_veg_emis, _ = read_raster_clipped(pct_veg_reproj_path, bounding_box_proj)
        pct_org_soil, _ = read_raster_clipped(pct_org_reproj_path, bounding_box_proj)
        pct_min_loss, _ = read_raster_clipped(pct_min_reproj_path, bounding_box_proj)

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


    ### Part 5: Agriculture gross emissions map
    if has_agriculture_inputs:
        main_logger.info("\n\n\n---Part 5: Agriculture gross emissions map")

        agri_core = "agriculture_gross_emis__ktCO2e_yr"
        jpeg_path_agri = render_unidirectional_map(
            data_agriculture, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Gross agriculture emissions\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(agri_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
            mask_positive=True,
        )
        main_logger.info(f"Part 5 done in {round(time.time() - start_time)}s: {uu.timestr()}")


    ### Part 6: AFOLU total maps (gross emissions | gross removals | net flux)
    if has_agriculture_inputs:
        main_logger.info("\n\n\n---Part 6: AFOLU total maps")

        afolu_emis_core = "AFOLU_gross_emis__ktCO2e_yr"
        jpeg_path_afolu_emis = render_unidirectional_map(
            data_afolu_emis, raster_extent, bounding_box_proj, country_shapefile,
            cn.emissions_colors_rgb, cn.emissions_percentiles,
            title_text=f"Gross AFOLU emissions\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(afolu_emis_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
            mask_positive=True,
        )

        afolu_remv_core = "AFOLU_gross_remv__ktCO2_yr"
        jpeg_path_afolu_remv = render_unidirectional_map(
            data_afolu_remv, raster_extent, bounding_box_proj, country_shapefile,
            cn.removals_colors_rgb, cn.removals_percentiles,
            title_text=f"Gross AFOLU removals\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$ yr$^{{-1}}$",
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(afolu_remv_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
            mask_positive=False,
        )

        afolu_net_core = "AFOLU_net_flux__ktCO2e_yr"
        jpeg_path_afolu_net = render_divergent_map(
            data_afolu_net, raster_extent, bounding_box_proj, country_shapefile,
            net_colors_rgb,
            title_text=f"Net AFOLU flux\n{cn.veg_outputs_years[0]}-{cn.veg_outputs_years[-1]}\nkt CO$_2$e yr$^{{-1}}$",
            veg_analysis_years=cn.veg_year_range_str,
            non_pres_folder=non_pres_folder, pres_folder=pres_folder,
            jpeg_name=jpeg_name(afolu_net_core, bounding_box_description),
            slide_text=lulucf_slide_text_with_disclaimer,
            logger=main_logger,
        )

        afolu_three_panel_core = "AFOLU_three_panel__emis_remv_net__ktCO2e_yr"
        jpeg_path_afolu_three_panel = f"{non_pres_folder}/{jpeg_name(afolu_three_panel_core, bounding_box_description)}.jpeg"
        mu.create_three_panel_map(
            jpeg_path_afolu_three_panel,
            jpeg_path_afolu_emis, jpeg_path_afolu_remv, jpeg_path_afolu_net,
            "", main_logger,
        )
        main_logger.info(f"Part 6 done in {round(time.time() - start_time)}s: {uu.timestr()}")


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

    main_logger.info("\nInput arguments:")
    main_logger.info(f"  lulucf_input_date:            {lulucf_input_date}")
    main_logger.info(f"  lulucf_model_type:            {lulucf_model_type}")
    main_logger.info(f"  lulucf_model_path_description:{lulucf_model_path_description}")
    main_logger.info(f"  parquet_path:                 {parquet_path}")
    main_logger.info(f"  veg_net_geotif:               {veg_net_geotif}")
    main_logger.info(f"  organic_soil_drained_s3:      {organic_soil_drained_s3}")
    main_logger.info(f"  organic_soil_burned_s3:       {organic_soil_burned_s3}")
    main_logger.info(f"  mineral_soil_s3:              {mineral_soil_s3}")
    main_logger.info(f"  veg_emis_geotif:              {veg_emis_geotif}")
    main_logger.info(f"  mineral_soil_loss_s3:         {mineral_soil_loss_s3}")
    main_logger.info(f"  cropland_geotif_s3:           {cropland_geotif_s3}")
    main_logger.info(f"  livestock_geotif_s3:          {livestock_geotif_s3}")
    main_logger.info(f"  center_latitude:              {center_latitude}")
    main_logger.info(f"  center_longitude:             {center_longitude}")
    main_logger.info(f"  lat_height:                   {lat_height}")
    main_logger.info(f"  bounding_box_description:     {bounding_box_description}")
    main_logger.info("\n")

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
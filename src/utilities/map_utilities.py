import math
import os
import boto3
import rasterio
import geopandas as gpd
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import time
import warnings

from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from scipy.stats import percentileofscore
from rasterio.windows import from_bounds
from pyproj import Transformer

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import log_utilities as lu
from src.utilities import universal_utilities as uu


def rgb_to_mpl(rgb):
    """
    Converts RGB from 0-255 range to matplotlib-compatible 0-1 range.
    :param rgb: Tuple of (R, G, B) in 0-255 range.
    :return: Tuple of (R, G, B) in 0-1 range.
    """
    return tuple(val / 255 for val in rgb)

def calculate_bbox_centered(main_logger, center_lat, center_lon, lat_height, aspect_ratio=2.0):
    """
    Given a center point (lat/lon), vertical height in degrees latitude,
    and desired width:height aspect ratio, returns a bounding box in degrees
    that maintains the visual proportions in the map projection.

    Returns: (lon_min, lat_min, lon_max, lat_max)
    """
    # Compute vertical range
    lat_min = center_lat - lat_height / 2
    lat_max = center_lat + lat_height / 2

    # Setup projection to Robinson (or your map projection)
    src_crs = "EPSG:4326"
    dst_crs = cn.Robinson_crs  # e.g. 'ESRI:54030'
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Project the vertical extent at the center longitude
    _, y_min = transformer.transform(center_lon, lat_min)
    _, y_max = transformer.transform(center_lon, lat_max)
    height_m = abs(y_max - y_min)

    # Desired width in projected meters
    width_m = height_m * aspect_ratio

    # Estimate how many degrees of longitude gives that width
    # Use small step to compute meters per degree lon at center_lat
    test_dx = 1.0
    x0, _ = transformer.transform(center_lon, center_lat)
    x1, _ = transformer.transform(center_lon + test_dx, center_lat)
    meters_per_degree_lon = abs(x1 - x0)

    # Required lon range in degrees
    lon_width = width_m / meters_per_degree_lon
    lon_min = center_lon - lon_width / 2
    lon_max = center_lon + lon_width / 2

    return (lon_min, lat_min, lon_max, lat_max)

def transform_bbox_to_robinson(bbox_deg, src_crs="EPSG:4326", dst_crs=None):
    """
    Transforms a bounding box from lat/lon (EPSG:4326) to Robinson projection.
    bbox_deg: (minx, miny, maxx, maxy) in degrees
    dst_crs: destination CRS (defaults to cn.Robinson_crs)
    Returns: (minx, miny, maxx, maxy) in meters (Robinson)
    """
    if dst_crs is None:
        dst_crs = cn.Robinson_crs  # e.g., 'ESRI:54030'

    minx, miny, maxx, maxy = bbox_deg
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Transform corners
    xmin_t, ymin_t = transformer.transform(minx, miny)
    xmax_t, ymax_t = transformer.transform(maxx, maxy)

    # Return projected bounding box
    return (min(xmin_t, xmax_t), min(ymin_t, ymax_t),
            max(xmin_t, xmax_t), max(ymin_t, ymax_t))

def reproject_raster(tif_unproj_s3, tif_reproj_local, main_logger):
    """
    Reprojects raster to Robinson projection if the output doesn't already exist.
    Only supports local output; input can be S3.
    """

    if not os.path.exists(tif_reproj_local):
        main_logger.info("  Reprojected raster does not exist. Reprojecting now...")

        with rasterio.open(tif_unproj_s3) as src:
            transform, width, height = calculate_default_transform(
                src.crs, cn.Robinson_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': cn.Robinson_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'nodata': 0,
                'compress': 'lzw'  # Optional: add compression
            })

            with rasterio.open(tif_reproj_local, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=cn.Robinson_crs,
                    resampling=Resampling.nearest
                )
    else:
        main_logger.info("  Reprojected raster already exists locally. Skipping reprojection.")

def check_and_reproject_shapefile(main_logger, shapefile_path, target_crs, reprojected_shapefile_path):
    """
    Checks if the shapefile is already projected to the target CRS.
    If not, reprojects the shapefile, saves it, and returns the reprojected shapefile.

    Parameters:
    - shapefile_path (str): Path to the input shapefile.
    - target_crs (str): The target CRS in PROJ format (e.g., "EPSG:4326" or "ESRI:54030").
    - reprojected_shapefile_path (str): Path to save the reprojected shapefile.

    Returns:
    - geopandas.GeoDataFrame: The original or reprojected shapefile.
    """

    # Checks if the reprojected shapefile already exists
    if os.path.exists(reprojected_shapefile_path):
        main_logger.info(f"  Reprojected shapefile already exists at {reprojected_shapefile_path}.")
        return gpd.read_file(reprojected_shapefile_path)

    # Loads the shapefile
    shapefile = gpd.read_file(shapefile_path)

    # Checks if the shapefile is already in the target CRS
    if shapefile.crs == target_crs:
        main_logger.info(f"  Shapefile is already projected to {target_crs}.")
        return shapefile

    # Reprojects the shapefile
    main_logger.info(f"  Reprojecting shapefile from {shapefile.crs} to {target_crs}.")
    shapefile = shapefile.to_crs(target_crs)

    # Saves the reprojected shapefile for future use
    shapefile.to_file(reprojected_shapefile_path)
    main_logger.info(f"  Reprojected shapefile saved to {reprojected_shapefile_path}.")

    return shapefile

def create_plot():
    """
    Creates matplotlib plot
    :return: ax and fig
    """
    fig, ax = plt.subplots(figsize=cn.panel_dims)
    return ax, fig

def remove_ticks(ax):
    """
    Removes ticks from matplotlib plot
    :param ax: graph
    :return: N/A
    """
    # Set map aesthetics
    # NOTE: can't use ax.set_axis_off() to remove axis ticks and labels because it also changes the background color back to white
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels

def create_divergent_legend_asymmetric(fig, vmin, vmax, title_text, tick_labels,
                                       year, colors_rgb, percentiles, percentile_0, main_logger,
                                       colorbar_height_multiplier=1.0, add_intermediate_ticks=False,
                                       show_direction_arrows=False, colorbar_left_offset=0.0):
    """
    Creates a vertical divergent colorbar legend whose colors match the map exactly.

    The map uses TwoSlopeNorm (neutral value always at 0.5 in colormap space) + 10 evenly-spaced
    colors via np.linspace.  This function replicates that approach so the legend and map agree:
    the same data value always shows the same color in both.  The neutral tick therefore sits at
    the visual midpoint of the bar (50% height), matching TwoSlopeNorm's behaviour.

    Parameters:
        fig: Matplotlib figure
        vmin: Minimum data value in display units (e.g., -14 kt)
        vmax: Maximum data value in display units (e.g., 17 kt)
        title_text: Text for the legend title
        tick_labels: Labels for the three anchor ticks (min, 0, max)
        year: Year string, for logging/debug
        colors_rgb: List of RGB tuples for the colormap (same length as used in the map)
        percentiles: (kept for backward compatibility; no longer used in legend construction)
        percentile_0: (kept for backward compatibility; no longer used in legend construction)
        colorbar_height_multiplier: Scale factor for colorbar height (default 1.0)
        add_intermediate_ticks: If True, adds ticks at 1/3 and 2/3 of each side's saturated value
        show_direction_arrows: If True, adds "Source"/"Sink" arrow annotations and strips direction
                               text (e.g., "(source)", "(sink)") from the existing tick labels
        colorbar_left_offset: Additional shift (in figure fraction) applied to the colorbar left
                              position. Use to move the bar right when direction arrows would otherwise
                              overflow the left edge of the panel (default 0.0).
    """
    main_logger.info(f"  Creating legend for {year}")

    # Optionally strips direction qualifiers from tick labels (e.g., "(source)", "(sink)", "(neutral)")
    if show_direction_arrows:
        tick_labels = [l.split('(')[0].strip() if isinstance(l, str) else l for l in tick_labels]

    # Converts RGB palette to hex
    net_colors_rgb_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors_rgb]

    # Builds colormap with evenly-spaced colors, exactly matching the map (map_net_flux uses
    # np.linspace(0, 1, n) as well).  Colors at equal intervals means each shade covers the same
    # fraction of the colourmap range, so the visual progression matches gross emissions colorbars.
    positions = np.linspace(0, 1, len(net_colors_rgb_hex))
    cmap = LinearSegmentedColormap.from_list("asymmetric_div", list(zip(positions, net_colors_rgb_hex)))

    # TwoSlopeNorm matches what the map uses: neutral (0) always maps to 0.5 in colormap space.
    # Using the same norm here ensures that any data value shows the same color in both map and legend.
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    main_logger.info(f"  Neutral tick at bar height 0.5 (TwoSlopeNorm convention)")

    # Adds the colorbar axis; height is scaled by colorbar_height_multiplier;
    # left position shifted by colorbar_left_offset to make room for direction arrows if needed
    cbar_ax = fig.add_axes([          # [left, bottom, width, height]
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2] + colorbar_left_offset,
        cn.colorbar_dimensions[1],
        cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[3] * colorbar_height_multiplier
    ])

    # Colorbar ticks are in DATA space (kt); TwoSlopeNorm positions them correctly on the bar
    cb = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")

    if add_intermediate_ticks:
        # 1/3 and 2/3 of the saturated values on each side, in data/display units.
        # These directly mirror the gross emissions intermediate ticks (1/3 and 2/3 of max range),
        # making the colour progressions comparable across panels.
        d_sink_1 = round(2 * vmin / 3)
        d_sink_2 = round(vmin / 3)
        d_src_1  = round(vmax / 3)
        d_src_2  = round(2 * vmax / 3)
        tick_data = [vmin, d_sink_1, d_sink_2, 0, d_src_1, d_src_2, vmax]
        tick_labels_full = [tick_labels[0],
                            f"{d_sink_1:.0f}", f"{d_sink_2:.0f}",
                            tick_labels[1],
                            f"{d_src_1:.0f}", f"{d_src_2:.0f}",
                            tick_labels[2]]
        cb.set_ticks(tick_data)
        cb.set_ticklabels(tick_labels_full, fontsize=cn.legend_fontsize)
    else:
        cb.set_ticks([vmin, 0, vmax])
        cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)

    # Adds "Source" and "Sink" arrow annotations to the left of the colorbar.
    # With TwoSlopeNorm, neutral is always at bar height 0.5, so arrow midpoints are fixed at
    # 0.25 (sink) and 0.75 (source).
    if show_direction_arrows:
        arrow_x = -0.6  # in axes fraction coords, to the left of the colorbar
        # Upward arrow indicating increasing source (top half of bar)
        cbar_ax.annotate('', xy=(arrow_x, 0.97), xytext=(arrow_x, 0.53),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=8),
                         annotation_clip=False)
        cbar_ax.text(arrow_x - 0.35, 0.75, 'Source', ha='center', va='center',
                     fontsize=cn.legend_fontsize, rotation=90,
                     transform=cbar_ax.transAxes, clip_on=False)
        # Downward arrow indicating increasing sink (bottom half of bar)
        cbar_ax.annotate('', xy=(arrow_x, 0.03), xytext=(arrow_x, 0.47),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=8),
                         annotation_clip=False)
        cbar_ax.text(arrow_x - 0.35, 0.25, 'Sink', ha='center', va='center',
                     fontsize=cn.legend_fontsize, rotation=90,
                     transform=cbar_ax.transAxes, clip_on=False)

    # Adds title above the bar
    cbar_ax.text(
        0, 1.05,  # x, y in axes coords
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cbar_ax.transAxes
    )

def create_unidirection_legend(fig, img, lower_lim_all_yrs, upper_lim_all_yrs, title_text, tick_labels,
                               year, colors_rgb, percentiles, main_logger,
                               colorbar_height_multiplier=1.0, add_intermediate_ticks=False,
                               label_divisor=1.0, colorbar_left_offset=0.0):
    """
    Creates a vertical colorbar legend with a left-aligned title above it.
    :param fig: The figure
    :param img: The image
    :param lower_lim_all_yrs: minimum value to use in scaling legend colors (raw data units)
    :param upper_lim_all_yrs: maximum value to use in scaling legend colors (raw data units)
    :param title_text: Title for legend
    :param tick_labels: Tick labels for legend (already in display units; used as-is for extremes)
    :param colorbar_height_multiplier: Scale factor for colorbar height (default 1.0)
    :param add_intermediate_ticks: If True, adds ticks at 1/3 and 2/3 of the range
    :param label_divisor: Divisor applied to intermediate tick data values before formatting as labels.
                          Use 1e3 when lower/upper limits are in Mg but labels should be in kt (default 1.0).
    :param colorbar_left_offset: Additional shift (in figure fraction) applied to the colorbar left
                                 position (default 0.0).
    :return: N/A
    """
    main_logger.info(f"  Creating legend for {year}")

    # Converts net flux RGB palette to hex
    # per https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/68d6d26f-b054-8323-98bb-731a86582e74
    net_colors_rgb_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors_rgb]

    # Add a vertical colorbar (legend) in the bottom-left of the map;
    # height scaled by colorbar_height_multiplier; position shifted right by colorbar_left_offset
    cbar_ax = fig.add_axes([          # [left, bottom, width, height]
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2] + colorbar_left_offset,
        cn.colorbar_dimensions[1],
        cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[3] * colorbar_height_multiplier
    ])
    cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")

    if add_intermediate_ticks:
        # Adds ticks at 1/3 and 2/3 of the range between lower and upper limits.
        # Tick positions are in raw data units (e.g. Mg); labels are divided by label_divisor
        # to convert to display units (e.g. kt when label_divisor=1e3).
        data_range = upper_lim_all_yrs - lower_lim_all_yrs
        t1 = lower_lim_all_yrs + data_range / 3
        t2 = lower_lim_all_yrs + 2 * data_range / 3
        tick_positions = [lower_lim_all_yrs, t1, t2, upper_lim_all_yrs]
        tick_labels_full = [tick_labels[0],
                            f"{round(t1 / label_divisor):.0f}",
                            f"{round(t2 / label_divisor):.0f}",
                            tick_labels[1]]
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(tick_labels_full, fontsize=cn.legend_fontsize)
    else:
        # Set custom ticks and labels for the colorbar
        cb.set_ticks([lower_lim_all_yrs, upper_lim_all_yrs])  # Set the ticks at the minimum, zero, and maximum
        cb.set_ticklabels(tick_labels, fontsize=cn.legend_fontsize)  # Format the labels

    # Add a left-aligned, multi-row title above the colorbar
    cbar_ax.text(
        0, 1.1,  # Adjust the x (horizontal) and y (vertical) coordinates for the title position
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",  # Horizontally align the text to the left
        va="bottom",  # Vertically align the text
        transform=cbar_ax.transAxes  # Use axes coordinates for positioning
    )


# Creates legend for categorical map of fraction of gross emissions from LULUCF components
def create_categorical_fraction_legend(fig, img, title_text, boundaries, class_labels, main_logger):
    main_logger.info(f"  Creating categorical fraction legend")

    cbar_ax = fig.add_axes([
        cn.colorbar_dimensions[0] + cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[1],
        cn.colorbar_dimensions[2],
        cn.colorbar_dimensions[3]
    ])
    cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")

    midpoints = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries) - 1)]
    cb.set_ticks(midpoints)
    cb.set_ticklabels(class_labels, fontsize=cn.legend_fontsize)

    cbar_ax.text(
        0, 1.1,
        title_text,
        fontsize=cn.legend_fontsize,
        ha="left",
        va="bottom",
        transform=cbar_ax.transAxes
    )


def rgb_to_mpl_palette(rgb_palette):
    """
    Converts a list of RGB colors from 0-255 range to 0-1 range for Matplotlib.

    Parameters:
    - rgb_palette (list of tuples): List of RGB tuples (R, G, B) in 0-255 range.

    Returns:
    - list: List of RGB tuples (R, G, B) in 0-1 range.
    """
    return [tuple(val / 255 for val in rgb) for rgb in rgb_palette]


def percentile_for_0(data):

    # Masks invalid values (e.g., NoData or zero values)
    valid_data = data[data != 0]  # Excludes zeros (or use np.ma.masked_invalid for general NoData masking)

    # Ensures valid_data is not empty
    if len(valid_data) == 0:
        raise ValueError("No valid data found in the raster.")

    # Calculates the percentile of 0
    percentile_0 = percentileofscore(valid_data, 0, kind="mean")

    return percentile_0

def set_ocean_color(ax):
    # Sets the background color of the map
    ax.set_facecolor(rgb_to_mpl(cn.ocean_color))  # Set the background color

def plot_country_polygons(ax, shapefile):
    """
    Plots the shapefile polygons or multipolygons with a specified color. zorder sets the order of drawing.
    :param ax: figure
    :param shapefile: shapefile to draw
    :return: N/A
    """

    for geom in shapefile.geometry:
        if isinstance(geom, Polygon):
            # Single Polygon
            x, y = geom.exterior.xy
            ax.fill(x, y, color=rgb_to_mpl(cn.land_bkgrnd), zorder=1)
        elif isinstance(geom, MultiPolygon):
            # MultiPolygon: Iterate through each Polygon in the MultiPolygon
            for part in geom.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, color=rgb_to_mpl(cn.land_bkgrnd), zorder=1)

def plot_raster(ax, cmap, extent, masked_data, norm):
    """
    Plots raster
    :param ax: figure
    :param cmap: colormap
    :param extent: raster extent
    :param masked_data: masked data (no NoData/0s) to plot
    :param norm: data normalization
    :return: image
    """

    # Turn off interpolation to prevent showing boundaries around pixel patches
    # (matters mostly for areas with sparse emissions in gross emissions map, like boreal forest)
    # https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6964d5d4-12b4-8325-aec3-c7bbed008ac9
    img = ax.imshow(masked_data, cmap=cmap, norm=norm, extent=extent,
                    origin='upper', interpolation='none', zorder=2)
    return img

def plot_country_boundaries(ax, shapefile):

    # Overlaya shapefile boundaries (e.g., country borders)
    # zorder determines the order of appearance in the figure
    shapefile.boundary.plot(ax=ax, edgecolor=rgb_to_mpl(cn.boundary_color), linewidth=cn.boundary_width, zorder=3)

def save_jpeg(out_jpeg, year, main_logger):

    main_logger.info(f"  Saving {out_jpeg} for {year}")
    plt.savefig(out_jpeg, dpi=cn.dpi_jpeg, bbox_inches="tight", pad_inches=0)

def save_pres_non_pres_jpegs(ax, out_jpeg, out_jpeg_for_pres, year, pres_text, main_logger):

    # Adds year label inside the plot, bottom center
    ax.text(
        0.5, 0.07, str(year),   # x=50% (center), slightly into the panel space
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=18, weight="bold", color="black"
    )

    # Saves jpeg without journal name and update notes in bottom right
    save_jpeg(out_jpeg, year, main_logger)

    # Note in bottom right of panel
    ax.text(0.99, 0.09, pres_text, transform=ax.transAxes, fontsize=7,   #Horizontal (lower value moves left), vertical (lower value moves down)
            ha="right", va="top", color="black")

    # Saves jpeg with journal name and update notes in bottom right
    save_jpeg(out_jpeg_for_pres, year, main_logger)
    plt.close()

    return out_jpeg_for_pres

# Creates gifs of timeseries (fast and slow)
def create_gif(out_maps_for_gif, main_logger, output_gif_path):
    """
    Create a GIF from JPEGs using a consistent color palette across frames.

    Parameters:
        jpeg_folder (str): Path to folder containing .jpeg frames.
        output_gif_path (str): Where to save the output .gif.
    """
    # Step 1: Loads JPEG frames
    frames = [Image.open(f).convert("RGB") for f in out_maps_for_gif]

    # Step 2: Builds a global color palette.
    # I was finding that even if the color palettes looked the same in each jpeg,
    # they were different by year in the gif for some reason.
    # Using a single color palette fixes that.
    main_logger.info("\n\n---Generating global color palette from all frames...")
    combined_height = frames[0].height * len(frames)
    combined = Image.new("RGB", (frames[0].width, combined_height))
    for i, frame in enumerate(frames):
        combined.paste(frame, (0, i * frames[0].height))

    # Gets adaptive palette from composite image
    palette_image = combined.convert("P", palette=Image.ADAPTIVE, colors=256)
    global_palette = palette_image.getpalette()

    # Step 3: Applies palette to each frame.
    # Needs this specific dithering to prevent the gif legend from being blocky,
    # per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/695e7ea8-60dc-832b-ba01-c4852bedbc57.
    # It was looking blocky otherwise.
    palettized_frames = [
        f.quantize(palette=palette_image, dither=Image.FLOYDSTEINBERG)
        for f in frames
    ]

    # Step 4: Saves GIF
    main_logger.info(f"Saving animated GIF to: {output_gif_path}")
    palettized_frames[0].save(
        f"{output_gif_path}_fast.gif",
        save_all=True,
        append_images=palettized_frames[1:],
        duration=1000,
        loop=0,
        optimize=False,
        disposal=2  # Clears previous frame
    )

    palettized_frames[0].save(
        f"{output_gif_path}_slow.gif",
        save_all=True,
        append_images=palettized_frames[1:],
        duration=2500,
        loop=0,
        optimize=False,
        disposal=2  # Clears previous frame
    )

# Saves the mean of years as geotif
def save_mean_annual_geotif(local_reproj_folder, pattern_segment, year_path_reproj, yearly_data_stack, main_logger):

    # Computes average across years (ignoring NaNs)
    # Known issue is that if there are only two years of values and they are equal with opposite signs (e.g., -0.5 and 0.5),
    # they cancel out, the mean is 0, and the pixel shows up as NoData (and doesn't get a color).
    # This is pretty rare and doesn't affect the global visualization.
    main_logger.info("  Computing and saving annual average raster")
    with warnings.catch_warnings():  # Suppresses RunTime warning about empty slices (for pixels that are NaN every year)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_data = np.nanmean(np.stack(yearly_data_stack, axis=0), axis=0)

    # Masks no-data for writing (0 as nodata to match others)
    mean_data_to_write = np.where(np.isnan(mean_data), 0, mean_data)

    # Gets metadata from any of the reprojected rasters (uses the last one) to save geotif of mean
    with rasterio.open(year_path_reproj) as src:
        profile = src.profile
        profile.update({
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'nodata': 0  # Match your other rasters
        })

        out_path_mean_geotiff = f"{local_reproj_folder}/{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}_mean_global_reproj.tif"

        with rasterio.open(out_path_mean_geotiff, 'w', **profile) as dst:
            dst.write(mean_data_to_write, 1)

    main_logger.info(f"  Saved mean annual GeoTIFF to: {out_path_mean_geotiff}")

    return mean_data


# Makes jpegs and gifs of net fluxes
def map_net_flux(s3_folders, model_type, model_path_description,
                 local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
                 colors_rgb, country_shapefile, main_logger, bounding_box=None, bounding_box_description=None):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    # First pass: Reprojects input rasters
    for i, year in enumerate(cn.interval_end_years_annual[0:]):
    # for i, year in enumerate(cn.interval_end_years_annual[2:3]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        main_logger.info(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        main_logger.info(f"Unprojected raster: {year_path_unproj}")
        main_logger.info(f"Reprojected raster: {year_path_reproj}")

        # Reprojects raster, if needed
        reproject_raster(year_path_unproj, year_path_reproj, main_logger)

    # Second pass: reads all reprojected year rasters, computes mean, derives shared legend limits from mean's non-zero pixels
    main_logger.info("\n\n---Computing mean raster to derive shared legend limits...")
    yearly_data_for_limits = []

    for i, year in enumerate(cn.interval_end_years_annual[0:]):
        s3_folder = s3_folders[i]
        parts = s3_folder.strip('/').split('/')

        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        interval_idx = parts.index("annual_intervals")
        interval_segment = parts[interval_idx + 1]

        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        with rasterio.open(year_path_reproj) as src:
            if bounding_box_proj is not None:
                window = from_bounds(*bounding_box_proj, src.transform)
                data = src.read(1, window=window)
            else:
                data = src.read(1)
        yearly_data_for_limits.append(data.astype('float32'))

    mean_for_limits = np.mean(np.stack(yearly_data_for_limits, axis=0), axis=0)
    non_zero_mean_for_limits = mean_for_limits[mean_for_limits != 0]

    percentile_for_saturation = 1
    breaks_all_yrs = np.percentile(non_zero_mean_for_limits, [percentile_for_saturation, (100 - percentile_for_saturation)])
    lower_lim_all_yrs = breaks_all_yrs[0]
    global_neutral = 0
    upper_lim_all_yrs = breaks_all_yrs[-1]

    main_logger.info("Legend limits from mean raster (non-zero pixels):")
    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_all_yrs}")
    main_logger.info(f"  neutral: {global_neutral}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_all_yrs}")

    rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100
    rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100
    tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}  (sink)",
                   f"{0}        (neutral)",
                   f"> {rounded_upper_lim_all_yrs:.0f}  (source)"]


    # Final pass: Iterates through modeled years to create the annual jpegs

    # Stores yearly arrays to compute annual average for annual average map
    yearly_data_stack = []
    yearly_masked_data = []

    for i, year in enumerate(cn.interval_end_years_annual[0:]):
    # for i, year in enumerate(cn.interval_end_years_annual[0:2]): # For testing a specific year
    # for i, year in enumerate(cn.interval_end_years_annual[0:4]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        main_logger.info(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reads raster data for year
        with rasterio.open(year_path_reproj) as src:

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
        percentile_0 = percentile_for_0(data)
        main_logger.info(f"  0 is at the {percentile_0}th percentile of the raster for {year}.")
        percentiles = [percentile_0*cn.net_percentiles[0], percentile_0*cn.net_percentiles[1], percentile_0*cn.net_percentiles[2],
                       percentile_0*cn.net_percentiles[3], percentile_0*cn.net_percentiles[4],
                       percentile_0*cn.net_percentiles[5], percentile_0*cn.net_percentiles[6], percentile_0*cn.net_percentiles[7],
                       percentile_0*cn.net_percentiles[8], percentile_0*cn.net_percentiles[9]]
        # print("percentiles:", percentiles)

        main_logger.info(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        main_logger.info(f"  Masking raster for {year} to non-0 values")
        masked_data = np.ma.masked_where(data == 0, data)
        yearly_masked_data.append(masked_data)

        # For map (not legend)
        norm = TwoSlopeNorm(
            vmin=lower_lim_all_yrs,
            vcenter=global_neutral,
            vmax=upper_lim_all_yrs
        )

        main_logger.info(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
        if bounding_box_proj is not None:
            bbox_geom = box(*bounding_box_proj)
            country_shapefile = country_shapefile.clip(bbox_geom)

        # Plots the country polygons first
        plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = list(raster_extent)

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, country_shapefile)

        # Explicitly sets the bounding box for the plot image
        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Modifies the legend title based on the input.
        if "all_gases" in pattern_segment:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
        else:
            title_text = f"Net greenhouse gas flux\nAll vegetation pools, CO2 only\nkt CO$_2$e yr$^{{-1}}$"

        # Creates legend
        create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                           title_text, tick_labels,
                                           year, colors_rgb, percentiles, percentile_0, main_logger,
                                           colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
                                           show_direction_arrows=True, colorbar_left_offset=0.05)

        # Removes axis ticks and labels
        remove_ticks(ax)

        pattern_segment_revised = pattern_segment.replace("MgCO2", "ktCO2")  # Replaces Mg with the mapped unit of kt
        core_jpeg_name = f"veg_{pattern_segment_revised}_yr__{year}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
        if bounding_box_description:  # Adds bounding box description to file name, if supplied
            core_jpeg_name = f"{core_jpeg_name}_{bounding_box_description}"
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year, cn.veg_pres_text, main_logger)

        out_maps_for_gif.append(out_jpeg_for_pres)

        # Adds annual data to stack for creating annual average map.
        # Uses 0s in calculation of mean (doesn't replace them with NaN).
        yearly_data_stack.append(data)

        # Saves the percentile for neutral flux (0) for use in the mean map.
        # Calculating the neutral flux for the mean map was problematic because of NaNs. This should be close enough.
        if i == 0:
            percentile_0_ref = percentile_0

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}"
    create_gif(
        out_maps_for_gif, main_logger,
        output_gif_path=f"{local_gif_folder}/{gif_base_name}"
    )


    ### Creates map of annual average

    main_logger.info(f"\n\n---Mapping mean across all years")
    mean_data = save_mean_annual_geotif(local_reproj_folder, pattern_segment, year_path_reproj, yearly_data_stack, main_logger)

    # Mask 0s from the mean raster for mapping only (e.g., water)-- not used for calculating percentiles
    masked_mean_data = np.ma.masked_where(mean_data==0, mean_data)

    main_logger.info("Plotting annual average map")

    ax, fig = create_plot()
    set_ocean_color(ax)

    plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)  # Use the extent from last year (they should all match)

    img = plot_raster(ax, cmap, extent, masked_mean_data, norm)
    plot_country_boundaries(ax, country_shapefile)

    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Title for average map
    if "all_gases" in pattern_segment:
        title_text = "Mean annual net greenhouse gas flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
    else:
        title_text = "Mean annual net greenhouse gas flux\nAll vegetation pools, CO$_2$ only\nkt CO$_2$e yr$^{{-1}}$"

    # 0th percentile for the mean of the timeseries
    percentile_0_mean = percentile_for_0(mean_data)
    main_logger.info(f"  0 is at the {percentile_0_mean}th percentile of the mean raster")
    percentiles_avg = [percentile_0_mean * cn.net_percentiles[0], percentile_0_mean * cn.net_percentiles[1], percentile_0_mean * cn.net_percentiles[2],
                       percentile_0_mean * cn.net_percentiles[3], percentile_0_mean * cn.net_percentiles[4],
                       percentile_0_mean * cn.net_percentiles[5], percentile_0_mean * cn.net_percentiles[6], percentile_0_mean * cn.net_percentiles[7],
                       percentile_0_mean * cn.net_percentiles[8], percentile_0_mean * cn.net_percentiles[9]]

    create_divergent_legend_asymmetric(fig, rounded_lower_lim_all_yrs, rounded_upper_lim_all_yrs,
                                       title_text, tick_labels,
                                       "avg", colors_rgb, percentiles_avg, percentile_0_mean, main_logger,
                                       colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
                                       show_direction_arrows=True, colorbar_left_offset=0.05)

    remove_ticks(ax)

    # Save the JPEG
    core_jpeg_name_avg = f"veg_{pattern_segment_revised}__mean_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_avg += f"_{bounding_box_description}"

    jpeg_path_avg = f"{local_jpeg_non_pres_folder}/{core_jpeg_name_avg}.jpeg"
    jpeg_for_pres_path_avg = f"{local_jpeg_pres_folder}/{core_jpeg_name_avg}__for_pres.jpeg"
    save_pres_non_pres_jpegs(ax, jpeg_path_avg, jpeg_for_pres_path_avg, "",
                             cn.veg_pres_text, main_logger)

    ### Creates 9-panel map (one panel per year, legend on first panel only)

    def _legend_fn_net(ax, _img):
        _vmin = rounded_lower_lim_all_yrs
        _vmax = rounded_upper_lim_all_yrs
        if "all_gases" in pattern_segment:
            _title = f"Net GHG flux\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
        else:
            _title = f"Net GHG flux\nAll vegetation pools, CO$_2$ only\nkt CO$_2$e yr$^{{-1}}$"
        _norm_legend = TwoSlopeNorm(vmin=_vmin, vcenter=0, vmax=_vmax)
        _sm = plt.cm.ScalarMappable(norm=_norm_legend, cmap=cmap)
        _sm.set_array([])
        cbar_ax = ax.inset_axes([0.01, 0.08, 0.05, 0.38])
        cb = plt.colorbar(_sm, cax=cbar_ax, orientation="vertical")
        d_sink_1 = round(2 * _vmin / 3)
        d_sink_2 = round(_vmin / 3)
        d_src_1  = round(_vmax / 3)
        d_src_2  = round(2 * _vmax / 3)
        cb.set_ticks([_vmin, d_sink_1, d_sink_2, 0, d_src_1, d_src_2, _vmax])
        cb.set_ticklabels([f"< {_vmin:.0f}", f"{d_sink_1:.0f}", f"{d_sink_2:.0f}", "0",
                           f"{d_src_1:.0f}", f"{d_src_2:.0f}", f"> {_vmax:.0f}"],
                          fontsize=cn.legend_fontsize)
        cbar_ax.text(0, 1.05, _title, fontsize=cn.legend_fontsize, ha="left", va="bottom",
                     transform=cbar_ax.transAxes)

    core_jpeg_name_nine = f"veg_{pattern_segment_revised}__9panel_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_nine += f"_{bounding_box_description}"
    nine_panel_jpeg = f"{local_jpeg_non_pres_folder}/{core_jpeg_name_nine}.jpeg"

    create_nine_panel_map(nine_panel_jpeg, yearly_masked_data, cmap, norm,
                          raster_extent, country_shapefile, cn.interval_end_years_annual,
                          bounding_box_proj, _legend_fn_net, main_logger)

    series_end_time = time.time()
    main_logger.info(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")


# Makes jpeg of gross fluxes
def map_gross(s3_folders, model_type, model_path_description,
              local_reproj_folder, local_jpeg_non_pres_folder, local_jpeg_pres_folder, local_gif_folder,
              colors_rgb, percentiles, main_logger, country_shapefile, bounding_box=None, bounding_box_description=None):

    series_start_time = time.time()

    out_maps_for_gif = []

    # If bounding_box was given in degrees, transforms to match the raster CRS (Robinson)
    if bounding_box is not None:
        bounding_box_proj = transform_bbox_to_robinson(bounding_box)
    else:
        bounding_box_proj = None

    # First pass: Reprojects input rasters
    for i, year in enumerate(cn.interval_end_years_annual[0:]):
    # for i, year in enumerate(cn.interval_end_years_annual[2:3]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input pattern
        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names before and after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_unproj = f"{s3_folder}{year_file}.tif"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        main_logger.info(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        main_logger.info(f"Unprojected raster: {year_path_unproj}")
        main_logger.info(f"Reprojected raster: {year_path_reproj}")

        # Reprojects raster, if needed
        reproject_raster(year_path_unproj, year_path_reproj, main_logger)

    # Second pass: reads all reprojected year rasters, computes mean, derives shared legend limits from mean's non-zero pixels
    main_logger.info("\n\n---Computing mean raster to derive shared legend limits...")
    yearly_data_for_limits = []

    for i, year in enumerate(cn.interval_end_years_annual[0:]):
        s3_folder = s3_folders[i]
        parts = s3_folder.strip('/').split('/')

        pattern_idx = parts.index(f"version_{cn.veg_model_version_underscore}__{model_type}__{model_path_description}")
        pattern_segment = parts[pattern_idx + 1]

        interval_idx = parts.index("annual_intervals")
        interval_segment = parts[interval_idx + 1]

        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        with rasterio.open(year_path_reproj) as src:
            if bounding_box_proj is not None:
                window = from_bounds(*bounding_box_proj, src.transform)
                data = src.read(1, window=window)
            else:
                data = src.read(1)
        yearly_data_for_limits.append(data.astype('float32'))

    mean_for_limits = np.mean(np.stack(yearly_data_for_limits, axis=0), axis=0)
    non_zero_mean_for_limits = mean_for_limits[mean_for_limits != 0]

    percentile_for_saturation = 1
    breaks_all_yrs = np.percentile(non_zero_mean_for_limits, [percentile_for_saturation, (100 - percentile_for_saturation)])
    lower_lim_all_yrs = breaks_all_yrs[0]
    upper_lim_all_yrs = breaks_all_yrs[-1]

    main_logger.info("Legend limits from mean raster (non-zero pixels):")
    main_logger.info(f"  lower limit ({percentile_for_saturation} percentile): {lower_lim_all_yrs}")
    main_logger.info(f"  upper limit ({(100 - percentile_for_saturation)} percentile): {upper_lim_all_yrs}")

    rounded_lower_lim_all_yrs = math.ceil(lower_lim_all_yrs / 10 ** 3 * 100) / 100
    rounded_upper_lim_all_yrs = math.floor(upper_lim_all_yrs / 10 ** 3 * 100) / 100

    # Legend labels depend on what exact input is displayed
    if "removals" in pattern_segment:
        tick_labels = [f"< {rounded_lower_lim_all_yrs:.0f}", 0]
        title_text = f"Gross removals\nAll vegetation pools\nkt CO$_2$ yr$^{{-1}}$"
    elif "all_gases" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, all gases\nkt CO$_2$e yr$^{{-1}}$"
    elif "non_CO2_only" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, non-CO$_2$ only\nkt CO$_2$e yr$^{{-1}}$"
    elif "CO2_only" in pattern_segment:
        tick_labels = [0, f"> {rounded_upper_lim_all_yrs:.0f}"]
        title_text = f"Gross emissions\nAll vegetation pools, CO$_2$ only\nkt CO$_2$ yr$^{{-1}}$"
    else:
        tick_labels = ["N/A", "N/A"]
        title_text = ""
        main_logger.info("Can't generate tick labels")
    main_logger.info(f"tick labels {tick_labels}")


    # Final pass: Iterates through modeled years to create the annual jpegs

    # Stores yearly arrays to compute annual average for annual average map
    yearly_data_stack = []
    yearly_masked_data = []

    for i, year in enumerate(cn.interval_end_years_annual[0:]):
    # for i, year in enumerate(cn.interval_end_years_annual[0:2]): # For testing a specific year
    # for i, year in enumerate(cn.interval_end_years_annual[0:4]): # For testing a specific year

        # The s3 folder to process for this year
        s3_folder = s3_folders[i]

        # All the components of the input s3 path
        parts = s3_folder.strip('/').split('/')

        # Gets the segment for the input interval
        interval_idx = parts.index(f"annual_intervals")
        interval_segment = parts[interval_idx + 1]

        # Names after reprojection
        year_file = f"{pattern_segment}{cn.flux_aggreg_pixel_meaning}_v{cn.veg_model_version_underscore}_{interval_segment}_global"
        year_path_reproj = f"{local_reproj_folder}/{year_file}_reproj.tif"

        main_logger.info(f"\n\n---Mapping {pattern_segment} for {year} from {year_file}")

        # Reads raster data for year
        with rasterio.open(year_path_reproj) as src:

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

        # Matches percentile breaks with colors.
        # Normalizes percentiles to a 0-1 scale.
        main_logger.info(f"  Calculating percentiles and breaks for {year}")

        # Converts RGB color palette to matplotlib color palette
        colors_matplotlib = rgb_to_mpl_palette(colors_rgb)

        # Matches percentile breaks with colors for the map.
        # Normalizes percentiles to a 0-1 scale.
        percentiles_normalized = np.linspace(0, 1, len(percentiles))
        # print("percentiles_normalized:", percentiles_normalized)
        cmap = LinearSegmentedColormap.from_list("custom_colormap", list(zip(percentiles_normalized, colors_matplotlib)))

        main_logger.info(f"  Masking raster to non-0 values for {year}")
        if "removals" in year_path_reproj:
            masked_data = np.ma.masked_where(data >= 0, data)

            # This colors all 0-value pixels, leaving non-0s white.
            # It clearly shows that more of Australia is non-0, but I just can't get it to be symbolized in any masking.
            # masked_data = np.ma.masked_where(data < 0, data)
        elif "emis" in year_path_reproj:
            masked_data = np.ma.masked_where(data <= 0, data)
        else:
            masked_data = np.ma.masked_where(data == 0, data)
            main_logger.info("Not using either emissions or removals")

        yearly_masked_data.append(masked_data)
        main_logger.info(f"  Normalizing for {year}")
        # Normalizes the data for the colormap
        norm = Normalize(vmin=lower_lim_all_yrs, vmax=upper_lim_all_yrs)

        main_logger.info(f"  Plotting map for {year}")
        ax, fig = create_plot()

        # Sets the ocean color
        set_ocean_color(ax)

        # Limits shapefile to focal extent (if requested)
        if bounding_box_proj is not None:
            bbox_geom = box(*bounding_box_proj)
            country_shapefile = country_shapefile.clip(bbox_geom)

        # Plots the country polygons first
        plot_country_polygons(ax, country_shapefile)

        # Raster extent
        extent = list(raster_extent)

        # Plots the raster next
        img = plot_raster(ax, cmap, extent, masked_data, norm)

        # Plots the country boundaries on top
        plot_country_boundaries(ax, country_shapefile)

        # Explicitly sets the bounding box for the plot image
        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        # Legend for gross fluxes
        create_unidirection_legend(fig, img, lower_lim_all_yrs, upper_lim_all_yrs,
                                   title_text, tick_labels,
                                   year, colors_rgb, percentiles, main_logger,
                                   colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
                                   label_divisor=1e3, colorbar_left_offset=0.05)

        # Removes axis ticks and labels
        remove_ticks(ax)

        pattern_segment_revised = pattern_segment.replace("MgCO2", "ktCO2")  # Replaces Mg with the mapped unit of kt
        core_jpeg_name = f"veg_{pattern_segment_revised}_yr__{year}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
        jpeg_path = f"{local_jpeg_non_pres_folder}/{core_jpeg_name}.jpeg"
        jpeg_for_pres_path = f"{local_jpeg_pres_folder}/{core_jpeg_name}__for_pres.jpeg"

        # Saves two versions of the map: without and with a source note in the bottom right
        out_jpeg_for_pres = save_pres_non_pres_jpegs(ax, jpeg_path, jpeg_for_pres_path, year, cn.veg_pres_text, main_logger)

        out_maps_for_gif.append(out_jpeg_for_pres)

        # Adds annual data to stack for creating annual average map.
        # Uses 0s in calculation of mean (doesn't replace them with NaN).
        yearly_data_stack.append(data)

    # Creates gifs of timeseries
    gif_base_name = f"veg_{pattern_segment_revised}__{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}"
    create_gif(out_maps_for_gif, main_logger, output_gif_path=f"{local_gif_folder}/{gif_base_name}")

    ### Creates map of annual average

    main_logger.info(f"\n\n---Mapping mean across all years")
    mean_data = save_mean_annual_geotif(local_reproj_folder, pattern_segment, year_path_reproj, yearly_data_stack, main_logger)

    # Mask 0s from the mean raster for mapping only (e.g., water)
    masked_mean_data = np.ma.masked_where(mean_data==0, mean_data)

    main_logger.info("  Plotting annual average map")

    ax, fig = create_plot()
    set_ocean_color(ax)

    plot_country_polygons(ax, country_shapefile)
    extent = list(raster_extent)  # Use the extent from last year (they should all match)

    img = plot_raster(ax, cmap, extent, masked_mean_data, norm)
    plot_country_boundaries(ax, country_shapefile)

    if bounding_box_proj is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Legend for gross fluxes
    create_unidirection_legend(fig, img, lower_lim_all_yrs, upper_lim_all_yrs,
                               title_text, tick_labels,
                               'avg', colors_rgb, percentiles, main_logger,
                               colorbar_height_multiplier=1.8, add_intermediate_ticks=True,
                               label_divisor=1e3, colorbar_left_offset=0.05)

    remove_ticks(ax)

    # Saves the JPEG
    core_jpeg_name_avg = f"veg_{pattern_segment_revised}__mean_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_avg += f"_{bounding_box_description}"

    jpeg_path_avg = f"{local_jpeg_non_pres_folder}/{core_jpeg_name_avg}.jpeg"
    jpeg_for_pres_path_avg = f"{local_jpeg_pres_folder}/{core_jpeg_name_avg}__for_pres.jpeg"
    save_pres_non_pres_jpegs(ax, jpeg_path_avg, jpeg_for_pres_path_avg, f'{cn.interval_end_years_annual[0]}-{cn.interval_end_years_annual[-1]}',
                             cn.veg_pres_text, main_logger)

    ### Creates 9-panel map (one panel per year, legend on first panel only)

    def _legend_fn_gross(ax, img):
        cbar_ax = ax.inset_axes([0.01, 0.08, 0.05, 0.38])
        cb = plt.colorbar(img, cax=cbar_ax, orientation="vertical")
        _range = upper_lim_all_yrs - lower_lim_all_yrs
        t1 = lower_lim_all_yrs + _range / 3
        t2 = lower_lim_all_yrs + 2 * _range / 3
        cb.set_ticks([lower_lim_all_yrs, t1, t2, upper_lim_all_yrs])
        cb.set_ticklabels([tick_labels[0], f"{round(t1 / 1e3):.0f}", f"{round(t2 / 1e3):.0f}", tick_labels[1]],
                          fontsize=cn.legend_fontsize)
        cbar_ax.text(0, 1.1, title_text, fontsize=cn.legend_fontsize, ha="left", va="bottom",
                     transform=cbar_ax.transAxes)

    core_jpeg_name_nine = f"veg_{pattern_segment_revised}__9panel_{cn.interval_end_years_annual[0]}_{cn.interval_end_years_annual[-1]}__v{cn.veg_model_version_underscore}__{uu.timestr()[0:8]}"
    if bounding_box_description:
        core_jpeg_name_nine += f"_{bounding_box_description}"
    nine_panel_jpeg = f"{local_jpeg_non_pres_folder}/{core_jpeg_name_nine}.jpeg"

    create_nine_panel_map(nine_panel_jpeg, yearly_masked_data, cmap, norm,
                          raster_extent, country_shapefile, cn.interval_end_years_annual,
                          bounding_box_proj, _legend_fn_gross, main_logger)

    series_end_time = time.time()
    main_logger.info(f"{pattern_segment} took {round(series_end_time - series_start_time)} seconds: {uu.timestr()}")

def create_three_panel_map(three_panel_jpeg, top_jpeg, middle_jpeg, bottom_jpeg, year, main_logger, panel_labels=None):
    """
    Creates a three-panel map showing emissions, removals, and net flux.
    """
    main_logger.info("Creating three-panel map")

    # Loads individual panel images
    top_img = plt.imread(top_jpeg)
    middle_img = plt.imread(middle_jpeg)
    bottom_img = plt.imread(bottom_jpeg)

    if panel_labels is None:
        panel_labels = ["a", "b", "c"]

    images = [top_img, middle_img, bottom_img]
    three_panel_dims = (cn.panel_dims[0], cn.panel_dims[1] * len(images))

    fig, axes = plt.subplots(nrows=len(images), ncols=1, figsize=three_panel_dims)
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax, img, label in zip(axes, images, panel_labels):
        ax.imshow(img, aspect='auto')
        ax.axis("off")
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="top", color="black")

    save_jpeg(three_panel_jpeg, year, main_logger)
    plt.close()


def create_four_panel_map(four_panel_jpeg, top_jpeg, second_jpeg, third_jpeg, bottom_jpeg, year, main_logger, panel_labels=None):
    """
    Creates a four-panel single-column map, e.g., for components of LULUCF and total LULUCF.
    """
    main_logger.info("Creating four-panel map")

    top_img = plt.imread(top_jpeg)
    second_img = plt.imread(second_jpeg)
    third_img = plt.imread(third_jpeg)
    bottom_img = plt.imread(bottom_jpeg)

    if panel_labels is None:
        panel_labels = ["a", "b", "c", "d"]

    images = [top_img, second_img, third_img, bottom_img]
    four_panel_dims = (cn.panel_dims[0], cn.panel_dims[1] * len(images))

    fig, axes = plt.subplots(nrows=len(images), ncols=1, figsize=four_panel_dims)
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax, img, label in zip(axes, images, panel_labels):
        ax.imshow(img, aspect='auto')
        ax.axis("off")
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
                ha="left", va="top", color="black")

    save_jpeg(four_panel_jpeg, year, main_logger)
    plt.close()


# From Claude (session '9-panel annual map JPEG')
def create_nine_panel_map(nine_panel_jpeg_path, yearly_masked_data, cmap, norm,
                          raster_extent, country_shapefile, years, bounding_box_proj,
                          legend_fn, main_logger):
    """
    Creates a 3x3 JPEG with one panel per year, reading left-to-right then top-to-bottom.
    The legend is added only to the first (top-left) panel via legend_fn(ax, img).
    """
    main_logger.info("\n\n---Creating 9-panel map")

    fig, axes = plt.subplots(3, 3, figsize=(cn.panel_dims[0] * 3, cn.panel_dims[1] * 3))
    fig.subplots_adjust(hspace=0, wspace=0)

    extent = list(raster_extent)

    if bounding_box_proj is not None:
        bbox_geom = box(*bounding_box_proj)
        cs = country_shapefile.clip(bbox_geom)
    else:
        cs = country_shapefile

    for idx, (ax, masked_d, yr) in enumerate(zip(axes.flat, yearly_masked_data, years)):
        set_ocean_color(ax)
        plot_country_polygons(ax, cs)
        img = plot_raster(ax, cmap, extent, masked_d, norm)
        plot_country_boundaries(ax, cs)
        remove_ticks(ax)

        if bounding_box_proj is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        ax.text(0.02, 0.98, str(yr), transform=ax.transAxes, fontsize=12, fontweight='bold',
                ha='left', va='top', color='black')

        if idx == 0:
            legend_fn(ax, img)

    year_range = f"{cn.interval_end_years_annual[0]}-{cn.interval_end_years_annual[-1]}"
    save_jpeg(nine_panel_jpeg_path, year_range, main_logger)
    plt.close()



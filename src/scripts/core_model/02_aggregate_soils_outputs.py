"""
02_aggregate_soils_outputs.py
Aggregate chunk-level organic-soils model outputs to 10x10 degree tiles.

Reads per-dataset slices from the global mega-zarr and writes GeoTIFFs to S3
under ``{BASE_URL}/{dataset}/...``.  The canonical packed-state dataset is
``combined_state``; archived mega-zarr stores that still use the legacy name
``emissions_state`` are handled via a read-time fallback (see
``create_10x10_outputs_from_zarr``).  New outputs always use the canonical
``combined_state`` path.
"""

import argparse
import dask
import numpy as np
import zarr

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import drainage_zarr_utilities as dzu
from src.scripts.utilities import universal_utilities as uu
from src.scripts.utilities import log_utilities as lu


def default_data_types(include_legacy_state_rasters: bool = False) -> list[str]:
    data_types = list(cn.drainage_outputs_to_zarr)
    if include_legacy_state_rasters:
        data_types.extend(list(getattr(cn, "drainage_optional_state_outputs", [])))
    if "combined_state" not in data_types:
        # Keep aggregation forward-compatible when constants lag behind model outputs.
        data_types.append("combined_state")
    return list(dict.fromkeys(data_types))


version = cn.model_version_underscore
BASE_URL = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{version}"
DEFAULT_OUTPUT_DATE = "20251007"


def _split_cli_items(values: list[str] | None) -> list[str] | None:
    """Normalize repeated, comma-delimited, or space-delimited CLI values."""
    if not values:
        return None

    items: list[str] = []
    for value in values:
        items.extend(
            item.strip()
            for item in str(value).replace(",", " ").split()
            if item.strip()
        )
    return list(dict.fromkeys(items)) or None


def get_inventory_periods(interval_type: str) -> list[str]:
    """Return inventory period labels for the selected interval type."""
    if interval_type == cn.intervals_annual:
        last_year = cn.five_year_inventory_periods[-1][1]
        return [
            f"{year}_{year}"
            for year in range(cn.annual_land_cover_start_year, last_year + 1)
        ]
    return [f"{start}_{end}" for start, end in cn.five_year_inventory_periods]


def get_output_folders(
    interval_type: str,
    output_pixel_resolution: str,
    run_name: str = "ogh_standard_model",
    output_date: str = DEFAULT_OUTPUT_DATE,
    inventory_periods: list[str] | None = None,
    data_types: list[str] | None = None,
) -> list:
    """Return list of S3 folders for organic soil outputs."""
    interval_folder = f"{interval_type}_intervals"
    if inventory_periods is None:
        inventory_periods = get_inventory_periods(interval_type)
    paths = []
    data_types = data_types or default_data_types()
    for period in inventory_periods:
        for dtype in data_types:
            path = (
                f"{BASE_URL}/{dtype}/{run_name}/"
                f"{interval_folder}/{period}/{output_pixel_resolution}/{output_date}"
            )
            paths.append(path)
    return paths


def filter_inventory_periods(
    inventory_periods: list[str], interval_end_years: list[int] | None
) -> list[str]:
    """Filter inventory periods by selected interval end years, preserving order."""
    if not interval_end_years:
        return inventory_periods

    selected_end_years = {int(year) for year in interval_end_years}
    filtered = [
        period
        for period in inventory_periods
        if int(period.split("_")[-1]) in selected_end_years
    ]
    available_end_years = {int(period.split("_")[-1]) for period in inventory_periods}
    missing = sorted(selected_end_years - available_end_years)
    if missing:
        raise ValueError(
            f"Requested interval end years are not available for this interval type: {missing}"
        )
    if not filtered:
        raise ValueError("No inventory periods matched the requested interval end years.")
    return filtered


def build_year_indices(interval_type: str) -> tuple[list[int], dict[int, int]]:
    year_index = dzu.full_model_year_index(interval_type)
    year_lookup = {year: idx for idx, year in enumerate(year_index)}
    return year_index, year_lookup


def ready_mega_zarr_year_index(zarr_path: str, logger) -> list[int]:
    """Return stored years only when a new-style model run is complete."""

    zarr_group = zarr.open_group(
        dzu.make_zarr_store(zarr_path, read_only=True),
        mode="r",
    )
    run_status = zarr_group.attrs.get("run_status")
    if run_status is not None and run_status != "complete":
        raise RuntimeError(
            f"Mega-zarr is not marked complete (run_status={run_status!r}): "
            f"{zarr_path}"
        )
    if run_status is None:
        logger.warning(
            "Mega-zarr has no run_status marker; treating it as a legacy store: %s",
            zarr_path,
        )
    return [int(year) for year in zarr_group["year"][:]]


def load_zarr_window(zarr_path: str, dataset: str, bounds, year_idx: int) -> np.ndarray:
    return dzu.open_zarr_window(zarr_path, dataset, bounds, year_idx)


def create_10x10_outputs_from_zarr(
    dataset: str,
    year_idx: int,
    year_value: int,
    period: str,
    tile_id: str,
    zarr_path: str,
    interval_type: str,
    output_pixel_resolution: str,
    run_name: str,
    output_date: str,
    is_final: bool,
    no_upload: bool,
    logger,
):
    """Read a single dataset/tile/period slice from the mega-zarr and write 10x10 GeoTIFFs.

    Output path contract
    --------------------
    New outputs are written under ``{BASE_URL}/{dataset}/...`` where *dataset*
    is the canonical name (e.g. ``combined_state``).  No new outputs are ever
    written under the legacy ``emissions_state`` path.

    Legacy read fallback
    --------------------
    When *dataset* is ``combined_state`` and the mega-zarr does not contain that
    variable (i.e. the store predates the rename), the function falls back to
    reading ``emissions_state`` and emits a warning.  The output is still
    written under the canonical ``combined_state`` path.
    """
    bounds = uu.get_10x10_tile_bounds(tile_id)
    chunk_px = uu.calc_chunk_length_pixels(bounds)
    bstr = uu.boundstr(bounds)

    dataset_for_read = dataset
    try:
        data_per_ha = load_zarr_window(zarr_path, dataset_for_read, bounds, year_idx)
    except Exception:
        if dataset == "combined_state":
            dataset_for_read = "emissions_state"
            logger.warning("Legacy input naming used for archived store: emissions_state -> combined_state")
            data_per_ha = load_zarr_window(zarr_path, dataset_for_read, bounds, year_idx)
        else:
            raise
    has_data = np.any(np.isfinite(data_per_ha) & (data_per_ha != 0))
    if not has_data:
        logger.info(
            "Skipping %s %s %s (no nonzero data in tile).",
            tile_id,
            dataset,
            period,
        )
        return f"Skipped empty {tile_id} {dataset} {period}", []

    dtype_str = cn.drainage_output_dtypes.get(dataset, "float32")
    is_numeric = dtype_str == "float32"

    data_per_pixel = None
    if is_numeric:
        pixel_area_uri = f"{cn.pixel_area_dir}{cn.pixel_area_pattern}_{tile_id}.tif"
        pixel_area = uu.get_tile_dataset_rio(
            pixel_area_uri,
            "Float32",
            bounds,
            chunk_px,
            is_final,
            logger,
            required=True,
        )[0]
        data_per_pixel = data_per_ha * pixel_area * cn.m2_to_ha

    output_interval_folder = f"{interval_type}_intervals"
    output_dir_per_ha = (
        f"{BASE_URL}/{dataset}/{run_name}/{output_interval_folder}/"
        f"{period}/{output_pixel_resolution}/{output_date}/"
    )
    output_name_per_ha = f"{tile_id}__{dataset}__{period}.tif"

    output_name_per_pixel = None
    output_dir_per_pixel = None
    if is_numeric:
        dataset_pixel = dataset.replace("_ha_", "_pixel_")
        output_dir_per_pixel = (
            f"{BASE_URL}/{dataset_pixel}/{run_name}/{output_interval_folder}/"
            f"{period}/{output_pixel_resolution}/{output_date}/"
        )
        output_name_per_pixel = f"{tile_id}__{dataset_pixel}__{period}.tif"

    if not no_upload:
        uu.save_and_upload_single_raster(
            bounds,
            chunk_px,
            tile_id,
            data_per_ha,
            data_per_ha.dtype.name,
            output_name_per_ha,
            output_dir_per_ha,
            is_final,
            logger,
        )
        if is_numeric and data_per_pixel is not None:
            uu.save_and_upload_single_raster(
                bounds,
                chunk_px,
                tile_id,
                data_per_pixel,
                data_per_pixel.dtype.name,
                output_name_per_pixel,
                output_dir_per_pixel,
                is_final,
                logger,
            )

    chunk_stats = []
    chunk_stats.append(
        uu.calculate_stats(
            data_per_ha,
            output_name_per_ha,
            bstr,
            tile_id,
            "output_layer",
            array_per_pixel=data_per_pixel if is_numeric else None,
            iv_start=None,
            iv_end=year_value,
        )
    )
    if is_numeric and output_name_per_pixel:
        chunk_stats.append(
            uu.calculate_stats(
                data_per_pixel,
                output_name_per_pixel,
                bstr,
                tile_id,
                "output_layer",
                array_per_pixel=data_per_pixel,
                iv_start=None,
                iv_end=year_value,
            )
        )

    return f"Success for {tile_id} {dataset} {period}", chunk_stats


def robust_create_10x10_outputs(*args, logger, **kwargs):
    dataset = args[0] if args else kwargs.get("dataset", "unknown")
    tile_id = args[4] if len(args) > 4 else kwargs.get("tile_id", "unknown")
    try:
        msg, stats = create_10x10_outputs_from_zarr(*args, **kwargs, logger=logger)
        logger.info(f"Successfully aggregated: {tile_id} {dataset}")
        return msg, stats
    except Exception as e:
        logger.error(f"Error aggregating {tile_id} {dataset}: {e}")
        return f"Error: {tile_id} {dataset} - {e}", None


def main(
    cluster_name,
    run_local: bool = False,
    no_upload: bool = False,
    no_log: bool = False,
    pixel_resolution: str = "4000_pixels",
    run_name: str = "ogh_standard_model",
    output_date: str = DEFAULT_OUTPUT_DATE,
    interval_type: str = cn.intervals_five_year,
    interval_end_years: list[int] | None = None,
    include_legacy_state_rasters: bool = False,
    tile_ids: list[str] | None = None,
    data_types: list[str] | None = None,
    final_output_naming: bool = False,
):
    logger = lu.setup_logging_main()

    cluster, client, run_local = uu.connect_to_cluster(
        cluster_name=cluster_name, run_local=run_local
    )

    stage = f"organic_soils_outputs_aggregated_to_10x10deg_{pixel_resolution}"

    start_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} started at: {start_time}", False, logger)

    chunk_size_pixels = int(pixel_resolution.replace("_pixels", ""))
    output_pixel_resolution = f"{cn.full_raster_dims}_pixels"
    inventory_periods = get_inventory_periods(interval_type)
    inventory_periods = filter_inventory_periods(inventory_periods, interval_end_years)
    logger.info(f"Inventory periods selected for aggregation: {inventory_periods}")

    zarr_path = dzu.create_mega_zarr_path(
        cn.drainage_outputs_path_mega_zarr,
        chunk_size_pixels,
        interval_type,
        run_name,
        output_date,
        logger,
    )
    year_index = ready_mega_zarr_year_index(zarr_path, logger)
    year_lookup = {year: index for index, year in enumerate(year_index)}
    missing_periods = [
        period
        for period in inventory_periods
        if int(period.split("_")[-1]) not in year_lookup
    ]
    if missing_periods:
        raise ValueError(
            "Selected inventory periods are absent from the mega-zarr year "
            f"coordinate {year_index}: {missing_periods}."
        )

    model_tile_ids = cn.get_tile_id_list()
    if tile_ids:
        unknown_tile_ids = [tile_id for tile_id in tile_ids if tile_id not in model_tile_ids]
        if unknown_tile_ids:
            raise ValueError(f"Unknown tile_ids requested: {unknown_tile_ids}")
        tile_ids = list(dict.fromkeys(tile_ids))
    else:
        tile_ids = list(model_tile_ids)

    is_final = bool(final_output_naming) or len(tile_ids) > 20

    available_data_types = default_data_types(
        include_legacy_state_rasters=include_legacy_state_rasters
    )
    if data_types:
        unknown_data_types = [
            data_type for data_type in data_types if data_type not in available_data_types
        ]
        if unknown_data_types:
            raise ValueError(
                "Unknown data_types requested: "
                f"{unknown_data_types}. Available: {available_data_types}"
            )
        data_types = list(dict.fromkeys(data_types))
    else:
        data_types = available_data_types

    logger.info("Aggregation tile selection (%d): %s", len(tile_ids), tile_ids)
    logger.info("Aggregation dataset selection (%d): %s", len(data_types), data_types)
    if final_output_naming and len(tile_ids) <= 20:
        logger.info("Using production output filenames for subset aggregation.")

    tasks = []
    for dataset in data_types:
        for period in inventory_periods:
            end_year = int(period.split("_")[-1])
            year_idx = year_lookup[end_year]
            for tile_id in tile_ids:
                tasks.append(
                    dask.delayed(robust_create_10x10_outputs)(
                        dataset,
                        year_idx,
                        year_index[year_idx],
                        period,
                        tile_id,
                        zarr_path,
                        interval_type,
                        output_pixel_resolution,
                        run_name,
                        output_date,
                        is_final,
                        no_upload,
                        logger=logger,
                    )
                )

    delayed_results = [
        task for task in tasks
    ]

    results = dask.compute(*delayed_results)
    lu.print_and_log(results, is_final, logger)

    errors = [
        result[0]
        for result in results
        if result and isinstance(result[0], str) and result[0].startswith("Error:")
    ]
    if errors:
        raise RuntimeError(
            f"Raster aggregation failed for {len(errors)} tasks; "
            f"first error: {errors[0]}"
        )

    success_count, all_stats = uu.count_successful_chunks(
        tile_ids, is_final, logger, results
    )

    output_folders = get_output_folders(
        interval_type,
        output_pixel_resolution,
        run_name,
        output_date,
        inventory_periods=inventory_periods,
        data_types=data_types,
    )
    output_folders_pixel = [
        folder.replace("_ha_", "_pixel_")
        for folder in output_folders
        if "_ha_" in folder
    ]

    if not no_upload:
        for folder in output_folders:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(
                folder
            )
            lu.print_and_log(
                f"Aggregated 10x10 deg outputs in {folder}: {file_count}",
                is_final,
                logger,
            )
        for folder in output_folders_pixel:
            geotiff_files, file_count = uu.list_raster_full_paths_in_s3_folder_and_count(
                folder
            )
            lu.print_and_log(
                f"Aggregated 10x10 deg per-pixel outputs in {folder}: {file_count}",
                is_final,
                logger,
            )

    if success_count > 0:
        uu.aggregate_10x10_chunk_stats(
            all_stats,
            stage,
            no_upload,
            logger,
            run_name=run_name,
            run_date=output_date,
        )

    end_time = uu.timestr()
    lu.print_and_log(f"Stage {stage} ended at: {end_time}", is_final, logger)
    uu.stage_duration(start_time, end_time, stage)

    if not run_local:
        lu.compile_worker_logs(no_log, cluster, stage, start_time, logger)

    if not run_local:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate LULUCF model outputs to 10x10 degree geotifs."
    )
    parser.add_argument("-cn", "--cluster_name", required=True, help="Cluster name")

    parser.add_argument("--run_local", action="store_true", help="Run locally without Dask/Coiled")
    parser.add_argument("--no_log", action="store_true", help="Do not create the combined log")
    parser.add_argument("--no_upload", action="store_true", help="Do not save and upload outputs to S3")
    parser.add_argument(
        "--pixel_resolution",
        choices=["4000_pixels", "8000_pixels"],
        default="4000_pixels",
        help="Input raster resolution to process",
    )
    parser.add_argument("--run_name", default="ogh_standard_model", help="Model run name")
    parser.add_argument(
        "--interval_type",
        choices=[cn.intervals_five_year, cn.intervals_annual],
        default=cn.intervals_five_year,
        help="Interval type for zarr outputs",
    )
    parser.add_argument(
        "--output_date",
        default=DEFAULT_OUTPUT_DATE,
        help="Date tag for selecting input datasets (YYYYMMDD)",
    )
    parser.add_argument(
        "--interval_end_years",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of interval end years to aggregate (e.g., 2010 2015 2020).",
    )
    parser.add_argument(
        "--include_legacy_state_rasters",
        action="store_true",
        help="Also aggregate drained_state and burned_state (default: omit from standard runs).",
    )
    parser.add_argument(
        "--tile_ids",
        nargs="+",
        default=None,
        help="Optional subset of 10x10 tile IDs. Supports spaces or commas.",
    )
    parser.add_argument(
        "--data_types",
        nargs="+",
        default=None,
        help="Optional subset of mega-zarr datasets to aggregate.",
    )
    parser.add_argument(
        "--final_output_naming",
        action="store_true",
        help="Use production GeoTIFF names without timestamp suffix for subset runs.",
    )

    args = parser.parse_args()

    main(
        cluster_name=args.cluster_name,
        run_local=args.run_local,
        no_upload=args.no_upload,
        no_log=args.no_log,
        pixel_resolution=args.pixel_resolution,
        run_name=args.run_name,
        output_date=args.output_date,
        interval_type=args.interval_type,
        interval_end_years=args.interval_end_years,
        include_legacy_state_rasters=args.include_legacy_state_rasters,
        tile_ids=_split_cli_items(args.tile_ids),
        data_types=_split_cli_items(args.data_types),
        final_output_naming=args.final_output_naming,
    )

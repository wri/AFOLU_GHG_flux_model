"""
Copy completed global COGs from AWS S3 to Google Cloud Storage.

Uses the same dataset x flux x year configuration and path construction as
create_cogs_download_tiles.py, but only copies already-created global COGs.

Examples:
    gcloud auth application-default login

    python -m src.utilities.create_cluster -cn S3_to_GCS_removals_cog_copy --n 8 -t 1 -m 4 --on_demand --gcp
    python -m src.LULUCF.scripts.postprocessing.GEE.S3_to_GCS_COG_copy -cn S3_to_GCS_removals_cog_copy -d vegetation -f removals -y 2016 2018 2019 2020 2021 2022 2023 2024

    python -m src.LULUCF.scripts.postprocessing.GEE.S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d vegetation -f emissions removals
    python -m src.LULUCF.scripts.postprocessing.GEE.S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d mineral_soil -f netflux
    python -m src.LULUCF.scripts.postprocessing.GEE.S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d organic_soil -f emissions
"""

import argparse
import json
from pathlib import Path

import dask.bag as db
import gcsfs
import s3fs
from dask.distributed import print

from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu


CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def build_dataset_config(datasets, fluxes, years):
    """Build the same dataset/flux/year configuration used for COG creation."""

    # TODO: update to cn path/version once the COG creation script is updated.
    veg_model_version_underscore = "1_0_5"

    veg_output_dir = f"{cn.LULUCF_dir}outputs_vegetation/version_{veg_model_version_underscore}__standard__global/"
    min_soil_output_dir = f"{cn.LULUCF_dir}outputs_soil_organic_carbon/version_{cn.SOC_model_version_underscore}__standard__global/"
    org_soil_output_dir = f"{cn.AFOLU_dir}organic_soils/outputs/version_{cn.organic_soil_model_version_underscore}/"

    veg_rundate = "20260130"
    min_soil_rundate = "20260611"
    org_soil_rundate = "20260525"

    emissions_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e"
    removals_pattern = "gross_removals__all_C_pools__MgCO2"
    netflux_pattern = "net_flux__all_C_pools__all_gases__MgCO2e"

    # These are the source TILE directories used by the COG script. The final
    # global COG path is derived below by replacing 40000_pixels -> global.
    emissions_path = f"{veg_output_dir}{emissions_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    removals_path = f"{veg_output_dir}{removals_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    netflux_path = f"{veg_output_dir}{netflux_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"

    min_soil_loss_path = f"{min_soil_output_dir}{cn.SOC_loss_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_gain_path = f"{min_soil_output_dir}{cn.SOC_gain_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_net_path = f"{min_soil_output_dir}{cn.SOC_net_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"

    org_soil_burned_path = f"{org_soil_output_dir}{cn.burned_organic_soils_total_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    org_soil_drained_path = f"{org_soil_output_dir}{cn.drained_organic_soils_total_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"

    # GCS output directories
    gcs_bucket = "wri-lcl-lgms"
    veg_asset_folder = f"vegetation/v{veg_model_version_underscore}"
    min_soil_asset_folder = f"soil/mineral/v{cn.SOC_model_version_underscore}"
    org_soil_asset_folder = f"soil/organic/v{cn.organic_soil_model_version_underscore}"

    emissions_gcs_dir = f"gs://{gcs_bucket}/{veg_asset_folder}/emissions/{emissions_pattern}/"
    removals_gcs_dir = f"gs://{gcs_bucket}/{veg_asset_folder}/removals/{removals_pattern}/"
    netflux_gcs_dir = f"gs://{gcs_bucket}/{veg_asset_folder}/net_flux/{netflux_pattern}/"

    min_soil_loss_gcs_dir = (
        f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_loss_min_soil_extent_pattern}/"
    )
    min_soil_gain_gcs_dir = (
        f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_gain_min_soil_extent_pattern}/"
    )
    min_soil_net_gcs_dir = (
        f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_net_min_soil_extent_pattern}/"
    )

    org_soil_burned_gcs_dir = (
        f"gs://{gcs_bucket}/{org_soil_asset_folder}/{cn.burned_organic_soils_total_pattern}/"
    )
    org_soil_drained_gcs_dir = (
        f"gs://{gcs_bucket}/{org_soil_asset_folder}/{cn.drained_organic_soils_total_pattern}/"
    )

    # Resolve requested calendar years to the model's stored soil intervals.
    if years is None:
        years = list(cn.veg_outputs_years)

    mineral_soil_years = sorted({cn.mineral_soil_year_map[year] for year in years})
    organic_soil_years = sorted({cn.organic_soil_year_map[year] for year in years})

    dataset_config = {
        "vegetation": {
            "emissions": {
                "s3_path": emissions_path,
                "pattern": f"{emissions_pattern}_ha_yr",
                "gcs_dir": emissions_gcs_dir,
                "years": years,
            },
            "removals": {
                "s3_path": removals_path,
                "pattern": f"{removals_pattern}_ha_yr",
                "gcs_dir": removals_gcs_dir,
                "years": years,
            },
            "netflux": {
                "s3_path": netflux_path,
                "pattern": f"{netflux_pattern}_ha_yr",
                "gcs_dir": netflux_gcs_dir,
                "years": years,
            },
        },
        "mineral_soil": {
            "emissions": {
                "s3_path": min_soil_loss_path,
                "pattern": f"{cn.SOC_loss_min_soil_extent_pattern}_ha_yr",
                "gcs_dir": min_soil_loss_gcs_dir,
                "years": mineral_soil_years,
            },
            "removals": {
                "s3_path": min_soil_gain_path,
                "pattern": f"{cn.SOC_gain_min_soil_extent_pattern}_ha_yr",
                "gcs_dir": min_soil_gain_gcs_dir,
                "years": mineral_soil_years,
            },
            "netflux": {
                "s3_path": min_soil_net_path,
                "pattern": f"{cn.SOC_net_min_soil_extent_pattern}_ha_yr",
                "gcs_dir": min_soil_net_gcs_dir,
                "years": mineral_soil_years,
            },
        },
        "organic_soil": {
            "emissions": [
                {
                    "component": "burned",
                    "s3_path": org_soil_burned_path,
                    "pattern": cn.burned_organic_soils_total_pattern,
                    "gcs_dir": org_soil_burned_gcs_dir,
                    "years": organic_soil_years,
                },
                {
                    "component": "drained",
                    "s3_path": org_soil_drained_path,
                    "pattern": cn.drained_organic_soils_total_pattern,
                    "gcs_dir": org_soil_drained_gcs_dir,
                    "years": organic_soil_years,
                },
            ],
        },
    }

    # Only return combinations the caller actually requested.
    return dataset_config, years, mineral_soil_years, organic_soil_years


def build_jobs(datasets, fluxes, years):
    """Build S3 global-COG -> GCS COG copy jobs."""
    dataset_config, veg_years, mineral_soil_years, organic_soil_years = (
        build_dataset_config(datasets, fluxes, years)
    )

    print(f"Vegetation years required: {veg_years}")
    print(f"Mineral soil output years required: {mineral_soil_years}")
    print(f"Organic soil intervals required: {organic_soil_years}")

    jobs = {}

    for dataset in datasets:
        for flux in fluxes:
            if flux not in dataset_config[dataset]:
                print(f"WARNING: {dataset} does not have a {flux} output. Skipping.")
                continue

            configs = dataset_config[dataset][flux]
            if not isinstance(configs, list):
                configs = [configs]

            for config in configs:
                for year in config["years"]:
                    s3_dir = config["s3_path"].replace("YYYY", str(year))
                    pattern = config["pattern"]
                    gcs_dir = config["gcs_dir"]
                    component = config.get("component")

                    # Match create_cogs_download_tiles.py exactly:
                    # mineral-soil input is stored under 2020 but output COG is
                    # named with the 2016_2020 interval. Organic-soil configs
                    # already contain interval strings such as 2016_2020.
                    output_year = (
                        cn.organic_soil_year_map[year]
                        if dataset == "mineral_soil"
                        else year
                    )

                    key = (
                        f"{dataset}_{flux}_{component}_{output_year}"
                        if component
                        else f"{dataset}_{flux}_{output_year}"
                    )

                    output_dir = s3_dir.replace("40000_pixels", "global").rstrip("/") + "/"
                    source_cog_s3_path = f"{output_dir}{pattern}_{output_year}.tif"
                    destination_cog_gcs_path = (
                        f"{gcs_dir.rstrip('/')}/{pattern}_{output_year}.tif"
                    )

                    jobs[key] = {
                        "key": key,
                        "dataset": dataset,
                        "flux": flux,
                        "component": component,
                        "year": output_year,
                        "src": source_cog_s3_path,
                        "dst": destination_cog_gcs_path,
                    }

    return jobs


def create_tasks(jobs, task_file, overwrite_existing=False):
    """Inspect S3/GCS and write transfer tasks to JSON."""
    s3_fs = s3fs.S3FileSystem()
    gcs_fs = gcsfs.GCSFileSystem()

    tasks = []

    for key, job in jobs.items():
        src_path = job["src"].replace("s3://", "", 1)
        dst_path = job["dst"].replace("gs://", "", 1)

        task = dict(job)
        task.update(
            {
                "src_size": None,
                "dest_size": None,
                "status": "pending",
                "error": None,
            }
        )

        try:
            src_info = s3_fs.info(src_path)
            task["src_size"] = src_info.get("size", 0)
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"Could not read S3 source: {exc}"
            tasks.append(task)
            print(f"{key}: source check failed: {job['src']}")
            continue

        if not overwrite_existing:
            try:
                dst_info = gcs_fs.info(dst_path)
                task["dest_size"] = dst_info.get("size", 0)

                if task["dest_size"] == task["src_size"]:
                    task["status"] = "completed"
                else:
                    task["status"] = "mismatched"
            except FileNotFoundError:
                pass
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = f"Could not check GCS destination: {exc}"

        tasks.append(task)
        print(
            f"{key}: {task['status']}\n"
            f"  S3:  {job['src']}\n"
            f"  GCS: {job['dst']}"
        )

    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    return tasks


def load_task_file(task_file):
    """Load pending/mismatched/failed tasks for resume."""
    with open(task_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    return [
        task
        for task in tasks
        if task["status"] in ("pending", "mismatched", "failed")
    ]


def copy_file(task, overwrite_existing=False):
    """Stream one completed COG from S3 to GCS and verify byte size."""
    s3_fs = s3fs.S3FileSystem()
    gcs_fs = gcsfs.GCSFileSystem()

    src_path = task["src"].replace("s3://", "", 1)
    dst_path = task["dst"].replace("gs://", "", 1)
    expected_size = task["src_size"]

    print(f"Starting {task['key']}:\n  {task['src']}\n  -> {task['dst']}")

    try:
        if not overwrite_existing and gcs_fs.exists(dst_path):
            dst_info = gcs_fs.info(dst_path)
            if dst_info.get("size", 0) == expected_size:
                task["status"] = "completed"
                task["dest_size"] = expected_size
                task["skipped"] = True
                print(f"{task['key']}: destination already matches; skipping.")
                return task

        bytes_copied = 0
        next_progress = 5

        with s3_fs.open(src_path, "rb") as src_f:
            with gcs_fs.open(dst_path, "wb") as dst_f:
                while True:
                    chunk = src_f.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    dst_f.write(chunk)
                    bytes_copied += len(chunk)

                    if expected_size:
                        percent = bytes_copied / expected_size * 100
                        if percent >= next_progress:
                            print(
                                f"{task['key']}: {percent:.1f}% "
                                f"({bytes_copied / 1e9:.2f} / "
                                f"{expected_size / 1e9:.2f} GB)"
                            )
                            next_progress += 5

        dst_info = gcs_fs.info(dst_path)
        task["dest_size"] = dst_info.get("size", 0)

        if task["dest_size"] != expected_size:
            raise ValueError(
                f"Destination size mismatch for {task['dst']}. "
                f"Expected {expected_size}, got {task['dest_size']}."
            )

        task["status"] = "completed"
        task["error"] = None

    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc)

    return task


def save_progress(task_file, updated_tasks):
    """Merge updated statuses into the full task file."""
    with open(task_file, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    task_map = {task["key"]: task for task in all_tasks}

    for task in updated_tasks:
        task_map[task["key"]] = task

    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(list(task_map.values()), f, indent=2)


def main(
    cluster_name,
    run_local,
    datasets,
    fluxes,
    years,
    task_file,
    resume,
    overwrite_existing,
):
    cluster, client, run_local = uu.connect_to_Coiled_cluster(cluster_name, run_local)

    if resume:
        if not Path(task_file).exists():
            raise FileNotFoundError(f"Task file does not exist: {task_file}")
        tasks = load_task_file(task_file)
    else:
        jobs = build_jobs(datasets, fluxes, years)
        tasks_all = create_tasks(
            jobs,
            task_file=task_file,
            overwrite_existing=overwrite_existing,
        )
        tasks = [
            task
            for task in tasks_all
            if task["status"] in ("pending", "mismatched")
        ]

    if not tasks:
        print("Nothing to copy. All requested destination COGs already match.")
        if client is not None:
            client.close()
        return

    valid_tasks = [task for task in tasks if task.get("src_size") is not None]
    failed_source_checks = len(tasks) - len(valid_tasks)

    total_size = sum(task["src_size"] for task in valid_tasks)
    print(
        f"\nReady to copy {len(valid_tasks)} COG(s), "
        f"total source size = {total_size / 1e9:.2f} GB."
    )
    if failed_source_checks:
        print(f"WARNING: {failed_source_checks} task(s) failed source validation.")

    if not valid_tasks:
        if client is not None:
            client.close()
        return

    choice = input("Continue with the copy? [y/N]: ")
    if choice.lower() not in ("y", "yes"):
        print("Aborting.")
        if client is not None:
            client.close()
        return

    # One Dask task per global COG. Multiple selected COGs can therefore copy
    # concurrently when the cluster has multiple available worker threads.
    bag = db.from_sequence(valid_tasks, npartitions=len(valid_tasks))
    results = bag.map(copy_file, overwrite_existing=overwrite_existing).compute()

    save_progress(task_file, results)

    completed = sum(task["status"] == "completed" for task in results)
    failed = sum(task["status"] == "failed" for task in results)

    print(f"\nCopy finished: {completed} succeeded, {failed} failed.")

    if failed:
        print(f"Re-run with --resume --task_file {task_file} to retry failures.")
        for task in results:
            if task["status"] == "failed":
                print(f"  {task['key']}: {task.get('error')}")

    if client is not None:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy completed global LULUCF COGs from S3 to GCS.")
    parser.add_argument("-cn", "--cluster_name", help="Coiled cluster name")
    parser.add_argument("--run_local", action="store_true", help="Run locally instead of on a Coiled cluster")
    parser.add_argument("-d", "--datasets", required=True, nargs="+", choices=["vegetation", "mineral_soil", "organic_soil"], help="Dataset(s) to copy.")
    parser.add_argument("-f", "--flux", required=True, nargs="+", choices=["emissions", "removals", "netflux"], help="Flux type(s) to copy.")
    parser.add_argument("-y", "--years", nargs="+", type=int, choices=range(2016, 2025), help="Calendar year(s) to copy. Defaults to all available years.")
    parser.add_argument("--task_file", default="s3_to_gcs_cog_tasks.json", help="JSON task/status file. Default: s3_to_gcs_cog_tasks.json")
    parser.add_argument("--resume", action="store_true", help="Resume pending/failed/mismatched tasks from --task_file.")
    parser.add_argument("--overwrite_existing", action="store_true", default=False, help="Re-copy GCS objects even if an object already exists.")

    args = parser.parse_args()
    main(args.cluster_name, args.run_local, args.datasets, args.flux, args.years, args.task_file, args.resume, args.overwrite_existing)

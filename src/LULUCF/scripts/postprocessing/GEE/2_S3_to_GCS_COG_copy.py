"""
Copy completed global COGs from AWS S3 to Google Cloud Storage.

Uses the same dataset x flux x gas x year configuration and path construction as
create_cogs_download_tiles.py, but only copies already-created global COGs.

Examples:
    gcloud auth application-default set-quota-project landandcarbon
    python -m src.utilities.create_cluster -cn S3_to_GCS_cog_copy --n 9 -t 1 -m 4 --on_demand --gcp
    python -m src.LULUCF.scripts.postprocessing.GEE.2_S3_to_GCS_COG_copy -cn S3_to_GCS_cog_copy -d vegetation -f removals -g all_gases -y 2016 2018 2019 2020 2021 2022 2023 2024

    python -m src.LULUCF.scripts.postprocessing.GEE.2_S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d vegetation -f emissions removals -g all_gases
    python -m src.LULUCF.scripts.postprocessing.GEE.2_S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d mineral_soil -f netflux -g all_gases
    python -m src.LULUCF.scripts.postprocessing.GEE.2_S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d organic_soil -f emissions -g all_gases
    python -m src.LULUCF.scripts.postprocessing.GEE.2_S3_to_GCS_COG_copy -cn S3_to_GCS_copy -d vegetation -f emissions -g CO2_only

Caution:
    - Registered COG-backed EE assets point at the exact version of each GCS object, so re-copying over a COG that is
      already registered breaks that asset until it is re-registered.
    - Safeguard: a COG that already exists in GCS is never copied over, even if its size differs from the S3 COG.
      Those are reported as skipped_exists. Only --overwrite_existing replaces them, after listing them and asking
      for confirmation. Re-register the EE assets for anything you overwrite.
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


def build_dataset_config(gases, years):
    """Build the same dataset/flux/gas/year configuration used for COG creation (create_cogs_download_tiles.py)."""

    #TODO: This script is behind the current model version (1.0.6) Change to cn paths after merging to updated model branch.
    # Keep the versions, patterns and paths below in sync with create_cogs_download_tiles.py and 3_COG_EE_Asset_Registration.ipynb
    veg_model_version_underscore = "1_0_5"

    veg_output_dir = f"{cn.LULUCF_dir}outputs_vegetation/version_{veg_model_version_underscore}__standard__global/"
    min_soil_output_dir = f"{cn.LULUCF_dir}outputs_soil_organic_carbon/version_{cn.SOC_model_version_underscore}__standard__global/"
    org_soil_output_dir = f"{cn.AFOLU_dir}organic_soils/outputs/version_{cn.organic_soil_model_version_underscore}/"

    veg_rundate = "20260130"
    min_soil_rundate = "20260611"
    org_soil_rundate = "20260525"

    # Vegetation patterns
    # Note: the patterns include veg_ at the beginning starting with v1.0.6
    veg_emissions_all_gases_pattern =    "gross_emissions__all_C_pools__all_gases__MgCO2e"
    veg_emissions_CO2_only_pattern =     "gross_emissions__all_C_pools__CO2_only__MgCO2"
    veg_emissions_non_CO2_only_pattern = "gross_emissions__all_C_pools__non_CO2_only__MgCO2e"

    veg_netflux_all_gases_pattern =    "net_flux__all_C_pools__all_gases__MgCO2e"
    veg_netflux_CO2_only_pattern =     "net_flux__all_C_pools__CO2_only__MgCO2"
    veg_netflux_non_CO2_only_pattern = "net_flux__all_C_pools__non_CO2_only__MgCO2e"

    veg_removals_pattern = "gross_removals__all_C_pools__MgCO2"  # CO2 only

    # Organic soil patterns
    org_soil_burned_all_gases_pattern = cn.burned_organic_soils_total_pattern
    org_soil_burned_CO2_only_pattern = "burned_co2_Mg_CO2_ha_yr"
    # org_soil_burned_non_CO2_only_pattern = # TODO: burned organic soils, non-CO2 only

    org_soil_drained_all_gases_pattern = cn.drained_organic_soils_total_pattern
    org_soil_drained_CO2_only_pattern = "drained_co2_Mg_CO2_ha_yr"
    # org_soil_drained_non_CO2_only_pattern = # TODO: drained organic soils, non-CO2 only

    # These are the source TILE directories used by the COG script. The final
    # global COG path is derived in build_jobs by replacing 40000_pixels -> global.

    # Vegetation input tile s3 directories
    veg_emissions_all_gases_path =    f"{veg_output_dir}{veg_emissions_all_gases_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    veg_emissions_CO2_only_path =     f"{veg_output_dir}{veg_emissions_CO2_only_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    veg_emissions_non_CO2_only_path = f"{veg_output_dir}{veg_emissions_non_CO2_only_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"

    veg_netflux_all_gases_path =    f"{veg_output_dir}{veg_netflux_all_gases_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    veg_netflux_CO2_only_path =     f"{veg_output_dir}{veg_netflux_CO2_only_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"
    veg_netflux_non_CO2_only_path = f"{veg_output_dir}{veg_netflux_non_CO2_only_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"

    veg_removals_path = f"{veg_output_dir}{veg_removals_pattern}/annual_intervals/YYYY/_ha_yr/40000_pixels/{veg_rundate}/"

    # Mineral soil input tile s3 directories (CO2 only)
    min_soil_loss_path = f"{min_soil_output_dir}{cn.SOC_loss_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_gain_path = f"{min_soil_output_dir}{cn.SOC_gain_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"
    min_soil_net_path =  f"{min_soil_output_dir}{cn.SOC_net_min_soil_extent_pattern}/YYYY/_ha_yr/40000_pixels/{min_soil_rundate}/"

    # Organic soil input tile s3 directories
    org_soil_burned_all_gases_path = f"{org_soil_output_dir}{org_soil_burned_all_gases_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    org_soil_burned_CO2_only_path =  f"{org_soil_output_dir}{org_soil_burned_CO2_only_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    # org_soil_burned_non_CO2_only_path = # TODO: burned organic soils, non-CO2 only

    org_soil_drained_all_gases_path = f"{org_soil_output_dir}{org_soil_drained_all_gases_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    org_soil_drained_CO2_only_path =  f"{org_soil_output_dir}{org_soil_drained_CO2_only_pattern}/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/YYYY/40000_pixels/{org_soil_rundate}/"
    # org_soil_drained_non_CO2_only_path = # TODO: drained organic soils, non-CO2 only

    # GCS output directories
    gcs_bucket = "wri-lcl-lgms"
    veg_asset_folder = f"vegetation/v{veg_model_version_underscore}"
    min_soil_asset_folder = f"soil/mineral/v{cn.SOC_model_version_underscore}"
    org_soil_asset_folder = f"soil/organic/v{cn.organic_soil_model_version_underscore}"

    # Resolve requested calendar years to the model's stored soil intervals.
    if years is None:
        years = list(cn.veg_outputs_years)

    mineral_soil_years = sorted({cn.mineral_soil_year_map[year] for year in years})
    organic_soil_years = sorted({cn.organic_soil_year_map[year] for year in years})

    # Define available dataset / flux combinations
    # Each flux holds a list of COGs, added below depending on which gas(es) were requested.

    dataset_config = {
        "vegetation":   {"emissions": [], "removals": [], "netflux": []},
        "mineral_soil": {"emissions": [], "removals": [], "netflux": []},
        "organic_soil": {"emissions": []},
    }

    if "all_gases" in gases:
        dataset_config["vegetation"]["emissions"].append(
            {"gas": "all_gases",
             "s3_path": veg_emissions_all_gases_path,
             "pattern": f"{veg_emissions_all_gases_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/emissions/{veg_emissions_all_gases_pattern}/",
             "years": years})
        dataset_config["vegetation"]["netflux"].append(
            {"gas": "all_gases",
             "s3_path": veg_netflux_all_gases_path,
             "pattern": f"{veg_netflux_all_gases_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/net_flux/{veg_netflux_all_gases_pattern}/",
             "years": years})
        dataset_config["organic_soil"]["emissions"].extend([
            {"gas": "all_gases",
             "component": "burned",
             "s3_path": org_soil_burned_all_gases_path,
             "pattern": org_soil_burned_all_gases_pattern,
             "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_burned_all_gases_pattern}/",
             "years": organic_soil_years},
            {"gas": "all_gases",
             "component": "drained",
             "s3_path": org_soil_drained_all_gases_path,
             "pattern": org_soil_drained_all_gases_pattern,
             "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_drained_all_gases_pattern}/",
             "years": organic_soil_years},
        ])

    if "CO2_only" in gases:
        dataset_config["vegetation"]["emissions"].append(
            {"gas": "CO2_only",
             "s3_path": veg_emissions_CO2_only_path,
             "pattern": f"{veg_emissions_CO2_only_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/emissions/{veg_emissions_CO2_only_pattern}/",
             "years": years})
        dataset_config["vegetation"]["netflux"].append(
            {"gas": "CO2_only",
             "s3_path": veg_netflux_CO2_only_path,
             "pattern": f"{veg_netflux_CO2_only_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/net_flux/{veg_netflux_CO2_only_pattern}/",
             "years": years})
        dataset_config["organic_soil"]["emissions"].extend([
            {"gas": "CO2_only",
             "component": "burned",
             "s3_path": org_soil_burned_CO2_only_path,
             "pattern": org_soil_burned_CO2_only_pattern,
             "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_burned_CO2_only_pattern}/",
             "years": organic_soil_years},
            {"gas": "CO2_only",
             "component": "drained",
             "s3_path": org_soil_drained_CO2_only_path,
             "pattern": org_soil_drained_CO2_only_pattern,
             "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_drained_CO2_only_pattern}/",
             "years": organic_soil_years},
        ])

    if "non_CO2_only" in gases:
        dataset_config["vegetation"]["emissions"].append(
            {"gas": "non_CO2_only",
             "s3_path": veg_emissions_non_CO2_only_path,
             "pattern": f"{veg_emissions_non_CO2_only_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/emissions/{veg_emissions_non_CO2_only_pattern}/",
             "years": years})
        dataset_config["vegetation"]["netflux"].append(
            {"gas": "non_CO2_only",
             "s3_path": veg_netflux_non_CO2_only_path,
             "pattern": f"{veg_netflux_non_CO2_only_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/net_flux/{veg_netflux_non_CO2_only_pattern}/",
             "years": years})
        # dataset_config["organic_soil"]["emissions"].extend([
        #     {"gas": "non_CO2_only",
        #      "component": "burned",
        #      "s3_path": org_soil_burned_non_CO2_only_path,
        #      "pattern": org_soil_burned_non_CO2_only_pattern,
        #      "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_burned_non_CO2_only_pattern}/",
        #      "years": organic_soil_years},
        #     {"gas": "non_CO2_only",
        #      "component": "drained",
        #      "s3_path": org_soil_drained_non_CO2_only_path,
        #      "pattern": org_soil_drained_non_CO2_only_pattern,
        #      "gcs_dir": f"gs://{gcs_bucket}/{org_soil_asset_folder}/{org_soil_drained_non_CO2_only_pattern}/",
        #      "years": organic_soil_years},
        # ])

    # Mineral soil and vegetation removals are CO2 only, so include them for all_gases or CO2_only (added once)
    if "all_gases" in gases or "CO2_only" in gases:
        dataset_config["vegetation"]["removals"].append(
            {"gas": "CO2_only",
             "s3_path": veg_removals_path,
             "pattern": f"{veg_removals_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{veg_asset_folder}/removals/{veg_removals_pattern}/",
             "years": years})
        dataset_config["mineral_soil"]["emissions"].append(
            {"gas": "CO2_only",
             "s3_path": min_soil_loss_path,
             "pattern": f"{cn.SOC_loss_min_soil_extent_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_loss_min_soil_extent_pattern}/",
             "years": mineral_soil_years})
        dataset_config["mineral_soil"]["removals"].append(
            {"gas": "CO2_only",
             "s3_path": min_soil_gain_path,
             "pattern": f"{cn.SOC_gain_min_soil_extent_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_gain_min_soil_extent_pattern}/",
             "years": mineral_soil_years})
        dataset_config["mineral_soil"]["netflux"].append(
            {"gas": "CO2_only",
             "s3_path": min_soil_net_path,
             "pattern": f"{cn.SOC_net_min_soil_extent_pattern}_ha_yr",
             "gcs_dir": f"gs://{gcs_bucket}/{min_soil_asset_folder}/{cn.SOC_net_min_soil_extent_pattern}/",
             "years": mineral_soil_years})

    return dataset_config, years, mineral_soil_years, organic_soil_years


def build_jobs(datasets, fluxes, gases, years):
    """Build S3 global-COG -> GCS COG copy jobs."""
    dataset_config, veg_years, mineral_soil_years, organic_soil_years = (
        build_dataset_config(gases, years)
    )

    print(f"Gas(es) requested: {gases}")
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
            if not configs:
                print(f"WARNING: {dataset} has no {flux} output for gas(es) {gases}. Skipping.")
                continue

            for config in configs:
                for year in config["years"]:
                    s3_dir = config["s3_path"].replace("YYYY", str(year))
                    pattern = config["pattern"]
                    gcs_dir = config["gcs_dir"]
                    component = config.get("component")
                    gas = config["gas"]

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
                        f"{dataset}_{flux}_{gas}_{component}_{output_year}"
                        if component
                        else f"{dataset}_{flux}_{gas}_{output_year}"
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
                        "gas": gas,
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

        # Safeguard: never copy over a COG that already exists in GCS unless --overwrite_existing is set.
        # Registered COG-backed EE assets break if the GCS object they point to is replaced.
        try:
            dest_exists = gcs_fs.exists(dst_path)
            if dest_exists:
                task["dest_size"] = gcs_fs.info(dst_path).get("size", 0)
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"Could not check GCS destination: {exc}"
            tasks.append(task)
            print(f"{key}: destination check failed: {job['dst']}")
            continue

        task["dest_exists"] = dest_exists

        if dest_exists and not overwrite_existing:
            if task["dest_size"] == task["src_size"]:
                task["status"] = "completed"
            else:
                task["status"] = "skipped_exists"
                task["error"] = (
                    f"COG already exists in GCS with a different size "
                    f"(S3={task['src_size']}, GCS={task['dest_size']}). Not copied. "
                    f"Use --overwrite_existing to replace it, then re-register its EE asset."
                )

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
        # Safeguard: re-check right before copying, so an existing COG is never replaced
        # unless --overwrite_existing is set (this also covers --resume runs).
        if not overwrite_existing and gcs_fs.exists(dst_path):
            task["dest_size"] = gcs_fs.info(dst_path).get("size", 0)
            task["skipped"] = True
            if task["dest_size"] == expected_size:
                task["status"] = "completed"
                task["error"] = None
                print(f"{task['key']}: destination already matches; skipping.")
            else:
                task["status"] = "skipped_exists"
                task["error"] = (
                    f"COG already exists in GCS with a different size "
                    f"(S3={expected_size}, GCS={task['dest_size']}). Not copied. "
                    f"Use --overwrite_existing to replace it, then re-register its EE asset."
                )
                print(f"{task['key']}: COG already exists in GCS with a different size; not copied.")
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
    gases,
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
        jobs = build_jobs(datasets, fluxes, gases, years)
        tasks_all = create_tasks(
            jobs,
            task_file=task_file,
            overwrite_existing=overwrite_existing,
        )
        tasks = [task for task in tasks_all if task["status"] == "pending"]

        skipped_existing = [task for task in tasks_all if task["status"] == "skipped_exists"]
        if skipped_existing:
            print(
                f"\nWARNING: {len(skipped_existing)} COG(s) already exist in GCS with a different size "
                f"and will NOT be copied over:"
            )
            for task in skipped_existing:
                print(f"  {task['dst']}")
            print("Use --overwrite_existing to replace them, then re-register their EE assets.")

    if not tasks:
        print("Nothing to copy. All requested destination COGs already exist in GCS or failed source checks.")
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

    to_overwrite = [task for task in valid_tasks if task.get("dest_exists")]
    if to_overwrite:
        print(
            f"\nWARNING: --overwrite_existing will replace {len(to_overwrite)} COG(s) already in GCS. "
            f"Any EE assets registered from them will break until they are re-registered:"
        )
        for task in to_overwrite:
            print(f"  {task['dst']}")

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
    skipped = sum(task["status"] == "skipped_exists" for task in results)

    print(f"\nCopy finished: {completed} succeeded, {failed} failed, {skipped} skipped (already in GCS).")

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
    parser.add_argument("-g", "--gas", required=True, nargs="+", choices=["all_gases", "CO2_only", "non_CO2_only"], help="GHG(s) to copy. Options: all_gases, CO2_only, non_CO2_only")
    parser.add_argument("-y", "--years", nargs="+", type=int, choices=range(2016, 2025), help="Calendar year(s) to copy. Defaults to all available years.")
    parser.add_argument("--task_file", default="s3_to_gcs_cog_tasks.json", help="JSON task/status file. Default: s3_to_gcs_cog_tasks.json")
    parser.add_argument("--resume", action="store_true", help="Resume pending/failed/mismatched tasks from --task_file.")
    parser.add_argument("--overwrite_existing", action="store_true", default=False, help="Replace COGs that already exist in GCS. Breaks any EE assets registered from them until they are re-registered.")

    args = parser.parse_args()
    main(args.cluster_name, args.run_local, args.datasets, args.flux, args.gas, args.years, args.task_file, args.resume, args.overwrite_existing)

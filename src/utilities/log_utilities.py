import boto3
import logging
import re
import os
import statistics
import sys
import time
import numpy as np

from dask.distributed import print

from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu


# Log for main function
# per https://chatgpt.com/share/e/67ae4ae8-ff64-800a-8b75-484d388e6a43
def setup_logging_main(log_filename=None):
    """Set up logging to log both to console and a file."""

    logger = logging.getLogger("flm_logger")
    logger.setLevel(logging.INFO)

    # Ensure no duplicate handlers
    if not logger.hasHandlers():
        formatter = logging.Formatter('flm: %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if filename is provided
        if log_filename:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# Populates the log for the main function with various header and run information
def populate_main_log_header(client, cluster, log_note, run_local, model_type, stage):

    main_log_name = f"{cn.combined_log}_main_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    main_log_local_path = f"{cn.local_log_path}{main_log_name}"
    os.makedirs("logs", exist_ok=True)  # Ensures logs directory exists
    main_logger = setup_logging_main(main_log_local_path)  # Sets up logging for the main function

    if run_local:
        worker_memory = "N/A- local run"
        n_workers = "N/A- local run"
        nthreads = "N/A- local run"
        dashboard_link = "N/A- local run"
    else:
        worker_memory, n_workers, nthreads, dashboard_link = uu.get_cluster_info(client, cluster)

    main_logger.info(f"Model type: {model_type}")
    main_logger.info(f"Stage: {stage}")
    main_logger.info(f"Number of workers: {n_workers}")
    main_logger.info(f"Memory per worker: {worker_memory}")
    main_logger.info(f"Coiled dashboard link: {dashboard_link}")
    main_logger.info(f"Threads per worker: {nthreads}")
    main_logger.info(f"Log note: {log_note}\n")

    return main_logger, main_log_local_path, n_workers

# Configure logging for the distributed workers
# https://chatgpt.com/share/e/6f80ccde-6a85-4837-94a0-4fcf09b96e43
def setup_logging_worker():
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Apply formatter to existing handlers (Dask may have already added handlers)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Ensure there's at least one handler
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Determines whether statement should be printed to the console as well as logged
def print_and_log(text, is_large_run, logger):
    logger.info(f"flm: {text}")
    if not is_large_run:
        print(text, flush=True)


# Log compilation and uploading
# From https://chatgpt.com/share/e/4fe1e9c8-05a0-4e9d-8eee-64168891b5e2
# Gets the logs for all workers
def compile_worker_logs(no_log, cluster, stage, start_time_str, logger):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    combined_worker_log_name = f"{cn.combined_log}_workers_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    worker_log_local_path = f"{cn.local_log_path}{combined_worker_log_name}"

    logger.info(f"Combining worker logs into {combined_worker_log_name}")

    # Recovers legs from Coiled
    logs = cluster.get_logs()

    # Filters lines containing 'flm',
    filtered_logs = []
    for worker_id, log in logs.items():
        for line in log.split('\n'):
            if 'flm' in line:
                filtered_logs.append(line)


    combined_filtered_logs = (
            "\n".join(filtered_logs) + "\n"
    )

    # Saves the filtered logs to a text file
    with open(worker_log_local_path, "w") as file:
        file.write(combined_filtered_logs)

    return worker_log_local_path


# Merges the log from main() with all the worker logs after all processing and uploads to s3
def merge_main_and_worker_upload_logs(no_log, main_log, worker_log, stage):

    combined_log_name = f"{cn.combined_log}_combined_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    combined_local_log = f"{cn.local_log_path}{combined_log_name}"

    # Open the output file in write mode
    with open(combined_local_log, "w") as outfile:
        with open(main_log, "r") as infile1:
            outfile.write(infile1.read())
            outfile.write("\n")  # Adds blank line between files

            # Only adds worker logs if no_log not selected
            if not no_log:

                with open(worker_log, "r") as infile2:
                    outfile.write(infile2.read())

        # Calculates average and standard deviation chunk processing times
        with open(combined_local_log, "r") as logfile:
            log_content = logfile.read()

        # Time extraction from https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/691f42e3-cd2c-800a-b9e1-715190ad3024
        # Extracts seconds from lines for core calculation processing
        calc_proc_times__sec = [int(m) for m in re.findall(r'Calculated.*?(\d+) seconds', log_content)]

        # Extract seconds from lines for zarr insertion
        zarr_insert_proc_times__sec = [int(m) for m in re.findall(r'Wrote outputs to global zarr.*?(\d+) seconds', log_content)]

        # Extract seconds from lines for geotif uploads
        uploads_proc_times__sec = [int(m) for m in re.findall(r'Uploads completed for.*?(\d+) seconds', log_content)]

        # Extract seconds from lines for total chunk processing
        total_chunk_proc_times__sec = [int(m) for m in re.findall(r'Total chunk processing.*?(\d+) seconds', log_content)]

        # Extract peak memory usage
        peak_memory__GB = [np.float32(m) for m in re.findall(r'Peak memory for [^:]+: ([0-9]+(?:\.[0-9]+)?) GB', log_content)]

        # Averages
        avg_calc_proc_times__sec = sum(calc_proc_times__sec) / len(calc_proc_times__sec) if calc_proc_times__sec else 0
        avg_zarr_pop_proc_times__sec = sum(zarr_insert_proc_times__sec) / len(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        avg_uploads_proc_times__sec = sum(uploads_proc_times__sec) / len(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        avg_total_chunk_proc_times__sec = sum(total_chunk_proc_times__sec) / len(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        avg_peak_memory__GB = sum(peak_memory__GB) / len(peak_memory__GB) if peak_memory__GB else 0

        # Standard deviations
        stdev_calc_proc_times__sec = statistics.stdev(calc_proc_times__sec) if len(calc_proc_times__sec) > 1 else 0
        stdev_zarr_pop_proc_times__sec = statistics.stdev(zarr_insert_proc_times__sec) if len(zarr_insert_proc_times__sec) > 1 else 0
        stdev_uploads_proc_times__sec = statistics.stdev(uploads_proc_times__sec) if len(uploads_proc_times__sec) > 1 else 0
        stdev_total_chunk_proc_times__sec = statistics.stdev(total_chunk_proc_times__sec) if len(total_chunk_proc_times__sec) > 1 else 0
        stdev_peak_memory__GB = statistics.stdev(peak_memory__GB) if len(peak_memory__GB) > 1 else 0

        # Mins
        min_calc_proc_times__sec = min(calc_proc_times__sec) if calc_proc_times__sec else 0
        min_zarr_pop_proc_times__sec = min(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        min_uploads_proc_times__sec = min(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        min_total_chunk_proc_times__sec = min(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        min_peak_memory__GB = min(peak_memory__GB) if peak_memory__GB else 0

        # Maxes
        max_calc_proc_times__sec = max(calc_proc_times__sec) if calc_proc_times__sec else 0
        max_zarr_pop_proc_times__sec = max(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        max_uploads_proc_times__sec = max(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        max_total_chunk_proc_times__sec = max(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        max_peak_memory__GB = max(peak_memory__GB) if peak_memory__GB else 0

        # Step 3: Append results to the log file
        with open(combined_local_log, "a") as outfile:
            outfile.write("\n")
            outfile.write("=== Chunk-level processing times (approximate because some of worker log may be missing) ===\n")
            outfile.write(f"Processing stats for calculation code ({len(calc_proc_times__sec)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_calc_proc_times__sec:.0f} seconds (stdev: {stdev_calc_proc_times__sec:.0f})\n")
            outfile.write(f"  Min and max: {min_calc_proc_times__sec:.0f}-{max_calc_proc_times__sec:.0f}\n")

            outfile.write(f"Processing stats for zarr insertion code ({len(zarr_insert_proc_times__sec)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_zarr_pop_proc_times__sec:.0f} seconds (stdev: {stdev_zarr_pop_proc_times__sec:.0f})\n")
            outfile.write(f"  Min and max: {min_zarr_pop_proc_times__sec:.0f}-{max_zarr_pop_proc_times__sec:.0f}\n")

            outfile.write(f"Processing stats for geotif upload code ({len(uploads_proc_times__sec)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_uploads_proc_times__sec:.0f} seconds (stdev: {stdev_uploads_proc_times__sec:.0f})\n")
            outfile.write(f"  Min and max: {min_uploads_proc_times__sec:.0f}-{max_uploads_proc_times__sec:.0f}\n")

            outfile.write(f"Processing stats for full tasks ({len(total_chunk_proc_times__sec)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_total_chunk_proc_times__sec:.0f} seconds (stdev: {stdev_total_chunk_proc_times__sec:.0f})\n")
            outfile.write(f"  Min and max: {min_total_chunk_proc_times__sec:.0f}-{max_total_chunk_proc_times__sec:.0f}\n")

            outfile.write(f"Peak memory usage for tasks ({len(peak_memory__GB)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_peak_memory__GB:.2f} GB (stdev: {stdev_peak_memory__GB:.2f})\n")
            outfile.write(f"  Min and max: {min_peak_memory__GB:.2f}-{max_peak_memory__GB:.2f}\n")

            outfile.write("--- End of log---\n")

            print(f"Combined log saved as {combined_local_log}")  # Does not go in the log because it's closed

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(combined_local_log, "gfw2-data", Key=f"{cn.s3_log_path}{combined_log_name}")

    # Removes the main log if the stage doesn't run in batches.
    # The main log must be kept for stages that run in batches because each batch uses the same main log.
    # The main log can be manually deleted after the run is done.
    if stage not in ["create_forest_age_2010_2015__1x1_deg", "vegetation_fluxes", "soil_carbon_densities_and_changes"]:
        os.remove(main_log)

    if not no_log:
        os.remove(worker_log)
"""Logging helpers required by the organic-soils core model."""

import logging
import os
import sys
import time

from dask.distributed import print as dask_print

from . import constants_and_names as cn


def setup_logging_main(log_filename=None):
    """
    Set up a main-function logger that logs both to console (stdout) and a file.
    """
    logger = logging.getLogger("flm_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.hasHandlers():
        formatter = logging.Formatter('flm: %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_filename:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def populate_main_log_header(bounding_box, use_shapefile, client, cluster, log_note,
                             run_local, model_type, stage):
    """
    Creates a main log for the top-level script. Records memory, workers, bounding box, etc.
    """
    from . import universal_utilities as uu

    os.makedirs(cn.local_log_path, exist_ok=True)
    main_log_name = f"{cn.combined_log}_main_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    main_log_local_path = f"{cn.local_log_path}{main_log_name}"
    main_logger = setup_logging_main(main_log_local_path)

    if run_local:
        worker_memory = "N/A - local run"
        n_workers = "N/A - local run"
        nthreads = "N/A - local run"
    else:
        worker_memory, n_workers, nthreads = uu.get_cluster_info(client, cluster)

    main_logger.info(f"Model type: {model_type}")
    main_logger.info(f"Stage: {stage}")
    main_logger.info(f"Model version: {cn.model_version}")
    main_logger.info(f"Number of workers: {n_workers}")
    main_logger.info(f"Memory per worker: {worker_memory}")
    main_logger.info(f"Threads per worker: {nthreads}")
    main_logger.info(f"Log note: {log_note}")

    if bounding_box:
        main_logger.info(f"Bounding box: {bounding_box}")

    if use_shapefile:
        main_logger.info(f"Using shapefile: {use_shapefile}")

    return main_logger, main_log_local_path


def setup_logging_worker():
    """
    Configure logging for distributed workers.
    """
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Apply formatter to existing handlers (Dask may have added some)
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def compile_worker_logs(no_log, cluster, stage, start_time_str, logger):
    """
    Retrieves logs from Coiled, filters lines with 'flm', writes to local file.
    This is used by the core runtime at the end of the run.
    """
    if no_log:
        return None

    os.makedirs(cn.local_log_path, exist_ok=True)
    worker_log_name = f"{cn.combined_log}_workers_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.log"
    worker_log_local_path = f"{cn.local_log_path}{worker_log_name}"

    logger.info(f"Preparing consolidated worker log: {worker_log_name}")

    # Retrieve logs from cluster
    try:
        logs = cluster.get_logs()
    except Exception as e:
        logger.error(f"Error retrieving logs from cluster: {e}")
        logs = {}

    filtered_logs = []
    for worker_id, log_str in logs.items():
        for line in log_str.split('\n'):
            if 'flm' in line:  # capturing lines containing 'flm'
                filtered_logs.append(line)

    combined_filtered_logs = "\n".join(filtered_logs) + "\n"

    # Save filtered logs
    try:
        with open(worker_log_local_path, "w") as f:
            f.write(combined_filtered_logs)
    except Exception as e:
        logger.error(f"Error saving worker logs: {e}")
        return None

    return worker_log_local_path


def print_and_log(text, is_final, logger):
    """
    Print to console if not is_final, and always log with 'flm:' prefix.
    """
    logger.info(f"flm: {text}")
    if not is_final:
        dask_print(f"flm: {text}", flush=True)

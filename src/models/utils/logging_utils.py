# utils/logging_utils.py

import logging
import os
from config.constants import project_dir, processed_dir
from datetime import datetime
import boto3
import time
import re
import pandas as pd


def setup_logging(log_file='processing.log'):
    """
    Set up logging to output to both console and a log file.

    Args:
        log_file (str): The filename for the log file. Defaults to 'processing.log'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('flm_logger')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        log_dir = os.path.join(project_dir, processed_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def print_and_log(text, is_final, logger):
    """
    Logs the text and optionally prints it to the console.

    Args:
        text (str): The message to log and print.
        is_final (bool): Determines if the message should be printed to the console.
        logger (logging.Logger): The logger instance to use.
    """
    logger.info(f"flm: {text}")
    if not is_final:
        print(f"flm: {text}")


def compile_and_upload_log(logs, stage, chunk_count, chunk_size_deg, start_time_str, end_time_str, log_note,
                           combined_log, log_path, model_version, coiled_client, coiled_cluster):
    """
    Compiles logs, filters relevant entries, and uploads the compiled log to S3.

    Args:
        logs (dict): Dictionary containing logs from workers.
        stage (str): Current stage of processing.
        chunk_count (int): Number of chunks processed.
        chunk_size_deg (float): Size of each chunk in degrees.
        start_time_str (str): Start time of the stage.
        end_time_str (str): End time of the stage.
        log_note (str): Additional note for the log.
        combined_log (str): Prefix for the combined log filename.
        log_path (str): S3 path where the log will be uploaded.
        model_version (str): Version of the model.
        coiled_client: Coiled Dask client instance.
        coiled_cluster: Coiled Dask cluster instance.
    """
    log_name = f"logs/{combined_log}_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.txt"

    # Converts the start time of the stage run from string to datetime so it can be compared to the log entries' times
    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")

    # Retrieves the number of workers
    n_workers = len(coiled_client.scheduler_info()['workers'])  # Get the number of connected workers

    # Retrieves scheduler info for other cluster properties
    scheduler_info = coiled_cluster.scheduler_info  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info['workers'][next(iter(scheduler_info['workers']))]['memory_limit']
        worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
    except KeyError:
        worker_memory = "Unknown"

    # Create header lines
    header_lines = [
        f"Stage: {stage}",
        f"Model version: {model_version}",
        f"Number of workers: {n_workers}",
        f"Memory per worker: {worker_memory}",
        f"Number of chunks: {chunk_count}",
        f"Chunk size (degrees): {chunk_size_deg}",
        f"Log note: {log_note}",
        f"Starting time: {start_time_str}",
        "",
        "Filtered logs:",
        ""
    ]

    # Filter lines containing both 'distributed.worker' and 'flm',
    # and where the datetime is greater than start_time
    filtered_logs = []
    tile_id_pattern = r"[0-9]{2}[NS]_[0-9]{3}[EW]"  # Example pattern, adjust as needed
    for worker_id, log in logs.items():
        for line in log.split('\n'):
            if 'distributed.worker' in line and 'flm' in line:
                # Extract the datetime from the end of the log line
                log_time_str = line.split()[-1]
                try:
                    log_time = datetime.strptime(log_time_str, "%Y%m%d_%H_%M_%S")
                    # Include the line only if log_time is greater than start_time
                    if log_time > start_time:
                        filtered_logs.append(line)
                except ValueError:
                    # If the datetime format is incorrect, skip this line
                    continue

    end_time = f"Stage ended at: {end_time_str}"

    # Combine the header and filtered logs into a single string
    combined_filtered_logs = "\n".join(header_lines) + "\n".join(filtered_logs) + "\n" + end_time

    # Save the filtered logs to a text file
    with open(log_name, "w") as file:
        file.write(combined_filtered_logs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(log_name, "gfw2-data", Key=f"{log_path}{log_name}")

    # Optionally, delete the local log file
    os.remove(log_name)


def timestr():
    """
    Returns the current time in Eastern US timezone as a formatted string.

    Returns:
        str: Current time formatted as "YYYYMMDD_HH_MM_SS".
    """
    import pytz
    from datetime import datetime

    # Define the Eastern Time timezone
    eastern = pytz.timezone('US/Eastern')

    # Get the current time in Eastern Time
    eastern_time = datetime.now(eastern)

    # Format the time as a string
    return eastern_time.strftime("%Y%m%d_%H_%M_%S")

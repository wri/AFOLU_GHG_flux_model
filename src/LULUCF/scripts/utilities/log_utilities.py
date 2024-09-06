import boto3
import logging
import time

from dask.distributed import print
from datetime import datetime


from . import constants_and_names as cn
from . import universal_utilities as uu


# Log compilation and uploading
# From https://chatgpt.com/share/e/4fe1e9c8-05a0-4e9d-8eee-64168891b5e2
# Gets the logs for all workers
#TODO Wait to run this until all entries have been added to the Coiled log--
# running this right after the model finishes means that final log entries haven't made it into Coiled yet.
def compile_and_upload_log(no_log, client, cluster, stage,
                           chunk_count, chunk_size_deg, start_time_str, end_time_str, log_note):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    #TODO Create log folder if it doesn't exist already
    log_name = f"{cn.combined_log}_{stage}_{time.strftime('%Y%m%d_%H_%M_%S')}.txt"
    local_log = f"{cn.local_log_path}{log_name}"

    print(f"Preparing consolidated log {log_name}")

    # Recovers legs from Coiled
    logs = cluster.get_logs()

    # Converts the start time of the stage run from string to datetime so it can be compared to the log entries' times
    start_time = datetime.strptime(start_time_str, "%Y%m%d_%H_%M_%S")

    # Retrieves the number of workers
    n_workers = len(client.scheduler_info()['workers'])  # Get the number of connected workers

    # Retrieves scheduler info for other cluster properties
    scheduler_info = cluster.scheduler_info  # Access scheduler info directly as a dictionary

    # Gets memory per worker.
    # Can't get it to report the worker instance type
    try:
        worker_memory_bytes = scheduler_info['workers'][next(iter(scheduler_info['workers']))]['memory_limit']
        worker_memory_gb = worker_memory_bytes / (1024 ** 3)  # Convert bytes to GB
        worker_memory = f"{worker_memory_gb:.2f} GB"  # Format to 2 decimal places
        # worker_type = coiled_cluster.config.get('worker_options', {}).get('instance_type', "Unknown")
    except KeyError:
        worker_memory = "Unknown"
        # worker_type = "Unknown"

    # Create header lines
    header_lines = [
        f"Stage: {stage}",
        f"Model version: {cn.model_version}",
        f"Number of workers: {n_workers}",
        f"Memory per worker: {worker_memory}",
        f"Number of chunks: {chunk_count}",
        f"Chunk size (degrees): {chunk_size_deg}",
        # f"Worker Type: {worker_type}",
        f"Log note: {log_note}",
        f"Starting time: {start_time_str}",
        "",
        "Filtered logs:",
        ""
    ]

    # Filter lines containing both 'distributed.worker' and 'flm',
    # and where the datetime is greater than start_time
    filtered_logs = []
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
    with open(local_log, "w") as file:
        file.write(combined_filtered_logs)

    s3_client = boto3.client("s3")  # Needs to be in the same function as the upload_file call
    s3_client.upload_file(local_log, "gfw2-data", Key=f"{cn.s3_log_path}{log_name}")

    print(f"Log uploaded to {cn.s3_log_path}{log_name}")


# Determines whether statement should be printed to the console as well as logged
def print_and_log(text, is_final, logger):

    logger.info(f"flm: {text}")
    if not is_final:
        print(f"flm: {text}")


# Configure logging for the distributed workers
# https://chatgpt.com/share/e/6f80ccde-6a85-4837-94a0-4fcf09b96e43
def setup_logging():
    logger = logging.getLogger('distributed.worker')
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
import boto3
import logging
import re
import os
import statistics
import sys
import time
import numpy as np
import gzip
from datetime import datetime, timedelta
import coiled
import pytz

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


### New approach for compiling worker logs, from Claude conversation 'Missing worker tasks in veg log'
# Patterns for selecting lines for the readable worker log.
# The raw worker log keeps everything regardless of these.
FLM_TAG = "flm:"
SEVERITY_PATTERN = re.compile(r"\b(WARNING|ERROR|CRITICAL)\b")
LIFECYCLE_PATTERN = re.compile(
    # Task failures
    r"KilledWorker|marked as failed|"
    # Worker process deaths and restarts (nanny)
    r"Worker process died unexpectedly|Nanny asking worker to close|Worker process still alive|"
    r"Failed to restart worker|Restarting worker|"
    # Scheduler removing workers
    r"Remove worker|Removing worker|Lost all workers|"
    # Memory
    r"memory budget|memory usage\. (Pausing|Resuming) worker|Unmanaged memory use is high|Worker memory limit|"
    # System-level (not from Dask; unverified against Coiled system logs)
    r"\bOOM\b|oom-kill|Out of memory|Segmentation fault|SIGSEGV|SIGKILL|"
    # Communication problems
    r"CommClosedError|Timed out",
    re.IGNORECASE,
)
TRACEBACK_START = "Traceback (most recent call last)"
TRACEBACK_CONTINUATION = re.compile(
    r"^\s|^$|^[\w.]+(Error|Exception|Exit|Interrupt|Warning)\b|"
    r"^During handling of the above exception|^The above exception was the direct cause"
)


# Converts a timestr()-format string (US/Eastern, %Y%m%d_%H_%M_%S) to epoch milliseconds
def _timestr_to_epoch_ms(time_str):
    eastern = pytz.timezone('US/Eastern')
    t = eastern.localize(datetime.strptime(time_str, "%Y%m%d_%H_%M_%S"))
    return int(t.timestamp() * 1000)


# Maps Coiled instance ids to readable labels (scheduler or worker name)
def _coiled_instance_labels(cluster_id, workspace):
    details = coiled.cluster_details(cluster_id, workspace=workspace)
    labels = {}
    scheduler_id = (details.get("scheduler") or {}).get("instance", {}).get("id")
    if scheduler_id:
        labels[scheduler_id] = "scheduler"
    for worker in details.get("workers", []):
        if worker.get("instance"):
            labels[worker["instance"]["id"]] = worker.get("name") or str(worker["instance"]["id"])
    return labels


# Yields all Coiled log events (dask + system, all instances) between since_ms and until_ms, oldest first.
# The endpoint returns a limited number of events per request, so this pages forward from the last
# timestamp received, skipping events already seen at that timestamp (same approach as `coiled cluster logs`).
def _iter_coiled_log_events(cluster_id, workspace, since_ms, until_ms, logger, progress_every=25):
    from_ts = since_ms
    seen_at_from_ts = set()
    pages = 0
    n_events = 0

    while True:
        events = coiled.better_cluster_logs(cluster_id=cluster_id, workspace=workspace,
                                            dask=True, system=True,
                                            since_ms=from_ts, until_ms=until_ms)

        new_events = [e for e in events
                      if not (e["timestamp"] == from_ts and
                              f"{e['timestamp']}#{e.get('instance_id')}#{e['message']}" in seen_at_from_ts)]
        if not new_events:
            break

        for e in new_events:
            yield e

        # Tracks events at the newest timestamp so they aren't repeated on the next page
        last_ts = new_events[-1]["timestamp"]
        last_keys = {f"{e['timestamp']}#{e.get('instance_id')}#{e['message']}"
                     for e in new_events if e["timestamp"] == last_ts}
        seen_at_from_ts = seen_at_from_ts | last_keys if last_ts == from_ts else last_keys
        from_ts = last_ts

        pages += 1
        n_events += len(new_events)
        if pages % progress_every == 0:
            t = datetime.fromtimestamp(from_ts / 1000, pytz.timezone('US/Eastern')).strftime("%Y%m%d_%H_%M_%S")
            logger.info(f"  Retrieved {n_events} log events in {pages} requests (up to {t}): {uu.timestr()}")


# Log compilation and uploading
# Gets the logs for the scheduler and all workers (including replaced workers) from Coiled for the whole stage.
# Writes two files:
#   1. Raw log (gzipped, uploaded to s3): every log line from every instance, unfiltered.
#   2. Readable log (returned, later merged with main log): flm lines, warnings/errors, tracebacks,
#      and worker lifecycle events (killed, lost, memory, etc.), labeled by instance.
def compile_worker_logs_new(no_log, cluster, stage, start_time_str, logger):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    log_time = time.strftime('%Y%m%d_%H_%M_%S')
    combined_worker_log_name = f"{cn.combined_log}_workers_{stage}_{log_time}.log"
    worker_log_local_path = f"{cn.local_log_path}{combined_worker_log_name}"
    raw_worker_log_name = f"{cn.combined_log}_workers_raw_{stage}_{log_time}.log.gz"
    raw_worker_log_local_path = f"{cn.local_log_path}{raw_worker_log_name}"

    logger.info(f"Combining worker logs into {combined_worker_log_name} (raw: {raw_worker_log_name}): {uu.timestr()}")

    cluster_id = cluster.cluster_id
    workspace = cluster.workspace
    instance_labels = _coiled_instance_labels(cluster_id, workspace)

    # Starts a bit before the stage start to catch worker startup messages
    since_ms = _timestr_to_epoch_ms(start_time_str) - int(timedelta(minutes=10).total_seconds() * 1000)
    until_ms = int(time.time() * 1000)

    eastern = pytz.timezone('US/Eastern')
    in_traceback = {}   # Per-instance flag for whether the current line continues a traceback
    n_raw = n_kept = n_flm = n_severity = n_tracebacks = 0

    with gzip.open(raw_worker_log_local_path, "wt") as raw_file, open(worker_log_local_path, "w") as readable_file:
        for event in _iter_coiled_log_events(cluster_id, workspace, since_ms, until_ms, logger):

            instance_id = event.get("instance_id")
            label = instance_labels.get(instance_id, str(instance_id or event.get("instance", "unknown")))
            t = datetime.fromtimestamp(event["timestamp"] / 1000, eastern).strftime("%Y-%m-%d %H:%M:%S")

            # An event can contain multiple lines (e.g., a traceback), so they are handled together
            lines = event["message"].split("\n")
            out_lines = [f"({label}) {t} {line}\n" for line in lines]

            for out_line in out_lines:
                raw_file.write(out_line)
            n_raw += len(lines)

            # Keeps the whole event if any of its lines match, so that multi-line records
            # (e.g., Dask's "Compute Failed" with Key/Exception/Traceback) stay intact
            has_flm = any(FLM_TAG in line for line in lines)
            has_severe = any(SEVERITY_PATTERN.search(line) for line in lines)
            has_lifecycle = any(LIFECYCLE_PATTERN.search(line) for line in lines)
            has_tb_start = any(TRACEBACK_START in line for line in lines)

            # Tracebacks split across separate events are followed until a non-traceback line
            continues_tb = in_traceback.get(instance_id) and TRACEBACK_CONTINUATION.match(lines[0])

            if has_tb_start:
                in_traceback[instance_id] = True
                n_tracebacks += 1
            elif not continues_tb:
                in_traceback[instance_id] = False

            if has_flm or has_severe or has_lifecycle or has_tb_start or continues_tb:
                readable_file.writelines(out_lines)
                n_kept += len(lines)
                n_flm += has_flm
                n_severity += has_severe

    # Uploads the raw log to s3 so that nothing is lost even if the readable filter is too narrow
    s3_client = boto3.client("s3")
    s3_client.upload_file(raw_worker_log_local_path, "gfw2-data", Key=f"{cn.s3_log_path}{raw_worker_log_name}")

    # Counts unique chunks that finished, as a quick check of log completeness
    with open(worker_log_local_path, "r") as f:
        finished_chunks = set(re.findall(r"Total chunk processing for (\S+)", f.read()))

    logger.info(f"Worker log lines: {n_raw} raw, {n_kept} kept in readable log "
                f"({n_flm} flm, {n_severity} WARNING/ERROR/CRITICAL, {n_tracebacks} tracebacks)")
    logger.info(f"Unique chunks with 'Total chunk processing' in worker log: {len(finished_chunks)}")
    logger.info(f"Raw worker log uploaded to s3://gfw2-data/{cn.s3_log_path}{raw_worker_log_name}: {uu.timestr()}")

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

        # Extract peak memory usage (could be MB or GB)
        peak_memory = [np.float32(m) for m in re.findall(r'Peak memory for [^:]+: ([0-9]+(?:\.[0-9]+)?)', log_content)]

        # Averages
        avg_calc_proc_times__sec = sum(calc_proc_times__sec) / len(calc_proc_times__sec) if calc_proc_times__sec else 0
        avg_zarr_pop_proc_times__sec = sum(zarr_insert_proc_times__sec) / len(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        avg_uploads_proc_times__sec = sum(uploads_proc_times__sec) / len(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        avg_total_chunk_proc_times__sec = sum(total_chunk_proc_times__sec) / len(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        avg_peak_memory = sum(peak_memory) / len(peak_memory) if peak_memory else 0

        # Standard deviations
        stdev_calc_proc_times__sec = statistics.stdev(calc_proc_times__sec) if len(calc_proc_times__sec) > 1 else 0
        stdev_zarr_pop_proc_times__sec = statistics.stdev(zarr_insert_proc_times__sec) if len(zarr_insert_proc_times__sec) > 1 else 0
        stdev_uploads_proc_times__sec = statistics.stdev(uploads_proc_times__sec) if len(uploads_proc_times__sec) > 1 else 0
        stdev_total_chunk_proc_times__sec = statistics.stdev(total_chunk_proc_times__sec) if len(total_chunk_proc_times__sec) > 1 else 0
        stdev_peak_memory = statistics.stdev(peak_memory) if len(peak_memory) > 1 else 0

        # Mins
        min_calc_proc_times__sec = min(calc_proc_times__sec) if calc_proc_times__sec else 0
        min_zarr_pop_proc_times__sec = min(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        min_uploads_proc_times__sec = min(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        min_total_chunk_proc_times__sec = min(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        min_peak_memory = min(peak_memory) if peak_memory else 0

        # Maxes
        max_calc_proc_times__sec = max(calc_proc_times__sec) if calc_proc_times__sec else 0
        max_zarr_pop_proc_times__sec = max(zarr_insert_proc_times__sec) if zarr_insert_proc_times__sec else 0
        max_uploads_proc_times__sec = max(uploads_proc_times__sec) if uploads_proc_times__sec else 0
        max_total_chunk_proc_times__sec = max(total_chunk_proc_times__sec) if total_chunk_proc_times__sec else 0
        max_peak_memory = max(peak_memory) if peak_memory else 0

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

            outfile.write(f"Peak memory usage for tasks ({len(peak_memory)} tasks):\n")
            outfile.write(f"  Average and stdev: {avg_peak_memory:.2f} MB or GB (stdev: {stdev_peak_memory:.2f})\n")
            outfile.write(f"  Min and max: {min_peak_memory:.2f}-{max_peak_memory:.2f}\n")

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
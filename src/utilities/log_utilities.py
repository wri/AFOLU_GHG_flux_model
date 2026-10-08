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
from coiled.v2 import cluster_details
import pytz
import builtins
from dask.distributed import get_worker, print

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
        formatter = logging.Formatter('%(message)s')

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


# Determines whether statement should be printed to the console as well as logged in workers.
# On a Dask worker, the message is sent only to the client's terminal.
# Off a Dask worker, it prints normally to the console.
# From Claude conversation 'Missing worker tasks in veg log'
def print_and_log(text, is_large_run, logger):
    # On Coiled, goes to the worker's log. In local runs, goes to the terminal (not the main log file).
    # This way, everything gets logged during a Coiled run.
    logger.info(f"flm: {text}")

    # Only affects what is printed in the terminal.
    # Essentially, things are printed to terminal only during small runs.
    if not is_large_run:
        try:  # Checks if running on a Dask worker (i.e., on Coiled)
            worker = get_worker()
        except ValueError:  # Not running on a worker, so just prints to terminal
            builtins.print(text, flush=True)
        else:  # On a worker for a small run: sends the message to the terminal, without printing it on the worker
            worker.log_event("print", {"args": (str(text),), "sep": " ", "end": "\n", "flush": True})


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
    # Shutdowns started outside Dask (e.g., Coiled stopping the cluster)
    r"Received signal SIG|Closing scheduler|"
    # Communication problems
    r"CommClosedError|Timed out",
    re.IGNORECASE,
)

# System messages from cloud machine startup that aren't relevant to the model.
# Excluded from the readable log (still in the raw log).
IGNORE_PATTERN = re.compile(r"snapd\[|networkd-dispatcher\[|amazon-ssm-agent|Removing Worker plugin")

TRACEBACK_START = "Traceback (most recent call last)"
TRACEBACK_CONTINUATION = re.compile(
    r"^\s|^$|^[\w.]+(Error|Exception|Exit|Interrupt|Warning)\b|"
    r"^During handling of the above exception|^The above exception was the direct cause"
)

# Parts of worker log lines removed from the readable log, so lines read "<worker id> <time> - <message>"
SYSTEM_LINE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)\s+\S+\s+")  # "<UTC time> <machine> " on system lines
FLM_LOGGER_PREFIX = re.compile(r"^distributed\.worker - INFO - (?=flm:)")  # Dask logger name and level, on flm lines only (logged by my code as opposed to Dask system)
FLM_TIMESTAMP_SUFFIX = re.compile(r":\s*\d{8}_\d{2}_\d{2}_\d{2}\s*$")  # ": 20261008_11_33_26" at the end of flm lines

# Removes repeated information (logger name, extra timestamps, machine address) from a line for the readable log
def _clean_readable_line(line):
    line = SYSTEM_LINE_PREFIX.sub("", line)
    line = FLM_LOGGER_PREFIX.sub("", line)
    return FLM_TIMESTAMP_SUFFIX.sub("", line)


# Converts a timestr()-format string (US/Eastern, %Y%m%d_%H_%M_%S) to epoch milliseconds
def _timestr_to_epoch_ms(time_str):
    eastern = pytz.timezone('US/Eastern')
    t = eastern.localize(datetime.strptime(time_str, "%Y%m%d_%H_%M_%S"))
    return int(t.timestamp() * 1000)


# Maps Coiled instance ids to readable labels (scheduler or worker name)
def _coiled_instance_labels(cluster_id, workspace):
    details = cluster_details(cluster_id, workspace=workspace)
    labels = {}
    scheduler_id = (details.get("scheduler") or {}).get("instance", {}).get("id")
    if scheduler_id:
        labels[scheduler_id] = "scheduler"
    for worker in details.get("workers", []):
        if worker.get("instance"):
            # Worker names are "<cluster name>-worker-<id>"; keeps just the id
            name = worker.get("name")
            labels[worker["instance"]["id"]] = f"id-{name.rsplit('-', 1)[-1]}" if name else f"id-{worker['instance']['id']}"
    return labels

# Key that identifies a log event, for skipping events that were already retrieved.
# (Coiled's event "id" field is empty, so time, instance, and message are used.)
def _event_key(e):
    return f"{e['timestamp']}#{e.get('instance_id')}#{e['message']}"

# Formats a duration in seconds as hours (2 decimals), matching stage_duration, e.g. "0.19 hours"
def _format_duration(seconds):
    return f"{seconds / 3600:.2f} hours"

# Yields all Coiled log events (dask + system, all instances) between since_ms and until_ms, oldest first, once each.
# The endpoint returns a limited number of events per request, so this pages forward from the newest timestamp received
# (same approach as `coiled cluster logs`).
# When queried soon after a run (cluster running or just shut down), Coiled has been seen to return the whole time range
# again on a later request, and to return two copies of each event in a single response. So each page is sorted by time,
# events older than the newest timestamp already yielded are skipped, and events at that timestamp are skipped if
# already yielded. Only keys at the newest timestamp are kept, so memory use stays small for large runs.
def _iter_coiled_log_events(cluster_id, workspace, since_ms, until_ms, logger, progress_interval_s=30):
    current_ts = since_ms          # Newest timestamp yielded so far (also where the next request starts)
    keys_at_current_ts = set()     # Events already yielded at current_ts
    requests = 0
    n_events = 0
    n_skipped = 0
    eastern = pytz.timezone('US/Eastern')
    total_range_ms = max(until_ms - since_ms, 1)
    retrieval_start = last_progress = time.time()

    while True:
        events = coiled.better_cluster_logs(cluster_id=cluster_id, workspace=workspace,
                                            dask=True, system=True,
                                            since_ms=current_ts, until_ms=until_ms)
        requests += 1
        n_new = 0

        for e in sorted(events, key=lambda e: e["timestamp"]):
            key = _event_key(e)

            # Skips events already yielded (older than the newest timestamp, or already seen at it)
            if e["timestamp"] < current_ts or (e["timestamp"] == current_ts and key in keys_at_current_ts):
                n_skipped += 1
                continue

            # Moves forward to a newer timestamp
            if e["timestamp"] > current_ts:
                current_ts = e["timestamp"]
                keys_at_current_ts = set()

            keys_at_current_ts.add(key)
            n_new += 1
            yield e

        # Stops when a request returns nothing that hasn't already been yielded
        if n_new == 0:
            break

        n_events += n_new

        # Progress report at most every progress_interval_s seconds.
        # Percent is the share of the log time range covered so far. It is approximate because log events
        # aren't spread evenly over time (most are written while chunks are processing).
        now = time.time()
        if now - last_progress >= progress_interval_s:
            last_progress = now
            pct = min(100.0, 100.0 * (current_ts - since_ms) / total_range_ms)
            t = datetime.fromtimestamp(current_ts / 1000, eastern).strftime("%Y%m%d_%H_%M_%S")
            elapsed = now - retrieval_start
            logger.info(f"  Log retrieval ~{pct:.0f}% of time range (up to {t}): {n_events} events in {requests} requests, "
                        f"{n_events / elapsed:.0f} events/s, {_format_duration(elapsed)} elapsed: {uu.timestr()}")

    if n_skipped:
        logger.info(f"  Skipped {n_skipped} log events that Coiled returned more than once")

    logger.info(f"  Retrieved {n_events} log events in {requests} requests in {_format_duration(time.time() - retrieval_start)}")


# Log compilation and uploading
# Gets the logs for the scheduler and all workers (including replaced workers) from Coiled for the whole stage.
# Writes two files:
#   1. Raw log (gzipped, saved locally): every log line from every instance, unfiltered.
#   2. Readable log (returned, later merged with main log): flm lines, warnings/errors, tracebacks,
#      and worker lifecycle events (killed, lost, memory, etc.), labeled by instance.
# Also summarizes worker restarts for exceeding the memory limit and unexpected worker deaths, so the cause of
# a failed run is visible in the main log.
# From Claude conversation 'Missing worker tasks in veg log'
def compile_worker_logs(no_log, cluster, stage, start_time_str, logger, end_time_str=None):

    # Only consolidates the worker logs and uploads to s3 if not deactivated
    if no_log:
        return

    compile_start = time.time()

    log_time = time.strftime('%Y%m%d_%H_%M_%S')
    combined_worker_log_name = f"{cn.combined_log}_workers_{stage}_{log_time}.log"
    worker_log_local_path = f"{cn.local_log_path}{combined_worker_log_name}"
    raw_worker_log_name = f"{cn.combined_log}_workers_raw_{stage}_{log_time}.log.gz"
    raw_worker_log_local_path = f"{cn.local_log_path}{raw_worker_log_name}"

    logger.info(f"Combining worker logs into {combined_worker_log_name} (raw: {raw_worker_log_name}): {uu.timestr()}")

    cluster_id = cluster.cluster_id
    workspace = cluster.workspace
    instance_labels = _coiled_instance_labels(cluster_id, workspace)

    # Time range of logs to download: from a bit before the stage start (to catch worker startup messages)
    since_ms = _timestr_to_epoch_ms(start_time_str) - int(timedelta(minutes=1).total_seconds() * 1000)

    # Downloads logs up to end_time_str if supplied (e.g., when the cluster was shrunk, so workers being shut down during cluster resize
    # aren't reported as dying); otherwise logs downloaded up to now
    until_ms = _timestr_to_epoch_ms(end_time_str) if end_time_str else int(time.time() * 1000)

    # Gives Coiled time to receive the last worker log lines before downloading
    logger.info(f"Waiting 30 s for Coiled to receive the last worker log lines: {uu.timestr()}")
    time.sleep(30)
    logger.info(f"Retrieving Coiled logs (progress reported every 30 s): {uu.timestr()}")

    eastern = pytz.timezone('US/Eastern')
    in_traceback = {}   # Per-instance flag for whether the current line continues a traceback
    n_raw = n_kept = n_flm = n_severity = n_tracebacks = 0
    n_memory_restarts = n_worker_deaths = 0

    with gzip.open(raw_worker_log_local_path, "wt") as raw_file, open(worker_log_local_path, "w") as readable_file:
        for event in _iter_coiled_log_events(cluster_id, workspace, since_ms, until_ms, logger):

            instance_id = event.get("instance_id")
            label = instance_labels.get(instance_id, str(instance_id or event.get("instance", "unknown")))
            t = datetime.fromtimestamp(event["timestamp"] / 1000, eastern).strftime("%Y-%m-%d %H:%M:%S")
            message = event["message"]

            # An event can contain multiple lines (e.g., a traceback), so they are handled together
            lines = message.split("\n")
            # Raw log keeps messages as they arrived; readable log has repeated information removed
            raw_file.writelines(f"{label} {t} - {line}\n" for line in lines)
            out_lines = [f"{label} {t} - {_clean_readable_line(line)}\n" for line in lines]

            n_raw += len(lines)

            # Counts worker restarts for exceeding the memory limit and unexpected worker deaths (for the summary below).
            # Only Dask's own log lines are counted, because some machines also copy these messages into their system log.
            if event.get("log_type") == "dask":
                if re.search(r"exceeded \d+% memory budget", message):
                    n_memory_restarts += 1
                if "Worker process died unexpectedly" in message:
                    n_worker_deaths += 1

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

            if not IGNORE_PATTERN.search(message) and (has_flm or has_severe or has_lifecycle or has_tb_start or continues_tb):
                readable_file.writelines(out_lines)
                n_kept += len(lines)
                n_flm += has_flm
                n_severity += has_severe

    # Counts unique chunks that finished, as a quick check of log completeness
    with open(worker_log_local_path, "r") as f:
        finished_chunks = set(re.findall(r"Total chunk processing for (\S+)", f.read()))

    logger.info(f"Worker log lines: {n_raw} raw, {n_kept} kept in readable log "
                f"({n_flm} flm, {n_severity} WARNING/ERROR/CRITICAL, {n_tracebacks} tracebacks)")
    logger.info(f"Worker restarts for exceeding memory limit: {n_memory_restarts}; "
                f"unexpected worker deaths: {n_worker_deaths}")
    logger.info(f"Unique chunks with 'Total chunk processing' in worker log: {len(finished_chunks)}")
    logger.info(f"Raw worker log saved locally to {raw_worker_log_local_path}: {uu.timestr()}")

    logger.info(f"Worker log compilation took {_format_duration(time.time() - compile_start)} "
                f"(including the 30 s wait for Coiled to receive the last log lines): {uu.timestr()}")

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
        outfile.write("=== Chunk-level processing times ===\n")
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
    if stage not in ["create_forest_age_2010_2015__1x1_deg", "soil_carbon_densities_and_changes"]:
        os.remove(main_log)

    if not no_log:
        os.remove(worker_log)
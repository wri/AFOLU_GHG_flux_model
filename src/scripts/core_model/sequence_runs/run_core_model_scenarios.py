"""Run the authoritative core model and aggregation stages on one cluster."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import coiled

from src.scripts.utilities import constants_and_names as cn
from src.scripts.utilities import local_output_paths as lop


DEFAULT_RUN_NAME = "ogh_mixed_f1_f15_f2_20260513"
DEFAULT_RUN_DATE = "20260525"
DEFAULT_THRESHOLD_CSV = "config/thresholds/ogh_20260513_mixed.csv"
DEFAULT_INTERVAL_END_YEARS = ("2005", "2010", "2015", "2020", "2024")
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple[str, ...]


def _shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in argv)


def build_steps(cluster_name: str, phases: Sequence[str]) -> list[Step]:
    selected = set(phases)
    steps: list[Step] = []

    if "model" in selected:
        steps.append(
            Step(
                label="model",
                command=(
                    sys.executable,
                    "-m",
                    "src.scripts.core_model.0_drainage_emissions_model",
                    "--cluster_name",
                    cluster_name,
                    "--full_model",
                    "--chunk_size",
                    "1",
                    "--all_five_year_periods",
                    "--count_burned_years",
                    "--peat_dataset",
                    "ogh",
                    "--peat_threshold",
                    "0.27",
                    "--peat_threshold_by_biome",
                    DEFAULT_THRESHOLD_CSV,
                    "--fscore_metric",
                    "mixed",
                    "--peat_threshold_scenario",
                    "baseline",
                    "--drainage_distance_threshold_m",
                    "500",
                    "--emission_factor_variant",
                    "default",
                    "--create_zarr",
                    "--run_date",
                    DEFAULT_RUN_DATE,
                    "--run_name",
                    DEFAULT_RUN_NAME,
                ),
            )
        )

    if "aggregate" in selected:
        steps.append(
            Step(
                label="aggregate",
                command=(
                    sys.executable,
                    "-m",
                    "src.scripts.core_model.02_aggregate_soils_outputs",
                    "-cn",
                    cluster_name,
                    "--run_name",
                    DEFAULT_RUN_NAME,
                    "--output_date",
                    DEFAULT_RUN_DATE,
                    "--interval_type",
                    "five_year",
                    "--pixel_resolution",
                    "4000_pixels",
                    "--interval_end_years",
                    *DEFAULT_INTERVAL_END_YEARS,
                ),
            )
        )

    return steps


def wait_for_cluster_ready(cluster_name: str, wait_seconds: int) -> None:
    deadline = time.monotonic() + max(wait_seconds, 0)
    while True:
        clusters = coiled.list_clusters(workspace=cn.Coiled_workspace)
        matching = [item for item in clusters if item.get("name") == cluster_name]
        if matching:
            state = matching[0].get("current_state", {}).get("state")
            if state == "ready":
                return
        else:
            state = "missing"

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Cluster {cluster_name!r} is not ready in workspace "
                f"{cn.Coiled_workspace!r}; last state: {state!r}."
            )
        time.sleep(30)


def run_step(step: Step, log_dir: Path, environment: dict[str, str]) -> None:
    log_path = log_dir / f"{step.label}.log"
    print(f">>> {step.label}: {_shell_join(step.command)}")
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.Popen(
            step.command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, step.command)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core organic-soils model and aggregation stages."
    )
    parser.add_argument("--cluster-name", required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("model", "aggregate"),
        default=("model", "aggregate"),
    )
    parser.add_argument("--cluster-ready-wait-seconds", type=int, default=900)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    steps = build_steps(args.cluster_name, args.phases)

    if args.dry_run:
        for step in steps:
            print(_shell_join(step.command))
        return 0

    wait_for_cluster_ready(args.cluster_name, args.cluster_ready_wait_seconds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or (
        Path(lop.pipeline_log_dir("core_model_scenarios"))
        / DEFAULT_RUN_DATE
        / timestamp
    )
    log_dir.mkdir(parents=True, exist_ok=False)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["AFOLU_CLUSTER_READY_WAIT_SECONDS"] = str(
        args.cluster_ready_wait_seconds
    )
    environment["AFOLU_REQUIRE_CLUSTER"] = "1"

    for step in steps:
        run_step(step, log_dir, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a Coiled cluster for the organic-soils core model."""

import argparse
import inspect
import sys
import coiled

from . import constants_and_names as cn


# ----------------------------
# Instance type presets
# ----------------------------
# IMPORTANT: All instance types (scheduler + workers) must share the same CPU architecture.
# Default is ARM (aarch64) to match your existing *g and x*gd selections.


ARM_WORKER_TYPES = {
    16:  ["x2gd.medium",  "r7g.large",   "r6g.large"],
    32:  ["x8g.large",    "r7g.xlarge",  "r6g.xlarge"],
    64:  ["x2gd.xlarge",  "r7g.2xlarge", "r6g.2xlarge"],
    128: ["x2gd.2xlarge", "r7g.4xlarge", "r6g.4xlarge"],
}

# Scheduler can be smaller; keep ARM-only fallbacks for availability.
ARM_SCHEDULER_TYPES_DEFAULT = ["m7g.large", "m6g.large", "t4g.large"]


# Optional x86 presets (only used if you pass --arch x86).
# NOTE: Keep these x86-only; do not mix with ARM types.
X86_WORKER_TYPES = {
    16:  ["r7i.large",   "r6i.large"],
    32:  ["r7i.xlarge",  "r6i.xlarge"],
    64:  ["r7i.2xlarge", "r6i.2xlarge"],
    128: ["r7i.4xlarge", "r6i.4xlarge"],
}
X86_SCHEDULER_TYPES_DEFAULT = ["m7i.large", "m6i.large", "t3.large"]


IDLE_TIMEOUT_MINUTES = {
    16: 25,
    32: 20,
    64: 15,
    128: 15,
}


def _get_instance_type_lists(worker_memory: int, arch: str,
                             scheduler_vm_types_override: list[str] | None = None) -> tuple[list[str], list[str]]:
    if worker_memory not in (16, 32, 64, 128):
        sys.exit("Memory argument must be one of: 16, 32, 64, 128 (GiB)")

    arch = arch.lower().strip()
    if arch not in ("arm", "x86"):
        sys.exit("arch must be one of: arm, x86")

    if arch == "arm":
        worker_vm_types = ARM_WORKER_TYPES[worker_memory]
        scheduler_vm_types = scheduler_vm_types_override or ARM_SCHEDULER_TYPES_DEFAULT
    else:
        worker_vm_types = X86_WORKER_TYPES[worker_memory]
        scheduler_vm_types = scheduler_vm_types_override or X86_SCHEDULER_TYPES_DEFAULT

    # Defensive: ensure user override doesn't accidentally mix architectures with the selected arch.
    # (We cannot perfectly validate arch locally without an AWS lookup; Coiled will also validate.)
    return worker_vm_types, scheduler_vm_types


def create_cluster(
    cluster_name: str,
    n_workers: int,
    threads_per_worker: int,
    worker_memory: int,
    *,
    workspace: str = cn.Coiled_workspace,
    region: str = "us-east-1",
    spot_policy: str = "spot_with_fallback",
    use_best_zone: bool = True,
    allow_cross_zone: bool = False,
    arch: str = "arm",
    scheduler_vm_types: list[str] | None = None,
    software: str | None = None,
    idle_timeout: str | None = None,
):
    """
    Create a Coiled cluster with availability-friendly instance-type fallbacks.
    """

    if not cluster_name:
        sys.exit("cluster_name is required")

    # Resolve instance-type lists (must be same architecture across scheduler + workers)
    worker_vm_types, scheduler_vm_types_final = _get_instance_type_lists(
        worker_memory=worker_memory,
        arch=arch,
        scheduler_vm_types_override=scheduler_vm_types,
    )

    if idle_timeout is None:
        idle_timeout_minutes = IDLE_TIMEOUT_MINUTES.get(worker_memory, 20)
        idle_timeout = f"{idle_timeout_minutes} minutes"

    cluster_kwargs = {
        "name": cluster_name,
        "software": software,
        "region": region,
        "n_workers": n_workers,
        # Availability knobs
        "spot_policy": spot_policy,        # "on-demand" | "spot" | "spot_with_fallback"
        "use_best_zone": use_best_zone,    # pick best AZ in-region for your requested instances
        "allow_cross_zone": allow_cross_zone,  # may improve capacity; may increase network cost
        "idle_timeout": idle_timeout,
        "tags": {"project": "AFOLU_flux_model"},
        # Instance types (lists prioritized in order)
        "scheduler_vm_types": scheduler_vm_types_final,
        "worker_vm_types": worker_vm_types,
        "worker_options": {"nthreads": int(threads_per_worker)},
    }
    cluster_params = inspect.signature(coiled.Cluster).parameters
    cluster_kwargs = {
        key: value
        for key, value in cluster_kwargs.items()
        if key in cluster_params
    }
    if "workspace" in cluster_params:
        cluster_kwargs["workspace"] = workspace
    else:
        cluster_kwargs["account"] = workspace

    cluster = coiled.Cluster(**cluster_kwargs)

    print(f"Cluster created with name: {cluster.name}")
    print(f"Workspace: {workspace}; Region: {region}; Arch: {arch}")
    print(f"Software environment: {software or 'package sync/default'}")
    print(f"Spot policy: {spot_policy}; use_best_zone: {use_best_zone}; allow_cross_zone: {allow_cross_zone}")
    print(f"Idle timeout: {idle_timeout}")
    print(f"Scheduler types (priority): {scheduler_vm_types_final}")
    print(f"Worker types (priority): {worker_vm_types}")
    print(f"Number of workers: {n_workers}; worker memory tier: {worker_memory}GiB; threads/worker: {threads_per_worker}")
    return cluster


def _csv_list(value: str) -> list[str]:
    # Parse a comma-separated list like "m7g.large,m6g.large,t4g.large"
    return [v.strip() for v in value.split(",") if v.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Coiled cluster with specified parameters.")

    parser.add_argument("-cn", "--cluster_name", type=str, required=True, help="Coiled cluster name")
    parser.add_argument("-n", "--n_workers", type=int, default=1, help="Number of workers for the cluster")
    parser.add_argument("-m", "--worker_memory", type=int, required=True, help="Worker memory tier (GiB): 16|32|64|128")
    parser.add_argument("-t", "--threads_per_worker", type=int, default=1, help="Number of threads per worker (default=1)")

    # New knobs
    parser.add_argument("--workspace", type=str, default=cn.Coiled_workspace, help="Coiled workspace (replaces account)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--spot-policy", type=str, default="spot_with_fallback",
                        choices=["on-demand", "spot", "spot_with_fallback"],
                        help="Worker purchase option (default: spot_with_fallback)")
    parser.add_argument("--cross-zone", action="store_true",
                        help="Allow scheduler/workers across multiple AZs (may improve capacity; may increase network cost)")
    parser.add_argument("--no-best-zone", action="store_true",
                        help="Disable use_best_zone (by default we enable it to improve spot availability)")
    parser.add_argument("--arch", type=str, default="arm", choices=["arm", "x86"],
                        help="CPU architecture for all instance types (default: arm)")
    parser.add_argument("--scheduler-vm-types", type=_csv_list, default=None,
                        help='Override scheduler instance types as CSV (must match --arch), e.g. "m7g.large,m6g.large,t4g.large"')
    parser.add_argument("--software", type=str, default=None,
                        help="Optional Coiled software environment name, e.g. coiled_20251119")
    parser.add_argument(
        "--idle-timeout",
        default=None,
        help=(
            "Optional Coiled idle timeout, e.g. '1 hour'. Defaults to the "
            "repo's memory-tier timeout."
        ),
    )

    args = parser.parse_args()

    create_cluster(
        cluster_name=args.cluster_name,
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        worker_memory=args.worker_memory,
        workspace=args.workspace,
        region=args.region,
        spot_policy=args.spot_policy,
        use_best_zone=(not args.no_best_zone),
        allow_cross_zone=args.cross_zone,
        arch=args.arch,
        scheduler_vm_types=args.scheduler_vm_types,
        software=args.software,
        idle_timeout=args.idle_timeout,
    )


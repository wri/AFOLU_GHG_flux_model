"""Portable local paths used by core runtime logs and statistics."""

from __future__ import annotations

import os
import platform
import posixpath
from typing import Callable, Mapping, Optional


LOCAL_OUTPUT_ROOT_ENV = "AFOLU_LOCAL_OUTPUT_ROOT"


def _as_posix_path(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path).replace("\\", "/")
    if text.endswith("/") and not _is_root_path(text):
        text = text.rstrip("/")
    return text


def _is_root_path(path: str) -> bool:
    return path == "/" or (len(path) == 3 and path[1:] == ":/")


def default_local_output_root(
    *,
    platform_system: Optional[str] = None,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> str:
    system = platform_system or platform.system()
    if system == "Windows":
        return "C:/tmp/afolu"
    if path_exists("/mnt/c/tmp"):
        return "/mnt/c/tmp/afolu"
    return "/tmp/afolu"


def local_output_root(env: Optional[Mapping[str, str]] = None) -> str:
    env_map = os.environ if env is None else env
    root = env_map.get(LOCAL_OUTPUT_ROOT_ENV) or default_local_output_root()
    return _as_posix_path(root)


def local_output_path(*parts: str, root: Optional[str] = None) -> str:
    base = _as_posix_path(root or local_output_root())
    return posixpath.join(base, *parts)


def pipeline_log_dir(pipeline_name: str = "core_model") -> str:
    return local_output_path("logs", "pipelines", pipeline_name)


def chunk_stats_root(
    run_name: Optional[str] = None,
    run_date: Optional[str] = None,
) -> str:
    parts = ["chunk_stats"]
    if run_name:
        parts.append(run_name)
    if run_date:
        parts.append(run_date)
    return local_output_path(*parts)

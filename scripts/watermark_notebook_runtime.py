#!/usr/bin/env python3
"""Fail-fast runtime configuration shared by executable watermark notebooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PLACEHOLDER_PARTS = ("/path/to/", "<path", "your/path", "change/me")


def configured_path(
    variable: str,
    *,
    default: str | Path | None = None,
    description: str | None = None,
) -> Path:
    """Resolve one environment-backed path without accepting tutorial placeholders."""

    raw = os.environ.get(variable)
    if raw is None and default is not None:
        raw = str(default)
    label = description or variable
    if raw is None or not raw.strip():
        raise RuntimeError(
            f"Missing configuration for {label}. Set the {variable} environment variable "
            "to the real local path before running this notebook cell."
        )
    if any(part in raw.lower() for part in PLACEHOLDER_PARTS):
        raise RuntimeError(
            f"{variable} still contains a placeholder: {raw!r}. "
            "Set it to the real local path before running this notebook cell."
        )
    # Keep the final path component unresolved so .venv/bin/python remains the
    # virtual-environment entry point instead of becoming the system Python.
    return Path(os.path.abspath(Path(raw).expanduser()))


def require_path(path: str | Path, description: str, *, executable: bool = False) -> Path:
    """Require a materialized file/directory and provide an actionable error."""

    resolved = Path(os.path.abspath(Path(path).expanduser()))
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    if executable and (not resolved.is_file() or not os.access(resolved, os.X_OK)):
        raise RuntimeError(f"{description} is not executable: {resolved}")
    return resolved


def require_commit(repository: str | Path, expected: str) -> str:
    """Validate a checkout only after confirming it is a real Git repository."""

    repository = require_path(repository, "repository checkout")
    if not (repository / ".git").exists():
        raise RuntimeError(
            f"Configured repository is not a Git checkout: {repository}. "
            "Point the corresponding environment variable at the cloned upstream repository."
        )
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise RuntimeError(f"Could not inspect Git commit for {repository}: {detail}")
    actual = completed.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"Expected commit {expected} in {repository}, found {actual}")
    return actual

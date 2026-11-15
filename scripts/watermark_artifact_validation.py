#!/usr/bin/env python3
"""Lightweight file checks shared by media runners and static validators."""

from __future__ import annotations

from pathlib import Path


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """Return whether a path is an unmaterialized Git LFS pointer."""

    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def require_materialized(path: Path, description: str = "artifact") -> Path:
    """Reject missing files and LFS pointers with an actionable checkout command."""

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty {description}: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the materialized {description}. "
            "From the repository root run `python3 scripts/prepare_mose_notebooks.py --all` "
            "after authenticating Git LFS, then retry validation."
        )
    return path

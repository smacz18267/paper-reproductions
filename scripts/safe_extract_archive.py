#!/usr/bin/env python3
"""Extract a zip or tar archive after rejecting unsafe member paths."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


def checked_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Archive member escapes destination: {member}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    destination = args.destination.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        print(f"Destination already populated; keeping it: {destination}")
        return

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                checked_target(destination, info.filename)
            source.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            members = source.getmembers()
            for member in members:
                checked_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Archive links are not accepted: {member.name}")
            source.extractall(destination, members=members)
    else:
        raise RuntimeError(f"Unsupported or corrupt archive: {archive}")

    print(f"Extracted {archive.name} to {destination}")


if __name__ == "__main__":
    main()

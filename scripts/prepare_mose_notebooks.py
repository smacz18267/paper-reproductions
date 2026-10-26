#!/usr/bin/env python3
"""Download and verify the Git LFS media used by the MOSE notebooks."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MEDIA_ROOT = REPO_ROOT / "evidence/mose-task1/review50/media"
DEFAULT_SAMPLES = ("02b72b8d", "0ab75563", "0b02886a")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def run_git(*arguments: str) -> None:
    subprocess.run(["git", "-C", str(REPO_ROOT), *arguments], check=True)


def tracked_videos(relative_roots: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", *relative_roots],
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(
        REPO_ROOT / raw.decode()
        for raw in result.stdout.split(b"\0")
        if raw and raw.decode().endswith(".mp4")
    )


def verify(videos: list[Path]) -> int:
    if not videos:
        raise RuntimeError("No tracked MP4 files were found for the requested samples.")

    total_bytes = 0
    failures: list[str] = []
    for path in videos:
        if not path.is_file():
            failures.append(f"missing: {path.relative_to(REPO_ROOT)}")
            continue
        header = path.read_bytes()[:128]
        if header.startswith(LFS_POINTER_PREFIX):
            failures.append(f"Git LFS pointer, not video: {path.relative_to(REPO_ROOT)}")
            continue
        if b"ftyp" not in header:
            failures.append(f"not an MP4 container: {path.relative_to(REPO_ROOT)}")
            continue
        total_bytes += path.stat().st_size

    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures[:20])
        remainder = len(failures) - 20
        if remainder > 0:
            details += f"\n  - ...and {remainder} more"
        raise RuntimeError(
            f"{len(failures)} video files are not ready:\n{details}\n"
            "Install Git LFS, then run this script again without --check-only."
        )
    return total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="download and verify all 50-sample review videos instead of the 48 notebook videos",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify local files without contacting the Git LFS remote",
    )
    args = parser.parse_args()

    if args.all:
        relative_roots = ["evidence/mose-task1/review50/media"]
        include = "evidence/mose-task1/review50/media/**"
        label = "complete 50-sample review"
    else:
        relative_roots = [
            f"evidence/mose-task1/review50/media/{sample_id}"
            for sample_id in DEFAULT_SAMPLES
        ]
        include = ",".join(f"{root}/**" for root in relative_roots)
        label = "three notebooks"

    videos = tracked_videos(relative_roots)
    if not args.check_only:
        try:
            run_git("lfs", "version")
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise SystemExit(
                "Git LFS is required. Install it (for example, `brew install git-lfs` "
                "on macOS or `sudo apt-get install git-lfs` on Ubuntu), then rerun this command."
            ) from error
        run_git("lfs", "install", "--local")
        run_git("lfs", "pull", f"--include={include}", "--exclude=")

    total_bytes = verify(videos)
    print(
        f"Ready: {len(videos)} playable MP4 containers for the {label} "
        f"({total_bytes / (1024 * 1024):.1f} MiB)."
    )
    if not args.all:
        print("Next: start Jupyter from the repository root and open notebooks/mose-watermark-samples/.")


if __name__ == "__main__":
    main()

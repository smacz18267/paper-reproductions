#!/usr/bin/env python3
"""Check dependencies, allowing only Decord's verified Linux wheel metadata bug."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

KNOWN_ERROR = "decord 0.6.0 is not supported on this platform"
KNOWN_TAG = "Tag: cp36-cp36m-manylinux2010_x86_64"
ISSUE = "https://github.com/dmlc/decord/issues/356"


def check_environment(source):
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    report = {"pip_check_exit_code": result.returncode, "pip_check_output": result.stdout,
              "known_decord_metadata_issue_accepted": False, "status": "failed"}
    try:
        if result.returncode == 0:
            report["status"] = "passed"
            return
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 1 or lines != [KNOWN_ERROR]:
            raise RuntimeError("Dependency checks failed; this is not the isolated Decord metadata issue")
        distribution = importlib.metadata.distribution("decord")
        wheel = distribution.read_text("WHEEL") or ""
        if (distribution.version != "0.6.0" or sys.platform != "linux"
                or platform.machine().lower() not in {"x86_64", "amd64"}
                or KNOWN_TAG not in wheel.splitlines()):
            raise RuntimeError("Decord metadata does not match the known Linux x86_64 packaging bug")
        report["decord_wheel_metadata"] = wheel
        report["upstream_issue"] = ISSUE

        # Check the native decoder instead of trusting the incorrect cp36 tag.
        # A missing shared library or a genuinely incompatible binary still fails.
        from decord import VideoReader, cpu
        checks = []
        for folder in ("FG_BG", "MASK"):
            path = source / "demo" / folder / "WILD_ENV006_00042.mp4"
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            if len(reader) < 81:
                raise RuntimeError(f"Decord found only {len(reader)} frames in {path}")
            shapes = []
            for index in (0, 40, 80):
                frame = reader[index].asnumpy()
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
                    raise RuntimeError(f"Decord returned an invalid frame {index} from {path}")
                shapes.append(list(frame.shape))
            checks.append({"video": str(path), "frames": len(reader),
                           "sample_indices": [0, 40, 80], "sample_shapes": shapes})
            del reader
        report["decord_decode_checks"] = checks
        report["known_decord_metadata_issue_accepted"] = True
        report["status"] = "passed_with_verified_metadata_exception"
        print("Decord decoded both supplied videos successfully. "
              "Accepting only its known wheel metadata error; all other dependency checks passed.")
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (source / "setup_dependency_checks.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    check_environment(args.source.resolve())

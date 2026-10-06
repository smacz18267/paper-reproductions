#!/usr/bin/env python3
"""Run the unchanged upstream entry point with a seed and preserve its prior video."""
import argparse
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args, upstream_args = parser.parse_known_args()
    source = args.source.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import diffueraser.diffueraser as implementation
    read_priori = implementation.read_priori

    def preserve_priori(path, *positional, **keywords):
        shutil.copyfile(path, Path(path).with_name("propainter_prior.mp4"))
        return read_priori(path, *positional, **keywords)

    implementation.read_priori = preserve_priori
    sys.argv = [str(source / "run_diffueraser.py"), *upstream_args]
    runpy.run_path(str(source / "run_diffueraser.py"), run_name="__main__")


if __name__ == "__main__":
    main()

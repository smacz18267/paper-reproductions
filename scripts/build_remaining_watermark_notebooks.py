#!/usr/bin/env python3
"""Build execution notebooks for the eight remaining watermark methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from remaining_watermark_methods import METHODS


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = code(
    '''from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
from IPython.display import JSON, Video, display

def find_repo_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "scripts/remaining_watermark_methods.py").is_file():
            return candidate
    raise FileNotFoundError("Start Jupyter inside the paper-reproductions-task1 checkout")

ROOT = find_repo_root()
SCRIPTS = ROOT / "scripts"

def run(command, *, cwd=None, env=None):
    command = [str(value) for value in command]
    print("+", shlex.join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)

def require(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path

def require_commit(repository, expected):
    actual = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Expected {expected} in {repository}, found {actual}")

def show_results(run_root):
    run_root = Path(run_root)
    for name in ("watermarked.mp4", "aligned-mask.mp4"):
        path = run_root / name
        if path.is_file():
            print(name)
            display(Video(str(path), embed=True, html_attributes="controls muted loop"))
    attack = run_root / "attack"
    for name in ("input-watermarked.mp4", "gt-mask.mp4", "propainter.mp4", "diffueraser.mp4"):
        path = attack / name
        if path.is_file():
            print(name)
            display(Video(str(path), embed=True, html_attributes="controls muted loop"))
    for path in sorted(run_root.rglob("summary.json")):
        print(path.relative_to(run_root))
        display(JSON(json.loads(path.read_text())))

print("Repository:", ROOT)'''
)


def intro(slug: str, extra: str) -> list[dict[str, Any]]:
    spec = next(method for method in METHODS if method.slug == slug)
    compatibility = (
        "This method can be applied to existing media, so the notebook uses the fixed MOSE sample."
        if spec.existing_video_compatible
        else "This method is coupled to generation, so the notebook uses only author-provided or author-generated native data. It does not fake a MOSE embedding."
    )
    return [
        markdown(
            f"""
# {spec.label}: ProPainter and DiffuEraser

Repository: [{spec.official_repository}]({spec.official_repository})

Paper: [{spec.paper_url}]({spec.paper_url})

Inspected commit: `{spec.verified_commit}`

Watermark stage: `{spec.watermark_stage}`
Domain: {spec.domain}

{compatibility}

The two removers receive the exact same watermarked frames and binary mask. One official DiffuEraser execution is used: its internal ProPainter prior is preserved as the ProPainter result, and its final refinement is the DiffuEraser result. Inputs shorter than 22 frames are padded by repeating only the last frame/mask to 24 for inference, then both reported outputs are trimmed to the authentic length.

{extra}
"""
        ),
        markdown(
            f"""
## Authenticity contract

- Watermark preparation: {spec.preparation_summary}
- Checkpoint status: {spec.checkpoint_status}
- Protocol note: {spec.protocol_note}
- No result cell fabricates metrics. Native extraction must finish and write its JSON before values are displayed.
- A method without a released spatial detector reports bit recovery only; it is not assigned a proxy localizer.
"""
        ),
        SETUP,
    ]


def common_attack_cells(
    *, method: str, sample_id: str, sample_source: str, source_kind: str,
    video_var: str = "WATERMARKED_VIDEO", mask_var: str = "ALIGNED_MASK",
) -> list[dict[str, Any]]:
    return [
        markdown(
            """
## Run the identical two-method attack

The preparation cell freezes hashes, frame count, dimensions, FPS, mask foreground counts, provenance, and any short-video padding. The next cell runs both methods from that frozen input.
"""
        ),
        code(
            f'''ATTACK_RUN = RUN_ROOT / "attack"
if not (ATTACK_RUN / "sample.json").is_file():
    run([
        DIFFUERASER_PYTHON, SCRIPTS / "prepare_watermark_removal_sample.py", "prepare",
        "--source-video", {video_var}, "--source-mask", {mask_var},
        "--output", ATTACK_RUN, "--watermark-method", "{method}",
        "--sample-id", "{sample_id}", "--sample-source", "{sample_source}",
        "--source-kind", "{source_kind}",
    ])
else:
    print("Prepared sample already exists:", ATTACK_RUN / "sample.json")'''
        ),
        code(
            '''if not (ATTACK_RUN / "removal-pair.json").is_file():
    run([
        DIFFUERASER_PYTHON, SCRIPTS / "run_watermark_removal_pair.py",
        "--run", ATTACK_RUN, "--diffueraser", DIFFUERASER,
        "--diffueraser-python", DIFFUERASER_PYTHON,
        "--device", GPU, "--seed", "2025",
    ])
else:
    print("Both removal outputs already exist:", ATTACK_RUN / "removal-pair.json")'''
        ),
    ]


def results_cell() -> dict[str, Any]:
    return code("show_results(RUN_ROOT)")


def rivagan() -> dict[str, Any]:
    cells = intro(
        "rivagan",
        "**Current blocker:** DAI-Lab publishes training code but no pretrained `model.pt`. The notebook is fully wired, but it deliberately stops before embedding unless you provide a trained checkpoint and its provenance. The RivaGAN sample committed by the VideoMark authors is not treated as natively decodable because its matching RivaGAN checkpoint/message are absent.",
    )
    cells += [
        code(
            '''RIVAGAN = Path("/path/to/RivaGAN")
RIVAGAN_PYTHON = RIVAGAN / ".venv/bin/python"
RIVAGAN_MODEL = Path("/path/to/rivagan/model.pt")  # required; not published upstream
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
MOSE_VIDEO = ROOT / "evidence/mose-task1/review50/media/02b72b8d/original.mp4"
MOSE_MASK = ROOT / "evidence/mose-task1/review50/media/02b72b8d/gt_mask.mp4"
RUN_ROOT = ROOT / "runs/remaining-watermarks/rivagan/02b72b8d"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = RUN_ROOT / "watermarked.mp4"
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"'''
        ),
        code(
            '''require_commit(RIVAGAN, "efffa72a4ca46d4d5051f6970c96424c2cdab441")
require(RIVAGAN_PYTHON, "RivaGAN Python")
require(RIVAGAN_MODEL, "trained RivaGAN model.pt")
require(MOSE_VIDEO, "MOSE video")
require(MOSE_MASK, "MOSE mask")'''
        ),
        code(
            '''run([
    RIVAGAN_PYTHON, SCRIPTS / "rivagan_video.py", "embed",
    "--source", RIVAGAN, "--checkpoint", RIVAGAN_MODEL,
    "--input-video", MOSE_VIDEO, "--input-mask", MOSE_MASK,
    "--output-video", WATERMARKED_VIDEO, "--output-mask", ALIGNED_MASK,
])'''
        ),
    ]
    cells += common_attack_cells(
        method="RivaGAN", sample_id="02b72b8d", sample_source="MOSE fixed sample",
        source_kind="mose_post_hoc",
    )
    cells += [
        code(
            '''run([
    RIVAGAN_PYTHON, SCRIPTS / "rivagan_video.py", "detect",
    "--source", RIVAGAN, "--checkpoint", RIVAGAN_MODEL,
    "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def wam() -> dict[str, Any]:
    cells = intro(
        "watermark-anything",
        "The official `wam_mit.pth` model embeds one fixed 32-bit payload into every MOSE frame. WAM's native watermark-presence map is inverted at threshold 0.5 to score the removed region; that definition is recorded in the detector JSON.",
    )
    cells += [
        code(
            '''WAM = Path("/path/to/watermark-anything")
WAM_PYTHON = WAM / ".venv/bin/python"
WAM_CHECKPOINT = WAM / "checkpoints/wam_mit.pth"
WAM_PARAMS = WAM / "checkpoints/params.json"
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
MOSE_VIDEO = ROOT / "evidence/mose-task1/review50/media/02b72b8d/original.mp4"
MOSE_MASK = ROOT / "evidence/mose-task1/review50/media/02b72b8d/gt_mask.mp4"
RUN_ROOT = ROOT / "runs/remaining-watermarks/watermark-anything/02b72b8d"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = RUN_ROOT / "watermarked.mp4"
ALIGNED_MASK = MOSE_MASK'''
        ),
        markdown("Download the official checkpoint with `wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/` from the WAM checkout."),
        code(
            '''require_commit(WAM, "2c08af04d037d5667c02f6ddebbda9ff04581c3e")
for path, label in [(WAM_PYTHON, "WAM Python"), (WAM_CHECKPOINT, "wam_mit.pth"), (WAM_PARAMS, "params.json")]:
    require(path, label)'''
        ),
        code(
            '''run([
    WAM_PYTHON, SCRIPTS / "watermark_anything_video.py", "embed",
    "--source", WAM, "--checkpoint", WAM_CHECKPOINT, "--params", WAM_PARAMS,
    "--input-video", MOSE_VIDEO, "--output-video", WATERMARKED_VIDEO,
])'''
        ),
    ]
    cells += common_attack_cells(
        method="Watermark Anything", sample_id="02b72b8d", sample_source="MOSE fixed sample",
        source_kind="mose_post_hoc",
    )
    cells += [
        code(
            '''run([
    WAM_PYTHON, SCRIPTS / "watermark_anything_video.py", "detect",
    "--source", WAM, "--checkpoint", WAM_CHECKPOINT, "--params", WAM_PARAMS,
    "--gt-mask", ATTACK_RUN / "gt-mask.mp4", "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def vine() -> dict[str, Any]:
    cells = intro(
        "vine",
        "VINE's official list encoder center-crops non-square images. The adapter applies that exact crop to every MOSE frame and mask, embeds one fixed 100-bit text payload, and records the geometry. The released decoder has no pixel localizer, so only bit recovery is reported.",
    )
    cells += [
        code(
            '''VINE = Path("/path/to/VINE")
VINE_PYTHON = VINE / ".venv/bin/python"
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
MOSE_VIDEO = ROOT / "evidence/mose-task1/review50/media/02b72b8d/original.mp4"
MOSE_MASK = ROOT / "evidence/mose-task1/review50/media/02b72b8d/gt_mask.mp4"
RUN_ROOT = ROOT / "runs/remaining-watermarks/vine/02b72b8d"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = RUN_ROOT / "watermarked.mp4"
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"'''
        ),
        code(
            '''require_commit(VINE, "d72f11a93f8de9bc45ef61202b9c8dda2a70cc77")
require(VINE_PYTHON, "VINE Python")'''
        ),
        code(
            '''run([
    VINE_PYTHON, SCRIPTS / "vine_video.py", "embed", "--source", VINE,
    "--input-video", MOSE_VIDEO, "--input-mask", MOSE_MASK,
    "--output-video", WATERMARKED_VIDEO, "--output-mask", ALIGNED_MASK,
    "--encoder-model", "Shilin-LU/VINE-R-Enc", "--message", "Hello World!",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="VINE", sample_id="02b72b8d", sample_source="MOSE fixed sample after upstream center crop",
        source_kind="mose_post_hoc",
    )
    cells += [
        code(
            '''run([
    VINE_PYTHON, SCRIPTS / "vine_video.py", "detect", "--source", VINE,
    "--decoder-model", "Shilin-LU/VINE-R-Dec", "--message", "Hello World!",
    "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def videomark() -> dict[str, Any]:
    cells = intro(
        "videomark",
        "This notebook uses the authors' committed `a_red_panda_eating_leaves` ModelScope watermarked MP4, committed PRC key, and committed shift value. Because VideoMark publishes no object annotation for that sample, the attack is a recorded 35% centered region mask—not an author mask.",
    )
    cells += [
        code(
            '''VIDEOMARK = Path("/path/to/VideoMark")
VIDEOMARK_PYTHON = VIDEOMARK / ".venv/bin/python"
MODELSCOPE_MODEL = Path("/path/to/damo-vilab/text-to-video-ms-1.7b")
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
SAMPLE = VIDEOMARK / "result/videomark/modelscope/512bit/a_red_panda_eating_leaves/wm.mp4"
SHIFT = VIDEOMARK / "result/videomark/modelscope/512bit/a_red_panda_eating_leaves/shift_value.npy"
KEYS = VIDEOMARK / "keys/64_64_512bit.pkl"
RUN_ROOT = ROOT / "runs/remaining-watermarks/videomark/a_red_panda_eating_leaves"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = SAMPLE
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"'''
        ),
        code(
            '''require_commit(VIDEOMARK, "9f8d78b73ab9f9f055651b1b4f37d68bdb05e7be")
for path, label in [(SAMPLE, "authors' watermarked sample"), (SHIFT, "shift value"), (KEYS, "PRC key"), (MODELSCOPE_MODEL, "ModelScope inversion model")]:
    require(path, label)'''
        ),
        code(
            '''run([
    DIFFUERASER_PYTHON, SCRIPTS / "make_native_removal_mask.py", "center-box",
    "--video", SAMPLE, "--output-video", ALIGNED_MASK,
    "--output-frames", RUN_ROOT / "aligned-mask-frames",
    "--width-fraction", "0.35", "--height-fraction", "0.35",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="VideoMark", sample_id="a_red_panda_eating_leaves",
        sample_source="VideoMark authors' committed ModelScope 512-bit sample",
        source_kind="authors_provided",
    )
    cells += [
        code(
            '''run([
    VIDEOMARK_PYTHON, SCRIPTS / "run_videomark_detection.py",
    "--source", VIDEOMARK, "--model-path", MODELSCOPE_MODEL,
    "--keys", KEYS, "--shift", SHIFT, "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def sigmark() -> dict[str, Any]:
    cells = intro(
        "sigmark",
        "The repository contains the authors' VBench2 prompt text but no generated watermarked MP4. This notebook generates a small native HunyuanVideo-T2V sample with SIGMark, preserving the maintained state and ground-truth message. The paper/repository state that the unquantized path needs roughly 50 GB GPU memory. A deterministic center-region mask is recorded because no object mask is released.",
    )
    cells += [
        code(
            '''SIGMARK = Path("/path/to/SIGMark-release")
SIGMARK_PYTHON = SIGMARK / ".venv/bin/python"
TORCHRUN = SIGMARK / ".venv/bin/torchrun"
MODEL_BASE = Path("/path/to/pretrained_models")
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
RUN_ROOT = ROOT / "runs/remaining-watermarks/sigmark/native-vbench2"
NATIVE = RUN_ROOT / "native-generation"
RUN_ROOT.mkdir(parents=True, exist_ok=True)'''
        ),
        code(
            '''require_commit(SIGMARK, "3713243f2e002cb21bbc3da896094d39b2b528c9")
require(TORCHRUN, "SIGMark torchrun")
require(MODEL_BASE / "HunyuanVideo-community", "HunyuanVideo-community model")'''
        ),
        code(
            '''if not list(NATIVE.rglob("*.mp4")):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = GPU
    run([
        TORCHRUN, "--nproc_per_node=1", SIGMARK / "main.py", "--mode=gen",
        "--model_base_path", MODEL_BASE, "--model_name", "HunyuanVideo-community",
        "--prompt_set", "VBench2_aug", "--watermark_method", "sigmark",
        "--ch_factor", "2", "--hw_factor", "8", "--fr_factor", "1",
        "--batch_size", "1", "--num_prompts_per_dimension", "1",
        "--num_videos_per_prompt", "1", "--num_prompts_diversity", "1",
        "--num_videos_per_prompt_diversity", "1", "--small_scale_test", "1",
        "--output_path", NATIVE,
    ], cwd=SIGMARK, env=env)
WATERMARKED_VIDEO = sorted(NATIVE.rglob("*.mp4"))[0]
SAMPLE_KEY = str(WATERMARKED_VIDEO.relative_to(NATIVE).with_suffix(""))
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"
print("Native sample:", WATERMARKED_VIDEO, "key:", SAMPLE_KEY)'''
        ),
        code(
            '''run([
    DIFFUERASER_PYTHON, SCRIPTS / "make_native_removal_mask.py", "center-box",
    "--video", WATERMARKED_VIDEO, "--output-video", ALIGNED_MASK,
    "--output-frames", RUN_ROOT / "aligned-mask-frames",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="SIGMark", sample_id="first VBench2_aug small-scale sample",
        sample_source="SIGMark authors' VBench2 prompt set and native HunyuanVideo pipeline",
        source_kind="authors_generated",
    )
    cells += [
        code(
            '''DETECT_CASES = {
    "original": ATTACK_RUN / "input-watermarked.mp4",
    "propainter": ATTACK_RUN / "propainter.mp4",
    "diffueraser": ATTACK_RUN / "diffueraser.mp4",
}
for name, video in DETECT_CASES.items():
    staged = RUN_ROOT / "detection" / name
    if not (staged / "staged-case.json").is_file():
        run([
            DIFFUERASER_PYTHON, SCRIPTS / "stage_sigmark_detection_case.py",
            "--generated-output", NATIVE, "--sample-key", SAMPLE_KEY,
            "--replacement-video", video, "--destination", staged,
        ])
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = GPU
    run([
        TORCHRUN, "--nproc_per_node=1", SIGMARK / "main.py", "--mode=extract",
        "--model_base_path", MODEL_BASE, "--model_name", "HunyuanVideo-community",
        "--prompt_set", "VBench2_aug", "--watermark_method", "sigmark",
        "--ch_factor", "2", "--hw_factor", "8", "--fr_factor", "1",
        "--batch_size", "1", "--num_prompts_per_dimension", "1",
        "--num_videos_per_prompt", "1", "--num_prompts_diversity", "1",
        "--num_videos_per_prompt_diversity", "1", "--small_scale_test", "1",
        "--output_path", staged,
    ], cwd=SIGMARK, env=env)'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def vidsig() -> dict[str, Any]:
    cells = intro(
        "vidsig",
        "The native sample is generated from the first prompt in the authors' `prompt.txt`, their ModelScope fine-tuned VAE, released 48-bit TorchScript decoder, published key, and seed 42. No object masks are released, so the notebook records a deterministic center-region mask.",
    )
    cells += [
        code(
            '''VIDSIG = Path("/path/to/Video-Signature")
VIDSIG_PYTHON = VIDSIG / ".venv/bin/python"
VAE_CHECKPOINT = VIDSIG / "ckpts/vae/modelscope/checkpoint.pth"
MESSAGE_DECODER = VIDSIG / "ckpts/msg_decoder/dec_48b_whit.torchscript.pt"
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
RUN_ROOT = ROOT / "runs/remaining-watermarks/vidsig/red-panda-seed42"
NATIVE = RUN_ROOT / "native-generation"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = NATIVE / "watermarked.mp4"
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"'''
        ),
        code(
            '''require_commit(VIDSIG, "9d219c47670c30fdaaccd17b24e7783c976a8c66")
require(VAE_CHECKPOINT, "authors' fine-tuned ModelScope VAE")
require(MESSAGE_DECODER, "authors' 48-bit TorchScript decoder")'''
        ),
        code(
            '''if not WATERMARKED_VIDEO.is_file():
    run([
        VIDSIG_PYTHON, SCRIPTS / "run_vidsig_native_sample.py", "--source", VIDSIG,
        "--vae-checkpoint", VAE_CHECKPOINT, "--message-decoder", MESSAGE_DECODER,
        "--output", NATIVE, "--prompt-file", VIDSIG / "prompt.txt",
        "--prompt-index", "0", "--seed", "42",
    ])
run([
    DIFFUERASER_PYTHON, SCRIPTS / "make_native_removal_mask.py", "center-box",
    "--video", WATERMARKED_VIDEO, "--output-video", ALIGNED_MASK,
    "--output-frames", RUN_ROOT / "aligned-mask-frames",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="VidSig", sample_id="red-panda-prompt-0-seed-42",
        sample_source="Video Signature authors' prompt, key, checkpoints, and ModelScope pipeline",
        source_kind="authors_generated",
    )
    cells += [
        code(
            '''run([
    VIDSIG_PYTHON, SCRIPTS / "run_vidsig_detection.py",
    "--source", VIDSIG, "--decoder", MESSAGE_DECODER,
    "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def lvmark() -> dict[str, Any]:
    cells = intro(
        "lvmark",
        "**Release blocker:** the repository's inference shell script contains placeholders and refers to `prompts/512_ours` plus a prompt/key JSON that are not checked in. The official 2.05 GB weight archive's central index contains four nested weight archives and no prompt/key data. Those two inputs must therefore be obtained separately from the authors. Once supplied, the notebook runs their unchanged DynamiCrafter entry point, attacks its native output, and applies the released LVMark decoder/BCH record.",
    )
    cells += [
        code(
            '''LVMARK = Path("/path/to/TIFS26_lvmark")
LVMARK_PYTHON = LVMARK / ".venv/bin/python"
WEIGHTS = Path("/path/to/lvmark_weights")
DYNAMICRAFTER_CKPT = Path("/path/to/dynamicrafter_512_v1/model.ckpt")
PROMPT_DIR = Path("/path/to/authors/prompts/512_ours")
PROMPT_JSON = Path("/path/to/authors/prompts_with_keys120_32bit.json")
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
RUN_ROOT = ROOT / "runs/remaining-watermarks/lvmark/native-32bit"
NATIVE = RUN_ROOT / "native-generation"
RUN_ROOT.mkdir(parents=True, exist_ok=True)'''
        ),
        code(
            '''require_commit(LVMARK, "da7ad67ba2d86ae9918dce2b42d3736e335561be")
for path, label in [
    (WEIGHTS / "vae_decoder.pth", "LVMark VAE decoder"),
    (WEIGHTS / "mapping_network.pth", "LVMark mapping network"),
    (WEIGHTS / "decoding_network.pth", "LVMark message decoder"),
    (DYNAMICRAFTER_CKPT, "DynamiCrafter checkpoint"),
    (PROMPT_DIR, "authors' prompt image directory"),
    (PROMPT_JSON, "authors' prompt/key JSON"),
]: require(path, label)'''
        ),
        code(
            '''if not list(NATIVE.rglob("*.avi")):
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = GPU
    run([
        LVMARK_PYTHON, LVMARK / "DynamiCrafter/scripts/evaluation/inference_watermark.py",
        "--seed", "42", "--ckpt_path", DYNAMICRAFTER_CKPT,
        "--config", LVMARK / "DynamiCrafter/configs/inference_512_v1.0_wm.yaml",
        "--savedir", NATIVE, "--n_samples", "1", "--bs", "1",
        "--height", "320", "--width", "512", "--unconditional_guidance_scale", "7.5",
        "--ddim_steps", "50", "--ddim_eta", "1.0", "--prompt_dir", PROMPT_DIR,
        "--text_input", "--video_length", "16", "--frame_stride", "24",
        "--timestep_spacing", "uniform_trailing", "--guidance_rescale", "0.7",
        "--int_dimension", "128", "--lr_mult", "1", "--phi_dimension", "32",
        "--vae_decoder_path", WEIGHTS / "vae_decoder.pth",
        "--mapping_path", WEIGHTS / "mapping_network.pth",
        "--msg_decoder_path", WEIGHTS / "decoding_network.pth",
        "--json_file", PROMPT_JSON, "--affine_list", LVMARK / "layer_noise_0.5.txt",
        "--bit_capacity", "32", "--bch_bits", "14",
    ], cwd=LVMARK / "DynamiCrafter", env=env)
WATERMARKED_VIDEO = sorted(NATIVE.rglob("*.avi"))[0]
BCH = sorted(NATIVE.rglob("bch.pkl"))[0]
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"
print(WATERMARKED_VIDEO, BCH)'''
        ),
        code(
            '''run([
    DIFFUERASER_PYTHON, SCRIPTS / "make_native_removal_mask.py", "center-box",
    "--video", WATERMARKED_VIDEO, "--output-video", ALIGNED_MASK,
    "--output-frames", RUN_ROOT / "aligned-mask-frames",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="LVMark", sample_id="first-authors-prompt-32bit",
        sample_source="LVMark authors' prompt/key data and native DynamiCrafter pipeline",
        source_kind="authors_generated",
    )
    cells += [
        code(
            '''run([
    LVMARK_PYTHON, SCRIPTS / "run_lvmark_detection.py",
    "--source", LVMARK, "--decoder", WEIGHTS / "decoding_network.pth", "--bch", BCH,
    "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


def genptw() -> dict[str, Any]:
    cells = intro(
        "genptw",
        "The paper/released training code supports post-generation embedding by VAE-encoding existing COCO images and decoding the same latent with `decode_wm`. This notebook applies that path framewise to MOSE. It records the 512 x 512 center-crop/resize adapter and uses GenPTW's native 64-bit decoder and localizer after both removals.",
    )
    cells += [
        code(
            '''GENPTW = Path("/path/to/GenPTW")
GENPTW_PYTHON = GENPTW / ".venv/bin/python"
CHECKPOINT = GENPTW / "Checkpoint"
PRETRAINED_VAE = Path("/path/to/stable-diffusion-2-base/vae")
CONVNEXT = Path("/path/to/convnext_tiny-983f1562.pth")
DIFFUERASER = Path("/path/to/DiffuEraser")
DIFFUERASER_PYTHON = DIFFUERASER / ".venv/bin/python"
GPU = "0"
MOSE_VIDEO = ROOT / "evidence/mose-task1/review50/media/02b72b8d/original.mp4"
MOSE_MASK = ROOT / "evidence/mose-task1/review50/media/02b72b8d/gt_mask.mp4"
RUN_ROOT = ROOT / "runs/remaining-watermarks/genptw/02b72b8d"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
WATERMARKED_VIDEO = RUN_ROOT / "watermarked.mp4"
ALIGNED_MASK = RUN_ROOT / "aligned-mask.mp4"'''
        ),
        code(
            '''require_commit(GENPTW, "8c22fc04dcf9d896eaeabd034d09918588e82cb3")
for path, label in [
    (CHECKPOINT / "msg_decoder.pth", "GenPTW decoder"),
    (CHECKPOINT / "localizer.pth", "GenPTW localizer"),
    (CHECKPOINT / "diffusion_pytorch_model.safetensors", "GenPTW watermarked VAE"),
    (PRETRAINED_VAE, "Stable Diffusion 2 VAE"), (CONVNEXT, "ConvNeXt checkpoint"),
]: require(path, label)'''
        ),
        code(
            '''run([
    GENPTW_PYTHON, SCRIPTS / "genptw_video.py", "embed",
    "--source", GENPTW, "--checkpoint", CHECKPOINT,
    "--pretrained-vae", PRETRAINED_VAE, "--convnext-checkpoint", CONVNEXT,
    "--input-video", MOSE_VIDEO, "--input-mask", MOSE_MASK,
    "--output-video", WATERMARKED_VIDEO, "--output-mask", ALIGNED_MASK,
    "--seed", "2026",
])'''
        ),
    ]
    cells += common_attack_cells(
        method="GenPTW", sample_id="02b72b8d", sample_source="MOSE fixed sample, GenPTW 512x512 image adapter",
        source_kind="mose_post_hoc",
    )
    cells += [
        code(
            '''run([
    GENPTW_PYTHON, SCRIPTS / "genptw_video.py", "detect",
    "--source", GENPTW, "--checkpoint", CHECKPOINT,
    "--pretrained-vae", PRETRAINED_VAE, "--convnext-checkpoint", CONVNEXT,
    "--gt-mask", ATTACK_RUN / "gt-mask.mp4", "--output", RUN_ROOT / "detection",
    "--case", f"original={ATTACK_RUN / 'input-watermarked.mp4'}",
    "--case", f"propainter={ATTACK_RUN / 'propainter.mp4'}",
    "--case", f"diffueraser={ATTACK_RUN / 'diffueraser.mp4'}",
])'''
        ),
        results_cell(),
    ]
    return notebook(cells)


BUILDERS = {
    "watermark_anything": wam,
    "vine": vine,
    "videomark": videomark,
    "sigmark": sigmark,
    "vidsig": vidsig,
    "genptw": genptw,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("notebooks/mose-watermark-samples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        path = args.output / f"{name}.ipynb"
        path.write_text(json.dumps(builder(), indent=1) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download only DiffuEraser's inference components; keep model revisions on retries."""
import argparse
import fnmatch
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL_SPECS = {
    "diffueraser": ("lixiaowen/diffuEraser", "weights/diffuEraser", [
        "brushnet/config.json", "brushnet/diffusion_pytorch_model.safetensors",
        "unet_main/config.json", "unet_main/diffusion_pytorch_model.safetensors"]),
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "weights/stable-diffusion-v1-5", [
        "model_index.json", "feature_extractor/*.json", "scheduler/*.json", "tokenizer/*",
        "text_encoder/config.json", "safety_checker/config.json"]),
    "vae": ("stabilityai/sd-vae-ft-mse", "weights/sd-vae-ft-mse", [
        "config.json", "diffusion_pytorch_model.safetensors"]),
    "pcm": ("wangfuyun/PCM_Weights", "weights/PCM_Weights", [
        "sd15/pcm_sd15_smallcfg_2step_converted.safetensors"]),
}
PROPAINTER_FILES = ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")
PROPAINTER_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def selected_files(key, available, patterns):
    selected = {name for name in available if any(fnmatch.fnmatchcase(name, p) for p in patterns)}
    if key == "sd15":
        for component in ("text_encoder", "safety_checker"):
            alternatives = (f"{component}/model.safetensors", f"{component}/pytorch_model.bin")
            name = next((name for name in alternatives if name in available), None)
            if name is None:
                raise RuntimeError(f"Missing {component} weights in SD 1.5")
            selected.add(name)
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in selected):
            raise RuntimeError(f"Missing required model component: {key}/{pattern}")
    return sorted(selected)


def download_models(source):
    api = HfApi()
    manifest_path = source / "model_downloads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, (repo_id, destination, patterns) in MODEL_SPECS.items():
        entry = manifest.get(key)
        if entry is None:
            revision = api.model_info(repo_id).sha
            available = api.list_repo_files(repo_id, revision=revision)
            entry = {"repo_id": repo_id, "revision": revision, "destination": destination,
                     "files": selected_files(key, available, patterns)}
            manifest[key] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if entry["repo_id"] != repo_id or entry["destination"] != destination:
            raise RuntimeError(f"Unexpected existing download manifest entry: {key}")
        print(f"Downloading {key}: {repo_id}@{entry['revision']}", flush=True)
        snapshot_download(repo_id, revision=entry["revision"], local_dir=source / destination,
                          allow_patterns=entry["files"])
        for name in entry["files"]:
            path = source / destination / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty downloaded file: {path}")

    target = source / "weights/propainter"
    target.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("propainter", {})
    existing = source.parent / "ProPainter/weights"
    for name in PROPAINTER_FILES:
        path = target / name
        if not path.is_file() or path.stat().st_size == 0:
            temp = target / (name + ".part")
            if (existing / name).is_file() and (existing / name).stat().st_size > 0:
                print(f"Reusing {existing / name}", flush=True)
                shutil.copyfile(existing / name, temp)
                manifest["propainter"][name] = {"copied_from": str(existing / name),
                                                "upstream_url": PROPAINTER_URL + name}
            else:
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(PROPAINTER_URL + name, timeout=120) as response, temp.open("wb") as output:
                    shutil.copyfileobj(response, output)
                manifest["propainter"][name] = {"upstream_url": PROPAINTER_URL + name}
            if temp.stat().st_size == 0:
                raise RuntimeError(f"Empty ProPainter checkpoint: {temp}")
            temp.replace(path)
        manifest["propainter"].setdefault(name, {"upstream_url": PROPAINTER_URL + name,
                                                 "existing_local_file": True})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    download_models(args.source.resolve())

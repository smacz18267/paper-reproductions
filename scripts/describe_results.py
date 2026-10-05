#!/usr/bin/env python3
"""Describe actual videos and hash run artifacts; never invent scores or timings."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def video_info(path):
    if shutil.which('ffprobe'):
        out = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
            '-of', 'json', str(path),
        ], text=True)
        return json.loads(out)['streams'][0]
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot read video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': fps, 'nb_frames': count,
        'duration': count / fps if fps else None,
    }
    cap.release()
    return info


def describe(root, source=None):
    result = {'artifacts': [], 'inputs': [], 'weights': [], 'benchmark_scores': None}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name in ('artifact_manifest.json', 'SUCCESS.txt'):
            continue
        item = {'path': path.relative_to(root).as_posix(),
                'bytes': path.stat().st_size, 'sha256': digest(path)}
        if path.suffix.lower() == '.mp4':
            item['video'] = video_info(path)
        result['artifacts'].append(item)
    if source:
        entries = [
            source/'inputs/video_completion/running_car.mp4',
            source/'inputs/video_completion/mask_square.png',
            source/'inputs/object_removal/bmx-trees',
            source/'inputs/object_removal/bmx-trees_mask',
        ]
        for entry in entries:
            files = sorted(p for p in entry.rglob('*') if p.is_file()) if entry.is_dir() else [entry]
            for path in files:
                result['inputs'].append({'path': path.relative_to(source).as_posix(),
                                         'sha256': digest(path), 'bytes': path.stat().st_size})
        for name in ['raft-things.pth', 'recurrent_flow_completion.pth', 'ProPainter.pth']:
            path = source/'weights'/name
            if path.exists():
                result['weights'].append({'path': f'weights/{name}',
                                          'sha256': digest(path), 'bytes': path.stat().st_size})
    target = root/'artifact_manifest.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    manifest = describe(args.directory.resolve(), args.source.resolve() if args.source else None)
    print(f"Recorded {len(manifest['artifacts'])} artifacts; no benchmark scores calculated.")

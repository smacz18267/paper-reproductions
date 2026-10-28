# Three MOSE watermark/removal case studies

Each notebook covers one fixed MOSE sample and displays every committed video
for that sample: the original, raw object mask, unwatermarked ProPainter and
DiffuEraser baselines, and all available VideoSeal, Safe-Sora, and VideoShield
artifacts.

These are the first three records in the fixed 50-sample manifest, rather than
samples selected after looking at the results.

| Sample | Notebook | Source properties |
| --- | --- | --- |
| `02b72b8d` | [Open notebook](sample_02b72b8d.ipynb) | 67 frames, 960 × 480, 25 FPS |
| `0ab75563` | [Open notebook](sample_0ab75563.ipynb) | 104 frames, 960 × 448, 25 FPS |
| `0b02886a` | [Open notebook](sample_0b02886a.ipynb) | 30 frames, 960 × 640, 25 FPS |

The notebooks explain the complete experiment order and every displayed metric
in plain language. They also preserve an important capability distinction:
VideoSeal has a derived crop-grid localization result, Safe-Sora reconstructs a
graphical watermark rather than a spatial mask, and VideoShield cannot
faithfully watermark an already-existing MOSE clip. VideoShield is therefore
shown with its explicit negative-control/availability record rather than with
fabricated ProPainter, DiffuEraser, mask, or IoU outputs.

## Fresh clone: download every video at once

The MP4 evidence is stored with Git LFS. A clone made without Git LFS can look
complete while containing only tiny pointer files, which cannot play. From the
repository root, download and verify all 48 videos used by these notebooks in
one operation:

```bash
python3 scripts/prepare_mose_notebooks.py
```

To download and verify all videos for the complete 50-sample review instead:

```bash
python3 scripts/prepare_mose_notebooks.py --all
```

The script initializes Git LFS for the checkout, downloads the real media in
one batch, rejects leftover pointer files, and verifies that every downloaded
file is an MP4 container. If `git lfs` is unavailable, install Git LFS first
(`brew install git-lfs` on macOS or `sudo apt-get install git-lfs` on Ubuntu).

After preparation, start Jupyter from the repository root, open one notebook,
and execute its cells to get inline video controls and per-video download
links. The notebook raises a clear Git LFS error if a pointer file is still
present instead of silently displaying a broken player.

Regenerate the notebooks after changing the evidence bundle with:

```bash
python3 scripts/build_mose_sample_notebooks.py
```

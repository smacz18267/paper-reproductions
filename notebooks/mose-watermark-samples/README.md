# MOSE watermark method notebooks

The results are organized into three notebooks. Each notebook covers one
watermark method across the same three fixed MOSE samples.

| Method | Notebook | Samples |
| --- | --- | --- |
| VideoSeal | [Open notebook](videoseal.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| Safe-Sora | [Open notebook](safe_sora.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| VideoShield | [Open notebook](videoshield.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |

The selected samples are the first three records in the fixed 50-sample
manifest. They were not selected after reviewing the results.

The VideoSeal notebook contains the source videos, masks, unwatermarked removal
references, watermarked videos, ProPainter and DiffuEraser outputs, effective
ground-truth masks, derived predicted masks, and localization metrics.

The Safe-Sora notebook contains the source videos, masks, unwatermarked removal
references, sampled inputs, aligned masks, watermarked videos, ProPainter and
DiffuEraser outputs, graphical watermark recovery images, and fidelity metrics.

The VideoShield notebook contains the source videos, masks, and negative-control
records. Authentic VideoShield embedding requires diffusion generation, so the
existing MOSE clips do not have VideoShield-watermarked or removal outputs.

## Download all notebook videos

The MP4 evidence is stored with Git LFS. From the repository root, download and
verify all 48 videos used by the notebooks with:

```bash
python3 scripts/prepare_mose_notebooks.py
```

To download and verify all videos for the complete 50-sample review:

```bash
python3 scripts/prepare_mose_notebooks.py --all
```

After preparation, start Jupyter from the repository root, open a notebook, and
run its cells. Each displayed video has a download link.

Regenerate the notebooks after changing the evidence bundle with:

```bash
python3 scripts/build_mose_sample_notebooks.py
```

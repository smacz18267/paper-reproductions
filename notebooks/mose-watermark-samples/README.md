# MOSE watermark method notebooks

The results are organized into five notebooks. EditGuard and OmniGuard were
evaluated across all 50 clips in the fixed MOSE selection, while their
notebooks display the same three fixed examples used by the other notebooks.
Their full 50-sample aggregate metrics and evidence are linked from each
notebook.

| Method | Notebook | Samples |
| --- | --- | --- |
| VideoSeal | [Open notebook](videoseal.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| Safe-Sora | [Open notebook](safe_sora.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| VideoShield | [Open notebook](videoshield.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| EditGuard | [Open notebook](editguard.ipynb) | 50 evaluated; `02b72b8d`, `0ab75563`, `0b02886a` displayed |
| OmniGuard | [Open notebook](omniguard.ipynb) | 50 evaluated; `02b72b8d`, `0ab75563`, `0b02886a` displayed |

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

The EditGuard notebook applies the released image method frame by frame using
its official 512 x 512 input protocol. It contains aligned sources and masks,
watermarked videos, both removal outputs, native predicted masks, effective
ground-truth masks, localization IoU, and 64-bit recovery accuracy.

The OmniGuard notebook applies the released image method frame by frame at the
native 960 x 480 MOSE size. It contains watermarked videos, both removal
outputs, native predicted masks, effective ground-truth masks, and localization
IoU. The public demo does not expose a recovered bit message.

## Download all notebook videos

The MP4 evidence is stored with Git LFS. From the repository root, download and
verify all videos used by the notebooks with:

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

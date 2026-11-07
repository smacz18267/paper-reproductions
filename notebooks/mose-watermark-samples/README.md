# Watermark removal notebooks

This directory contains the five completed result notebooks plus the six
execution notebooks in the current experiment batch. The six new notebooks
run the requested ProPainter and DiffuEraser attacks, preserve provenance, and
invoke each method's native detector/extractor.

RivaGAN and LVMark are intentionally deferred and are not represented as
completed execution notebooks.

Post-hoc methods use the fixed MOSE sample. Methods whose watermark is coupled
to diffusion generation use only an authors' provided sample or a sample
generated with the authors' released prompts, keys, checkpoints, and pipeline.
They are never represented as having watermarked an existing MOSE clip.

| Method | Notebook | Protocol/status |
| --- | --- | --- |
| VideoSeal | [Open notebook](videoseal.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| Safe-Sora | [Open notebook](safe_sora.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| VideoShield | [Open notebook](videoshield.ipynb) | `02b72b8d`, `0ab75563`, `0b02886a` |
| EditGuard | [Open notebook](editguard.ipynb) | 50 evaluated; `02b72b8d`, `0ab75563`, `0b02886a` displayed |
| OmniGuard | [Open notebook](omniguard.ipynb) | 50 evaluated; `02b72b8d`, `0ab75563`, `0b02886a` displayed |
| Watermark Anything | [Open notebook](watermark_anything.ipynb) | Fixed 32-bit message, framewise MOSE adapter, native presence/bit detector |
| VINE | [Open notebook](vine.ipynb) | Fixed 100-bit message, upstream center-crop MOSE adapter, native bit decoder |
| VideoMark | [Open notebook](videomark.ipynb) | Authors' committed ModelScope sample, shift, and 512-bit PRC key |
| SIGMark | [Open notebook](sigmark.ipynb) | Authors' VBench2 prompt data and native HunyuanVideo generation/extraction |
| VidSig / Video Signature | [Open notebook](vidsig.ipynb) | Authors' prompt, key, fine-tuned VAE, and native 48-bit decoder |
| GenPTW | [Open notebook](genptw.ipynb) | Released post-generation image path applied framewise at 512 x 512 |

The selected samples are the first three records in the fixed 50-sample
manifest. They were not selected after reviewing the results.

The VideoSeal notebook contains the source videos, masks, unwatermarked removal
references, watermarked videos, ProPainter, DiffuEraser, and EffectErase outputs, effective
ground-truth masks, derived predicted masks, and localization metrics.

The Safe-Sora notebook contains the source videos, masks, unwatermarked removal
references, sampled inputs, aligned masks, watermarked videos, ProPainter,
DiffuEraser, and EffectErase outputs, graphical watermark recovery images, and fidelity metrics.

The VideoShield notebook contains the source videos, masks, and negative-control
records. Authentic VideoShield embedding requires diffusion generation, so the
existing MOSE clips do not have VideoShield-watermarked or removal outputs.

The EditGuard notebook applies the released image method frame by frame using
its official 512 x 512 input protocol. It contains aligned sources and masks,
watermarked videos, all three removal outputs, native predicted masks, effective
ground-truth masks, localization IoU, and 64-bit recovery accuracy.

The OmniGuard notebook applies the released image method frame by frame at the
native 960 x 480 MOSE size. It contains watermarked videos, all three removal
outputs, native predicted masks, effective ground-truth masks, and localization
IoU. The public demo does not expose a recovered bit message.

## Remaining-method execution contract

The six new notebooks share these scripts:

- `prepare_watermark_removal_sample.py` validates and freezes the authentic
  watermarked input and aligned binary mask. Short sequences are padded only
  for DiffuEraser compatibility and later trimmed to their authentic length.
- `run_watermark_removal_pair.py` runs the official DiffuEraser pipeline once,
  preserving its internal ProPainter prior and final DiffuEraser output. This
  guarantees identical input, mask, dilation, seed, and processing geometry.
- Method-specific adapters call the released embedder and detector/extractor.
  They record hashes, commits, messages/keys, geometry adapters, and detector
  outputs. Methods without a released spatial detector report bit recovery
  only rather than inventing a proxy localizer.

For authors' native samples without published object annotations, the notebook
uses a deterministic centered region mask and labels it as experiment-defined.
It is not presented as an author-provided object mask.

Every applicable notebook uses the official
[lixiaowen-xw/DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser)
implementation for its DiffuEraser column. This is the same checkout used by
the batch runner; it is not a separate third remover.

The five completed result notebooks also use the official
[FudanCVL/EffectErase](https://github.com/FudanCVL/EffectErase) implementation
as a third removal column. EffectErase is evaluated on the same fixed clips and
masks as ProPainter and DiffuEraser. The six new execution notebooks implement
only the two removers requested here: ProPainter and DiffuEraser.

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

Regenerate the six current-batch execution notebooks with:

```bash
python3 scripts/build_remaining_watermark_notebooks.py
```

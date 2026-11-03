# Correct MOSE removal and watermark-localization protocol

This file is the execution contract for the experiment. Do not change the
ordering of its stages.

## Stage 1 — manual removal review

Use the fixed 50-clip MOSEv1 training selection in `selection50.json`. For each
clip, the review view must show:

1. the exact original MOSE video;
2. the selected MOSE object mask, with white meaning remove;
3. the ProPainter result; and
4. the DiffuEraser result.

Both removal methods must receive the same selected MOSE object. The reviewer
records separately whether ProPainter and DiffuEraser removed that object
satisfactorily. The watermark-first outputs are generated for all 50 fixed
clips so the review does not change or cherry-pick the evaluated sample set.

The committed `scripts/build_mose_review.py` creates a single viewer covering
all 50 clips and exports the manual decisions as CSV.

## Stage 2 — watermark before tampering

For every watermark method and every selected clip, the order is:

```text
original MOSE video
  -> watermark embedding
  -> watermarked video
  -> ProPainter or DiffuEraser using the selected object mask
  -> attacked/tampered video
  -> the watermark method's detector
  -> predicted tamper mask, when that method has a native localization output
  -> IoU against the effective GT tamper mask
```

Watermarking an already inpainted video is the wrong experiment and must not be
used.

Each final clip record must keep one shared original and shared source mask,
then method-specific watermarked and attacked outputs. Do not duplicate the
shared media once per watermark method.

## GT mask used for IoU

The raw MOSE object mask is the human-readable object-selection mask. The IoU
GT must be the *effective mask actually consumed by the corresponding removal
method*, because the saved Task 1 runs expanded the mask at inference:

- ProPainter resized with nearest-neighbour interpolation and then used SciPy
  binary dilation for 8 iterations.
- DiffuEraser resized with nearest-neighbour interpolation, eroded once with a
  3 x 3 rectangular kernel, and then dilated 8 times with that kernel.

Therefore ProPainter and DiffuEraser need separate effective GT masks for IoU,
even though both start from the same MOSE object annotation. Report global
pixel IoU over all frames, matching the VideoShield paper implementation:

```text
IoU = sum(predicted AND GT) / sum(predicted OR GT)
```

Also retain per-frame IoU values so failures can be located during review.

## Recorded compatibility adapters

Two adapters are allowed only where their per-clip metadata records them:

- Safe-Sora's official protocol produces 16 frames, while DiffuEraser rejects
  inputs shorter than 22 frames. Repeat the final watermarked frame and mask to
  24 frames only during inference, then trim the result back to the 16 genuine
  frames shown and reported.
- Clip `c062f4d7` is 190 frames at 960 x 848 and did not fit beside another
  user's 33 GB GPU process. Its VideoSeal/ProPainter inference uses an 800 x 704
  internal size and a 10-frame window, then restores the result to 960 x 848.
  Its effective GT mask must follow the same resize, dilation, and restoration
  path before IoU. No other clip uses this spatial adapter unless its metadata
  explicitly says so.
- Safe-Sora's released recovery demo can decode duplicate patch positions after
  a strong attack. Its nearest-vacancy loop can then assign multiple duplicates
  to the same vacancy and leave invalid `-1` indices. The runner preserves the
  released highest-confidence exact matches, then assigns the remaining source
  patches one-to-one to their nearest vacant positions in descending confidence
  order. Each `recovery.json` records this deterministic completion. It affects
  only graphical watermark reconstruction, not a spatial mask or IoU.

## Method capability constraint

The three previously selected methods are not interchangeable:

- VideoSeal can post-process an existing MOSE video. The available released
  checkpoint exposes a global watermark-presence detector, not a native
  pixelwise localization map. This experiment derives a labelled 4 x 4
  patch-grid absence mask by running that detector on spatial crops. The result
  and its IoU must not be described as native VideoSeal localization.
- VideoShield provides the paper's native spatial tamper mask and IoU, but it
  embeds into a diffusion generator's initial noise. The released code has no
  faithful path for adding the watermark to an already-existing MOSE video.
  This was checked by tracing the embed/extract code and by running the
  documented DDIM-inversion strength sweep in
  `videoshield-adapter-check/README.md`. The literal method generates unrelated
  content; the experimental adapter obtains strong detection only after
  visibly changing or destroying the input content.
- Safe-Sora recovers a graphical watermark. Its released method does not output
  a spatial tamper mask, so it can be evaluated for watermark recovery after
  removal but cannot truthfully produce Pred mask or IoU.

Do not fabricate Pred-mask/IoU fields for a method that does not provide spatial
localization. A cross-method comparison must either use a separate common
forensic localizer (a different experiment) or report `not applicable`.

## Stage 1 commands

On the original GPU server, where the selected source frames already exist:

```bash
cd /home/smayan/research/paper-reproductions

/home/smayan/research/DiffuEraser/.venv/bin/python \
  scripts/build_mose_review.py \
  --selection /home/smayan/research/datasets/MOSEv1/selection50.json \
  --mask-metadata /home/smayan/research/datasets/MOSEv1/masks50.json \
  --source-frames /home/smayan/research/datasets/MOSEv1/selected50/JPEGImages \
  --results-archive /home/smayan/research/datasets/MOSEv1/mose_task1_results_20261013.tar \
  --output evidence/mose-task1/review50

python3 -m http.server 8765 --directory evidence/mose-task1/review50
```

Then open `http://localhost:8765`, review all 50 clips, and use **Export review
CSV**. The review records qualitative removal success but does not filter the
fixed 50-clip watermark experiment.

## Stage 2 and final validation commands

On the GPU server, after copying the scripts to the runner directory:

```bash
cd /home/smayan/research/mose50-corrected
GPU=1 bash scripts/run_mose_watermark_pipeline.sh
```

After transferring `watermark-review50` back to the repository host:

```bash
python3 scripts/merge_mose_watermark_results.py \
  --results-root /path/to/watermark-review50 \
  --review-root evidence/mose-task1/review50

python3 scripts/validate_mose_review.py evidence/mose-task1/review50
```

The validator decodes every video, checks frame count, resolution and frame
rate, rejects empty GT masks, enforces explicit N/A fields, and recomputes each
displayed VideoSeal IoU from the committed Pred-mask and effective-GT videos.

## VideoShield existing-video compatibility command

The separate compatibility check reuses the released watermark object, first
through its canonical generation path and then through a clearly labelled
experimental DDIM-inversion adapter:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/smayan/research/VideoShield/.venv/bin/python -u \
  /home/smayan/research/mose50-corrected/scripts/test_mose_videoshield_adapter.py \
  --source /home/smayan/research/VideoShield \
  --model-path /home/smayan/research/VideoShield/models/modelscope \
  --input-video /home/smayan/research/datasets/MOSEv1/batch_inputs/02b72b8d/video.mp4 \
  --output-dir /home/smayan/research/videoshield-mose-adapter/02b72b8d-final \
  --device cuda:0
```

See `videoshield-adapter-check/README.md` for the exact latent formula, all
strengths, full metrics, videos, contact sheet, and the scope of the conclusion.

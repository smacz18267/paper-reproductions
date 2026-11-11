# VideoShield existing-video compatibility check

This check answers a specific question raised during review: can the released
VideoShield watermark function be reused to watermark an already-existing MOSE
video instead of generating a new video?

## What the released code does

The pinned upstream source is commit
`a61efa73abb15d30f50ee535c3301cea7dec3075`.

- `watermark.py::VideoShield.create_watermark_and_return_w()` returns an entire
  encrypted, truncated-Gaussian **starting-noise tensor** `w` with shape
  `[1, 4, frames, height/8, width/8]`. It does not accept or modify video
  pixels.
- `watermark_embedding_and_extraction.py` passes that tensor to ModelScope as
  `latents=init_latents_w`. The watermarked video is therefore created by the
  diffusion generator from marked noise.
- Detection VAE-encodes a received video and DDIM-inverts it to estimate the
  original starting noise before sign testing, decryption and voting.
- The repository contains no frame-domain `watermark(video)` entry point and no
  released inversion/edit path for retaining the content of an arbitrary input
  video.

The normal VideoShield path can therefore produce a valid marked generated
video, but that video is not the input MOSE clip. The compatibility experiment
below also tests a reasonable inversion-based adapter instead of stopping at
that source-code observation.

## Adapter tested

Clip `02b72b8d` was decoded, sampled uniformly to 16 frames and resized to the
official ModelScope setting of 256 x 256. The experiment then:

1. VAE-encoded the MOSE frames;
2. DDIM-inverted them for 25 steps with the empty prompt, matching the released
   detector's unknown-prompt setting;
3. created an actual VideoShield encrypted sign template with
   `create_watermark_and_return_w()`;
4. imposed that sign template on the inverted noise while retaining the
   inverted noise magnitudes;
5. interpolated between the unmodified inversion and sign-imposed noise;
6. regenerated a video, re-inverted it, and ran VideoShield's released recovery
   metric; and
7. measured frame-aligned PSNR and SSIM against the sampled MOSE input and SSIM
   against the unwatermarked inversion round trip.

For strength `alpha`, the experimental adapter is:

```text
z_marked = sign(w) * abs(z_inverted)
z_alpha  = normalize((1 - alpha) * z_inverted + alpha * z_marked)
```

This adapter is deliberately labelled experimental. It is not in the
VideoShield paper or official repository.

## Result

| Route | Strength | Watermark accuracy | PSNR vs MOSE | SSIM vs MOSE | SSIM vs unmarked round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unmarked inversion control | 0.00 | 0.4707 | 20.69 dB | 0.7208 | 1.0000 |
| Experimental adapter | 0.05 | 0.5000 | 20.29 dB | 0.7022 | 0.9288 |
| Experimental adapter | 0.10 | 0.5820 | 19.58 dB | 0.6828 | 0.8682 |
| Experimental adapter | 0.15 | 0.6387 | 18.36 dB | 0.6530 | 0.7998 |
| Experimental adapter | 0.20 | 0.7148 | 17.12 dB | 0.6048 | 0.7126 |
| Experimental adapter | 0.25 | 0.7949 | 16.48 dB | 0.5832 | 0.6566 |
| Experimental adapter | 0.50 | 0.9668 | 14.81 dB | 0.4929 | 0.4850 |
| Experimental adapter | 0.75 | 0.5996 | 13.28 dB | 0.4778 | 0.4532 |
| Experimental adapter | 1.00 | 0.6562 | 13.03 dB | 0.4691 | 0.4468 |
| Released generation path | canonical `w` | 1.0000 | 12.03 dB | 0.1297 | 0.1292 |

The literal released route recovers perfectly but generates unrelated content.
The inversion adapter shows the expected trade-off: settings that roughly
preserve the scene are near chance or weakly detectable, while the strongest
result (`alpha=0.50`, accuracy 0.9668) visibly destroys the fish and aquarium
content. Stronger sign forcing is not monotonic after regeneration because the
empty-prompt generator has already moved far away from the input and DDIM
inversion no longer recovers the imposed template reliably.

The visual change is clear in
[`02b72b8d/contact-sheet.png`](02b72b8d/contact-sheet.png). Every corresponding
video is retained in the same directory, and
[`02b72b8d/metrics.json`](02b72b8d/metrics.json) contains the full-precision
measurements and sampling indices.

## Conclusion for the 50-sample experiment

The released VideoShield method is generation-time watermarking, not a faithful
post-processor for existing videos. A custom DDIM-inversion adapter can trade
input fidelity for detectability, but it does not preserve the MOSE video well
enough to serve as the requested "same video, now watermarked" input. Running
ProPainter or DiffuEraser after that adapter would confound object removal with
content regenerated or removed by the adapter itself.

For that reason, the 50-sample review keeps VideoShield explicitly marked
`not_applicable` rather than presenting experimental regenerated videos as
official VideoShield outputs. The native VideoShield generation/localization
demonstration remains separately reproduced under `evidence/videoshield/`.
This is a method-scope limitation supported by source inspection and an actual
compatibility test, not an assumption that the code was impossible to run.
The numerical sweep is a one-clip compatibility probe on the released
ModelScope path, not a 50-clip quality benchmark or a claim about an unreleased
future post-processing variant.

## Exact command

The committed script is `scripts/test_mose_videoshield_adapter.py`. The final
sweep was run on an NVIDIA RTX A6000 as follows:

```bash
cd /home/smayan/research
mkdir -p videoshield-mose-adapter/02b72b8d-final

CUDA_VISIBLE_DEVICES=0 \
  VideoShield/.venv/bin/python -u \
  mose50-corrected/scripts/test_mose_videoshield_adapter.py \
  --source /home/smayan/research/VideoShield \
  --model-path /home/smayan/research/VideoShield/models/modelscope \
  --input-video /home/smayan/research/datasets/MOSEv1/batch_inputs/02b72b8d/video.mp4 \
  --output-dir /home/smayan/research/videoshield-mose-adapter/02b72b8d-final \
  --device cuda:0
```

The default sweep uses strengths `0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50,
0.75, 1.0` plus the canonical released generation path. The script saves
`wm_info.bin` beside a local run so that run can be rechecked, but the file is
deliberately not committed because it contains the experiment's encryption key
and nonce. A fresh run creates a fresh VideoShield key, as the upstream code
does.

# MOSEv1 batch results

I ran the same 50 MOSEv1 clips through ProPainter and DiffuEraser. One annotated object was removed from each clip using the same mask for both methods.

## Result

- ProPainter: 50/50 valid outputs
- DiffuEraser: 50/50 valid outputs
- Total: 100 videos, 3,351 frames per method
- Final failures: 0

Every output was decoded and checked for frame count, resolution and 25 FPS. The final line in [batch_supervisor_v2.log](batch_supervisor_v2.log) records the completed 50/50 result.

## Settings

- Maximum side: 960 px
- Mask dilation: 8
- ProPainter: FP16
- GPU: NVIDIA RTX A6000
- Object choice: largest annotated object by total pixel area
- Selection seed: 42

## Timing

The first clip was the pilot, so the timing comparison below uses the other 49 paired runs.

| Method | Total | Mean per clip | Seconds per frame |
| --- | ---: | ---: | ---: |
| ProPainter | 33.34 min | 40.82 s | 0.609 |
| DiffuEraser | 73.43 min | 89.91 s | 1.342 |

DiffuEraser took about 2.2 times as long as ProPainter. On the pilot clip, both removed the target fish, but DiffuEraser reconstructed the overlapping white fish more cleanly. ProPainter was softer around that area.

## Review all 50 clips

The old deliverable exposed only one pilot clip. The corrected deliverable is a
single local review view at `review50/index.html` covering all 50 samples. Each
sample shows the exact original MOSE video, raw selected-object mask, ProPainter
result and DiffuEraser result. Playback can be synchronized, manual pass/fail
decisions are stored locally, and the completed review can be exported as CSV.

Start the review view from the repository root:

```bash
python3 -m http.server 8765 --directory evidence/mose-task1/review50
```

Then open `http://localhost:8765`. The pull request includes the complete media
bundle so the review works after checkout. `scripts/build_mose_review.py`
rebuilds the shared original/removal view, and
`scripts/merge_mose_watermark_results.py` adds the corrected watermark-first
outputs and metadata.

See [RESULTS_GUIDE.md](RESULTS_GUIDE.md) for the result directory tree and a
plain-language description of every MP4, PNG and JSON filename.

The method picker contains all three requested watermark methods:

- VideoSeal: watermarked input, both post-watermark removal videos, effective
  GT masks, derived patch-grid predicted masks, and IoU for both attacks.
- Safe-Sora: its official aligned 16-frame original, watermarked video, GT mask,
  and both post-watermark removal videos. Pred mask and IoU are marked not
  applicable because the released method does not provide spatial localization.
- VideoShield: an explicit per-clip availability record and unwatermarked
  negative control. It is marked not applicable because the released embedder
  creates a diffusion generator's initial noise and provides no faithful
  post-processing path for an existing MOSE video. This was verified with the
  source-level and empirical
  [existing-video adapter check](videoshield-adapter-check/README.md), not
  assumed from the paper title.

## Final validated bundle

The corrected review contains 50 fixed samples, 800 MP4 files and 100 Safe-Sora
recovery images. Validation decoded 40,861 video frames, rejected empty GT
masks, and independently recomputed all 100 displayed VideoSeal IoUs from the
committed predicted-mask and effective-GT videos.

- VideoSeal: 50/50 complete for both attacks. The released global detector did
  not produce a positive absence region through the documented 4 x 4 crop
  proxy, so all derived predicted masks are empty and both mean IoUs are 0.0.
- Safe-Sora: 50/50 complete for embedding, both attacks and graphical recovery.
  Mean recovery PSNR is 8.5305 dB after ProPainter and 8.6739 dB after
  DiffuEraser. These are watermark-recovery results, not spatial localization.
- VideoShield: 50/50 explicit not-applicable records. Its mean unwatermarked
  negative-control bit accuracy is 0.5028; this is not an embedding or
  localization result. A separate compatibility sweep on MOSE clip `02b72b8d`
  tested both the literal released generation path and a custom DDIM-inversion
  adapter. The literal path generated unrelated content; the adapter reached
  0.9668 recovery only after fidelity fell to 14.81 dB PSNR / 0.4929 SSIM.

The complete review directory is about 815 MB. See `validation.json` for the
per-file decoder inventory and `watermark-summary.json` for aggregate results.

The earlier base-only 100-video archive is 464 MiB. Its checksum is in
[full-archive.sha256](full-archive.sha256). The selection, mask metadata,
runner, batch state and event history are included in this folder.

The required experiment ordering, IoU definition and watermark-method
capability constraints are in [PROTOCOL.md](PROTOCOL.md). In particular, the
original video must be watermarked *before* either inpainting method is applied.

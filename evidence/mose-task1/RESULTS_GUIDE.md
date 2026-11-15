# Result files and directory map

All 50 reviewed samples are in `review50/media/<clip_id>/`. Open
`review50/index.html` to view them without navigating the folders manually.

```text
review50/
├── index.html                  # browser review for all 50 clips
├── manifest.json               # clip-to-file paths and displayed metrics
├── validation.json             # decoder/shape/FPS validation for every result
├── watermark-summary.json      # historical fixed-MOSE aggregate
└── media/<clip_id>/
    ├── original.mp4            # original MOSE clip
    ├── gt_mask.mp4             # raw MOSE object mask; white = object to remove
    ├── propainter.mp4           # ProPainter on the unwatermarked original
    ├── diffueraser.mp4          # DiffuEraser on the unwatermarked original
    └── methods/
        ├── videoseal/
        │   ├── watermarked.mp4
        │   ├── propainter.mp4, diffueraser.mp4, effecterase.mp4, e2fgvi.mp4
        │   ├── effective-gt-mask-{propainter,diffueraser,effecterase,e2fgvi}.mp4
        │   ├── pred-mask-{propainter,diffueraser,effecterase,e2fgvi}.mp4
        │   ├── embed.json and one JSON record per remover
        │   └── localization.json
        ├── safe-sora/
        │   ├── sampled-original.mp4, watermarked.mp4, gt-mask.mp4
        │   ├── propainter.mp4, diffueraser.mp4, effecterase.mp4, e2fgvi.mp4
        │   ├── recovery-{propainter,diffueraser,effecterase,e2fgvi}.png
        │   ├── embed.json and one JSON record per remover
        │   └── recovery.json
        └── videoshield/
            └── availability.json
```

## What each method file means

| File | Meaning |
| --- | --- |
| `watermarked.mp4` | Watermark applied to the original **before** object removal. |
| `propainter.mp4` / `diffueraser.mp4` / `effecterase.mp4` / `e2fgvi.mp4` | That watermarked video after the named removal attack. |
| `effective-gt-mask-*.mp4` | Exact dilated/resized mask consumed by that remover; white is the attacked region used for IoU. |
| `pred-mask-*.mp4` | VideoSeal's derived 4×4 detector mask; white is predicted tampering. |
| `embed.json` | Embedding settings, timing, checksums and fidelity measurements. |
| `propainter.json` / `diffueraser.json` | Run metadata for the corresponding MP4, including input, output, timing and any compatibility adapter. |
| `localization.json` | VideoSeal detector threshold, per-attack/per-frame measurements and global IoU. |
| `recovery-*.png` | Safe-Sora comparison: original graphical watermark, recovered watermark, and amplified recovery error. |
| `recovery.json` | Safe-Sora patch-recovery details and fidelity scores after each attack. |
| `availability.json` | VideoShield N/A reason plus the clearly labelled unwatermarked negative-control score. It does not represent embedding or localization. |

The cross-method full-50 machine record is
`evidence/full50-watermark-evaluation-summary.json`. It contains 50 per-sample
rows per runnable method, four remover states per row, and aggregates for every
method-specific key metric. The full-50 notebook cells verify that file's hash
before displaying per-sample values.

Safe-Sora has no predicted spatial mask or IoU because its released output is a
recovered graphical watermark. VideoShield's separate existing-video test is in
`videoshield-adapter-check/`: `input-sampled.mp4` is the MOSE input,
`alpha-*.mp4` are adapter strengths, `canonical.mp4` is the released generation
path, `contact-sheet.png` is the visual comparison, and `metrics.json` contains
the full detection/fidelity results.

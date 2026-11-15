# Watermark removal notebooks

This directory contains exactly 13 Jupyter notebooks, one for every watermarking method in the study. The completed minimum is one authentic clip per applicable watermarking method, attacked with the same four inpainting methods:

1. ProPainter
2. DiffuEraser
3. EffectErase
4. E2FGVI-HQ

The recorded one-clip matrix has 44 completed pair cells across 11 watermarking methods. The expanded contract is 11 methods × 50 authentic inputs × 4 removers = 2,200 evaluation cells. RivaGAN's four one-clip cells are intentionally deferred at the user's request. LVMark's four one-clip cells are explicitly blocked: its repository omits the referenced prompt inputs, and its official 2.05 GB weight archive was served to Zeppelin at roughly 25 KB/s (more than 24 hours estimated). Neither method is part of the 11-method full-50 contract, and no missing result is fabricated.

Every runnable notebook now also has a `Full-50 evaluation: key metrics` section generated from one exact summary hash. It gives completion counts and mean, population standard deviation, minimum, and maximum for every method-specific key metric. Its adjacent cell displays all 50 × 4 per-sample rows after verifying the summary SHA-256. Work that has not run is labelled `pending`, and unavailable downstream detector metrics are labelled `not evaluated` rather than silently omitted.

| Watermarking method | Notebook | One-clip source | Four-inpainter status |
|---|---|---|---:|
| EditGuard | [editguard.ipynb](editguard.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| OmniGuard | [omniguard.ipynb](omniguard.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| Watermark Anything | [watermark_anything.ipynb](watermark_anything.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| VINE | [vine.ipynb](vine.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| GenPTW | [genptw.ipynb](genptw.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| VideoSeal | [videoseal.ipynb](videoseal.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| Safe-Sora | [safe_sora.ipynb](safe_sora.ipynb) | Fixed MOSE `02b72b8d` | Complete |
| VideoShield | [videoshield.ipynb](videoshield.ipynb) | Authors' native ModelScope red-panda sample | Complete |
| VideoMark | [videomark.ipynb](videomark.ipynb) | Authors' committed ModelScope sample | Complete |
| SIGMark | [sigmark.ipynb](sigmark.ipynb) | Authors' VBench2 native generation | Complete |
| VidSig / Video Signature | [vidsig.ipynb](vidsig.ipynb) | Authors' prompt/key native generation | Complete |
| RivaGAN | [rivagan.ipynb](rivagan.ipynb) | Not executed | Deferred by request |
| LVMark | [lvmark.ipynb](lvmark.ipynb) | Not generated | Blocked, reason recorded |

Generation-coupled watermark methods are not represented as having watermarked MOSE. VideoShield, VideoMark, SIGMark, and VidSig use author-provided or author-generated native media. Every completed pair preserves the input/mask provenance and records any method-specific temporal padding, resizing, or mask dilation in JSON.

## Full-50 execution

The exact study plan is generated from the fixed MOSE selection plus four native-input manifests:

```bash
python3 scripts/build_full50_watermark_plan.py \
  --native-manifest videoshield=/path/to/videoshield-50.json \
  --native-manifest videomark=/path/to/videomark-50.json \
  --native-manifest sigmark=/path/to/sigmark-50.json \
  --native-manifest vidsig=/path/to/vidsig-50.json \
  --require-ready
```

The manifest schema is documented in [native-watermark-manifests.md](../../config/native-watermark-manifests.md). Copy [full50-watermark.example.json](../../config/full50-watermark.example.json), fill only real local paths, then run:

```bash
# The seven methods not already represented by the fixed-MOSE result tree.
python3 scripts/run_full50_watermark_jobs.py --config /path/to/full50-watermark.json

# Add E2FGVI-HQ to the four existing fixed-MOSE methods, then run their official detectors.
python3 scripts/run_mose_e2fgvi_full50.py --help
python3 scripts/run_mose_full50_detection.py --config /path/to/full50-watermark.json

# Rebuild the machine summary and all 11 notebook metric sections.
python3 scripts/collect_full50_watermark_results.py \
  --mose-output-root /path/to/watermark-review50 --require-complete
python3 scripts/update_full50_watermark_notebooks.py
python3 scripts/validate_full50_watermark_notebooks.py --require-complete
```

All runners are resumable. Existing complete metadata causes a skip unless `--force` is explicitly supplied.

The authoritative machine-readable status is [four-inpainter-matrix.json](../../evidence/four-inpainter-matrix.json). Regenerate and validate the notebooks from the repository root with:

```bash
python3 scripts/build_remaining_watermark_notebooks.py
python3 scripts/build_browser_video_previews.py
python3 scripts/finalize_four_inpainter_notebooks.py
python3 scripts/validate_remaining_watermark_notebooks.py
python3 scripts/validate_four_inpainter_matrix.py
python3 scripts/validate_full50_watermark_notebooks.py
python3 scripts/build_browser_video_previews.py --check
```

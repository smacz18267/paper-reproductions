# Watermark removal notebooks

This directory contains exactly 13 Jupyter notebooks, one for every watermarking method in the study. The completed minimum is one authentic clip per applicable watermarking method, attacked with the same four inpainting methods:

1. ProPainter
2. DiffuEraser
3. EffectErase
4. E2FGVI-HQ

The recorded matrix has 44 completed pair cells across 11 watermarking methods. RivaGAN's four cells are intentionally deferred at the user's request. LVMark's four cells are explicitly blocked: its repository omits the referenced prompt inputs, and its official 2.05 GB weight archive was served to Zeppelin at roughly 25 KB/s (more than 24 hours estimated). No results are fabricated for either method.

Every notebook puts a canonical equal four-inpainter comparison near the beginning. Each completed remover is shown in the same order with status, recorded runtime, frame count, dimensions, FPS, byte size, video, and JSON metadata. The adjacent code cell displays each of the four output videos and metadata records through the same loop. Older multi-sample MOSE tables also retain an explicit E2FGVI-HQ column; values outside the required one-clip run are labelled `not run`, and unavailable downstream detector metrics are labelled `not evaluated` rather than silently omitted.

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

The authoritative machine-readable status is [four-inpainter-matrix.json](../../evidence/four-inpainter-matrix.json). Regenerate and validate the notebooks from the repository root with:

```bash
python3 scripts/build_remaining_watermark_notebooks.py
python3 scripts/finalize_four_inpainter_notebooks.py
python3 scripts/validate_remaining_watermark_notebooks.py
python3 scripts/validate_four_inpainter_matrix.py
```

# Paper implementations

Paper implementations and execution results by Smayan Gupta.

| Paper | Implementation | Results |
| --- | --- | --- |
| ProPainter (ICCV 2023) | [Source](papers/propainter/source) · [Paper details](papers/propainter/README.md) | [Video completion](evidence/propainter/submitted/video-completion/README.md) · [Object removal](evidence/propainter/submitted/object-removal/README.md) |
| EffectErase (CVPR 2026) | [Implementation and setup](papers/effecterase/README.md) | [Recorded inference](evidence/effecterase/recorded/20261006T094029Z-3256661/README.md) |
| DiffuEraser (2025) | [Implementation and setup](papers/diffueraser/README.md) | [Recorded inference](evidence/diffueraser/recorded/20261006T110108Z-3306048/README.md) |
| VideoSeal (arXiv 2024) | [Reading notes and implementation](papers/videoseal/README.md) | [Recorded inference](evidence/videoseal/recorded/20261013T114148Z-1298956/README.md) |
| VideoShield (ICLR 2025) | [Reading notes and implementation](papers/videoshield/README.md) | [Recorded inference and localization](evidence/videoshield/recorded/20261013T115409Z-1323688/README.md) |
| Safe-Sora (NeurIPS 2025) | [Reading notes and implementation](papers/safe-sora/README.md) | [Recorded graphical-watermark demonstration](evidence/safe-sora/recorded/20261013T124831Z-1384467/README.md) |

[MOSEv1 50-clip removal and watermark review](evidence/mose-task1/README.md): all 50 originals, GT masks, ProPainter and DiffuEraser outputs, plus corrected watermark-first results for VideoSeal, Safe-Sora, and VideoShield capability records.

[Three MOSE watermark method notebooks](notebooks/mose-watermark-samples/README.md): one notebook each for VideoSeal, Safe-Sora, and VideoShield, with the same three fixed samples in every notebook.

After a fresh clone, run `python3 scripts/prepare_mose_notebooks.py` once to download and verify all 48 notebook videos through Git LFS. Add `--all` to fetch the complete 50-sample video review in one batch.

[Recorded ProPainter execution](papers/propainter/README.md#execution-evidence)

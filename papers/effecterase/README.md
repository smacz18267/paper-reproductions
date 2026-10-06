# EffectErase

**EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing**, CVPR 2026 Highlight.

Yang Fu, Yike Zheng, Ziyun Dai, and Henghui Ding.

[Paper](https://arxiv.org/abs/2603.19224) · [Official implementation](https://github.com/FudanCVL/EffectErase) · [Pinned source](https://github.com/FudanCVL/EffectErase/tree/bcee0a5da5ef387c2ba39390dc4d579503669fb8)

## Implementation

The setup script installs the authors' implementation in a separate virtual environment,
downloads Wan2.1-Fun-1.3B-InP and EffectErase.ckpt, and copies the inference source,
its supporting DiffSynth package, license, and supplied input video and mask into
`papers/effecterase/source`. The checkpoints remain in the external working checkout.
Hugging Face model revisions are recorded when downloaded and reused on setup retries.

The supplied example removes an object and its related visual effects. Its mask is
already included, so this demonstration does not require installing SAM2.1.

```bash
cd ~/research/paper-reproductions
bash scripts/setup_effecterase.sh "$HOME/research/EffectErase" 1
bash scripts/capture_effecterase.sh "$HOME/research/EffectErase" 1
```

The capture script uses GPU 1 and the authors' default settings: BF16, 81 frames,
832 × 480, 50 diffusion steps, seed 2025, CFG 1, and LoRA alpha 1.
It writes to a new evidence directory for every attempt rather than overwriting
the authors' bundled output. No TeaCache or additional VAE tiling is enabled.
The selected dependency constraints are in [environment-constraints.txt](environment-constraints.txt);
they have not yet been validated by a GPU inference run. Setup and each run save
the complete installed environment.

Decord 0.6.0 has a [known Linux wheel metadata error](https://github.com/dmlc/decord/issues/356):
its internal Python 3.6 tag contradicts the wheel's Python 3 filename. Setup accepts
only this isolated error on Linux x86_64, after the installed native decoder reads
frames 0, 40 and 80 from both supplied videos. Other dependency errors and decoding
failures still stop setup. The original check output, metadata and decoding checks
are saved in `setup_dependency_checks.json` and copied into each run's evidence.

## Execution evidence

<!-- effecterase-results:start -->

The setup and capture scripts are prepared. No EffectErase inference run has yet
been recorded in this repository. The authors' supplied result is not counted as
a personal reproduction.

<!-- effecterase-results:end -->

Every attempted run records its command, log, source commit and changes, Python
packages, GPU details, and exit status. A successful run also includes `removed.mp4`,
sampled input/mask/output comparisons, model and artifact SHA-256 hashes, and decoded
video metadata. `SUCCESS.txt` is written only after inference and output validation
both finish. These are pretrained inference demonstrations, not benchmark replication.

## Citation and license

The authors' code and checkpoint use [CC BY-NC 4.0](https://github.com/FudanCVL/EffectErase/blob/main/LICENSE).
The source snapshot retains their license and attribution.

```bibtex
@inproceedings{fu2026EffectErase,
  title={EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing},
  author={Fu, Yang and Zheng, Yike and Dai, Ziyun and Ding, Henghui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

# DiffuEraser

**DiffuEraser: A Diffusion Model for Video Inpainting**, 2025 technical report.

Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Tongyi Lab, Alibaba Group.

[Paper](https://arxiv.org/abs/2501.10018) · [Official implementation](https://github.com/lixiaowen-xw/DiffuEraser) · [Pinned source](https://github.com/lixiaowen-xw/DiffuEraser/tree/8e6f279ac7531e27ad1849c6f8dab5372a8597e7)

## Implementation

DiffuEraser refines a ProPainter reconstruction using a diffusion UNet and a
BrushNet branch with temporal attention. Setup downloads its pretrained weights,
the required Stable Diffusion 1.5 components, the fine-tuned VAE, and the 2-step
PCM LoRA used by the upstream inference entry point. Existing ProPainter
checkpoints under `~/research/ProPainter/weights` are copied when available.
Model revisions are recorded and reused when setup is rerun.

```bash
cd ~/research/paper-reproductions
bash scripts/setup_diffueraser.sh "$HOME/research/DiffuEraser" 1
bash scripts/capture_diffueraser.sh "$HOME/research/DiffuEraser" 1
```

The separate `.venv` uses the upstream dependency pins, including PyTorch 2.3.1,
torchvision 0.18.1, Diffusers 0.29.2, and NumPy 1.26.4. The authors recommend
Python 3.9.19; this setup also permits the server's Python 3.12 and checks actual
upstream imports and a CUDA deformable convolution before downloading models.
Additional dependency pins are in [environment-constraints.txt](environment-constraints.txt).
The server completed setup, CUDA checks, inference and artifact validation.
The recorded output, comparison and execution evidence are included below.

Setup archives the pinned upstream implementation and examples to
`papers/diffueraser/source`, retaining the upstream Apache 2.0 license and the
[ProPainter license](ProPainter-LICENSE). Large checkpoints stay in the external
working checkout and are not committed.

The recorded run uses upstream example3, up to 10 seconds, maximum side length
960, FP16, 2-step PCM, mask dilation 8, reference stride 10, neighbour length 10,
and subvideo length 50. A small wrapper seeds Python, NumPy, and PyTorch with
2025 and copies the intermediate ProPainter video before upstream deletes it.
The upstream inference code and model computation are otherwise unchanged.
The seed does not guarantee identical output across hardware or library versions.

The ProPainter prior is encoded at its smaller upstream resolution. Validation
accounts for MP4 encoding cropping an odd final row or column; the final
DiffuEraser output is still checked against its exact expected dimensions.
An existing run with inference exit code 0 can be validated without rerunning
the models:

```bash
bash scripts/finalize_diffueraser.sh "$HOME/research/DiffuEraser" /path/to/existing/run
```

Finalization preserves previous validation logs and exit status under
`validation_attempts`, records the validator used, and writes `SUCCESS.txt`
only after the existing videos pass validation.

The input and mask must be aligned MP4 videos with matching FPS.
The supplied example already includes a mask, but the two files can have
different total frame counts. As in the upstream code, capture processes the
common leading sequence up to 10 seconds, with at least 22 frames required.
Original frame counts and excluded tails are recorded in the input metadata.
Masks are resized to the processed video dimensions with nearest-neighbour
sampling by the unchanged upstream code.
The upstream memory estimate for a 960 × 540, 250-frame
video is around 20 GB; setup prints current GPU usage before selecting GPU 1.

## Execution evidence

<!-- diffueraser-results:start -->

Recorded inference: 250 frames at 960 × 536, 25 FPS. [Outputs and execution evidence](../../evidence/diffueraser/recorded/20261006T110108Z-3306048/README.md).

<!-- diffueraser-results:end -->

Each attempt records the exact command, log, source commit and local changes,
reproduction script commit and changes, Python environment, GPU details, model
revisions, and exit status. Successful runs also save the original input and mask,
the ProPainter prior, the final output, sampled comparisons, and SHA-256 hashes
of model files and artifacts. `SUCCESS.txt` is written only after inference and
decoded-output validation both pass. These are pretrained demonstrations, not
quantitative reproduction of the paper's benchmarks.

## Citation

```bibtex
@misc{li2025diffueraserdiffusionmodelvideo,
  title={DiffuEraser: A Diffusion Model for Video Inpainting},
  author={Xiaowen Li and Haolan Xue and Peiran Ren and Liefeng Bo},
  year={2025},
  eprint={2501.10018},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2501.10018}
}
```

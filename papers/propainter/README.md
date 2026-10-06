# ProPainter

**ProPainter: Improving Propagation and Transformer for Video Inpainting**, ICCV 2023.

Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, and Chen Change Loy.

[Paper](https://arxiv.org/abs/2309.03897) · [Official repository](https://github.com/sczhou/ProPainter) · [Implementation](source)

## Implementation

This directory contains the authors' ProPainter implementation. The pretrained
inference pipeline was run for video completion and object removal on an NVIDIA
RTX A6000, with FP16 inference and a processing resolution of 432 × 240.

The Python environment used Python 3.12.3, PyTorch 2.5.1, torchvision 0.20.1,
and CUDA 12.1 wheels. Dependency versions are recorded in
[environment-pins.txt](environment-pins.txt); the recorded run includes the
[complete package environment](../../evidence/propainter/recorded/20261005T120400Z-2328389/environment.freeze.txt).

## Results

| Example | Frames | Saved FPS | Resolution | Duration | Output |
| --- | ---: | ---: | --- | ---: | --- |
| Running car, square masks | 292 | 25 | 432 × 240 | 11.68 s | [Video completion](../../evidence/propainter/submitted/video-completion/README.md) |
| BMX cyclist removal | 80 | 24 | 432 × 240 | 3.33 s | [Object removal](../../evidence/propainter/submitted/object-removal/README.md) |

The result folders include the completed videos, masked-input visualizations,
sampled comparisons, and artifact hashes. The BMX example uses frame images;
24 FPS is the output-video setting. Both video pairs were fully decoded without
errors. These runs demonstrate pretrained inference. No training or quantitative
benchmark evaluation was performed for these submitted results.

## Execution evidence

The recorded rerun completed both examples successfully on 5 October 2026.
Its [source commit](../../evidence/propainter/recorded/20261005T120400Z-2328389/source_commit.txt),
[GPU and runtime details](../../evidence/propainter/recorded/20261005T120400Z-2328389/torch_runtime.txt),
[artifact hashes](../../evidence/propainter/recorded/20261005T120400Z-2328389/artifact_manifest.json),
and original command/log files are included:

| Example | Command | Execution log | Output |
| --- | --- | --- | --- |
| Video completion | [Command](../../evidence/propainter/recorded/20261005T120400Z-2328389/video-completion/command.sh) | [Log](../../evidence/propainter/recorded/20261005T120400Z-2328389/video-completion/run.log) | [Video](../../evidence/propainter/recorded/20261005T120400Z-2328389/video-completion/videos/running_car/inpaint_out.mp4) |
| Object removal | [Command](../../evidence/propainter/recorded/20261005T120400Z-2328389/object-removal/command.sh) | [Log](../../evidence/propainter/recorded/20261005T120400Z-2328389/object-removal/run.log) | [Video](../../evidence/propainter/recorded/20261005T120400Z-2328389/object-removal/videos/bmx-trees/inpaint_out.mp4) |

## Citation and license

The implementation retains the authors' [S-Lab License](source/LICENSE).

```bibtex
@inproceedings{zhou2023propainter,
  title={ProPainter: Improving Propagation and Transformer for Video Inpainting},
  author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
  booktitle={Proceedings of IEEE International Conference on Computer Vision},
  year={2023}
}
```

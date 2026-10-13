# Safe-Sora

**Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking**,
NeurIPS 2025.

Zihan Su, Xuerui Qiu, Hongbin Xu, Tangyu Jiang, Junhao Zhuang, Chun Yuan,
Ming Li, Shengfeng He, and Fei Richard Yu.

[Paper](https://arxiv.org/abs/2505.12667) · [Official implementation](https://github.com/Zihan-Su/Safe-Sora) · [Pinned source](https://github.com/Zihan-Su/Safe-Sora/tree/dc11a35aca3c594c99063c14793410a6df0c9532)

## Reading notes

Safe-Sora embeds an entire graphical watermark, not only a short bit string, into
a generated video. It divides a 256 × 256 watermark into patches and uses a
coarse-to-fine adaptive router to assign each patch to the most visually compatible
frame and latent region. This reduces visible interference because watermark
content is placed where it best matches the cover video.

The embedding and recovery networks use Mamba state-space blocks enhanced with
2D/3D wavelet decompositions. A spatiotemporal local scan models relationships
across frames and spatial regions while remaining more efficient than global
attention. Position bits travel with every patch, allowing the recovered patches
to be reordered into the original image.

## Reproduction workflow

```bash
cd ~/research/paper-reproductions
bash scripts/setup_safe_sora.sh "$HOME/research/Safe-Sora" 1
bash scripts/capture_safe_sora.sh "$HOME/research/Safe-Sora" 1
```

Setup downloads the authors' checkpoint archive and their supplied Mamba and
causal-convolution sources, installs the pinned environment, and derives a clean
256 × 256 demonstration watermark from the first bundled result figure. The large
training datasets are not required for pretrained inference. Capture follows the
official 16-frame, 320 × 512 VideoCrafter2 evaluation path and records the original,
watermarked and amplified-difference video alongside the original, recovered and
difference watermark image.

The setup selects CUDA 12.2 with PyTorch CUDA 12.1 when available, otherwise CUDA
11.8 with the matching PyTorch build, and compiles the authors' modified Mamba
sources locally. On other layouts, set `SAFE_SORA_CUDA_HOME`, `SAFE_SORA_CC`, and
`SAFE_SORA_CXX` to compatible toolkit and compiler paths.

The [recorded run](../../evidence/safe-sora/recorded/20261013T124831Z-1384467/README.md)
validated all 16 frames. The watermarked video retained **40.85 dB PSNR**, while
the recovered graphical watermark reached **39.68 dB PSNR** against the original.

The official repository does not include an explicit software license at the
pinned commit. For that reason its source is cloned for local execution but is not
redistributed inside this repository. Checkpoints and auxiliary CUDA sources also
remain in the external checkout. The workflow is a pretrained example, not a
reproduction of training or the full robustness tables.

## Citation

```bibtex
@article{su2025safe,
  title={Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking},
  author={Su, Zihan and Qiu, Xuerui and Xu, Hongbin and Jiang, Tangyu and Zhuang, Junhao and Yuan, Chun and Li, Ming and He, Shengfeng and Yu, Fei Richard},
  journal={arXiv preprint arXiv:2505.12667},
  year={2025}
}
```

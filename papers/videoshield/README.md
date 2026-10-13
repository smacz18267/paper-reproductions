# VideoShield

**VideoShield: Regulating Diffusion-based Video Generation Models via
Watermarking**, ICLR 2025.

Runyi Hu, Jie Zhang, Yiming Li, Jiwei Li, Qing Guo, Han Qiu, and Tianwei Zhang.

[Paper](https://arxiv.org/abs/2501.14195) · [Official implementation](https://github.com/hurunyi/VideoShield) · [Pinned source](https://github.com/hurunyi/VideoShield/tree/a61efa73abb15d30f50ee535c3301cea7dec3075)

## Reading notes

VideoShield is a generative watermark. It places a repeated and ChaCha20-encrypted
binary template into the signs of the diffusion model's initial Gaussian noise.
Positive and negative half-Gaussian sampling preserves the expected marginal noise
distribution. The video generator then turns that marked noise into visible frames.

Detection encodes the received frames and applies DDIM inversion to estimate the
initial noise. Sign tests, decryption and majority voting recover the payload.
Frame-specific template patterns support temporal localization of inserted,
dropped or reordered frames. Spatial localization compares the recovered template
with the original one and uses hierarchical spatial-temporal refinement to pool
noisy evidence at several scales.

## Reproduction workflow

The lower-resolution ModelScope path is used: 16 frames at 256 × 256, 25 generation
steps and the authors' first prompt. It exercises embedding, extraction, temporal
localization and spatial localization.

```bash
cd ~/research/paper-reproductions
bash scripts/setup_videoshield.sh "$HOME/research/VideoShield" 1
bash scripts/capture_videoshield.sh "$HOME/research/VideoShield" 1
```

Setup pins the official source and the exact Hugging Face model revision. Capture
seeds Python, NumPy and PyTorch, but CUDA kernels and diffusion libraries may still
produce small cross-platform differences. Every attempt records the commands,
environment, GPU, source/model revisions, logs and decoded artifacts. Successful
runs include watermark recovery accuracy, three temporal-localization accuracies,
spatial F1/precision/recall/IoU/AUC, generated videos and masks.

This is a pretrained demonstration rather than a full paper benchmark. The
VideoShield code is MIT-licensed. The ModelScope generator is non-commercial and
its model card currently lists CC BY-NC 4.0; the downloaded weights remain outside
this repository.

## Citation

```bibtex
@inproceedings{hu2025videoshield,
  title={VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking},
  author={Hu, Runyi and Zhang, Jie and Li, Yiming and Li, Jiwei and Guo, Qing and Qiu, Han and Zhang, Tianwei},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

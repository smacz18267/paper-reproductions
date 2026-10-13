# VideoSeal

**Video Seal: Open and Efficient Video Watermarking**, arXiv 2024.

Pierre Fernandez, Hady Elsahar, I. Zeki Yalniz, and Alexandre Mourachko.

[Paper](https://arxiv.org/abs/2412.09492) · [Official implementation](https://github.com/facebookresearch/videoseal) · [Pinned source](https://github.com/facebookresearch/videoseal/tree/870ca7fb33578b90f14c602016b6c2788096226e)

## Reading notes

VideoSeal is a post-processing watermark: it embeds a binary payload into an
existing image or video and later extracts the payload from the possibly
distorted media. The embedder and extractor are trained jointly with image and
video transformations between them. This makes extraction robust to operations
such as resizing, cropping and video compression while perceptual losses and a
just-noticeable-difference attenuation map keep the signal difficult to see.

Its main efficiency idea is temporal watermark propagation. The expensive
embedder runs on sparse frames and propagates a low-resolution watermark signal
through neighbouring frames, rather than independently watermarking every
high-resolution frame. The released VideoSeal v1.0 model carries 256 bits and
exposes a strength parameter: increasing it improves robustness but can make the
watermark more visible.

## Reproduction workflow

```bash
cd ~/research/paper-reproductions
bash scripts/setup_videoseal.sh "$HOME/research/VideoSeal" 1
bash scripts/capture_videoseal.sh "$HOME/research/VideoSeal" 1
```

Setup creates an isolated environment, checks the selected CUDA device, downloads
the authors' VideoSeal v1.0 checkpoint and records its SHA-256 digest. Capture uses
the authors' supplied demonstration video and the official pretrained model. A
small PyAV runner is used because the server does not provide a system `ffmpeg`
binary; model embedding and detection still call the unchanged VideoSeal API.

Each run records the input, 256-bit message, watermarked MP4, recovered bit
accuracy, decoded video properties, model/source revisions, environment, GPU and
artifact hashes. The run is a pretrained inference demonstration, not a
reproduction of the paper's complete robustness benchmark.

## License and citation

The code and model are released under the MIT license. The checkpoint remains in
the external working checkout and is not committed here.

```bibtex
@article{fernandez2024videoseal,
  title={Video Seal: Open and Efficient Video Watermarking},
  author={Fernandez, Pierre and Elsahar, Hady and Yalniz, I. Zeki and Mourachko, Alexandre},
  journal={arXiv preprint arXiv:2412.09492},
  year={2024}
}
```

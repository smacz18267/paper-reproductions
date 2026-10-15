# Task 3: watermark baselines on inpainting results

The three Task 2 watermark baselines were evaluated on all 100 Task 1 outputs: 50 ProPainter videos and the matching 50 DiffuEraser videos.

## Completion

- VideoSeal: 100/100 succeeded
- Safe-Sora: 100/100 succeeded
- VideoShield negative control: 100/100 succeeded
- Total baseline runs: 300
- Failed runs: 0
- Generated result tree: 589 MiB
- Server location: `/home/smayan/research/task3-baselines/results`

Every generated media artifact was decoded during metric calculation. The final validator found all 300 metric records, checked the declared artifact sizes and SHA-256 checksums, and exited successfully.

## Aggregate results

| Baseline and metric | DiffuEraser (50) | ProPainter (50) | All (100) |
| --- | ---: | ---: | ---: |
| VideoSeal bit accuracy | 99.9219% | 99.9766% | 99.9492% |
| VideoSeal decoded-video PSNR | 37.51 dB | 37.49 dB | 37.50 dB |
| Safe-Sora video PSNR | 41.52 dB | 41.45 dB | 41.48 dB |
| Safe-Sora recovered-watermark PSNR | 40.22 dB | 40.19 dB | 40.20 dB |
| VideoShield reference-key accuracy | 50.19% | 50.18% | 50.18% |

The two inpainting methods produced nearly identical watermark results. VideoSeal recovered almost every embedded bit, and Safe-Sora retained high video and watermark fidelity.

## Protocols

- **VideoSeal:** post-process each complete Task 1 video, encode it as H.264, decode the result and recover the embedded message. All frames are processed in chunks of eight.
- **Safe-Sora:** uniformly sample 16 frames, resize them to 512 x 320, encode through the official VAE, embed and recover the fixed Task 2 watermark, and record video and watermark PSNR.
- **VideoShield:** this method embeds a watermark in the initial noise of diffusion generation, so it cannot truthfully add a watermark to videos already rendered by ProPainter or DiffuEraser. The existing videos were therefore evaluated as an explicitly labelled unwatermarked negative control using 16 frames at 256 x 256, 25-step DDIM inversion, and the fixed Task 2 reference key. The observed 50.18% mean accuracy is the expected chance-level result; it is not a watermark robustness score.

Full distributions are in [summary.json](summary.json). The reproducible batch runners and validator are in `scripts/`.

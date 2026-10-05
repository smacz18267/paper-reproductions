# BMX object removal

These files were supplied from Smayan Gupta's completed ProPainter server run on 5 October 2026.

- [Completed video](inpaint_out.mp4)
- [Masked-input visualization](masked_in.mp4)
- [File metadata and hashes](artifact_manifest.json)

![Sampled input/output comparison](comparison.png)

Both videos have 80 frames at 24 saved FPS and a resolution of 432 x 240.
Both were fully decoded without errors, and the comparison uses corresponding sampled frames.
The left column is the masked-input visualization; the right is the completed output.
The green visualization overlay is not the ground-truth background.
No quantitative benchmark quality score was calculated. The original complete runtime log
and exact source commit were not recorded; the separate capture script records future runs.

# Watermark evaluation briefing: 44 completed cases

## Problems and status first

The evaluation stage is complete for all **44 runnable combinations**: 11 watermarking methods × 4 inpainting methods. There are no unresolved failures among those 44 cases.

Two of the 13 watermarking methods are excluded, accounting for the remaining 8 cells:

- **RivaGAN — deferred/not runnable (4 cells).** The official repository includes training code but no pretrained `model.pt`. The available sample also has no matching checkpoint and embedded message. This is an upstream release-resource problem, not a lack of Zeppelin access. Running the notebook locally will not fix it unless a trained checkpoint and matching message record are provided.
- **LVMark — blocked/not runnable (4 cells).** The released inference script refers to prompt images and a prompt/key JSON that are absent. The official 2.05 GB weight archive was also served to Zeppelin at roughly 25 KB/s, an estimated download time above 24 hours. A local run is useful only if the complete weights and missing input record are supplied.

Recoverable infrastructure issues during the 44-case run were resolved: an expired VideoSeal checkpoint certificate was handled by enforcing the checkpoint's recorded SHA-256, missing declared dependencies were installed, and Safe-Sora's CUDA extensions were compiled using Zeppelin's complete GCC/G++ 11 toolchain.

## How to read the results

- Lower payload/bit/watermark accuracy means **less watermark survived** the inpainting attack.
- Higher localization IoU means the detector **localized the inpainted region better**. IoU does not measure payload survival.
- For Safe-Sora, lower PSNR means **poorer graphical-watermark recovery**.
- These are method-native metrics on a mixture of the fixed MOSE sample and author-native generated samples. Compare the four inpainters **within each row**, not absolute values between watermarking methods.

Abbreviations in the table: **payload** = video-level or majority-vote bit accuracy; **frame** = mean per-frame bit accuracy; **IoU** = spatial localization IoU; **p** = mean predicted tamper probability.

## Every runnable watermarking method × all four inpainters

| Watermark method | ProPainter | DiffuEraser | EffectErase | E2FGVI-HQ |
|---|---|---|---|---|
| **EditGuard** | payload **82.42%**; IoU **0.0276** | payload **71.64%**; IoU **0.0297** | payload **51.49%**; IoU **0.0168** | payload **71.20%**; IoU **0.0223** |
| **GenPTW** | payload **100%**; frame 89.23%; IoU 0.0013 | payload **100%**; frame 97.46%; IoU 0.0011 | payload **73.44%**; frame 59.98%; IoU 0.0013 | payload **100%**; frame 98.88%; IoU 0.0012 |
| **OmniGuard** | p **0.0125**; IoU **0.0011** | p **0.0134**; IoU **0.0015** | p **0.0201**; IoU **0.0006** | p **0.0133**; IoU **0.0006** |
| **Safe-Sora** | recovery PSNR **8.232 dB**; MAE 87.54 | recovery PSNR **8.544 dB**; MAE 82.65 | recovery PSNR **8.586 dB**; MAE 83.39 | recovery PSNR **8.091 dB**; MAE 90.94 |
| **SIGMark** | bit accuracy **49.89%** | bit accuracy **61.89%** | bit accuracy **50.07%** | bit accuracy **68.25%** |
| **VideoMark** | **2/16** frames detected (12.5%); bit 6.46% | **1/16** (6.25%); bit 3.36% | **0/16** (0%); bit 0% | **1/16** (6.25%); bit 3.35% |
| **VideoSeal** | derived localization IoU **0.0000** | derived localization IoU **0.0000** | derived localization IoU **0.0000** | derived localization IoU **0.0000** |
| **VideoShield** | watermark accuracy **63.28%**; IoU **0.7709** | accuracy **76.95%**; IoU **0.8427** | accuracy **54.88%**; IoU **0.7490** | accuracy **79.49%**; IoU **0.8729** |
| **VidSig / Video Signature** | payload **60.42%**; frame 58.98% (29/48 bits) | payload **89.58%**; frame 90.49% (43/48) | payload **54.17%**; frame 54.30% (26/48) | payload **91.67%**; frame 91.93% (44/48) |
| **VINE-R** | payload **100%**; frame 99.88% | payload **100%**; frame 99.99% | payload **82.00%**; frame 66.55% | payload **100%**; frame 99.99% |
| **Watermark Anything** | payload **90.63%**; frame 86.15%; IoU 0.0232 | payload **90.63%**; frame 91.98%; IoU 0.0552 | payload **75.00%**; frame 64.97%; IoU 0.0173 | payload **100%**; frame 95.66%; IoU 0.0669 |

VideoSeal needs careful wording: its released global-presence detector was adapted to a 4×4 patch grid and returned an empty attacked-region map in all four cases. Therefore IoU is zero for every inpainter. This is a **failure to localize the changed region**, not a direct claim that zero watermark bits survived.

OmniGuard also does not expose message-bit recovery in the released detector. Its reported output is the mean tamper probability and localization IoU.

## Professor/supervisor speaking summary

1. **All 44 runnable evaluations are finished.** The only missing 8 cells are RivaGAN and LVMark, both excluded because their official releases lack resources needed for an authentic run.
2. **EffectErase was usually the strongest payload-removal attack.** It produced the lowest residual payload/watermark accuracy for EditGuard, GenPTW, Watermark Anything, VINE, VideoMark, VidSig, and VideoShield. The exception was SIGMark, where ProPainter was marginally lower than EffectErase: 49.89% versus 50.07%.
3. **E2FGVI-HQ usually preserved recoverability best.** It retained 100% payload accuracy for GenPTW, VINE, and Watermark Anything, and had the highest residual scores for SIGMark, VideoShield, and VidSig. For Safe-Sora, however, it produced the poorest graphical-watermark recovery at 8.09 dB.
4. **Spatial localization varied much more by watermark method than by inpainter.** VideoShield localized strongly (IoU 0.749–0.873), while GenPTW, EditGuard, OmniGuard, and the derived VideoSeal patch map were weak on this sample. These numbers should not be ranked across methods because their detectors, samples, and mask definitions differ.
5. **The most defensible conclusion is within-method:** EffectErase most consistently damages recoverable payloads, while E2FGVI-HQ most often leaves them recoverable. Cross-method claims require a normalized common detector and sample protocol.

The exact unrounded metrics and execution notes are in `evidence/watermark-evaluation-44.json`.

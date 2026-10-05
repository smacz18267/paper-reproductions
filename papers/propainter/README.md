# ProPainter

Paper: **ProPainter: Improving Propagation and Transformer for Video Inpainting**, ICCV 2023.
Authors: Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan and Chen Change Loy.
Official source: https://github.com/sczhou/ProPainter
Paper: https://arxiv.org/abs/2309.03897

## Current result

The official pretrained inference pipeline was run on the supplied video-completion
and object-removal examples. The submitted artifacts are stored under
`evidence/propainter/submitted/`. This is demonstration inference, not a
from-scratch implementation, newly trained model, or reproduction of benchmark scores.

| Case | Frames | Saved FPS | Resolution | Duration |
| --- | ---: | ---: | --- | ---: |
| Running car with square masks | 292 | 25 | 432 x 240 | 11.68 seconds |
| BMX cyclist removal | 80 | 24 | 432 x 240 | 3.333333 seconds |

The BMX demo uses image frames as input, so 24 FPS is the inference script's
saved-video default; it is not a measured original camera frame rate.
Both output/input visualization pairs were fully decoded without errors.
Sampled frames show reconstructed masked regions and removal of the cyclist/bike.
No PSNR, SSIM, VFID, inference runtime or peak GPU memory was measured for these
submitted runs. Benchmark scores are left unset.

## Save the exact source used

From the repository root on the server:

```bash
bash scripts/import_propainter.sh "$HOME/research/ProPainter"
```

This imports the existing checkout's committed files, including its LICENSE,
records its commit, and exports the installed environment if available. It does
not copy the virtual environment, downloaded weights, results or Git metadata.
It stops if tracked source files have uncommitted changes or a snapshot already exists.

## Record real execution evidence

Choose an available GPU according to the server's allocation instructions. Then:

```bash
bash scripts/capture_propainter.sh "$HOME/research/ProPainter" 1
```

This reruns both supplied examples in the existing working environment and
creates a timestamped evidence directory. It saves source commit/diff, package
versions, GPU information, exact commands, genuine process output, timestamps,
input/weight/output hashes and both saved videos. `SUCCESS.txt` appears only
after both commands and artifact recording succeed. A failed command stops the
script and leaves any collected logs for diagnosis.

## Recreate the environment for an imported source snapshot

From the repository root, with Python 3.12 available:

```bash
cd papers/propainter/source
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -r ../environment-pins.txt imageio tqdm
```

If `environment.freeze.txt` has been exported, retain it as the exact package
record of the existing installation. The CUDA wheel index also matters when
recreating that environment.

## Attribution and license

The source snapshot remains the authors' implementation. Keep its upstream
README and LICENSE intact and describe custom changes separately. ProPainter's
S-Lab License permits non-commercial use and redistribution under its conditions;
do not apply a blanket MIT license to all third-party code in this collection.

```bibtex
@inproceedings{zhou2023propainter,
  title={ProPainter: Improving Propagation and Transformer for Video Inpainting},
  author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
  booktitle={Proceedings of IEEE International Conference on Computer Vision},
  year={2023}
}
```

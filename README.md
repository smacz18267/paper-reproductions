# Paper reproductions

Smayan Gupta's collection of paper implementations, execution evidence and results.

| Paper | Status | Evidence |
| --- | --- | --- |
| [ProPainter](papers/propainter/README.md) | Official pretrained inference run on two supplied examples | [Video completion](evidence/propainter/submitted/video-completion/README.md), [object removal](evidence/propainter/submitted/object-removal/README.md) |

Each paper gets its own source snapshot, environment record, run instructions
and results. Keep authors' implementations and original licenses identified.
Record whether a result is a demonstration, a benchmark evaluation, training,
or a custom reimplementation.

## What this starter contains

- Actual videos uploaded from the two completed ProPainter demonstrations.
- Comparisons made from corresponding sampled video frames.
- The provided console screenshot for the running-car execution.
- Video metadata and SHA-256 hashes of the supplied artifacts.
- Scripts to import the source from the existing server checkout and capture
  fresh runs with real logs and exact environment/source/input records.

The original runs' complete logs and source commit were not captured, so this
starter does not invent them. Run the capture script to obtain those records.
No GPU inference was executed while constructing this starter on a separate
machine; the submitted output files came from the user's completed server runs.

## Start on the existing server

Extract this archive in ~/research. Then:

```bash
cd ~/research/paper-reproductions
bash scripts/import_propainter.sh "$HOME/research/ProPainter"
bash scripts/capture_propainter.sh "$HOME/research/ProPainter" 1
```

The capture script uses the existing virtual environment and runs both examples
again on physical GPU 1. Check its availability under your server's allocation
instructions. Evidence is saved below evidence/propainter/recorded/ with a new
run ID, so previously submitted results remain separate.

## Create the GitHub repository

1. Visit https://github.com/new and create an empty repository called
   `paper-reproductions`. Choose its visibility; do not initialize a README,
   .gitignore or license because this starter already supplies files.
2. On the server, replace the email and username placeholders below with your
   GitHub commit email and username:

```bash
cd ~/research/paper-reproductions
git init -b main
git config user.name "Smayan Gupta"
git config user.email "YOUR_GITHUB_COMMIT_EMAIL"
git add .
git add -f papers/propainter/source
git status --short
git commit -m "Add ProPainter source, reproducible demos and execution evidence"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/paper-reproductions.git
git push -u origin main
```

The forced add is only for the initial imported snapshot, which contains
committed upstream files; it preserves files the upstream ignore rules might
otherwise hide. Do not force-add a working directory containing downloaded data,
model checkpoints or virtual environments.

For HTTPS authentication, use GitHub's credential flow or a personal access
token at the password prompt. The CSE SSH key is not automatically a GitHub
credential. Repository-local Git name/email settings do not configure login.

## Add the next paper

Use papers/<paper-name>/ for its attributed source, README, configs and environment
record. Use evidence/<paper-name>/<run-id>/ for results and real execution logs.
Use a separate Python environment when dependency versions differ. Document the
same inputs/settings for cross-paper comparisons and link result previews from
the main table. Commit code plus selected small evidence files after each run.

Full datasets, virtual environments and model checkpoints stay out of ordinary
Git. GitHub blocks ordinary Git files above 100 MiB. Keep representative videos,
comparisons, logs and measured score tables in the repository; link larger runs
from suitable artifact storage. Retain the provenance/hash records for them.

## Dataset notes

[MOSE](datasets/MOSE.md) is useful for harder qualitative object-removal examples.
It is separate from the ProPainter paper's benchmark evaluation protocol.

# Lightweight explainable LoRA-ViT — 12-lead ECG arrhythmia task package

An agent-benchmark task package for 7-class arrhythmia classification from 12-lead ECG. An
agent is given `instruction.md`, a corpus and a GPU; it must build a parameter-efficient,
explainable classifier and emit a fixed set of artefacts. `tests/` grades them. `solution/`
is the reference implementation.

```
.
├── solution_explanation.md   how an expert approaches the task, and why
├── VALIDATION.md             transcripts of the two required checks
├── DECLARATION.md            authorship statement
├── docker-compose.yml        build / solve / verify / shell, with GPU passthrough
├── Makefile                  the same, as named targets
├── scripts/
│   ├── run_local.ps1         full pipeline on Windows + CUDA, no Docker
│   ├── run_local.sh          same, on Linux
│   └── checks.sh             harbor nop + oracle, transcripts captured to logs/
└── tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit/
    ├── instruction.md        the task as the agent sees it
    ├── task.toml             environment contract, timeouts, artefacts
    ├── environment/
    │   ├── Dockerfile        CUDA 12.8, every version pinned
    │   └── data/             SNOMED vocabulary, class map, split spec, corpus
    ├── solution/
    │   ├── solve.sh          reference entry point
    │   └── src/ecgvit/       11 modules: config, labels, preprocess, data, lora,
    │                         model, train, evaluate, xai, stats, cli
    └── tests/
        ├── test.sh           three-tier marking entry point
        ├── expected/         golden files
        ├── fixtures/         seeded synthetic WFDB generator
        └── test_*.py         219 tests
```

## Quick start

Docker is the reproducible path:

```bash
docker compose build
docker compose run --rm solve      # reference pipeline end to end
docker compose run --rm verify     # grade whatever is in outputs/
```

On Windows this needs Docker Desktop with WSL2 integration and the NVIDIA Container Toolkit.
Check the GPU reaches a container before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

If that does not print your GPU, the problem is Docker, not this repository.

### Without Docker

Faster to iterate on a workstation that already has a CUDA build of torch:

```powershell
.\scripts\run_local.ps1 -ChapmanRoot "C:\H Research Projects\WFDB_ChapmanShaoxing"
```

```bash
scripts/run_local.sh --chapman-root /path/to/WFDB_ChapmanShaoxing
```

Both take `-Stage` / `--stage` (`verify`, `train`, `evaluate`, `explain`, `stats`) to run one
step, and `-Smoke` / `--smoke` for a 2+1-epoch wiring check whose metrics are meaningless by
design.

## The two checks that gate submission

```bash
scripts/checks.sh          # or: make checks
```

- **NOP must be 0.000.** Proves the tests are real — if doing nothing scores above zero, the
  tests can be passed without doing the work.
- **Oracle must be 1.000.** Proves the task is solvable and the reference solution passes
  marking in full.

NOP runs first because it is fast and fails for structural reasons that would otherwise waste
an Oracle run. Both transcripts go into `VALIDATION.md`.

## What the task asks for

Denoise (Butterworth 0.5–40 Hz, 50 Hz notch, per-lead z-score, 10 s windows), resolve
multi-label SNOMED CT annotations to one of seven classes through a declared resolution
order, train a joint-lead transformer over CNN-tokenised patches, freeze it and adapt it with
rank-8 LoRA, then produce Grad-CAM, Integrated Gradients, Gradient SHAP and an
insertion/deletion faithfulness test, plus McNemar and DeLong against a no-LoRA control.

Built to the reference architecture this is **1,648,839** backbone parameters and
**131,072** trainable of **1,779,911** total — a 92.05% reduction against full fine-tuning.

## Environment

Everything is pinned: a CUDA 12.8.1 / cuDNN base image, Python 3.11, and thirteen exact
Python versions including `torch==2.8.0` from the cu128 index. The image build fails rather
than ships if the installed torch has no kernels for the target GPU architecture — the
RTX 50-series is `sm_120`, and wheels built against CUDA ≤ 12.6 contain no kernels for it.

The task runs with `network_mode = "no-network"`: every input is baked into the image, so
nothing can drift under it between runs.

## Data

The corpus is open access under CC-BY-4.0 (Zheng et al., *Scientific Data* 7:48, 2020). A
checksum-pinned subset lives in `environment/data/corpus/`; see `environment/data/README.md`
for provenance and the manifest. No patient-identifiable data is in this repository, and
`.gitignore` excludes every WFDB signal and header extension outside that vendored subset as
a second line of defence.

## Status

Outstanding before submission:

- Build and commit the corpus subset (`environment/data/corpus/` and its manifest).
- Calibrate `BALANCED_ACC_MIN` / `MACRO_F1_MIN` in `tests/test_artifacts.py` from a measured
  reference run rather than from a published claim, then re-run Oracle.
- Run both checks and fill in `VALIDATION.md`.
- Write and sign `DECLARATION.md`.
- Three assertions in `tests/test_artifacts.py` pin exact parameter counts that
  `instruction.md` no longer states. Oracle passes either way, but no other agent could hit
  an undisclosed integer — convert them to property checks before difficulty testing.

## Licence

MIT. See `LICENSE`.

## Not for clinical use

Research artefact only. Not a medical device, not validated for clinical use, not evaluated
in a reader study. Concordance between model attributions and AHA/ACC and ESC lead criteria
is a sanity check on the attributions, not evidence of diagnostic validity.

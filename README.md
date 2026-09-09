# Lightweight explainable LoRA-ViT for 12-lead ECG arrhythmia task package

An agent-benchmark task package for 7-class arrhythmia classification from 12-lead ECG. An
agent is given `instruction.md`, a corpus and a GPU; it must build a parameter-efficient,
explainable classifier and emit a fixed set of artefacts. `tests/` grades them. `solution/`
is the reference implementation.

```
.
├── RESULTS.md                test-split metrics and confusion matrix of a graded run
├── solution_explanation.md   how an expert approaches the task, and why
├── VALIDATION.md             transcripts of the two required checks
├── DECLARATION.md            authorship statement
├── docker-compose.yml        build / solve / verify / test / shell — runs anywhere
├── docker-compose.gpu.yml    GPU passthrough overlay, on by default via .env
├── .env / .env.cpu           GPU defaults / CPU-only overrides
├── Makefile                  the same, as named targets (plus *-cpu variants)
├── scripts/
│   ├── report.py             regenerates RESULTS.md from a run's artefacts
│   ├── docker/entrypoint.sh  installs the reference package into the container
│   ├── run_local.ps1         full pipeline on Windows + CUDA, no Docker
│   ├── run_local.sh          same, on Linux
│   └── checks.sh             harbor nop + oracle, transcripts captured to logs/
└── tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit/
    ├── instruction.md        the task as the agent sees it
    ├── task.toml             environment contract, timeouts, artefacts
    ├── environment/
    │   ├── Dockerfile        CUDA 12.8 by default, every version pinned
    │   ├── requirements-cu128.txt / requirements-cpu.txt  the two torch builds
    │   ├── docker-compose.yaml  GPU attachment (Harbor's Docker provider has none)
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

## Results

[`RESULTS.md`](RESULTS.md) carries the test-split metrics, the per-class classification
report and the confusion matrix of a graded run, generated from that run's own artefacts by
`scripts/report.py`. Headline: balanced accuracy **0.7438**, macro F1 **0.7433**, macro AUC
**0.9507**, with **92.05%** fewer trainable parameters than full fine-tuning.

Reviewers: expect to land *near* those numbers, not on them cuDNN kernel selection is not
pinned, so results vary with GPU and driver. The check is the thresholds in `tests/`, and
`docker compose run --rm verify` writing `1` to `logs/verifier/reward.txt` is the
confirmation. `RESULTS.md` sets this out in full.

## Quick start

Docker is the reproducible path:

```bash
docker compose build
docker compose run --rm test       # unit tier — no GPU, no prior run needed
docker compose run --rm solve      # reference pipeline end to end
docker compose run --rm verify     # grade whatever is in outputs/
docker compose run --rm shell      # interactive
```

`make test`, `make solve`, `make verify`, `make shell` are the same commands as named
targets, and `make smoke` is a 2 + 1-epoch wiring check.

Start with `test`. It runs the unit tier alone no corpus records, no artefacts from a
previous run, no GPU so it is the fastest proof that the image and the package are sound,
and it is the one check that passes on any machine. `verify` grades a completed run and
needs one to have happened.

`solve` needs a GPU. On Windows that means Docker Desktop with WSL2 integration and the
NVIDIA Container Toolkit; check the GPU reaches a container before anything else:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi   # or: make gpu-check
```

If that does not print your GPU, the problem is Docker, not this repository. Only `solve`,
`verify` and `shell` request a device — `test` and `build` do not, so they still work while
you sort the toolkit out.

### Running against the full corpus

The default run uses the checksum-pinned subset vendored in the image, which is the graded
corpus. Point `CHAPMAN_HOST_DIR` at a full PhysioNet `WFDBRecords` tree to run over the
whole database; it is bind-mounted read-only onto the same `/app/data/corpus`, so
`CHAPMAN_ROOT` stays exactly what `task.toml` declares:

```bash
CHAPMAN_HOST_DIR=/mnt/d/WFDB_ChapmanShaoxing docker compose run --rm solve
```

Every other knob works the same way — `SMOKE`, `PRETRAIN_EPOCHS`, `LORA_EPOCHS`,
`BATCH_SIZE`, `DEVICE`, `RESOLUTION_ORDER`. The defaults live in `.env`.

### On a machine with no NVIDIA GPU

There is a CPU-only image. Same code, same `torch==2.8.0` pin, CPU kernels, a separate tag
(`ecgvit:1.0.0-cpu`) so it cannot be confused with the graded one:

```bash
make build-cpu && make test-cpu     # or the full docker compose --env-file .env.cpu form
```

This is a wiring check, not a result. Training on CPU takes hours and its metrics are not
the manuscript's which is why `.env.cpu` sets `SMOKE=1`.

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

- **NOP must be 0.000.** Proves the tests are real if doing nothing scores above zero, the
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
**131,072** trainable of **1,779,911** total a 92.05% reduction against full fine-tuning.

## Environment

Everything is pinned: a CUDA 12.8.1 / cuDNN base image, Python 3.10 — Ubuntu 22.04's own
interpreter, from main and thirteen exact
Python versions including `torch==2.8.0` from the cu128 index. The image build fails rather
than ships if the installed torch has no kernels for the target GPU architecture the
RTX 50-series is `sm_120`, and wheels built against CUDA ≤ 12.6 contain no kernels for it.

The task runs with `network_mode = "no-network"`: every input is baked into the image, so
nothing can drift under it between runs.

The same Dockerfile also builds the CPU image, through three build arguments `BASE_IMAGE`,
`ACCELERATOR`, `TORCH_REQUIREMENTS` whose defaults reproduce the graded CUDA image exactly.
Harbor passes no build arguments, so the contract build is unchanged; `.env.cpu` overrides
all three. A second Dockerfile would have been the obvious alternative and the wrong one: the
two would have drifted at the first dependency bump.

Three things about this environment are not obvious. The first two were found by reading
Harbor's source rather than its documentation.

*The build context is `environment/`, not the task directory.* Harbor points the compose
build context at the environment directory, so every `COPY` in the Dockerfile is relative
to `environment/` and nothing above it can be copied at all. `docker-compose.yml` and the
`Makefile` build the same way, so a hand build and a Harbor build cannot diverge.

*Harbor's Docker provider cannot allocate GPUs.* `DockerEnvironment.capabilities` leaves
`gpus` at its `False` default, so `[environment] gpus = 1` does not request a device it
aborts the trial outright, before the container is created:

```
RuntimeError: Task requires 1 GPU(s) but EnvironmentType.DOCKER environment does not
support GPU allocation.
```

Only the cloud providers set `gpus=True`, and none is reachable under `no-network`. So
`gpus` is left unset and the device is attached by `environment/docker-compose.yaml`, which
Harbor merges after its own build override. `solution/solve.sh` then fails the run outright
if no CUDA device is visible, so a host missing the NVIDIA Container Toolkit says so in
seconds instead of spending an hour training on CPU.

*The environment image must not contain `solution/` or `tests/`, and that is why the
container needs an entrypoint.* Baking either one in would hand the reference implementation
to every agent, including the do-nothing agent, which would then score above zero. Under
Harbor this costs nothing: it uploads `solution/` at run time for the oracle agent only, and
the verifier inherits the same container the agent installed the package into. A local
`docker compose run --rm verify` gets a fresh container instead, so nothing had ever
installed `ecgvit` and `test.sh` failed its own precondition `no importable package named
'ecgvit'`, which reads like a broken submission and is really a missing install.
`scripts/docker/entrypoint.sh` installs it from the mounted read-only `/solution` when it is
not already importable, in the container's writable layer that `--rm` discards. The image on
disk stays clean.

One smaller trap, worth knowing before adding rules: Docker reads `<context>/.dockerignore`,
and the build context is `environment/`. The exclusions that actually shape the image are in
`environment/.dockerignore`; the one at the repository root only covers an ad-hoc
`docker build .`.

## Data

The corpus is open access under CC-BY-4.0 (Zheng et al., *Scientific Data* 7:48, 2020). A
checksum-pinned subset lives in `environment/data/corpus/`; see `environment/data/README.md`
for provenance and the manifest. No patient-identifiable data is in this repository, and
`.gitignore` excludes every WFDB signal and header extension outside that vendored subset as
a second line of defence.

## Status

Done:

- The corpus subset is built and committed 6,873 records, 792 MB, with `manifest.sha256`.
- `BALANCED_ACC_MIN` / `MACRO_F1_MIN` are calibrated from a measured reference run on the
  vendored subset rather than from a published claim.

Outstanding before submission:

- Run both checks and fill in `VALIDATION.md`.
- Write and sign `DECLARATION.md`.
- Three assertions in `tests/test_artifacts.py` pin exact parameter counts that
  `instruction.md` no longer states. Oracle passes either way, but no other agent could hit
  an undisclosed integer convert them to property checks before difficulty testing.

## Licence

MIT. See `LICENSE`.

## Not for clinical use

Research artefact only. Not a medical device, not validated for clinical use, not evaluated
in a reader study. Concordance between model attributions and AHA/ACC and ESC lead criteria
is a sanity check on the attributions, not evidence of diagnostic validity.

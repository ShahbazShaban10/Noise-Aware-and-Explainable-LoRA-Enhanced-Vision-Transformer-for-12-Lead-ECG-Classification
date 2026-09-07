#!/usr/bin/env bash
# Reference solution entry point (Harbor oracle agent).
#
# Harbor uploads solution/ to /solution and runs this script. Two things matter about
# that: /solution is NOT on the Python path, and the verifier runs in a separate process
# afterwards, so exporting PYTHONPATH here would not survive. The package is therefore
# *installed* into the image's Python, which is what makes `import ecgvit` work for the
# verifier as well as for this script.
#
# The environment provides everything else: the corpus at $CHAPMAN_ROOT, the reference
# inputs at $DATA_DIR, and $OUTPUT_DIR to write to. There is no network.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-/app/outputs}"
export DATA_DIR="${DATA_DIR:-/app/data}"
export CHAPMAN_ROOT="${CHAPMAN_ROOT:-/app/data/corpus}"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

log() { printf '\n\033[1m[solve]\033[0m %s\n' "$*"; }
die() { printf '\n[solve] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
[ -d "$CHAPMAN_ROOT" ] || die "corpus not found at $CHAPMAN_ROOT"
[ -f "$DATA_DIR/class_map_7.json" ] || die "class map not found in $DATA_DIR"
command -v python >/dev/null 2>&1 || die "python not found on PATH"

log "installing the reference package so the verifier can import it too"
# Build from a writable copy, never from $SCRIPT_DIR itself. With --no-build-isolation the
# setuptools backend writes build/lib/ inside the source tree, so a read-only /solution
# fails with "could not create 'build/lib/ecgvit': Read-only file system". Harbor uploads
# solution/ as a writable copy and never hits this, but docker-compose.yml mounts it :ro --
# and a grader mounting the reference implementation read-only is doing the sensible thing,
# not the wrong thing. The copy costs a few hundred kilobytes.
BUILD_TMP="$(mktemp -d)"
trap 'rm -rf "$BUILD_TMP"' EXIT
BUILD_SRC="$BUILD_TMP/ecgvit-src"
cp -r "$SCRIPT_DIR" "$BUILD_SRC" || die "could not copy $SCRIPT_DIR to a writable location"
python -m pip install --no-deps --no-build-isolation "$BUILD_SRC" \
  || die "could not install the ecgvit package from $SCRIPT_DIR"
python -c "import ecgvit, pathlib; print('  ecgvit at', pathlib.Path(ecgvit.__file__).parent)"

log "checking torch and GPU"
python - <<'PY'
import os, sys, torch
print(f"  torch          {torch.__version__}  (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    # Hard failure by default. Harbor's Docker provider ignores [environment] gpus, so the
    # device arrives only through environment/docker-compose.yaml, which needs the NVIDIA
    # Container Toolkit on the host. If that is missing the container starts perfectly well
    # and simply has no GPU -- and a silent CPU fallback would spend an hour producing
    # metrics that look like a modelling failure instead of a missing runtime.
    if os.environ.get("ALLOW_CPU") == "1":
        print("  WARNING: no CUDA device visible; ALLOW_CPU=1 is set, so this schedule "
              "runs on CPU and will take hours.", file=sys.stderr)
        sys.exit(0)
    print("\n  ERROR: no CUDA device is visible inside the container.\n"
          "  Verify the host can pass a GPU through Docker at all:\n"
          "    docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi\n"
          "  Set ALLOW_CPU=1 to run anyway.", file=sys.stderr)
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
cap = f"sm_{p.major}{p.minor}"
archs = torch.cuda.get_arch_list()
print(f"  gpu            {p.name} ({cap}, {p.total_memory/1024**3:.1f} GB)")
print(f"  build archs    {archs}")
if not any(cap in a for a in archs):
    print(f"\n  ERROR: this torch build has no kernels for {cap}.", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Hyperparameters. Defaults come from [solution.env] in task.toml; the manuscript's full
# 150 + 30 schedule is reachable by overriding them.
# ---------------------------------------------------------------------------
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
LORA_EPOCHS="${LORA_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"

if [ "${SMOKE:-0}" = "1" ]; then
  log "SMOKE=1: 2 + 1 epochs. Wiring check only; the resulting metrics are meaningless."
  PRETRAIN_EPOCHS=2
  LORA_EPOCHS=1
fi

# The taxonomy is part of the contract, not a preference. Passed explicitly rather than
# left to the library default, because that is exactly how this script and
# scripts/run_local.* silently disagreed: run_local passes canonical_v2, solve.sh passed
# nothing, so Harbor trained on OTHER-labelled records and was graded against VE.
RESOLUTION_ORDER="${RESOLUTION_ORDER:-canonical_v2}"

COMMON=(--chapman-root     "$CHAPMAN_ROOT"
        --output-dir       "$OUTPUT_DIR"
        --data-dir         "$DATA_DIR"
        --device           "$DEVICE"
        --batch-size       "$BATCH_SIZE"
        --resolution-order "$RESOLUTION_ORDER")

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
log "1/5 verifying corpus"
python -m ecgvit.cli "${COMMON[@]}" verify-data

log "2/5 training  (stage 1: ${PRETRAIN_EPOCHS} epochs full fine-tune; stage 2: ${LORA_EPOCHS} epochs LoRA r=8)"
python -m ecgvit.cli "${COMMON[@]}" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" --lora-epochs "$LORA_EPOCHS" train

log "3/5 evaluating on the held-out test split"
python -m ecgvit.cli "${COMMON[@]}" evaluate

log "4/5 explainability: Grad-CAM, Integrated Gradients, Gradient SHAP, faithfulness, t-SNE"
python -m ecgvit.cli "${COMMON[@]}" explain

log "5/5 statistical validation: McNemar and DeLong (LoRA vs no-LoRA)"
python -m ecgvit.cli "${COMMON[@]}" stats

log "done. Artefacts in $OUTPUT_DIR"
find "$OUTPUT_DIR" -maxdepth 2 -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.npy' -o -name '*.pt' \) | sort

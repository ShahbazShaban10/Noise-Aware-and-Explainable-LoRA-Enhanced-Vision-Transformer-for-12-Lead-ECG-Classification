#!/usr/bin/env bash
# Run the full pipeline on this machine, without Docker.
#
#   scripts/run_local.sh --chapman-root /path/to/WFDB_ChapmanShaoxing
#
# Docker is the reproducible path; this is the fast one for iterating on a workstation that
# already has a working CUDA build of torch.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="$REPO/tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit"

CHAPMAN_ROOT="${CHAPMAN_ROOT:-}"
RESOLUTION_ORDER="canonical_v2"
RECIPE="v2"
PRETRAIN_EPOCHS=40
LORA_EPOCHS=15
BATCH_SIZE=64
NUM_WORKERS=8
DEVICE=auto
STAGE=all

while [ $# -gt 0 ]; do
  case "$1" in
    --chapman-root)     CHAPMAN_ROOT="$2"; shift 2;;
    --resolution-order) RESOLUTION_ORDER="$2"; shift 2;;
    --recipe)           RECIPE="$2"; shift 2;;
    --pretrain-epochs)  PRETRAIN_EPOCHS="$2"; shift 2;;
    --lora-epochs)      LORA_EPOCHS="$2"; shift 2;;
    --num-workers)      NUM_WORKERS="$2"; shift 2;;
    --stage)            STAGE="$2"; shift 2;;
    --smoke)            PRETRAIN_EPOCHS=2; LORA_EPOCHS=1; shift;;
    -h|--help)          sed -n '2,10p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

[ -n "$CHAPMAN_ROOT" ] || { echo "set --chapman-root or \$CHAPMAN_ROOT" >&2; exit 2; }
[ -d "$CHAPMAN_ROOT" ] || { echo "corpus not found: $CHAPMAN_ROOT" >&2; exit 2; }

export PYTHONPATH="$TASK/solution/src:${PYTHONPATH:-}"
export DATA_DIR="$TASK/environment/data"
export OUTPUT_DIR="${OUTPUT_DIR:-$TASK/outputs}"
export PYTHONHASHSEED=0 MPLBACKEND=Agg
mkdir -p "$OUTPUT_DIR"

step() { printf '\n\033[1m[run]\033[0m %s\n' "$*"; }

step "torch and GPU"
python - <<'PY'
import sys, torch
print(f"  torch  {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA device visible; this will be very slow.", file=sys.stderr); sys.exit(0)
p = torch.cuda.get_device_properties(0); cap = f"sm_{p.major}{p.minor}"
print(f"  gpu    {p.name} ({cap}, {p.total_memory/1024**3:.1f} GB)")
if not any(cap in a for a in torch.cuda.get_arch_list()):
    print(f"  ERROR: this torch build has no kernels for {cap}", file=sys.stderr); sys.exit(1)
PY

COMMON=(--chapman-root "$CHAPMAN_ROOT" --output-dir "$OUTPUT_DIR" --data-dir "$DATA_DIR"
        --resolution-order "$RESOLUTION_ORDER" --recipe "$RECIPE"
        --device "$DEVICE" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS")
run() { step "$1"; shift; python -m ecgvit.cli "${COMMON[@]}" "$@"; }

case "$STAGE" in
  all)      run "verifying corpus" verify-data
            run "training ($PRETRAIN_EPOCHS + $LORA_EPOCHS epochs)" \
                --pretrain-epochs "$PRETRAIN_EPOCHS" --lora-epochs "$LORA_EPOCHS" train
            run "evaluating" evaluate
            run "explainability" explain
            run "statistics" stats ;;
  verify)   run "verifying corpus" verify-data ;;
  train)    run "training" --pretrain-epochs "$PRETRAIN_EPOCHS" --lora-epochs "$LORA_EPOCHS" train ;;
  evaluate) run "evaluating" evaluate ;;
  explain)  run "explainability" explain ;;
  stats)    run "statistics" stats ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2;;
esac

step "artefacts in $OUTPUT_DIR"

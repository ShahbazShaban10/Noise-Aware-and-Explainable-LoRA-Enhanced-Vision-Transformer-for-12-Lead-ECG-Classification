#!/usr/bin/env bash
# Harbor verifier entry point.
#
# Harbor copies tests/ to /tests and runs this file from the environment's working
# directory (/app). It must write the reward to /logs/verifier/reward.txt on every code
# path, and it must never trust a reward file the agent may have left behind.
#
# Reward is binary: 1 if every tier passes, 0 otherwise. A partial run is a failure --
# a submission that produced no artefacts must not score above zero.

set -uo pipefail

REWARD_DIR="/logs/verifier"
REWARD_FILE="${REWARD_DIR}/reward.txt"
mkdir -p "$REWARD_DIR"

# Overwrite immediately. If this script dies at any later point -- timeout, OOM, a bad
# import -- the reward on disk is already 0 rather than whatever was there before.
echo 0 > "$REWARD_FILE"

export DATA_DIR="${DATA_DIR:-/app/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/app/outputs}"
export CHAPMAN_ROOT="${CHAPMAN_ROOT:-/app/data/corpus}"
export PYTHONPATH="/app:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

TESTS_DIR="/tests"
PYTEST_ARGS=(-rA -p no:cacheprovider --timeout=900)

# CTRF gives Harbor a machine-readable per-test report. It is optional: if the plugin is
# not in the image, run without it rather than failing the whole verifier.
if python -c "import pytest_ctrf" 2>/dev/null || python -c "import pytest_json_ctrf" 2>/dev/null; then
  HAVE_CTRF=1
else
  HAVE_CTRF=0
fi

run_tier() {  # run_tier <name> <marker-expression>
  local name="$1" marker="$2"
  local args=("${PYTEST_ARGS[@]}")
  [ "$HAVE_CTRF" = 1 ] && args+=(--ctrf "${REWARD_DIR}/ctrf-${name}.json")
  python -m pytest "$TESTS_DIR" "${args[@]}" -m "$marker"
}

hr()  { printf '%s\n' "------------------------------------------------------------"; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

fail() { echo 0 > "$REWARD_FILE"; say "Result"; hr; echo "FAIL: $*"; exit 0; }

# ---------------------------------------------------------------------------
# Preconditions. Each is a genuine reason the submission cannot be graded, so each
# leaves the reward at 0 rather than skipping a tier.
# ---------------------------------------------------------------------------
command -v python >/dev/null 2>&1 || fail "python not found on PATH"
python -c "import pytest" 2>/dev/null || fail "pytest is not installed in the image"

say "Environment"
hr
printf '  workdir      %s\n' "$(pwd)"
printf '  data dir     %s\n' "$DATA_DIR"
printf '  output dir   %s\n' "$OUTPUT_DIR"
printf '  corpus       %s\n' "$CHAPMAN_ROOT"
python - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec("ecgvit")
print(f"  ecgvit       {'importable at ' + str(spec.origin) if spec else 'NOT IMPORTABLE'}")
try:
    import torch
    print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
except Exception as e:
    print(f"  torch        unavailable: {e}", file=sys.stderr)
PY
hr

[ -d "$CHAPMAN_ROOT" ]           || fail "corpus missing at $CHAPMAN_ROOT (image is broken, not the submission)"
python -c "import ecgvit" 2>/dev/null \
  || fail "no importable package named 'ecgvit' -- nothing was built"
[ -f "${OUTPUT_DIR}/metrics.json" ] \
  || fail "${OUTPUT_DIR}/metrics.json not found -- the run produced no artefacts"

RC=0

say "Tier 1/3  unit  (labels, preprocessing, model, LoRA, XAI, statistics)"
hr
run_tier unit "not corpus and not artifacts" || RC=1

say "Tier 2/3  artifacts  (grades \$OUTPUT_DIR, recomputes the reported metrics)"
hr
run_tier artifacts "artifacts" || RC=1

say "Tier 3/3  corpus  (real recordings)"
hr
run_tier corpus "corpus" || RC=1

say "Result"
hr
if [ $RC -eq 0 ]; then
  echo 1 > "$REWARD_FILE"
  echo "PASS -- all three tiers."
else
  echo 0 > "$REWARD_FILE"
  echo "FAIL -- at least one tier failed."
fi

# Exit 0 regardless: the reward file, not this exit code, carries the grade.
exit 0

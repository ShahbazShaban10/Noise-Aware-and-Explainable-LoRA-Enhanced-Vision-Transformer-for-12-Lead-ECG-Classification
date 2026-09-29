#!/usr/bin/env bash
# The two checks that must pass before submission, with their transcripts captured.
#
#   scripts/checks.sh
#
# NOP runs first: it is fast, needs no GPU time, and fails for structural reasons that
# would waste an Oracle run.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit"
cd "$REPO"; mkdir -p logs

command -v harbor >/dev/null 2>&1 || {
  echo "harbor not on PATH. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  uv tool install harbor"
  exit 2; }

echo "=== NOP (mean must be 0.000) ==="
harbor run -p "$TASK" -a nop 2>&1 | tee logs/nop.txt
echo
echo "=== ORACLE (mean must be 1.000) ==="
harbor run -p "$TASK" -a oracle 2>&1 | tee logs/oracle.txt
echo
echo "Transcripts: logs/nop.txt and logs/oracle.txt"
echo "Paste both into VALIDATION.md, with the mean line from each."

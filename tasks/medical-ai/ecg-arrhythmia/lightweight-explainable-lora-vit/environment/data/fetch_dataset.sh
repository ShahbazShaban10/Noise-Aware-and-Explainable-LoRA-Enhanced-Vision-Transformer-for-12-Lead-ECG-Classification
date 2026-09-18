#!/usr/bin/env bash
# Fetch and verify the Chapman-Shaoxing ECG corpus.
#
# This script exists because the corpus must NOT live in the repository. It is 1.2 GB of
# open-access patient signal data; environment/data/README.md has said from the start that
# it is "fetched by link", and a previous revision of this package ignored that and
# committed 13,746 signal files, which took the repository to 1,176 MB and failed review.
#
# SOURCE. PhysioNet/CinC Challenge 2021 training data, chapman_shaoxing cohort: 10,247
# records, flat JS-prefixed WFDB, grouped g1..g11 at 1,000 records per folder.
#
# NOT the PhysioNet `ecg-arrhythmia` 1.0.0 release, which earlier revisions of dataset.yaml
# and README.md wrongly cited. That release is 45,152 records because it merges
# Chapman-Shaoxing WITH Ningbo, and its records are JS-prefixed too -- so "JS-prefixed"
# does not isolate the cohort. Training on it would silently use a 4.4x larger, different
# population and nothing in the manuscript would reproduce.
#
#   bash fetch_dataset.sh [destination]      # default: $CHAPMAN_ROOT, else /app/data/corpus
#
# Safe to re-run: wget resumes and skips files already present.

set -euo pipefail

BASE="https://physionet.org/files/challenge-2021/1.0.3"
COHORT="training/chapman_shaoxing"
EXPECTED_RECORDS=10247

DEST="${1:-${CHAPMAN_ROOT:-/app/data/corpus}}"
die() { printf '\n[fetch-dataset] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[fetch-dataset] %s\n' "$*"; }

command -v wget >/dev/null 2>&1 || die "wget is required but not on PATH"
mkdir -p "$DEST"
cd "$DEST"

log "downloading $COHORT from PhysioNet (about 1.2 GB; resumes if interrupted)"
# --cut-dirs=5 strips files/challenge-2021/1.0.3/training/chapman_shaoxing so the g*/
# folders land directly in $DEST. The loader scans recursively, so the nested layout is
# fine and keeps the paths aligned with the published checksums.
wget -r -N -c -np -nH --cut-dirs=5 -R "index.html*" -e robots=off \
     --progress=dot:giga "$BASE/$COHORT/" \
  || die "download failed -- check network access to physionet.org"

log "verifying checksums against the published SHA256SUMS.txt"
wget -q -O /tmp/SHA256SUMS.full.txt "$BASE/SHA256SUMS.txt" \
  || die "could not fetch SHA256SUMS.txt"
# The published file covers the whole 1.0.3 release and its paths are relative to the
# release root; keep only this cohort and rewrite the paths to match $DEST.
# PhysioNet separates digest and path with ONE space. sha256sum writes two, so a filter
# written for two spaces matches nothing here and the build dies with "no entries found".
# Accept either, and re-emit the two-space form `sha256sum -c` expects.
sed -n "s#^\([0-9a-f]\{64\}\)[ *]\{1,2\}$COHORT/#\1  #p" \
    /tmp/SHA256SUMS.full.txt > /tmp/SHA256SUMS.cohort.txt
n_sums=$(wc -l < /tmp/SHA256SUMS.cohort.txt)
[ "$n_sums" -gt 0 ] || die "no $COHORT entries found in SHA256SUMS.txt -- has the layout changed?"
if ! sha256sum -c --quiet /tmp/SHA256SUMS.cohort.txt 2>/tmp/sha.err; then
    head -5 /tmp/sha.err >&2
    die "checksum verification failed -- the download is incomplete or corrupt"
fi

n_records=$(find "$DEST" -name '*.hea' | wc -l)
log "verified $n_sums files; $n_records records present"
[ "$n_records" -eq "$EXPECTED_RECORDS" ] \
  || die "expected $EXPECTED_RECORDS records, found $n_records"

rm -f /tmp/SHA256SUMS.full.txt /tmp/SHA256SUMS.cohort.txt /tmp/sha.err
log "corpus ready at $DEST"

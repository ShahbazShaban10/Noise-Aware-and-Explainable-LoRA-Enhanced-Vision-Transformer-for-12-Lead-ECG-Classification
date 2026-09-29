#!/usr/bin/env bash
# Fetch and verify the Chapman-Shaoxing ECG corpus from PhysioNet.
#
# This script exists because the corpus must NOT live in the repository. It is 1.2 GB of
# open-access patient signal data; environment/data/README.md has said from the start that
# it is "fetched by link", and a previous revision of this package ignored that and
# committed 13,746 signal files, which took the repository to 1,176 MB and failed review.
#
# SOURCE. PhysioNet/CinC Challenge 2021 training data, chapman_shaoxing cohort: 10,247
# records, JS-prefixed WFDB, grouped g1..g11 at 1,000 records per folder. The public
# dataset is the only source: nothing is re-hosted.
#
# NOT the PhysioNet `ecg-arrhythmia` 1.0.0 release, which earlier revisions of dataset.yaml
# and README.md wrongly cited. That release is 45,152 records because it merges
# Chapman-Shaoxing WITH Ningbo, and its records are JS-prefixed too -- so "JS-prefixed"
# does not isolate the cohort. Training on it would silently use a 4.4x larger, different
# population and nothing in the manuscript would reproduce.
#
# SPEED. physionet.org rate-limits downloads. Measured from one client: a single
# connection moves about 0.2 files/s, so a `wget -r` crawl of this cohort takes about a
# day. Fetching an explicit file list 16 at a time reaches ~6.5 headers/s and ~1.5
# signal files/s -- about 2.5 hours for the cohort, which is why task.toml allows the
# image build 6 hours. More than 16 connections is no faster and starts being refused.
#
#   bash fetch_dataset.sh [destination]      # default: $CHAPMAN_ROOT, else /app/data/corpus
#   FETCH_PARALLEL=8 bash fetch_dataset.sh   # fewer connections on a fragile link
#
# Safe to re-run: verified files are kept, missing or corrupt ones are fetched again.

set -euo pipefail

BASE="https://physionet.org/files/challenge-2021/1.0.3"
COHORT="training/chapman_shaoxing"
EXPECTED_RECORDS=10247
PARALLEL="${FETCH_PARALLEL:-16}"

DEST="${1:-${CHAPMAN_ROOT:-/app/data/corpus}}"
die() { printf '\n[fetch-dataset] ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[fetch-dataset] %s\n' "$*"; }

command -v wget >/dev/null 2>&1 || die "wget is required but not on PATH"
mkdir -p "$DEST"
cd "$DEST"
TMP="$(mktemp -d)"
PROGRESS_PID=""
cleanup() { [ -n "$PROGRESS_PID" ] && kill "$PROGRESS_PID" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

# Retry rate-limit and transient server errors rather than failing the whole build on one.
WGET_OPTS=(-q --tries=20 --waitretry=10 --retry-connrefused --retry-on-http-error=429,500,502,503,504 --timeout=60)

log "fetching PhysioNet's published SHA256SUMS.txt (it lists every file, with its digest)"
wget "${WGET_OPTS[@]}" -O "$TMP/SHA256SUMS.full.txt" "$BASE/SHA256SUMS.txt" \
  || die "could not fetch SHA256SUMS.txt -- check network access to physionet.org"

# The published file covers the whole 1.0.3 release and its paths are relative to the
# release root; keep only this cohort and rewrite the paths to match $DEST.
# PhysioNet separates digest and path with ONE space. sha256sum writes two, so a filter
# written for two spaces matches nothing here and the build dies with "no entries found".
# Accept either, and re-emit the two-space form `sha256sum -c` expects.
sed -n "s#^\([0-9a-f]\{64\}\)[ *]\{1,2\}$COHORT/#\1  #p" \
    "$TMP/SHA256SUMS.full.txt" > "$TMP/SHA256SUMS.cohort.txt"
n_files=$(wc -l < "$TMP/SHA256SUMS.cohort.txt")
[ "$n_files" -gt 0 ] || die "no $COHORT entries found in SHA256SUMS.txt -- has the layout changed?"
cut -d' ' -f3- "$TMP/SHA256SUMS.cohort.txt" > "$TMP/files.txt"

PASSES="${FETCH_PASSES:-8}"
log "downloading $n_files files, $PARALLEL at a time (about 1.2 GB; expect ~2.5 hours -- PhysioNet rate-limits)"
( while sleep 60; do
    printf '[fetch-dataset] %s / %s files present\n' \
      "$(find . -type f \( -name '*.hea' -o -name '*.mat' -o -name RECORDS \) | wc -l)" "$n_files"
  done ) &
PROGRESS_PID=$!

cut -d/ -f1 "$TMP/files.txt" | sort -u | xargs mkdir -p

# Self-healing passes. Each pass throws away anything on disk that does not match its
# published digest (a partial file from a dropped connection, a corrupt one), fetches
# whatever is missing, and checks again. One flaky connection costs a pass, not the build;
# `wget -c` alone cannot do this, because it cannot repair a file that is complete but wrong.
for pass in $(seq 1 "$PASSES"); do
    sha256sum -c "$TMP/SHA256SUMS.cohort.txt" 2>/dev/null \
      | sed -n 's/: FAILED.*$//p' > "$TMP/bad.txt" || true
    [ -s "$TMP/bad.txt" ] || break
    [ "$pass" -gt 1 ] && log "pass $pass: $(wc -l < "$TMP/bad.txt") files still missing or wrong; fetching them again"
    xargs rm -f < "$TMP/bad.txt"
    # -O names each file exactly as the checksum list does, so a failed or partial
    # download is simply caught by the next pass's verification.
    xargs -P "$PARALLEL" -I{} wget "${WGET_OPTS[@]}" -O {} "$BASE/$COHORT/{}" \
      < "$TMP/bad.txt" || true
done

kill "$PROGRESS_PID" 2>/dev/null || true
PROGRESS_PID=""

log "verifying every file against PhysioNet's published digests"
if ! sha256sum -c --quiet "$TMP/SHA256SUMS.cohort.txt" 2>"$TMP/sha.err"; then
    head -5 "$TMP/sha.err" >&2
    die "files still missing or wrong after $PASSES passes -- re-run this script to continue"
fi

n_records=$(find "$DEST" -name '*.hea' | wc -l)
log "verified $n_files files; $n_records records present"
[ "$n_records" -eq "$EXPECTED_RECORDS" ] \
  || die "expected $EXPECTED_RECORDS records, found $n_records"

log "corpus ready at $DEST"

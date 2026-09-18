# `environment/data/` — inputs for the task

This directory holds every **small** input the agent needs. The ECG signal corpus itself is
**not vendored** — it is an open-access PhysioNet database and is fetched by link. Nothing in
this repository contains patient signal data.

## Contents

| File | What it is |
| --- | --- |
| `snomed_conditions.csv` | 130 SNOMED CT concept ids with abbreviation, full name and the Chapman-Shaoxing record count. Transcribed from the official PhysioNet/CinC Challenge 2021 diagnosis tables. |
| `class_map_7.json` | Canonical SNOMED → 7-class mapping, resolution order for multi-label records, and per-class clinical rationale. Versioned; the tests assert against it byte-for-byte. |
| `dataset.yaml` | Download URL, licence, citation, expected record count, expected signal geometry. |
| `fetch_dataset.sh` | Downloads and verifies the corpus into `$CHAPMAN_ROOT`. |
| `splits/split_spec.yaml` | The split protocol (seed, proportions, stratification, grouping rule). Splits are **derived deterministically** from this spec, not stored as record lists. |

## The dataset

The **Chapman-Shaoxing** cohort — all 10,247 twelve-lead records — as distributed in the
PhysioNet/CinC Challenge 2021 training data, version 1.0.3.

- Landing page: <https://physionet.org/content/challenge-2021/1.0.3/>
- Files: <https://physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/>
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Original description: Zheng, J. et al. *A 12-lead electrocardiogram database for arrhythmia
  research covering more than 10,000 patients.* Scientific Data 7, 48 (2020).

This is **not** the PhysioNet `ecg-arrhythmia` 1.0.0 release, which earlier revisions of this
file cited. That release holds 45,152 records because it merges Chapman-Shaoxing with Ningbo,
and its records are JS-prefixed too, so the two cannot be told apart by name. Training on it
would silently use a 4.4× larger, different population. The manuscript's cross-dataset
generalisation experiment additionally uses PTB-XL,
<https://physionet.org/content/ptb-xl/1.0.3/> — optional here, see `dataset.yaml`.

## Getting the data

Nothing is needed for the graded run: the Docker image downloads the corpus while it is being
built (`RUN bash /app/data/fetch_dataset.sh /app/data/corpus`), so it is inside the image
before the container starts with no network. To fetch it outside Docker:

```bash
export CHAPMAN_ROOT=/data/chapman_shaoxing
bash fetch_dataset.sh            # ~1.2 GB; resumes if interrupted
```

The script verifies every file against PhysioNet's published `SHA256SUMS.txt`, fails on any
mismatch, and asserts exactly 10,247 records.

If you already have the corpus (for example a `WFDB_ChapmanShaoxing` folder from an earlier
project), skip the download and just point at it:

```bash
export CHAPMAN_ROOT="/mnt/d/H Research Projects/WFDB_ChapmanShaoxing"
python -m ecgvit.cli verify-data          # geometry + label sanity check, no training
```

`verify-data` is the gate: the training entry point refuses to run until it passes.

## Expected layout

`CHAPMAN_ROOT` is scanned recursively for `*.hea`, so both a flat directory and the
`g1/ … g11/` folders `fetch_dataset.sh` produces work. Each record is a WFDB header plus a
MATLAB signal file:

```
$CHAPMAN_ROOT/
├── g1/
│   ├── JS00001.hea
│   ├── JS00001.mat
│   └── …
├── …
└── g11/
```

A conforming header looks like this (verified against the real corpus):

```
JS00001 12 500 5000 23-Mar-2021 20:20:47
JS00001.mat 16+24 1000/mV 16 0 -254 21756 0 I
…                                          (11 more lead lines)
#Age: 85
#Sex: Male
#Dx: 164889003,59118001,164934002
#Rx: Unknown
#Hx: Unknown
#Sx: Unknown
```

Invariants the loader enforces:

- 12 leads, in the order `I II III aVR aVL aVF V1 V2 V3 V4 V5 V6`
- sampling frequency 500 Hz, 5000 samples (10 s)
- ADC gain `1000/mV`, 16-bit
- `#Dx:` present and comma-separated SNOMED CT concept ids

Records that violate an invariant are **excluded and reported**, never silently coerced. The
one exception is duration: shorter records are zero-padded and longer ones centre-cropped to
5000 samples, as specified in the manuscript (§3.2), and every such record is logged.

## What is *not* here

No `.mat`, no `.hea`, no cached tensors, no trained checkpoints. `.gitignore` blocks them.
If you find signal data under this directory, something has gone wrong — do not commit it.

## `ConditionNames_SNOMED-CT.csv`

The acronym → SNOMED CT table distributed **by the dataset developers** with
Chapman-Shaoxing (63 conditions, CC BY 4.0). It is the authoritative mapping for the
published `canonical_hmgmedformer` grouping, so the loader resolves that grouping's acronym
buckets against it and, where both files describe a code, its abbreviation wins over
`snomed_conditions.csv`. The two are merged rather than one replacing the other: this file
covers 63 conditions, the PhysioNet/CinC 2021 table covers 133, and codes outside this file
must still resolve.

**A collision worth knowing about.** In this file `AF` is *atrial flutter* and `AFIB` is
*atrial fibrillation*. In the PhysioNet/CinC 2021 tables `AF` is *atrial fibrillation* and
`AFL` is *atrial flutter*. So `AF` means different conditions in the two vocabularies, and
code that mixes them silently swaps fibrillation for flutter. Both land in the same `AFIB`
class here, so nothing downstream is affected — but any analysis that treats them
separately must fix the vocabulary first. The same file also assigns `SAAWR` to 195101003,
which the Challenge tables call `WAP` (wandering atrial pacemaker).

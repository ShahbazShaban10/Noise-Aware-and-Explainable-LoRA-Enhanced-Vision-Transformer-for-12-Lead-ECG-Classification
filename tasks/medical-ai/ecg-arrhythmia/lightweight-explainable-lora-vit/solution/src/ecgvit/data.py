"""Record indexing, deterministic splitting, and the torch Dataset.

Design notes
------------
* The split is a pure function of the record id (SHA-256 bucketing, salt from
  `split_spec.yaml`). No stored record list, no reshuffle when the corpus grows, and two
  machines agree byte-for-byte.
* Splitting happens BEFORE any balancing. Oversampling the training split cannot leak into
  validation or test, which is the failure mode in the notebook this task derives from:
  there, augmented copies were generated first and `train_test_split` then scattered
  near-duplicates of the same record across train and test.
* Preprocessing is cached to `.npy` on first read, keyed by a hash of the preprocessing
  config, so changing a filter parameter invalidates the cache instead of silently reusing
  signals filtered with the old settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import (CLASS_NAMES, N_LEADS, N_SAMPLES, PipelineConfig, PreprocessConfig,
                     active_class_names)
from .labels import ClassMap, get_class_map, load_snomed_vocabulary, parse_dx_codes
from .preprocess import preprocess_signal

log = logging.getLogger("ecgvit.data")

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Record index
# ---------------------------------------------------------------------------
@dataclass
class Record:
    record_id: str
    header_path: Path
    signal_path: Path
    fs_hz: float
    n_samples: int
    n_leads: int
    codes: Tuple[str, ...]
    label: str
    label_index: int
    split: str = ""

    def to_row(self) -> Dict[str, object]:
        return {
            "record_id": self.record_id,
            "header_path": str(self.header_path),
            "fs_hz": self.fs_hz,
            "n_samples": self.n_samples,
            "n_leads": self.n_leads,
            "codes": ";".join(self.codes),
            "label": self.label,
            "label_index": self.label_index,
            "split": self.split,
        }


def _parse_header_geometry(header_path: Path) -> Tuple[int, float, int, List[str]]:
    """Read line 1 (`name nsig fs nsamp ...`) and the lead names, without wfdb.

    Fast: scanning 10k headers with wfdb.rdheader takes minutes, this takes seconds.
    """
    lines = header_path.read_text(errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty header: {header_path}")
    parts = lines[0].split()
    if len(parts) < 4:
        raise ValueError(f"malformed header line in {header_path}: {lines[0]!r}")
    n_sig = int(parts[1])
    fs = float(parts[2])
    n_samp = int(parts[3])
    leads: List[str] = []
    for line in lines[1 : 1 + n_sig]:
        if line.startswith("#") or not line.strip():
            break
        leads.append(line.split()[-1])
    return n_sig, fs, n_samp, leads


def _split_bucket(record_id: str, salt: str, hex_chars: int = 16) -> float:
    digest = hashlib.sha256(f"{salt}:{record_id}".encode()).hexdigest()[:hex_chars]
    return int(digest, 16) / float(1 << (4 * hex_chars))


def assign_split(
    record_id: str,
    salt: str,
    proportions: Dict[str, float],
    hex_chars: int = 16,
) -> str:
    b = _split_bucket(record_id, salt, hex_chars)
    acc = 0.0
    for name in SPLITS:
        acc += proportions[name]
        if b < acc:
            return name
    return SPLITS[-1]


def assign_fold(record_id: str, salt: str, n_folds: int, hex_chars: int = 16) -> int:
    """Deterministic fold index in [0, n_folds).

    The same SHA-256 bucketing as `assign_split`, so folds share its two properties: a
    record's fold is a pure function of its id (no stored list, no reshuffle when the corpus
    grows) and two machines agree byte-for-byte.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    b = _split_bucket(record_id, salt, hex_chars)
    return min(int(b * n_folds), n_folds - 1)


def fold_split(fold: int, n_folds: int) -> Dict[int, str]:
    """Map fold index -> split name for cross-validation round `fold`.

    Fold `f` is the test set and fold `(f+1) mod k` is validation, so every fold serves as
    test exactly once and as validation exactly once, and validation is never drawn from
    the test fold. Model selection therefore never sees the fold it is scored on.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} outside [0, {n_folds})")
    out = {i: "train" for i in range(n_folds)}
    out[fold] = "test"
    out[(fold + 1) % n_folds] = "val"
    return out


def load_split_spec(data_dir: Path) -> dict:
    import yaml

    path = Path(data_dir) / "splits" / "split_spec.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"split spec not found: {path}")
    return yaml.safe_load(path.read_text())


@dataclass
class IndexReport:
    n_headers: int = 0
    n_indexed: int = 0
    n_missing_signal: int = 0
    n_bad_geometry: int = 0
    n_no_dx: int = 0
    n_unassigned: int = 0     # matched no class bucket under a null-fallback ordering
    rejected: Dict[str, List[str]] = None  # reason -> record ids (capped)
    unknown_codes: Counter = None
    label_counts: Counter = None
    split_counts: Dict[str, Counter] = None
    unassigned_codes: Counter = None

    def __post_init__(self) -> None:
        self.rejected = defaultdict(list)
        self.unknown_codes = Counter()
        self.label_counts = Counter()
        self.split_counts = {s: Counter() for s in SPLITS}
        self.unassigned_codes = Counter()

    def to_dict(self) -> dict:
        return {
            "n_headers": self.n_headers,
            "n_indexed": self.n_indexed,
            "n_missing_signal": self.n_missing_signal,
            "n_bad_geometry": self.n_bad_geometry,
            "n_no_dx": self.n_no_dx,
            "n_unassigned": self.n_unassigned,
            "rejected_examples": {k: v[:20] for k, v in self.rejected.items()},
            "unknown_snomed_codes": dict(self.unknown_codes.most_common(50)),
            # Which SNOMED codes the excluded records carried. An ordering that declares no
            # fallback drops whatever it cannot claim, so this is the audit trail for what
            # left the corpus and why.
            "unassigned_snomed_codes": dict(self.unassigned_codes.most_common(50)),
            "label_counts": dict(self.label_counts),
            "split_counts": {s: dict(c) for s, c in self.split_counts.items()},
        }


def build_index(
    chapman_root: Path,
    data_dir: Path,
    resolution_order: str = "clinical_specificity",
    strict_geometry: bool = True,
    cv_fold: Optional[int] = None,
    n_folds: int = 0,
) -> Tuple[List[Record], IndexReport, ClassMap]:
    """Scan CHAPMAN_ROOT, resolve labels, assign splits. No signal is read."""
    chapman_root = Path(chapman_root)
    if not chapman_root.is_dir():
        raise FileNotFoundError(f"CHAPMAN_ROOT does not exist: {chapman_root}")

    cmap = get_class_map(data_dir, resolution_order)
    vocab = load_snomed_vocabulary(data_dir)
    known = set(vocab)
    spec = load_split_spec(data_dir)
    salt = spec["hash"]["salt"]
    hex_chars = int(spec["hash"]["digest_hex_chars"])
    props = {k: float(v) for k, v in spec["proportions"].items()}

    report = IndexReport()
    by_class: Dict[str, List[Record]] = defaultdict(list)

    headers = sorted(chapman_root.rglob("*.hea"))
    report.n_headers = len(headers)
    if not headers:
        raise FileNotFoundError(f"no .hea files under {chapman_root}")

    for hea in headers:
        rid = hea.stem
        mat = hea.with_suffix(".mat")
        dat = hea.with_suffix(".dat")
        sig = mat if mat.is_file() else (dat if dat.is_file() else None)
        if sig is None:
            report.n_missing_signal += 1
            report.rejected["missing_signal_file"].append(rid)
            continue
        try:
            n_sig, fs, n_samp, leads = _parse_header_geometry(hea)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the scan
            report.n_bad_geometry += 1
            report.rejected[f"unreadable_header:{type(exc).__name__}"].append(rid)
            continue

        if strict_geometry and n_sig != N_LEADS:
            report.n_bad_geometry += 1
            report.rejected["not_12_lead"].append(rid)
            continue

        codes = parse_dx_codes(hea)
        if not codes:
            report.n_no_dx += 1
            report.rejected["no_dx_field"].append(rid)
            continue

        res = cmap.resolve(codes, record_id=rid, known_codes=known)
        for c in res.unknown_codes:
            report.unknown_codes[c] += 1

        if not res.is_assigned:
            # The ordering declares no fallback and this record matched no bucket. Dropping
            # it is the honest outcome -- the alternative is to label it with a class whose
            # definition it does not meet -- but it must be counted and its codes recorded.
            report.n_unassigned += 1
            report.rejected["unassigned_no_matching_class"].append(rid)
            for c in res.all_codes:
                report.unassigned_codes[c] += 1
            continue

        rec = Record(
            record_id=rid,
            header_path=hea,
            signal_path=sig,
            fs_hz=fs,
            n_samples=n_samp,
            n_leads=n_sig,
            codes=res.all_codes,
            label=res.label,
            label_index=res.label_index,
        )
        by_class[res.label].append(rec)
        report.label_counts[res.label] += 1

    # Stratified assignment: bucket within each class so proportions hold per class.
    # In cross-validation mode the same hash instead selects a fold, and `fold_split` turns
    # the requested round into train/val/test. Both paths are pure functions of record id.
    use_cv = cv_fold is not None and n_folds >= 2
    fold_to_split = fold_split(cv_fold, n_folds) if use_cv else {}
    records: List[Record] = []
    for label, recs in by_class.items():
        for rec in recs:
            if use_cv:
                rec.split = fold_to_split[assign_fold(rec.record_id, salt, n_folds, hex_chars)]
            else:
                rec.split = assign_split(rec.record_id, salt, props, hex_chars)
            report.split_counts[rec.split][label] += 1
            records.append(rec)

    records.sort(key=lambda r: r.record_id)
    report.n_indexed = len(records)
    if not records:
        raise RuntimeError(
            f"indexed 0 usable records from {report.n_headers} headers under "
            f"{chapman_root}; see the rejection reasons in the index report."
        )
    return records, report, cmap


# ---------------------------------------------------------------------------
# Signal loading + cache
# ---------------------------------------------------------------------------
def _preproc_key(cfg: PreprocessConfig, n_samples: int) -> str:
    payload = json.dumps(
        {
            "low": cfg.bandpass_low_hz, "high": cfg.bandpass_high_hz,
            "order": cfg.bandpass_order, "notch": cfg.notch_freq_hz,
            "q": cfg.notch_q, "eps": cfg.zscore_eps, "apply_notch": cfg.apply_notch,
            "n": n_samples, "v": 2,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def read_signal(record: Record) -> Tuple[np.ndarray, float]:
    """Return (12, n) physical-unit signal and fs, using wfdb so ADC gain/baseline apply."""
    import wfdb

    rec = wfdb.rdrecord(str(record.header_path.with_suffix("")))
    sig = np.asarray(rec.p_signal, dtype=np.float64).T  # (leads, samples)
    fs = float(getattr(rec, "fs", record.fs_hz) or record.fs_hz)
    return sig, fs


class ECGDataset:
    """Map-style dataset over indexed records.

    Deliberately NOT a subclass of `torch.utils.data.Dataset`. Two reasons:

    * `labels` and `preprocess` must stay importable without torch, so the unit test tier
      runs in a minimal environment. Subclassing would force a torch import at module load.
    * DataLoader only requires `__getitem__` and `__len__` for a map-style dataset; it does
      not check `isinstance(..., Dataset)`. Building a subclass dynamically inside a factory
      function (the obvious workaround) produces a class that `pickle` cannot resolve by
      qualname, which breaks `num_workers > 0` on any platform that spawns rather than forks
      workers -- Windows and macOS. This class is module-level and pickles cleanly.
    """

    def __init__(
        self,
        records: Sequence[Record],
        preprocess_cfg: Optional[PreprocessConfig] = None,
        cache_dir: Optional[Path] = None,
        n_samples: int = N_SAMPLES,
        augment: bool = False,
        rng_seed: int = 42,
        augment_strength: str = "basic",
        rate_defined_classes: Sequence[str] = ("NSR", "SB", "ST", "SVT"),
    ) -> None:
        self.records = list(records)
        self.cfg = preprocess_cfg or PreprocessConfig()
        self.n_samples = n_samples
        self.augment = augment
        if augment_strength not in ("basic", "physio"):
            raise ValueError(f"augment_strength must be basic|physio, got {augment_strength!r}")
        self.augment_strength = augment_strength
        self.rate_defined_classes = tuple(rate_defined_classes)
        self._rng = np.random.default_rng(rng_seed)
        self.cache_dir: Optional[Path] = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / _preproc_key(self.cfg, n_samples)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rec: Record) -> np.ndarray:
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{rec.record_id}.npy"
            if cached.is_file():
                return np.load(cached)
        sig, fs = read_signal(rec)
        x, _ = preprocess_signal(sig, fs=fs, cfg=self.cfg, n_samples=self.n_samples)
        if self.cache_dir is not None:
            # Write to a unique temp file, then rename: concurrent DataLoader workers
            # touch the same record (the balanced training split repeats records), so a
            # shared temp name would let one worker's partial write clobber another's.
            # np.save() appends '.npy' unless the path already ends in it, so the handle
            # is opened explicitly rather than passing a '.tmp' path.
            final = self.cache_dir / f"{rec.record_id}.npy"
            tmp = self.cache_dir / f"{rec.record_id}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with tmp.open("wb") as fh:
                    np.save(fh, x, allow_pickle=False)
                os.replace(tmp, final)
            except OSError:
                tmp.unlink(missing_ok=True)   # a cache failure must not fail the epoch
        return x

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Signal-level jitter used only for training-split balancing."""
        which = int(self._rng.integers(0, 3))
        if which == 0:
            x = x + 0.01 * self._rng.standard_normal(x.shape).astype(np.float32)
        elif which == 1:
            x = x * np.float32(self._rng.uniform(0.9, 1.1))
        else:
            shift = int(self._rng.integers(-int(0.05 * x.shape[-1]), int(0.05 * x.shape[-1]) + 1))
            x = np.roll(x, shift, axis=-1)
        return x.astype(np.float32)

    def _augment_physio(self, x: np.ndarray, label: str) -> np.ndarray:
        """Physiologically motivated augmentation (12, T).

        Each transform models a real source of variation in a recorded ECG, so an augmented
        copy is a plausible recording of the same patient rather than an arbitrary
        perturbation. All are applied independently with their own probability; unlike the
        `basic` jitter this composes, so the copies are not near-duplicates of each other.

        Time warping is SUPPRESSED for the rate-defined classes. Stretching the time axis
        of a sinus bradycardia recording changes its heart rate, which is the exact quantity
        that defines the label -- a 55 bpm strip warped by 1.15x is 63 bpm and is no longer
        bradycardic. That is label corruption, not augmentation.
        """
        rng = self._rng
        x = np.asarray(x, dtype=np.float32)
        n_leads, n = x.shape

        # 1. Per-lead amplitude scaling: electrode placement and body habitus change lead
        #    gains independently, not globally.
        if rng.random() < 0.5:
            x = x * rng.uniform(0.85, 1.15, size=(n_leads, 1)).astype(np.float32)

        # 2. Lead dropout: a disconnected or saturated electrode is a routine artefact, and
        #    forcing the model to survive it discourages reliance on any single lead.
        if rng.random() < 0.3:
            k = int(rng.integers(1, 3))
            x = x.copy()
            x[rng.choice(n_leads, size=k, replace=False), :] = 0.0

        # 3. Baseline wander: respiration and motion, 0.15-0.5 Hz. The 0.5 Hz high-pass in
        #    preprocessing attenuates but does not remove it.
        if rng.random() < 0.4:
            f = float(rng.uniform(0.15, 0.5))
            t = np.arange(n, dtype=np.float32) / 500.0
            phase = rng.uniform(0, 2 * np.pi, size=(n_leads, 1)).astype(np.float32)
            x = x + (0.05 * np.sin(2 * np.pi * f * t[None, :] + phase)).astype(np.float32)

        # 4. Additive noise at a realistic amplitude relative to the z-scored signal.
        if rng.random() < 0.5:
            x = x + (rng.uniform(0.005, 0.02) *
                     rng.standard_normal(x.shape)).astype(np.float32)

        # 5. Time warp -- rate-defined classes excluded, see the docstring.
        if label not in self.rate_defined_classes and rng.random() < 0.3:
            factor = float(rng.uniform(0.9, 1.1))
            src = np.linspace(0.0, n - 1.0, int(round(n * factor)), dtype=np.float32)
            warped = np.stack([np.interp(src, np.arange(n, dtype=np.float32), ch)
                               for ch in x]).astype(np.float32)
            if warped.shape[1] >= n:
                start = int(rng.integers(0, warped.shape[1] - n + 1))
                x = warped[:, start:start + n]
            else:
                pad = n - warped.shape[1]
                x = np.pad(warped, ((0, 0), (0, pad)), mode="edge")

        # 6. Circular shift: R-peak phase within the 10 s window carries no diagnostic
        #    information, so the model should be invariant to it.
        if rng.random() < 0.5:
            x = np.roll(x, int(rng.integers(-n // 20, n // 20 + 1)), axis=-1)

        return np.ascontiguousarray(x, dtype=np.float32)

    def __getitem__(self, idx: int):
        import torch

        rec = self.records[idx]
        x = self._load(rec)
        if self.augment:
            x = (self._augment_physio(x, rec.label)
                 if self.augment_strength == "physio" else self._augment(x))
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.tensor(rec.label_index, dtype=torch.long),
            idx,
        )


def make_torch_dataset(*args, **kwargs) -> "ECGDataset":
    """Build a dataset for DataLoader.

    Returns a plain `ECGDataset`. See that class's docstring for why it is not wrapped in a
    dynamically-created `torch.utils.data.Dataset` subclass: such a class is unpicklable and
    would make `num_workers > 0` fail on spawn-based platforms (Windows, macOS).
    """
    return ECGDataset(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training-split balancing
# ---------------------------------------------------------------------------
@dataclass
class BalanceReport:
    unique_support: Dict[str, int]
    support: Dict[str, int]
    duplication_factor: Dict[str, float]
    target: int
    strategy: str

    def to_dict(self) -> dict:
        return {
            "unique_support": self.unique_support,
            "support": self.support,
            "duplication_factor": self.duplication_factor,
            "target": self.target,
            "strategy": self.strategy,
        }


def balance_indices(
    records: Sequence[Record],
    spec: dict,
    rng_seed: int = 42,
    strategy: Optional[str] = None,
) -> Tuple[List[int], BalanceReport]:
    """Oversample the training split up to a capped target. Returns indices INTO `records`.

    Refuses, loudly, to inflate a class with fewer than `min_unique_samples_per_class`
    unique records: multiplying 1 record into 1000 does not create a class, it creates a
    memorised template that inflates balanced accuracy without any diagnostic ability.

    `strategy="none"` returns the natural training distribution untouched -- every record
    exactly once. Use it together with `TrainConfig.class_weighting`, which corrects the
    prior in the loss instead of in the sampler. On the completed run the duplication route
    gave OTHER 5.6 copies of each of its 184 records and left it with precision 0.468
    against recall 0.595: the model over-predicted a class it had seen 5.6 times over.
    """
    aug = spec["augmentation"]
    class_names = active_class_names()

    if strategy == "none":
        counts_all = Counter(r.label for r in records)
        return (
            list(range(len(records))),
            BalanceReport(
                unique_support={c: counts_all.get(c, 0) for c in class_names},
                support={c: counts_all.get(c, 0) for c in class_names},
                duplication_factor={c: 1.0 for c in class_names},
                target=0,
                strategy="none_natural_distribution",
            ),
        )

    min_unique = int(aug.get("min_unique_samples_per_class", 0))
    max_dup = float(aug.get("max_duplication_factor", 8))
    rng = np.random.default_rng(rng_seed)

    by_class: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_class[r.label].append(i)

    counts = {c: len(v) for c, v in by_class.items()}
    present = {c: n for c, n in counts.items() if n > 0}
    if not present:
        raise RuntimeError("training split is empty")

    too_small = {c: n for c, n in present.items() if n < min_unique}
    if too_small and str(aug.get("on_below_minimum", "error")).lower() == "error":
        raise RuntimeError(
            "Refusing to balance: class(es) "
            + ", ".join(f"{c} (n={n})" for c, n in sorted(too_small.items()))
            + f" have fewer than {min_unique} unique training records.\n\n"
              "Oversampling them to parity would make the class TRAINABLE without making "
              "it EVALUABLE: the test split is never augmented, so it still holds only "
              "~15% of that same handful, and a per-class metric computed on it is noise. "
              "Balancing the training split cannot fix a class that is too small to "
              "measure.\n\n"
              "Diagnose it first -- this is usually a labelling artefact, not a rare "
              "class:\n"
              "    python -m ecgvit.cli label-audit\n\n"
              "That compares every resolution order on your corpus and shows how many "
              "records CARRY each class's codes versus how many the ordering actually "
              "awards it. Then either switch --resolution-order (conduction_first gives "
              "CD its full population), merge the class into OTHER and report 6 classes, "
              "or lower min_unique_samples_per_class in split_spec.yaml and report that "
              "class's metric as not evaluable. See docs/CLASS_MAPPING.md."
        )

    target_mode = str(aug.get("target", "median_class_count"))
    values = sorted(present.values())
    if target_mode == "median_class_count":
        target = int(np.median(values))
    elif target_mode == "max_class_count":
        target = int(max(values))
    else:
        target = int(target_mode)

    out: List[int] = []
    support: Dict[str, int] = {}
    for cls in class_names:
        idxs = by_class.get(cls, [])
        if not idxs:
            support[cls] = 0
            continue
        cap = int(len(idxs) * max_dup)
        want = min(max(target, len(idxs)), cap)
        out.extend(idxs)
        extra = want - len(idxs)
        if extra > 0:
            out.extend(rng.choice(idxs, size=extra, replace=True).tolist())
        support[cls] = want

    rng.shuffle(out)
    report = BalanceReport(
        unique_support={c: counts.get(c, 0) for c in class_names},
        support={c: support.get(c, 0) for c in class_names},
        duplication_factor={
            c: round(support.get(c, 0) / counts[c], 3) if counts.get(c) else 0.0
            for c in class_names
        },
        target=target,
        strategy=str(aug.get("strategy", "capped_oversample_with_jitter")),
    )
    return out, report


def class_priors(records: Sequence[Record], n_classes: int) -> np.ndarray:
    """Empirical class prior over `records`, as a length-`n_classes` float array.

    Used by logit adjustment and inverse-frequency weighting. A class with no records gets
    a floor of 1 rather than 0, so log(prior) stays finite; that class is unlearnable
    either way and the floor keeps the failure visible instead of producing -inf logits.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for r in records:
        if 0 <= r.label_index < n_classes:
            counts[r.label_index] += 1
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def split_records(records: Sequence[Record]) -> Dict[str, List[Record]]:
    out: Dict[str, List[Record]] = {s: [] for s in SPLITS}
    for r in records:
        out[r.split].append(r)
    return out


def write_index_csv(records: Sequence[Record], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].to_row().keys()))
        w.writeheader()
        for r in records:
            w.writerow(r.to_row())


__all__ = [
    "Record", "IndexReport", "BalanceReport", "SPLITS",
    "build_index", "assign_split", "assign_fold", "fold_split", "load_split_spec",
    "read_signal", "class_priors",
    "ECGDataset", "make_torch_dataset", "balance_indices", "split_records",
    "write_index_csv",
]

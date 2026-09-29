"""Record indexing, deterministic splitting, and leakage-free balancing."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest

from ecgvit.config import CLASS_NAMES
from ecgvit.data import (assign_split, balance_indices, build_index,
                         load_split_spec, split_records)


# ---------------------------------------------------------------------------
# Split determinism
# ---------------------------------------------------------------------------
def test_split_assignment_is_a_pure_function_of_the_record_id(data_dir):
    spec = load_split_spec(data_dir)
    salt = spec["hash"]["salt"]
    props = {k: float(v) for k, v in spec["proportions"].items()}
    for rid in ("JS00001", "JS04242", "FX00007", "anything"):
        a = assign_split(rid, salt, props)
        b = assign_split(rid, salt, props)
        assert a == b
        assert a in ("train", "val", "test")


def test_split_proportions_hold_over_many_ids(data_dir):
    spec = load_split_spec(data_dir)
    salt = spec["hash"]["salt"]
    props = {k: float(v) for k, v in spec["proportions"].items()}
    counts = Counter(assign_split(f"JS{i:05d}", salt, props) for i in range(20000))
    for name, want in props.items():
        got = counts[name] / 20000
        assert abs(got - want) < 0.02, f"{name}: {got:.3f} vs target {want:.2f}"


def test_a_different_salt_produces_a_different_split(data_dir):
    props = {"train": 0.7, "val": 0.15, "test": 0.15}
    ids = [f"JS{i:05d}" for i in range(2000)]
    a = [assign_split(i, "salt-a", props) for i in ids]
    b = [assign_split(i, "salt-b", props) for i in ids]
    assert a != b


def test_adding_records_does_not_reassign_existing_ones(data_dir):
    """The point of hash bucketing over a stored shuffle: growing the corpus must not
    move a record from test into train, which would invalidate every earlier result."""
    spec = load_split_spec(data_dir)
    salt, props = spec["hash"]["salt"], {k: float(v) for k, v in spec["proportions"].items()}
    before = {f"JS{i:05d}": assign_split(f"JS{i:05d}", salt, props) for i in range(100)}
    after = {f"JS{i:05d}": assign_split(f"JS{i:05d}", salt, props) for i in range(500)}
    for rid, split in before.items():
        assert after[rid] == split


# ---------------------------------------------------------------------------
# Indexing against real files
# ---------------------------------------------------------------------------
def test_index_builds_from_fixture_corpus(fixture_corpus, fixture_labels, data_dir):
    records, report, cmap = build_index(fixture_corpus, data_dir, "rhythm_first")
    assert len(records) == len(fixture_labels)
    got = {r.record_id: r.label for r in records}
    assert got == fixture_labels


def test_index_rejects_malformed_records_with_reasons(fixture_corpus, data_dir):
    """The fixture corpus deliberately contains three broken records. Each must be
    rejected for the right reason, not silently coerced into the training set."""
    records, report, _ = build_index(fixture_corpus, data_dir)
    ids = {r.record_id for r in records}

    assert "FX90001" not in ids and report.n_no_dx >= 1
    assert "FX90001" in report.rejected["no_dx_field"]

    assert "FX90002" not in ids and report.n_bad_geometry >= 1
    assert "FX90002" in report.rejected["not_12_lead"]

    assert "FX90003" not in ids and report.n_missing_signal >= 1
    assert "FX90003" in report.rejected["missing_signal_file"]

    assert report.n_headers == len(records) + 3


def test_index_records_carry_correct_geometry(fixture_corpus, data_dir):
    records, _, _ = build_index(fixture_corpus, data_dir)
    for r in records:
        assert r.n_leads == 12
        assert r.fs_hz == 500.0
        assert r.n_samples == 5000
        assert r.signal_path.is_file()


def test_index_is_stable_across_calls(fixture_corpus, data_dir):
    a, _, _ = build_index(fixture_corpus, data_dir)
    b, _, _ = build_index(fixture_corpus, data_dir)
    assert [(r.record_id, r.label, r.split) for r in a] == \
           [(r.record_id, r.label, r.split) for r in b]


def test_index_raises_on_a_missing_root(tmp_path, data_dir):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope", data_dir)


def test_index_raises_when_no_headers_present(tmp_path, data_dir):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .hea"):
        build_index(tmp_path / "empty", data_dir)


# ---------------------------------------------------------------------------
# Balancing: the leakage guard
# ---------------------------------------------------------------------------
def _fake_records(counts):
    from ecgvit.data import Record
    from pathlib import Path

    out, i = [], 0
    for cls, n in counts.items():
        for _ in range(n):
            out.append(
                Record(f"R{i:05d}", Path("x.hea"), Path("x.mat"), 500.0, 5000, 12,
                       (), cls, CLASS_NAMES.index(cls), "train")
            )
            i += 1
    return out


def test_balancing_only_duplicates_and_never_invents(data_dir):
    spec = load_split_spec(data_dir)
    recs = _fake_records({"NSR": 300, "AFIB": 200, "SB": 500, "ST": 180,
                          "SVT": 120, "CD": 150, "OTHER": 100})
    idxs, rep = balance_indices(recs, spec, rng_seed=42)
    assert set(idxs) <= set(range(len(recs)))
    assert len(set(idxs)) == len(recs), "every original record must still appear"
    for cls in CLASS_NAMES:
        assert rep.support[cls] >= rep.unique_support[cls]


def test_balancing_reports_duplication_so_it_cannot_hide(data_dir):
    spec = load_split_spec(data_dir)
    recs = _fake_records({"NSR": 300, "AFIB": 200, "SB": 500, "ST": 180,
                          "SVT": 120, "CD": 150, "OTHER": 60})
    _, rep = balance_indices(recs, spec, rng_seed=42)
    d = rep.to_dict()
    assert set(d) >= {"unique_support", "support", "duplication_factor"}
    assert d["duplication_factor"]["OTHER"] > 1.0
    assert d["duplication_factor"]["SB"] == pytest.approx(1.0)


def test_balancing_refuses_to_fabricate_a_class_from_one_record(data_dir):
    """The manuscript's Chapman distribution has CD: 1 and OTHER: 17. Oversampling those
    to 1098 each would produce an impressive balanced accuracy that measures nothing.
    The pipeline must refuse rather than comply."""
    spec = load_split_spec(data_dir)
    recs = _fake_records({"NSR": 800, "AFIB": 600, "SB": 900, "ST": 500,
                          "SVT": 400, "CD": 1, "OTHER": 17})
    with pytest.raises(RuntimeError, match="Refusing to balance"):
        balance_indices(recs, spec, rng_seed=42)


def test_duplication_is_capped(data_dir):
    spec = copy.deepcopy(load_split_spec(data_dir))
    spec["augmentation"]["max_duplication_factor"] = 3
    spec["augmentation"]["min_unique_samples_per_class"] = 10
    recs = _fake_records({"NSR": 900, "AFIB": 900, "SB": 900, "ST": 900,
                          "SVT": 900, "CD": 900, "OTHER": 50})
    _, rep = balance_indices(recs, spec, rng_seed=42)
    assert rep.support["OTHER"] <= 50 * 3
    assert rep.duplication_factor["OTHER"] <= 3.0


def test_balancing_is_deterministic_for_a_given_seed(data_dir):
    spec = load_split_spec(data_dir)
    recs = _fake_records({c: 200 for c in CLASS_NAMES})
    a, _ = balance_indices(recs, spec, rng_seed=7)
    b, _ = balance_indices(recs, spec, rng_seed=7)
    c, _ = balance_indices(recs, spec, rng_seed=8)
    assert a == b and a != c


def test_split_spec_applies_balancing_to_train_only(data_dir):
    """Guard on the spec itself. If someone adds 'val' or 'test' to apply_to, augmented
    duplicates reach the evaluation split and every reported metric becomes meaningless."""
    spec = load_split_spec(data_dir)
    assert spec["augmentation"]["apply_to"] == ["train"]


# ---------------------------------------------------------------------------
# DataLoader worker compatibility
# ---------------------------------------------------------------------------
def test_dataset_is_picklable_for_spawned_workers(fixture_corpus, data_dir, tmp_path):
    """Windows and macOS spawn DataLoader workers instead of forking, which pickles the
    dataset. A dataset built from a dynamically-created class raises
    'Can't pickle local object' there while working fine on Linux -- so this must be
    checked explicitly, not left to whoever runs it on a laptop first.
    """
    import pickle

    from ecgvit.data import make_torch_dataset

    records, _, _ = build_index(fixture_corpus, data_dir)
    ds = make_torch_dataset(records, cache_dir=tmp_path / "cache")
    restored = pickle.loads(pickle.dumps(ds))
    assert len(restored) == len(ds)
    assert restored.records[0].record_id == ds.records[0].record_id


def test_dataset_loads_through_spawned_workers(fixture_corpus, data_dir, tmp_path):
    """End-to-end guard: run a DataLoader under the spawn start method, which is what
    Windows uses, and confirm batches actually arrive."""
    import multiprocessing as mp

    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from ecgvit.data import make_torch_dataset

    records, _, _ = build_index(fixture_corpus, data_dir)
    ds = make_torch_dataset(records[:8], cache_dir=tmp_path / "cache")
    loader = DataLoader(
        ds, batch_size=4, num_workers=2,
        multiprocessing_context=mp.get_context("spawn"),
    )
    batches = list(loader)
    assert len(batches) == 2
    x, y, idx = batches[0]
    assert x.shape == (4, 12, 5000)
    assert y.shape == (4,) and idx.shape == (4,)


def test_splits_are_disjoint(fixture_corpus, data_dir):
    records, _, _ = build_index(fixture_corpus, data_dir)
    by = split_records(records)
    ids = {k: {r.record_id for r in v} for k, v in by.items()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == len(records)

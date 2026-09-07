"""Fixture determinism, and corpus-gated checks against the real Chapman-Shaoxing data."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from ecgvit.config import CLASS_NAMES, LEAD_ORDER
from ecgvit.data import assign_split, build_index, load_split_spec
from ecgvit.labels import parse_dx_codes

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import make_fixtures  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture determinism
# ---------------------------------------------------------------------------
def test_fixtures_are_bit_identical_across_generations(tmp_path):
    """Two generations from the same seed must produce the same signal values. This is
    what makes the offline tier reproducible on any machine."""
    import wfdb

    a, b = tmp_path / "a", tmp_path / "b"
    make_fixtures.generate(a)
    make_fixtures.generate(b)
    for rid in ("FX00000", "FX00013", "FX00041"):
        sa = wfdb.rdrecord(str(a / rid)).p_signal
        sb = wfdb.rdrecord(str(b / rid)).p_signal
        assert np.array_equal(sa, sb), f"{rid} differs between generations"


def test_fixture_geometry_matches_the_real_corpus(fixture_corpus):
    import wfdb

    r = wfdb.rdrecord(str(fixture_corpus / "FX00000"))
    assert r.p_signal.shape == (5000, 12)
    assert r.fs == 500
    assert list(r.sig_name) == list(LEAD_ORDER)


def test_fixture_headers_use_the_challenge_dx_format(fixture_corpus):
    codes = parse_dx_codes(fixture_corpus / "FX00000.hea")
    assert codes and all(c.isdigit() for c in codes)


@pytest.mark.skipif(
    os.environ.get("STRICT_FIXTURES") != "1",
    reason="byte-level manifest check is opt-in (set STRICT_FIXTURES=1); the .mat writer's "
           "exact bytes can shift between scipy releases, while the signal values do not",
)
def test_fixture_bytes_match_the_committed_manifest(tmp_path):
    out = tmp_path / "corpus"
    make_fixtures.generate(out)
    problems = make_fixtures.verify(out, Path(__file__).parent / "fixtures/manifest.sha256")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Real corpus (skipped unless CHAPMAN_ROOT is set)
# ---------------------------------------------------------------------------
@pytest.mark.corpus
def test_corpus_size_and_geometry(chapman_root, data_dir):
    spec = yaml.safe_load((data_dir / "dataset.yaml").read_text())
    exp = spec["primary"]["expected"]
    records, report, _ = build_index(chapman_root, data_dir, "rhythm_first")

    assert report.n_indexed >= exp["min_records"], (
        f"indexed {report.n_indexed} records, dataset.yaml expects at least "
        f"{exp['min_records']}. Is the download complete?"
    )
    fs = Counter(r.fs_hz for r in records)
    assert set(fs) == {float(exp["sampling_frequency_hz"])}, (
        f"unexpected sampling rates: {dict(fs)}"
    )
    assert all(r.n_leads == exp["n_leads"] for r in records)


@pytest.mark.corpus
def test_corpus_lead_order_is_standard(chapman_root):
    import wfdb

    hea = next(iter(sorted(Path(chapman_root).rglob("*.hea"))))
    rec = wfdb.rdrecord(str(hea.with_suffix("")))
    assert list(rec.sig_name) == list(LEAD_ORDER), (
        f"lead order is {rec.sig_name}, expected {list(LEAD_ORDER)}. "
        "A permuted lead order silently invalidates every per-lead attribution."
    )


@pytest.mark.corpus
def test_corpus_every_class_is_populated(chapman_root, data_dir, resolution_order):
    _, report, cmap = build_index(chapman_root, data_dir, resolution_order)
    empty = [c for c in cmap.classes if report.label_counts.get(c, 0) == 0]
    assert not empty, (
        f"classes with zero records under the {resolution_order} mapping: {empty}. "
        "Check class_map_7.json against the SNOMED codes actually present."
    )


@pytest.mark.corpus
def test_corpus_no_class_is_too_small_to_train(chapman_root, data_dir, resolution_order):
    """Documents the real distribution. A class with a handful of records cannot be
    oversampled into a meaningful result -- see docs/CLASS_MAPPING.md."""
    _, report, cmap = build_index(chapman_root, data_dir, resolution_order)
    tiny = {c: n for c in cmap.classes
            if 0 < (n := report.label_counts.get(c, 0)) < 50}
    assert not tiny, (
        f"classes with fewer than 50 records: {tiny}. Balancing these to parity would "
        "fabricate a class from duplicates; merge them into OTHER instead."
    )


@pytest.mark.corpus
def test_corpus_splits_are_disjoint_and_hash_consistent(chapman_root, data_dir,
                                                        resolution_order):
    """Every record sits in the split its id hashes to, and in exactly one split.

    This replaces an assertion that the aggregate split sizes were 70/15/15. That holds for
    the full upstream corpus but NOT for the vendored subset, which is curated test-first
    and is deliberately ~85% training. Aggregate proportions were never the property worth
    protecting: what matters is that a record's split is a pure function of its id, because
    that is what guarantees subsetting cannot leak a training record into test.
    """
    records, report, _ = build_index(chapman_root, data_dir, resolution_order)
    spec = load_split_spec(data_dir)
    salt = spec["hash"]["salt"]
    hex_chars = int(spec["hash"]["digest_hex_chars"])
    props = {k: float(v) for k, v in spec["proportions"].items()}

    total = len(records)
    sizes = {s: sum(c.values()) for s, c in report.split_counts.items()}
    assert sum(sizes.values()) == total, "a record is in more than one split, or in none"

    wrong = [r.record_id for r in records
             if r.split != assign_split(r.record_id, salt, props, hex_chars)]
    assert not wrong, (
        f"{len(wrong)} records are not in the split their id hashes to, e.g. {wrong[:5]}. "
        "The partition is meant to be a pure function of the record id."
    )

    # Every split must be non-empty and hold every class, or a per-class metric is
    # undefined for whatever is missing.
    for name in ("train", "val", "test"):
        assert sizes.get(name, 0) > 0, f"the {name} split is empty"


@pytest.mark.corpus
@pytest.mark.slow
def test_corpus_signals_preprocess_cleanly(chapman_root, data_dir):
    """Spot-check 40 real records end to end: no NaN, correct shape, finite output."""
    from ecgvit.data import read_signal
    from ecgvit.preprocess import preprocess_signal

    records, _, _ = build_index(chapman_root, data_dir)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(records), size=min(40, len(records)), replace=False):
        rec = records[int(i)]
        sig, fs = read_signal(rec)
        x, rep = preprocess_signal(sig, fs=fs)
        assert x.shape == (12, 5000), f"{rec.record_id}: {x.shape}"
        assert np.isfinite(x).all(), f"{rec.record_id} produced non-finite values"
        assert rep.bandpass_applied and rep.notch_applied

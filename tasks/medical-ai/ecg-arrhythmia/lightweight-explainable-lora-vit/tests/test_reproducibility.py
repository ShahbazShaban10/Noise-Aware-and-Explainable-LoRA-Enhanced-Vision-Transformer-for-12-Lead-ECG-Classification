"""Cross-machine reproducibility.

The other test files check that the code is self-consistent *on the machine running them*.
These check something different and stronger: that this machine agrees with the golden values
committed to the repository. Self-consistency is cheap — a pipeline that shuffles differently
on every OS is perfectly self-consistent and still unreproducible.

What is pinned here, and what deliberately is not:

  pinned    split assignment, class mapping, filter response, parameter counts,
            preprocessed signal values
  NOT       trained weights and metrics. cuDNN autotuning and non-deterministic reduction
            kernels mean two runs on the same GPU differ in the third decimal. Pinning them
            would produce a test that fails for no reason; see the note in task.toml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ecgvit.config import LEAD_ORDER, PreprocessConfig
from ecgvit.data import assign_split, load_split_spec
from ecgvit.preprocess import preprocess_signal

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_split_assignments_match_the_golden_file(data_dir, expected_dir):
    """Every machine must place the same record in the same split.

    If this fails, the split logic, the salt or the proportions changed. Any result reported
    before the change is no longer comparable to any result after it — the test set is
    literally a different set of records.
    """
    golden = json.loads((expected_dir / "split_assignments.expected.json").read_text())
    spec = load_split_spec(data_dir)

    assert spec["hash"]["salt"] == golden["salt"], (
        "the split salt changed; this reshuffles every record in the corpus"
    )
    props = {k: float(v) for k, v in spec["proportions"].items()}
    assert props == golden["proportions"], "split proportions changed"

    mismatches = {
        rid: (want, got)
        for rid, want in golden["assignments"].items()
        if (got := assign_split(rid, golden["salt"], props)) != want
    }
    assert not mismatches, (
        f"{len(mismatches)} record(s) landed in a different split than the golden file: "
        f"{dict(list(mismatches.items())[:5])}"
    )


def test_split_golden_covers_all_three_splits(expected_dir):
    """A golden file where every id happens to land in `train` would pass trivially."""
    golden = json.loads((expected_dir / "split_assignments.expected.json").read_text())
    assert set(golden["assignments"].values()) == {"train", "val", "test"}
    assert len(golden["assignments"]) >= 50


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_preprocessed_signal_matches_the_golden_file(fixture_corpus, expected_dir):
    """The denoising pipeline must produce the same numbers here as it did on the authoring
    machine. Tolerance is 2e-3: far tighter than any genuine change to the filter design,
    and looser than float and BLAS variation across platforms and SciPy builds.
    """
    import wfdb

    golden = json.loads((expected_dir / "preprocessed_fixture.expected.json").read_text())
    raw = wfdb.rdrecord(str(fixture_corpus / golden["record"])).p_signal.T
    x, rep = preprocess_signal(raw, fs=golden["fs_hz"], cfg=PreprocessConfig())

    assert list(x.shape) == golden["shape"]
    assert rep.bandpass_applied and rep.notch_applied and rep.zerophase

    problems = []
    for i, lead in enumerate(LEAD_ORDER):
        want = golden["per_lead"][lead]
        got = {
            "mean": float(x[i].mean()),
            "std": float(x[i].std()),
            "min": float(x[i].min()),
            "max": float(x[i].max()),
            "abs_mean": float(np.abs(x[i]).mean()),
        }
        for key, w in want.items():
            if key == "samples_at_500_1500_2500_3500":
                for k, wv in zip((500, 1500, 2500, 3500), w):
                    if abs(float(x[i][k]) - wv) > 2e-3:
                        problems.append(f"{lead}[{k}]: {float(x[i][k]):.4f} vs {wv:.4f}")
            elif abs(got[key] - w) > 2e-3:
                problems.append(f"{lead}.{key}: {got[key]:.6f} vs {w:.6f}")

    assert not problems, (
        "preprocessed signal differs from the golden values:\n  "
        + "\n  ".join(problems[:12])
        + "\n\nThe filter design or its application changed. Regenerate the golden file "
          "only if that change was intentional."
    )


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def test_parameter_counts_are_stable():
    """The three counts the manuscript reports, asserted independently of any run."""
    pytest.importorskip("torch")
    from ecgvit.config import LoRAConfig, ModelConfig
    from ecgvit.model import build_model

    _, acct = build_model(ModelConfig(), LoRAConfig(rank=8, alpha=16))
    assert acct.baseline_trainable_parameters == 1_648_839
    assert acct.trainable_parameters == 131_072
    assert acct.total_parameters == 1_779_911
    assert round(acct.trainable_reduction_pct, 2) == 92.05
    # The manuscript's Table 8 states 13.6x. The reduction against full fine-tuning is
    # 1,648,839 / 131,072 = 12.58x; 13.58x is total-over-trainable, which counts the
    # adapters on both sides. See docs/CLASS_MAPPING.md.
    assert round(acct.reduction_factor, 2) == 12.58


def test_class_map_and_config_agree_on_class_order():
    """Class order fixes the meaning of every confusion-matrix axis and every saved
    probability column. Reordering it silently invalidates saved predictions."""
    from ecgvit.config import CLASS_NAMES

    assert CLASS_NAMES == ("NSR", "AFIB", "SB", "ST", "SVT", "CD", "OTHER")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def test_fixture_signals_are_reproducible_here(tmp_path):
    """Same seed, same signals — on this machine, this SciPy, this NumPy."""
    import wfdb

    import make_fixtures

    a, b = tmp_path / "a", tmp_path / "b"
    make_fixtures.generate(a)
    make_fixtures.generate(b)
    for rid in ("FX00000", "FX00020", "FX00041"):
        assert np.array_equal(
            wfdb.rdrecord(str(a / rid)).p_signal,
            wfdb.rdrecord(str(b / rid)).p_signal,
        ), f"{rid} is not reproducible from its seed"

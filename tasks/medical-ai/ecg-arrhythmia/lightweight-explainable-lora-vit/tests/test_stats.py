"""McNemar and DeLong: paired significance tests, and their degenerate cases.

Both tests are here because the comparison is *paired* -- the two models are evaluated on
the same records. The failure mode being guarded against is reporting a confident p-value
in a situation where none is estimable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecgvit.config import CLASS_NAMES
from ecgvit.stats import delong_per_class, delong_roc_test, mcnemar_test


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def test_mcnemar_contingency_counts():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    a = np.array([0, 1, 1, 0, 2, 2, 0, 1])   # wrong on 2 (idx 1, 3)
    b = np.array([0, 0, 0, 0, 2, 1, 0, 1])   # wrong on 3 (idx 2, 3, 5)
    #        idx   0  1  2  3  4  5  6  7
    #   a correct  T  F  T  F  T  T  T  T
    #   b correct  T  T  F  F  T  F  T  T
    r = mcnemar_test(y, a, b, "a", "b")
    assert r["n_samples"] == 8
    assert r["both_correct"] == 4           # idx 0, 4, 6, 7
    assert r["a_correct_b_wrong"] == 2      # idx 2, 5
    assert r["b_correct_a_wrong"] == 1      # idx 1
    assert r["both_wrong"] == 1             # idx 3
    assert r["a_correct_b_wrong"] + r["b_correct_a_wrong"] == r["discordant_pairs"]
    assert (r["both_correct"] + r["a_correct_b_wrong"]
            + r["b_correct_a_wrong"] + r["both_wrong"]) == 8


def test_mcnemar_identical_models_are_not_significant():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 500)
    p = rng.integers(0, 7, 500)
    r = mcnemar_test(y, p, p.copy())
    assert r["discordant_pairs"] == 0
    assert r["p_value"] == 1.0
    assert r["significant_at_0.05"] is False
    assert r["method"] == "degenerate_no_discordant_pairs"


def test_mcnemar_detects_a_clearly_better_model():
    y = np.zeros(400, dtype=int)
    a = y.copy()                 # perfect
    b = y.copy()
    b[:80] = 1                   # wrong on 80 samples
    r = mcnemar_test(y, a, b)
    assert r["p_value"] < 1e-10
    assert r["significant_at_0.05"] is True
    # The interpretation must name WHICH model is better, not merely that they differ:
    # "error patterns differ significantly" leaves the reader to work out the direction.
    assert "model_a makes fewer errors" in r["interpretation"]
    assert "80 of the 80 disagreements" in r["interpretation"]


def test_mcnemar_uses_the_exact_test_for_few_discordant_pairs():
    """The chi-square approximation is unreliable below ~25 discordant pairs. Reporting
    a chi-square p-value there overstates confidence."""
    y = np.zeros(200, dtype=int)
    a = y.copy()
    b = y.copy()
    b[:6] = 1
    r = mcnemar_test(y, a, b)
    assert r["method"] == "exact_binomial"
    assert 0.0 < r["p_value"] <= 1.0


def test_mcnemar_reproduces_a_hand_worked_chi_square():
    """15 / 11 discordant, continuity-corrected chi2 = (|15-11|-1)^2 / 26 = 9/26."""
    n_both_ok, a_only, b_only = 1976, 15, 11
    n_both_wrong = 130
    y, a, b = [], [], []
    for _ in range(n_both_ok):
        y.append(0); a.append(0); b.append(0)
    for _ in range(a_only):
        y.append(0); a.append(0); b.append(1)
    for _ in range(b_only):
        y.append(0); a.append(1); b.append(0)
    for _ in range(n_both_wrong):
        y.append(0); a.append(1); b.append(1)
    r = mcnemar_test(np.array(y), np.array(a), np.array(b))
    assert r["statistic"] == pytest.approx(9 / 26, abs=1e-6)
    assert r["method"] == "chi2_continuity_corrected"
    assert r["p_value"] > 0.05


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def test_delong_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    pa = rng.random(500) * 0.4 + y * 0.4
    pb = rng.random(500) * 0.4 + y * 0.3
    auc_a, auc_b, z, p = delong_roc_test(y, pa, pb)
    assert auc_a == pytest.approx(roc_auc_score(y, pa), abs=1e-6)
    assert auc_b == pytest.approx(roc_auc_score(y, pb), abs=1e-6)
    assert np.isfinite(z) and 0.0 <= p <= 1.0


def test_delong_handles_ties_via_midranks():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])   # all tied
    auc_a, auc_b, _, _ = delong_roc_test(y, p, p)
    assert auc_a == pytest.approx(0.5)


def test_delong_identical_predictions_give_zero_difference():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    auc_a, auc_b, z, pv = delong_roc_test(y, p, p.copy())
    assert auc_a == pytest.approx(auc_b)
    assert not np.isfinite(z), "zero variance must not yield a finite z"


def test_delong_single_class_is_not_estimable():
    y = np.ones(50, dtype=int)
    p = np.random.default_rng(3).random(50)
    auc_a, _, _, _ = delong_roc_test(y, p, p)
    assert np.isnan(auc_a)


def test_delong_per_class_marks_degenerate_classes_honestly():
    """A class both models classify perfectly has zero bootstrap variance. The output must
    say so rather than emit a placeholder p-value."""
    rng = np.random.default_rng(4)
    n = 400
    y = rng.integers(0, len(CLASS_NAMES), n)
    probs = rng.random((n, len(CLASS_NAMES)))
    probs /= probs.sum(1, keepdims=True)
    # Make class 5 perfectly separable and identical in both models.
    probs[y == 5] = 0.0
    probs[y == 5, 5] = 1.0
    res = delong_per_class(y, probs, probs.copy(), CLASS_NAMES, n_bootstrap=100, seed=0)

    by = {r["class"]: r for r in res["per_class"]}
    assert by[CLASS_NAMES[5]]["estimable"] is False
    assert by[CLASS_NAMES[5]]["p_value"] is None
    assert by[CLASS_NAMES[5]]["ci95_low"] is None
    assert "not estimable" in by[CLASS_NAMES[5]]["reason"]
    assert by[CLASS_NAMES[5]]["significant_at_0.05"] is False


def test_delong_per_class_handles_an_absent_class():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.integers(0, 6, n)          # class index 6 never appears
    probs = rng.random((n, len(CLASS_NAMES)))
    probs /= probs.sum(1, keepdims=True)
    res = delong_per_class(y, probs, probs.copy(), CLASS_NAMES, n_bootstrap=50, seed=0)
    absent = [r for r in res["per_class"] if r["class"] == CLASS_NAMES[6]][0]
    assert absent["estimable"] is False
    assert "absent" in absent["reason"]


def test_delong_per_class_output_shape():
    rng = np.random.default_rng(6)
    n = 300
    y = rng.integers(0, len(CLASS_NAMES), n)
    pa = rng.random((n, len(CLASS_NAMES))); pa /= pa.sum(1, keepdims=True)
    pb = rng.random((n, len(CLASS_NAMES))); pb /= pb.sum(1, keepdims=True)
    res = delong_per_class(y, pa, pb, CLASS_NAMES, n_bootstrap=50, seed=0,
                           name_a="no_lora", name_b="lora")
    assert len(res["per_class"]) == len(CLASS_NAMES)
    assert "macro_auc_no_lora" in res and "macro_auc_lora" in res
    assert "do not aggregate" in res["note"], (
        "the output must state that per-class DeLong statistics do not combine into a "
        "single macro-level test"
    )


# ---------------------------------------------------------------------------
# McNemar reporting (regression: an impossible line reached a run's console)
# ---------------------------------------------------------------------------
def _paired(n=1532, k=7, a_wrong=0, b_wrong=0):
    y = np.arange(n) % k
    a, b = y.copy(), y.copy()
    a[:a_wrong] = (a[:a_wrong] + 1) % k
    b[a_wrong:a_wrong + b_wrong] = (b[a_wrong:a_wrong + b_wrong] + 1) % k
    return y, a, b


def test_mcnemar_labels_its_statistic_correctly():
    """The exact-binomial branch does NOT produce a chi-square.

    The observed run printed "McNemar: chi2=11.0, p=1.0000", which is impossible for a
    chi-square and would have gone into the paper that way. 11 was min(b, c) from an exact
    binomial test on 22 discordant pairs split 11/11.
    """
    from ecgvit.stats import mcnemar_test

    y, a, b = _paired(a_wrong=11, b_wrong=11)
    r = mcnemar_test(y, a, b, "no_lora", "lora")
    assert r["method"] == "exact_binomial"
    assert "chi" not in r["statistic_name"].lower()
    assert r["statistic"] == 11.0
    assert r["discordant_pairs"] == 22
    assert r["p_value"] == pytest.approx(1.0)

    y2, a2, b2 = _paired(a_wrong=60, b_wrong=0)
    r2 = mcnemar_test(y2, a2, b2, "no_lora", "lora")
    assert r2["method"] == "chi2_continuity_corrected"
    assert "chi-square" in r2["statistic_name"]


def test_mcnemar_does_not_claim_identical_errors_when_the_models_disagree():
    """The defect this pins.

    With 22 discordant pairs the models DO misclassify different samples; they simply do so
    symmetrically. The old text said "no evidence the two models misclassify different
    samples" for both that case and the genuinely degenerate one, which is false for the
    first and hides the fact that the LoRA stage had come alive.
    """
    from ecgvit.stats import mcnemar_test

    y, a, b = _paired(a_wrong=11, b_wrong=11)
    disagree = mcnemar_test(y, a, b, "no_lora", "lora")["interpretation"]
    assert "22" in disagree
    assert "disagree" in disagree
    assert "identical" not in disagree.lower()
    # A non-significant result is not evidence of equivalence, and must say so.
    assert "equivalence test" in disagree

    y0 = np.arange(700) % 7
    same = mcnemar_test(y0, y0.copy(), y0.copy(), "no_lora", "lora")
    assert same["discordant_pairs"] == 0
    assert "IDENTICAL" in same["interpretation"]
    assert "degenerate" in same["interpretation"]


def test_mcnemar_names_the_better_model_when_the_difference_is_significant():
    from ecgvit.stats import mcnemar_test

    y, a, b = _paired(a_wrong=60, b_wrong=0)
    r = mcnemar_test(y, a, b, "no_lora", "lora")
    assert r["significant_at_0.05"] is True
    assert "lora makes fewer errors" in r["interpretation"]
    assert "no_lora makes fewer errors" not in r["interpretation"]

"""Statistical validation (manuscript section 4.3, Tables 4 and 5).

McNemar's test  -- do the two models misclassify the SAME samples?
DeLong's test   -- do they differ in ranking ability (AUC), accounting for the fact that
                   both AUCs are computed on the same records?

Both are paired tests. Comparing two accuracies with an unpaired test on the same test set
overstates significance, which is why the manuscript uses these and why they are here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .config import CLASS_NAMES


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str = "model_a",
    name_b: str = "model_b",
    exact_threshold: int = 25,
) -> Dict[str, object]:
    """Paired comparison of two classifiers' error patterns on identical samples.

    Uses the exact binomial test when the number of discordant pairs is small
    (<= `exact_threshold`), where the chi-square approximation is unreliable, and the
    continuity-corrected chi-square otherwise. The manuscript reports the chi-square form.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true

    both_correct = int(np.sum(a_ok & b_ok))
    a_only = int(np.sum(a_ok & ~b_ok))   # A correct, B wrong
    b_only = int(np.sum(~a_ok & b_ok))   # B correct, A wrong
    both_wrong = int(np.sum(~a_ok & ~b_ok))
    n = int(y_true.size)
    discordant = a_only + b_only

    if discordant == 0:
        statistic, p_value, method = 0.0, 1.0, "degenerate_no_discordant_pairs"
        statistic_name = "none (no discordant pairs)"
    elif discordant <= exact_threshold:
        p_value = float(stats.binomtest(a_only, discordant, 0.5).pvalue)
        statistic = float(min(a_only, b_only))
        # NOT a chi-square. Reporting it under a chi2 label -- as this did -- produces
        # impossible-looking lines such as "chi2=11.0, p=1.0000" in a paper.
        statistic_name = "min(b, c), exact binomial"
        method = "exact_binomial"
    else:
        statistic = float((abs(a_only - b_only) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
        statistic_name = "chi-square with continuity correction, df=1"
        method = "chi2_continuity_corrected"

    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": both_correct,
        f"{name_a}_correct_{name_b}_wrong": a_only,
        f"{name_b}_correct_{name_a}_wrong": b_only,
        "both_wrong": both_wrong,
        "n_samples": n,
        "discordant_pairs": discordant,
        "disagreement_rate": round(discordant / n, 6) if n else 0.0,
        "statistic": round(statistic, 6),
        "statistic_name": statistic_name,
        "p_value": p_value,
        "method": method,
        "significant_at_0.05": bool(p_value < 0.05),
        # Three genuinely different outcomes. The previous text collapsed the first two,
        # so a run with 22 discordant pairs was reported as "no evidence the two models
        # misclassify different samples" -- which is false: they demonstrably do, they
        # simply do so symmetrically.
        "interpretation": (
            f"the two models produce IDENTICAL predictions on all {n} samples; "
            "the comparison is degenerate and no test is possible"
            if discordant == 0 else
            f"{name_a} is right on {a_only} of the {discordant} disagreements and "
            f"{name_b} on {b_only}; the difference is significant (p = {p_value:.4g}), so "
            f"{(name_a if a_only > b_only else name_b)} makes fewer errors"
            if p_value < 0.05 else
            f"the models disagree on {discordant} of {n} samples "
            f"({name_a} right on {a_only}, {name_b} on {b_only}) but neither is "
            f"systematically more accurate (p = {p_value:.4g}). Note this is a failure to "
            "reject a difference, not a demonstration of equivalence -- for that, "
            "pre-specify a margin and run an equivalence test"
        ),
    }


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    """Sun & Xu (2014) fast DeLong. First `m` columns are the positive samples."""
    k, total = predictions_sorted_transposed.shape
    n = total - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, total))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(
    y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray
) -> Tuple[float, float, float, float]:
    """Returns (auc_a, auc_b, z, p) for one binary problem on paired predictions."""
    y = np.asarray(y_true).astype(int)
    order = (-y).argsort(kind="mergesort")   # positives first, stable
    label_sorted = y[order]
    m = int(label_sorted.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    preds = np.vstack([np.asarray(probs_a)[order], np.asarray(probs_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        # Identical rankings, or a perfect AUC in both -> zero variance, p undefined.
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = diff / np.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def delong_per_class(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    n_bootstrap: int = 1000,
    seed: int = 42,
    name_a: str = "no_lora",
    name_b: str = "lora",
) -> Dict[str, object]:
    """Per-class one-vs-rest DeLong plus a bootstrap CI on the AUC difference (Table 5)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    rows: List[Dict[str, object]] = []

    for i, cname in enumerate(class_names):
        yi = (y_true == i).astype(int)
        if yi.sum() == 0 or yi.sum() == len(yi):
            rows.append(
                {"class": cname, "auc_" + name_a: None, "auc_" + name_b: None,
                 "difference": None, "ci95_low": None, "ci95_high": None,
                 "z": None, "p_value": None, "estimable": False,
                 "reason": "class absent from this split"}
            )
            continue

        auc_a, auc_b, z, p = delong_roc_test(yi, probs_a[:, i], probs_b[:, i])

        diffs = np.full(n_bootstrap, np.nan)
        idx_pos = np.where(yi == 1)[0]
        idx_neg = np.where(yi == 0)[0]
        from sklearn.metrics import roc_auc_score

        for b in range(n_bootstrap):
            sp = rng.choice(idx_pos, size=idx_pos.size, replace=True)
            sn = rng.choice(idx_neg, size=idx_neg.size, replace=True)
            s = np.concatenate([sp, sn])
            ys = yi[s]
            if ys.sum() in (0, ys.size):
                continue
            diffs[b] = roc_auc_score(ys, probs_a[s, i]) - roc_auc_score(ys, probs_b[s, i])
        finite = diffs[np.isfinite(diffs)]
        var_zero = bool(finite.size and np.allclose(finite, 0.0))

        rows.append(
            {
                "class": cname,
                f"auc_{name_a}": round(auc_a, 6),
                f"auc_{name_b}": round(auc_b, 6),
                "difference": round(auc_a - auc_b, 6),
                "ci95_low": None if var_zero or not finite.size else round(float(np.quantile(finite, 0.025)), 6),
                "ci95_high": None if var_zero or not finite.size else round(float(np.quantile(finite, 0.975)), 6),
                "z": None if not np.isfinite(z) else round(z, 6),
                "p_value": None if not np.isfinite(p) else p,
                "estimable": bool(np.isfinite(p)) and not var_zero,
                "reason": (
                    "perfect or identical AUC in both models -> zero bootstrap variance; "
                    "CI and p-value are not estimable"
                    if var_zero or not np.isfinite(p)
                    else ""
                ),
                "significant_at_0.05": bool(np.isfinite(p) and p < 0.05 and not var_zero),
            }
        )

    est = [r for r in rows if r.get("estimable")]
    macro_a = np.mean([r[f"auc_{name_a}"] for r in rows if r.get(f"auc_{name_a}") is not None])
    macro_b = np.mean([r[f"auc_{name_b}"] for r in rows if r.get(f"auc_{name_b}") is not None])
    return {
        "model_a": name_a,
        "model_b": name_b,
        "n_bootstrap": n_bootstrap,
        "per_class": rows,
        "macro_auc_" + name_a: float(macro_a),
        "macro_auc_" + name_b: float(macro_b),
        "macro_difference": float(macro_a - macro_b),
        "n_estimable_classes": len(est),
        "note": (
            "Per-class DeLong statistics do not aggregate to a single macro-level test; "
            "the macro AUCs are a summary average only."
        ),
    }


__all__ = ["mcnemar_test", "delong_roc_test", "delong_per_class"]

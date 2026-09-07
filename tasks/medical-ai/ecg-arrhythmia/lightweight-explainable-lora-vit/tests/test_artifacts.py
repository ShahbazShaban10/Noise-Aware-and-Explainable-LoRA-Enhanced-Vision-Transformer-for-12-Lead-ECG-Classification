"""Grades the artefacts a completed run wrote to $OUTPUT_DIR.

Every test here is marked `artifacts` and is skipped when no run has happened. The point of
this tier is not to re-run the model but to check that what was reported is internally
consistent, complete, and meets the stated bar -- including the checks that catch a result
that looks good because the evaluation was bent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.artifacts

BALANCED_ACC_MIN = 0.85
MACRO_F1_MIN = 0.85
REDUCTION_PCT_MIN = 90.0
BASE_PARAMS = 1_648_839
LORA_TRAINABLE = 131_072
LORA_TOTAL = 1_779_911

REQUIRED = [
    "metrics.json", "confusion_matrix.npy", "predictions.csv",
    "param_efficiency.json", "model/model.pt", "model/base_no_lora.pt",
    "labels/label_index.json", "index_report.json", "balance_report.json",
    "training_history.json", "preprocessing_report.json",
    "xai/integrated_gradients_lead_importance.csv",
    "xai/shap_lead_importance.csv",
    "xai/shap_global_lead_importance.json",
    "xai/faithfulness.json", "xai/clinical_concordance.json", "xai/tsne.npz",
    "stats/mcnemar.json", "stats/delong_auc.json",
]


def _load(output_dir: Path, name: str):
    p = output_dir / name
    if not p.is_file():
        pytest.fail(f"required artefact missing: {name}")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Presence and schema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", REQUIRED)
def test_required_artifact_exists(output_dir, rel):
    p = output_dir / rel
    assert p.is_file(), f"missing artefact: {rel}"
    assert p.stat().st_size > 0, f"empty artefact: {rel}"


@pytest.fixture(scope="session")
def class_names(output_dir):
    """The classes THIS run used.

    Read from the run's own artefacts rather than from config.CLASS_NAMES: the class list
    belongs to the resolution order (canonical_v2 renames OTHER -> VE), so a hard-coded
    constant would fail a correct run. What the tests check instead is that every artefact
    agrees with every other artefact about what the classes were.
    """
    emitted = _load(output_dir, "labels/label_index.json")["classes"]
    assert emitted, "labels/label_index.json declares no classes"
    return list(emitted)


def test_every_artifact_agrees_on_the_class_list(output_dir, class_names):
    metrics = _load(output_dir, "metrics.json")
    assert metrics["class_names"] == class_names
    bal = _load(output_dir, "balance_report.json")
    assert sorted(bal["support"]) == sorted(class_names)
    idx = _load(output_dir, "index_report.json")
    assert set(idx["label_counts"]) <= set(class_names)


def test_metrics_schema(metrics, class_names):
    required = {"class_names", "n_samples", "overall_accuracy", "balanced_accuracy",
                "macro_f1", "per_class", "confusion_matrix", "inference_ms_per_sample"}
    missing = required - set(metrics)
    assert not missing, f"metrics.json is missing {sorted(missing)}"
    assert metrics["class_names"] == class_names
    for c in class_names:
        assert c in metrics["per_class"], f"per_class is missing {c}"
        for field in ("precision", "recall", "f1", "support"):
            assert field in metrics["per_class"][c]


def test_metric_values_are_in_range(metrics):
    for k in ("overall_accuracy", "balanced_accuracy", "macro_f1"):
        assert 0.0 <= metrics[k] <= 1.0, f"{k}={metrics[k]} is not a proportion"
    for c, d in metrics["per_class"].items():
        for k in ("precision", "recall", "f1"):
            assert 0.0 <= d[k] <= 1.0, f"{c}.{k}={d[k]}"


# ---------------------------------------------------------------------------
# Performance bar
# ---------------------------------------------------------------------------
def test_balanced_accuracy_meets_the_bar(metrics):
    assert metrics["balanced_accuracy"] >= BALANCED_ACC_MIN, (
        f"balanced accuracy {metrics['balanced_accuracy']:.4f} < {BALANCED_ACC_MIN}"
    )


def test_macro_f1_meets_the_bar(metrics):
    assert metrics["macro_f1"] >= MACRO_F1_MIN, (
        f"macro F1 {metrics['macro_f1']:.4f} < {MACRO_F1_MIN}"
    )


def test_no_class_is_silently_absent_from_the_test_split(metrics):
    """A 7-class balanced accuracy computed over 5 present classes is not a 7-class result."""
    absent = metrics.get("classes_absent_from_split") or [
        c for c, d in metrics["per_class"].items() if d["support"] == 0
    ]
    assert not absent, (
        f"classes with zero test support: {absent}. Balanced accuracy over the remaining "
        "classes is not comparable to a 7-class number."
    )


def test_balanced_accuracy_is_not_just_the_majority_class(metrics):
    """Guard against a model that predicts SB for everything and still posts a high plain
    accuracy. Balanced accuracy should be close to overall accuracy on a balanced test set,
    and every class should have non-trivial recall."""
    weak = {c: d["recall"] for c, d in metrics["per_class"].items()
            if d["support"] > 0 and d["recall"] < 0.5}
    assert not weak, f"classes with recall below 0.5: {weak}"


# ---------------------------------------------------------------------------
# Internal consistency -- the checks that catch a bent evaluation
# ---------------------------------------------------------------------------
def test_confusion_matrix_matches_metrics(output_dir, metrics, class_names):
    cm_npy = np.load(output_dir / "confusion_matrix.npy")
    cm_json = np.asarray(metrics["confusion_matrix"])
    assert cm_npy.shape == (len(class_names), len(class_names))
    assert np.array_equal(cm_npy, cm_json), (
        "confusion_matrix.npy and metrics.json disagree"
    )


def test_confusion_matrix_reproduces_the_reported_accuracy(metrics):
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    total = cm.sum()
    assert total == metrics["n_samples"], (
        f"confusion matrix totals {total:.0f} but n_samples is {metrics['n_samples']}"
    )
    acc = np.trace(cm) / total
    assert acc == pytest.approx(metrics["overall_accuracy"], abs=1e-6), (
        f"confusion matrix implies accuracy {acc:.6f}, metrics.json reports "
        f"{metrics['overall_accuracy']:.6f}"
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        recalls = np.divide(np.diag(cm), cm.sum(1),
                            out=np.full(cm.shape[0], np.nan), where=cm.sum(1) > 0)
    bacc = np.nanmean(recalls)
    assert bacc == pytest.approx(metrics["balanced_accuracy"], abs=1e-6)


def test_predictions_csv_agrees_with_the_confusion_matrix(output_dir, metrics, class_names):
    rows = list(csv.DictReader((output_dir / "predictions.csv").open()))
    assert len(rows) == metrics["n_samples"], (
        f"predictions.csv has {len(rows)} rows, metrics.json reports "
        f"{metrics['n_samples']} samples"
    )
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for r in rows:
        cm[idx[r["true_label"]], idx[r["pred_label"]]] += 1
    assert np.array_equal(cm, np.asarray(metrics["confusion_matrix"])), (
        "the confusion matrix cannot be rebuilt from predictions.csv"
    )


def test_predicted_probabilities_are_valid_and_consistent(output_dir, class_names):
    rows = list(csv.DictReader((output_dir / "predictions.csv").open()))
    for r in rows[:2000]:
        probs = np.array([float(r[f"prob_{c}"]) for c in class_names])
        assert np.all(probs >= -1e-6) and np.all(probs <= 1 + 1e-6)
        assert probs.sum() == pytest.approx(1.0, abs=1e-4), "probabilities do not sum to 1"
        assert class_names[int(np.argmax(probs))] == r["pred_label"], (
            f"argmax of the probability vector is not the reported prediction "
            f"for record {r['record_id']}"
        )


def test_test_records_are_unique(output_dir):
    """No record may appear twice in the test split. Duplicates are the signature of
    augmentation applied before splitting."""
    rows = list(csv.DictReader((output_dir / "predictions.csv").open()))
    ids = [r["record_id"] for r in rows]
    dupes = {i for i in ids if ids.count(i) > 1} if len(ids) < 5000 else (
        {i for i, n in __import__("collections").Counter(ids).items() if n > 1}
    )
    assert not dupes, f"duplicate records in the test split: {sorted(dupes)[:10]}"


def test_no_evaluation_record_was_augmented(output_dir, class_names):
    """Validation and test must contain only unique, unaugmented records."""
    bal = _load(output_dir, "balance_report.json")
    idx = _load(output_dir, "index_report.json")
    n_test = sum(idx["split_counts"]["test"].values())
    rows = sum(1 for _ in csv.DictReader((output_dir / "predictions.csv").open()))
    assert rows == n_test, (
        f"the index reports {n_test} test records but predictions.csv has {rows}; "
        "the evaluation split appears to have been resampled"
    )
    for c in class_names:
        assert bal["support"][c] >= bal["unique_support"][c]


def test_balance_report_exposes_duplication(output_dir):
    bal = _load(output_dir, "balance_report.json")
    for field in ("support", "unique_support", "duplication_factor"):
        assert field in bal, f"balance_report.json is missing {field}"
    inflated = {c: f for c, f in bal["duplication_factor"].items() if f > 8.0}
    assert not inflated, (
        f"classes oversampled by more than 8x: {inflated}. Per-class metrics for these "
        "are dominated by duplicates and do not measure diagnostic ability."
    )


# ---------------------------------------------------------------------------
# Parameter efficiency
# ---------------------------------------------------------------------------
def test_parameter_efficiency_schema_and_bar(output_dir):
    pe = _load(output_dir, "param_efficiency.json")
    for field in ("total_parameters", "trainable_parameters",
                  "baseline_trainable_parameters", "trainable_reduction_pct",
                  "lora_rank", "lora_alpha", "lora_target_modules", "n_lora_layers"):
        assert field in pe, f"param_efficiency.json is missing {field}"
    assert pe["trainable_reduction_pct"] >= REDUCTION_PCT_MIN, (
        f"reduction {pe['trainable_reduction_pct']:.2f}% < {REDUCTION_PCT_MIN}%"
    )
    assert pe["trainable_parameters"] < pe["total_parameters"]


def test_parameter_counts_match_the_manuscript(output_dir):
    """Table 8, exactly -- for the reference configuration.

    Two documented departures are allowed, and both must be visible in the artefact rather
    than inferred:

    * `--patch-len 50` (the QRS-width ablation) changes the tokeniser and so the counts;
      `model_config.is_reference_geometry` says which geometry produced the file.
    * `--train-head` adds the 903-parameter classifier head to the trainable set, moving
      the reduction from 92.05% to 92.00%. The adapter count itself never moves.
    """
    pe = _load(output_dir, "param_efficiency.json")
    mc = pe.get("model_config", {"is_reference_geometry": True})
    assert pe["lora_parameters"] == LORA_TRAINABLE, "the adapters are not r=8 over 32 layers"

    head = pe.get("trainable_parameters", 0) - pe["lora_parameters"]
    assert head in (0, 903), (
        f"trainable parameters exceed the adapters by {head}, which is neither 0 (frozen "
        "head) nor 903 (trainable head)"
    )
    if mc.get("is_reference_geometry", True):
        assert pe["baseline_trainable_parameters"] == BASE_PARAMS
        assert pe["total_parameters"] == LORA_TOTAL
    else:
        # The adapters live in the encoder blocks, which no tokeniser change touches.
        assert pe["total_parameters"] - pe["baseline_trainable_parameters"] == LORA_TRAINABLE


def test_reduction_percentage_is_arithmetically_consistent(output_dir):
    pe = _load(output_dir, "param_efficiency.json")
    expected = 100.0 * (1 - pe["trainable_parameters"] / pe["baseline_trainable_parameters"])
    assert pe["trainable_reduction_pct"] == pytest.approx(expected, abs=0.01), (
        "the reported reduction does not follow from the reported parameter counts"
    )


def test_lora_targets_all_four_projection_families(output_dir):
    pe = _load(output_dir, "param_efficiency.json")
    assert set(pe["lora_target_modules"]) == {"qkv", "proj", "fc1", "fc2"}
    assert pe["n_lora_layers"] == 32, "8 blocks x 4 targets"


# ---------------------------------------------------------------------------
# Explainability artefacts
# ---------------------------------------------------------------------------
def test_faithfulness_deletion_beats_insertion(output_dir):
    """The load-bearing XAI check. If deleting the top-attributed patches does not hurt
    more than inserting them helps, the attributions are not explaining the decision."""
    f = _load(output_dir, "xai/faithfulness.json")
    for k in ("insertion_auc", "deletion_auc", "insertion_accuracy",
              "deletion_accuracy", "baseline_accuracy"):
        assert k in f, f"faithfulness.json is missing {k}"
    assert f["deletion_auc"] < f["insertion_auc"], (
        f"deletion AUC {f['deletion_auc']:.4f} >= insertion AUC {f['insertion_auc']:.4f}; "
        "the reported attributions are not faithful to the model"
    )
    assert f.get("faithful") is True


def test_faithfulness_curves_are_monotone_in_the_right_direction(output_dir):
    f = _load(output_dir, "xai/faithfulness.json")
    ins = f["insertion_accuracy"]
    dele = f["deletion_accuracy"]
    assert ins[-1] >= ins[0], "inserting more evidence should not reduce accuracy overall"
    assert dele[-1] <= dele[0], "deleting more evidence should not increase accuracy overall"


def test_lead_importance_csv_is_complete(output_dir, class_names):
    for name in ("xai/integrated_gradients_lead_importance.csv",
                 "xai/shap_lead_importance.csv"):
        rows = list(csv.DictReader((output_dir / name).open()))
        assert len(rows) == len(class_names) * 12, (
            f"{name} should hold one row per class x lead"
        )
        for r in rows[:12]:
            assert float(r["mean_abs_attribution"]) >= 0.0
            assert float(r["ci95_low"]) <= float(r["mean_abs_attribution"]) + 1e-9
            assert float(r["ci95_high"]) >= float(r["mean_abs_attribution"]) - 1e-9


def test_global_lead_importance_covers_twelve_leads(output_dir):
    g = _load(output_dir, "xai/shap_global_lead_importance.json")
    from ecgvit.config import LEAD_ORDER

    assert set(g) == set(LEAD_ORDER)
    assert all(v >= 0 for v in g.values())


def test_gradcam_arrays_are_normalised(output_dir):
    cams = sorted((output_dir / "xai").glob("gradcam_*.npy"))
    assert cams, "no Grad-CAM arrays were written"
    for p in cams[:20]:
        cam = np.load(p)
        assert cam.ndim == 1 and cam.shape[0] == 50, f"{p.name}: expected 50 patch values"
        assert cam.min() >= -1e-6 and cam.max() <= 1 + 1e-6, f"{p.name} is not in [0,1]"


def test_tsne_artifact_shape(output_dir):
    z = np.load(output_dir / "xai/tsne.npz")
    assert "Z" in z and "labels" in z
    assert z["Z"].ndim == 2 and z["Z"].shape[1] == 2
    assert z["Z"].shape[0] == z["labels"].shape[0]


def test_clinical_concordance_is_reported_with_a_caveat(output_dir):
    c = _load(output_dir, "xai/clinical_concordance.json")
    assert "integrated_gradients" in c and "gradient_shap" in c
    note = c.get("note", "")
    assert "not evidence" in note or "sanity check" in note, (
        "clinical concordance must be qualified; overlapping with expected leads is not "
        "clinical validation"
    )
    for row in c["integrated_gradients"]:
        assert row["clinical_criterion"], f"{row['class']} has no cited criterion"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_mcnemar_artifact(output_dir):
    m = _load(output_dir, "stats/mcnemar.json")
    for k in ("both_correct", "both_wrong", "n_samples", "discordant_pairs",
              "statistic", "p_value", "method", "interpretation"):
        assert k in m, f"mcnemar.json is missing {k}"
    assert 0.0 <= m["p_value"] <= 1.0
    assert m["discordant_pairs"] <= m["n_samples"]


def test_delong_artifact_reports_estimability(output_dir, class_names):
    d = _load(output_dir, "stats/delong_auc.json")
    assert len(d["per_class"]) == len(class_names)
    for row in d["per_class"]:
        assert "estimable" in row
        if not row["estimable"]:
            assert row["p_value"] is None, (
                f"{row['class']} is flagged not estimable but still reports a p-value"
            )
            assert row["reason"], f"{row['class']} is not estimable with no reason given"


def test_training_history_covers_both_stages(output_dir):
    h = _load(output_dir, "training_history.json")
    stages = {s["stage"] for s in h}
    assert stages == {"pretrain", "lora"}, (
        f"expected both training stages, found {stages}. The LoRA model must adapt a "
        "pretrained backbone, not a random one."
    )
    for s in h:
        assert s["epochs"], f"stage {s['stage']} recorded no epochs"
        assert s["best_epoch"] >= 1


def test_label_index_matches_the_shipped_class_map(output_dir, data_dir, class_names):
    from ecgvit.labels import get_class_map

    emitted = _load(output_dir, "labels/label_index.json")
    # The artefact names the ordering it used. Guessing one here (the old default was
    # "rhythm_first") silently compared a run against a mapping it never used, so the check
    # passed or failed for reasons unrelated to the run.
    order = emitted.get("resolution_order_name")
    assert order, (
        "labels/label_index.json does not record which resolution order produced it; "
        "the mapping cannot be verified without it"
    )
    expected = get_class_map(data_dir, order)
    assert emitted["classes"] == class_names
    assert emitted["class_to_codes"] == expected.fingerprint()["class_to_codes"], (
        "the mapping actually used differs from environment/data/class_map_7.json"
    )


def test_metrics_carry_the_evaluability_gate(metrics, class_names):
    """A per-class metric must say whether its test split can support it.

    class_map_7.json sets min_unique_test_records_for_reporting, but until this run that
    gate lived only in `label-audit`, so metrics.json printed a point estimate for a class
    with 37 test records whose recall 95% CI is 30 percentage points wide. The gate now
    travels with the numbers a reader actually quotes.
    """
    gate = metrics.get("evaluability")
    assert gate, "metrics.json has no evaluability block"
    min_test = int(gate["min_unique_test_records_for_reporting"])
    assert min_test > 0
    for c in class_names:
        entry = metrics["per_class"][c]
        assert "evaluable" in entry, f"{c} has no evaluable flag"
        assert entry["evaluable"] == (entry["support"] >= min_test)
        lo, hi = entry["recall_ci95"]
        assert 0.0 <= lo <= entry["recall"] <= hi <= 1.0, (
            f"{c}: recall {entry['recall']} outside its own CI [{lo}, {hi}]"
        )
        if not entry["evaluable"]:
            assert c in gate["not_evaluable"]
    # Every flagged class must actually be under the threshold, not merely listed.
    for c in gate["not_evaluable"]:
        assert metrics["per_class"][c]["support"] < min_test


def test_confidence_intervals_widen_as_support_shrinks(metrics, class_names):
    """A sanity check on the intervals themselves: fewer records, wider interval."""
    widths = {
        c: (metrics["per_class"][c]["recall_ci95"][1]
            - metrics["per_class"][c]["recall_ci95"][0])
        for c in class_names if metrics["per_class"][c]["support"] > 0
    }
    supports = {c: metrics["per_class"][c]["support"] for c in widths}
    smallest = min(supports, key=supports.get)
    largest = max(supports, key=supports.get)
    if supports[largest] > 2 * supports[smallest]:
        assert widths[smallest] > widths[largest], (
            f"{smallest} (n={supports[smallest]}) has a narrower interval than "
            f"{largest} (n={supports[largest]})"
        )


def test_parameter_artifact_declares_its_geometry(output_dir):
    """A stride ablation must be self-describing.

    Without this, a reader comparing an ablation's 1,636,039 parameters against Table 8's
    1,648,839 concludes the run is broken rather than that it is a different tokeniser.
    """
    pe = _load(output_dir, "param_efficiency.json")
    mc = pe.get("model_config")
    assert mc, "param_efficiency.json does not record the model geometry"
    assert mc["patch_len"] * mc["n_patches"] == 5000
    assert mc["is_reference_geometry"] == (mc["patch_len"] == 100)
    if mc["is_reference_geometry"]:
        assert pe["no_lora"]["total_parameters"] == 1_648_839
    assert "recipe" in pe and "resolution_order" in pe


def test_training_history_records_how_the_checkpoint_was_chosen(output_dir):
    """Selection metric and stop reason are part of the result, not folklore.

    The published run's LoRA stage saved its epoch-1 adapters because val balanced
    accuracy peaked there while val loss kept improving; nothing in the artefacts said so.
    """
    for stage in _load(output_dir, "training_history.json"):
        assert stage["select_metric"] in {"val_loss", "val_balanced_acc", "val_macro_auc"}
        assert stage["epochs_run"] == len(stage["epochs"])
        assert stage["epochs_run"] <= stage["epochs_planned"]
        assert isinstance(stage["stopped_early"], bool)
        if stage["stopped_early"]:
            assert stage["epochs_run"] < stage["epochs_planned"]
        series = [e[stage["select_metric"]] for e in stage["epochs"]]
        best = series[stage["best_epoch"] - 1]
        if stage["select_metric"] == "val_loss":
            assert best == min(series), "checkpoint is not the minimum-val-loss epoch"
        else:
            assert best == max(series), "checkpoint is not the best-scoring epoch"

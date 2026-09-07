"""Evaluation and metric reporting (manuscript section 4.1, Table 3, Figures 5 and 6)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import CLASS_NAMES


@dataclass
class Predictions:
    record_ids: List[str]
    y_true: np.ndarray          # (N,)
    y_pred: np.ndarray          # (N,)
    probs: np.ndarray           # (N, K)
    embeddings: Optional[np.ndarray] = None   # (N, D) CLS descriptors, for t-SNE
    inference_ms_per_sample: float = 0.0


@torch.no_grad()
def predict(
    model,
    loader,
    device,
    record_ids: Optional[Sequence[str]] = None,
    return_embeddings: bool = True,
) -> Predictions:
    model.eval().to(device)
    ys, ps, probs, embs, idxs = [], [], [], [], []
    total_s, n = 0.0, 0

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        idx = batch[2] if len(batch) > 2 else None

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = model.forward_features(x)
        logits = model.head(feats)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_s += time.perf_counter() - t0
        n += x.size(0)

        p = torch.softmax(logits.float(), dim=1)
        probs.append(p.cpu().numpy())
        ps.append(p.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        if return_embeddings:
            embs.append(feats.float().cpu().numpy())
        if idx is not None:
            idxs.append(idx.numpy())

    y_true = np.concatenate(ys)
    order_ids: List[str]
    if record_ids is not None and idxs:
        flat = np.concatenate(idxs)
        order_ids = [str(record_ids[i]) for i in flat]
    elif record_ids is not None:
        order_ids = [str(r) for r in record_ids][: len(y_true)]
    else:
        order_ids = [str(i) for i in range(len(y_true))]

    return Predictions(
        record_ids=order_ids,
        y_true=y_true,
        y_pred=np.concatenate(ps),
        probs=np.concatenate(probs),
        embeddings=np.concatenate(embs) if embs else None,
        inference_ms_per_sample=1000.0 * total_s / max(n, 1),
    )


def compute_metrics(
    pred: Predictions,
    class_names: Sequence[str] = CLASS_NAMES,
    support_detail: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    K = len(class_names)
    labels = list(range(K))
    y, p = pred.y_true, pred.y_pred

    prec, rec, f1, sup = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, p, labels=labels)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        # Per-class accuracy in the manuscript's Table 3 sense: recall on that class.
        support_i = int(sup[i])
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": support_i,
            "accuracy": float(rec[i]),
        }
        if support_detail and name in support_detail:
            per_class[name].update(support_detail[name])

    # One-vs-rest AUC, skipping degenerate classes rather than reporting a fake 0.5.
    auc: Dict[str, Optional[float]] = {}
    for i, name in enumerate(class_names):
        yi = (y == i).astype(int)
        if yi.sum() == 0 or yi.sum() == len(yi):
            auc[name] = None
        else:
            auc[name] = float(roc_auc_score(yi, pred.probs[:, i]))
    scored = [v for v in auc.values() if v is not None]

    return {
        "class_names": list(class_names),
        "n_samples": int(len(y)),
        "overall_accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", labels=labels, zero_division=0)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_auc": float(np.mean(scored)) if scored else None,
        "per_class": per_class,
        "per_class_auc": auc,
        "confusion_matrix": cm.tolist(),
        "inference_ms_per_sample": round(pred.inference_ms_per_sample, 4),
        "classes_absent_from_split": [
            class_names[i] for i in range(K) if int(sup[i]) == 0
        ],
    }


def write_predictions_csv(pred: Predictions, path: Path, class_names=CLASS_NAMES) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["record_id", "true_label", "pred_label", "true_index", "pred_index"]
            + [f"prob_{c}" for c in class_names]
        )
        for i, rid in enumerate(pred.record_ids):
            w.writerow(
                [
                    rid,
                    class_names[int(pred.y_true[i])],
                    class_names[int(pred.y_pred[i])],
                    int(pred.y_true[i]),
                    int(pred.y_pred[i]),
                ]
                + [f"{v:.6f}" for v in pred.probs[i]]
            )


def _agg_pyplot():
    """Return pyplot with a non-interactive backend, tolerantly.

    Written this way because `import matplotlib; matplotlib.use("Agg")` raised
    AttributeError on a working matplotlib 3.10 install -- and a cosmetic figure must never
    abort a training run that has already produced its graded artefacts. MPLBACKEND is the
    documented, import-order-independent way to select a backend; use() is only a fallback.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    try:
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
    except Exception:                      # noqa: BLE001 - backend selection is best-effort
        pass
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(cm, class_names, path: Path, title: str) -> None:
    plt = _agg_pyplot()

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black", fontsize=9,
            )
    ax.set_xlabel("Predicted label", fontweight="bold")
    ax.set_ylabel("True label", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_precision_recall(pred: Predictions, class_names, path: Path) -> Dict[str, float]:
    plt = _agg_pyplot()
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    aps: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        yi = (pred.y_true == i).astype(int)
        if yi.sum() == 0:
            continue
        p, r, _ = precision_recall_curve(yi, pred.probs[:, i])
        ap = float(average_precision_score(yi, pred.probs[:, i]))
        aps[name] = ap
        ax.step(r, p, where="post", label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title("Precision-Recall (one-vs-rest)", fontweight="bold")
    ax.legend(fontsize="small", loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return aps


def plot_training_curves(histories, path: Path) -> None:
    plt = _agg_pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for hist in histories:
        ep = [e["epoch"] if isinstance(e, dict) else e.epoch for e in hist["epochs"]]
        tl = [e["train_loss"] for e in hist["epochs"]]
        vl = [e["val_loss"] for e in hist["epochs"]]
        vb = [e["val_balanced_acc"] for e in hist["epochs"]]
        axes[0].plot(ep, tl, label=f"{hist['stage']} train")
        axes[0].plot(ep, vl, "--", label=f"{hist['stage']} val")
        axes[1].plot(ep, vb, label=f"{hist['stage']} val balanced acc")
    axes[0].set_title("Loss", fontweight="bold")
    axes[1].set_title("Validation balanced accuracy", fontweight="bold")
    for a in axes:
        a.set_xlabel("Epoch")
        a.legend(fontsize="small")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


__all__ = [
    "Predictions", "predict", "compute_metrics", "write_predictions_csv",
    "plot_confusion_matrix", "plot_precision_recall", "plot_training_curves",
]

"""Multifaceted explainability layer (manuscript sections 4.2, 4.3 and Figures 3, 4, 7-10).

Four complementary views, deliberately not interchangeable:

  Grad-CAM              WHERE in time the model looked, per lead panel   (Fig. 3)
  Integrated Gradients  WHICH lead drove each class, with 95% CI, eq. 19 (Fig. 7)
  Gradient SHAP         signed local contributions + global lead ranking (Figs. 4, 8)
  insertion / deletion  whether those attributions are FAITHFUL          (Fig. 10)

The last one is the part that makes the other three worth reporting. A saliency map that
looks physiological but whose top-ranked samples can be deleted without hurting accuracy is
a picture, not an explanation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import CLASS_NAMES, LEAD_ORDER, N_LEADS, N_SAMPLES, XAIConfig

log = logging.getLogger("ecgvit.xai")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class ViTGradCAM:
    """Grad-CAM over transformer tokens.

    Hooks the pre-norm LayerNorm of the last encoder block, which carries (B, 1+N, D)
    token activations. Channel weights are the token-averaged gradients; the CAM is the
    weighted channel sum, ReLU'd, CLS token dropped, min-max normalised to [0, 1].

    The hook handles are released by `close()` / the context manager. Registering hooks and
    never removing them leaks activations across calls and silently corrupts a long XAI
    sweep -- a real failure mode in the notebook version of this code.
    """

    def __init__(self, model, target_layer=None) -> None:
        self.model = model
        self.layer = target_layer if target_layer is not None else model.blocks[-1].norm1
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activation),
            self.layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0]

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ViTGradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(
        self, x: torch.Tensor, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Returns (cam over N patches in [0,1], predicted class, softmax probabilities)."""
        if x.shape[0] != 1:
            raise ValueError("ViTGradCAM operates on one record at a time")
        was_training = self.model.training
        self.model.eval()

        x = x.detach().requires_grad_(True)
        logits = self.model(x)
        target = int(logits.argmax(1).item()) if class_idx is None else int(class_idx)

        self.model.zero_grad(set_to_none=True)
        logits[0, target].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check the target layer")

        weights = self.gradients.mean(dim=1, keepdim=True)          # (1, 1, D)
        cam = (weights * self.activations).sum(dim=2)               # (1, 1+N)
        cam = F.relu(cam)
        if self.model.cls_token is not None:
            cam = cam[:, 1:]                                        # drop CLS
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        if was_training:
            self.model.train()
        return cam.detach().cpu().numpy()[0], target, probs


def upsample_cam(cam: np.ndarray, patch_len: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Patch-level CAM -> per-sample envelope."""
    out = np.repeat(np.asarray(cam, dtype=np.float32), patch_len)
    if out.size < n_samples:
        out = np.pad(out, (0, n_samples - out.size), mode="edge")
    return out[:n_samples]


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


def plot_gradcam_12lead(
    signal: np.ndarray,
    cam_samples: np.ndarray,
    path: Path,
    title: str,
    fs: float = 500.0,
) -> None:
    """Figure 3 style: 12 stacked lead panels with the CAM envelope shaded."""
    plt = _agg_pyplot()

    t = np.arange(signal.shape[-1]) / fs
    fig, axes = plt.subplots(N_LEADS, 1, figsize=(11, 13), sharex=True)
    for i, ax in enumerate(axes):
        s = signal[i]
        lo, hi = float(s.min()), float(s.max())
        ax.fill_between(t, lo, lo + cam_samples * (hi - lo), color="red", alpha=0.35, lw=0)
        ax.plot(t, s, color="black", lw=0.6)
        ax.set_ylabel(LEAD_ORDER[i], rotation=0, labelpad=22, fontsize=9, va="center")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Integrated Gradients / Gradient SHAP
# ---------------------------------------------------------------------------
def _require_captum():
    try:
        from captum.attr import GradientShap, IntegratedGradients  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "captum is required for the attribution layer.\n"
            "    pip install captum==0.9.0\n"
            "NOT 0.8.0: it declares numpy<2.0 and cannot resolve against the numpy 2.1.3 "
            "this project pins, so the install fails with ResolutionImpossible. "
            "environment/requirements.txt has the correct pin."
        ) from exc


def integrated_gradients(
    model, x: torch.Tensor, target: int, n_steps: int = 50, baseline: Optional[torch.Tensor] = None
) -> np.ndarray:
    """IG attribution map A in R^{12 x T} for one record against a zero baseline."""
    _require_captum()
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)
    base = torch.zeros_like(x) if baseline is None else baseline
    attr = ig.attribute(x.requires_grad_(True), baselines=base, target=target, n_steps=n_steps)
    return attr.detach().squeeze(0).float().cpu().numpy()


def gradient_shap(
    model,
    x: torch.Tensor,
    target: int,
    background: torch.Tensor,
    n_samples: int = 128,
) -> np.ndarray:
    """Gradient SHAP attribution against a background distribution of real records."""
    _require_captum()
    from captum.attr import GradientShap

    model.eval()
    gs = GradientShap(model)
    attr = gs.attribute(
        x.requires_grad_(True),
        baselines=background,
        target=target,
        n_samples=n_samples,
        stdevs=0.09,
    )
    return attr.detach().squeeze(0).float().cpu().numpy()


@dataclass
class LeadImportance:
    """Equation (19): I_c(l) = (1/M) sum_m (1/T) sum_t |A_c^(m)(l, t)|."""

    method: str
    class_names: List[str]
    lead_names: List[str]
    mean: np.ndarray          # (K, 12)
    ci_low: np.ndarray        # (K, 12)
    ci_high: np.ndarray       # (K, 12)
    n_samples: Dict[str, int]

    def to_rows(self) -> List[Dict[str, object]]:
        rows = []
        for ci, cname in enumerate(self.class_names):
            order = np.argsort(-self.mean[ci])
            rank = {int(l): r + 1 for r, l in enumerate(order)}
            for li, lname in enumerate(self.lead_names):
                rows.append(
                    {
                        "method": self.method,
                        "class": cname,
                        "lead": lname,
                        "lead_index": li,
                        "mean_abs_attribution": float(self.mean[ci, li]),
                        "ci95_low": float(self.ci_low[ci, li]),
                        "ci95_high": float(self.ci_high[ci, li]),
                        "rank_within_class": rank[li],
                        "n_records": self.n_samples.get(cname, 0),
                    }
                )
        return rows

    def top_k(self, k: int = 3) -> Dict[str, List[str]]:
        return {
            c: [self.lead_names[i] for i in np.argsort(-self.mean[ci])[:k]]
            for ci, c in enumerate(self.class_names)
        }


def per_class_lead_importance(
    attributions: Dict[int, List[np.ndarray]],
    method: str,
    class_names: Sequence[str] = CLASS_NAMES,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LeadImportance:
    """Aggregate per-record (12, T) maps into per-class lead importance with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    K, L = len(class_names), N_LEADS
    mean = np.zeros((K, L))
    lo = np.zeros((K, L))
    hi = np.zeros((K, L))
    counts: Dict[str, int] = {}

    alpha = (1.0 - confidence) / 2.0
    for ci, cname in enumerate(class_names):
        maps = attributions.get(ci, [])
        counts[cname] = len(maps)
        if not maps:
            continue
        # (M, 12): mean |attribution| over time, per record
        per_record = np.stack([np.abs(m).mean(axis=-1) for m in maps], axis=0)
        mean[ci] = per_record.mean(axis=0)
        if per_record.shape[0] < 2:
            lo[ci] = hi[ci] = mean[ci]
            continue
        idx = rng.integers(0, per_record.shape[0], size=(n_bootstrap, per_record.shape[0]))
        boots = per_record[idx].mean(axis=1)          # (n_bootstrap, 12)
        lo[ci] = np.quantile(boots, alpha, axis=0)
        hi[ci] = np.quantile(boots, 1.0 - alpha, axis=0)

    return LeadImportance(
        method=method,
        class_names=list(class_names),
        lead_names=list(LEAD_ORDER),
        mean=mean, ci_low=lo, ci_high=hi, n_samples=counts,
    )


def plot_lead_importance(imp: LeadImportance, path: Path, title: str) -> None:
    plt = _agg_pyplot()

    K, L = imp.mean.shape
    width = 0.8 / max(K, 1)
    xs = np.arange(L)
    fig, ax = plt.subplots(figsize=(13, 5))
    for ci, cname in enumerate(imp.class_names):
        if imp.n_samples.get(cname, 0) == 0:
            continue
        off = (ci - (K - 1) / 2) * width
        err = np.vstack(
            [
                np.maximum(imp.mean[ci] - imp.ci_low[ci], 0),
                np.maximum(imp.ci_high[ci] - imp.mean[ci], 0),
            ]
        )
        ax.bar(xs + off, imp.mean[ci], width=width, label=cname, yerr=err,
               capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(xs, imp.lead_names)
    ax.set_xlabel("ECG lead", fontweight="bold")
    ax.set_ylabel(f"Mean |{imp.method}| attribution", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(ncol=4, fontsize="small")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_global_lead_importance(imp: LeadImportance, path: Path) -> Dict[str, float]:
    """Figure 8: SHAP-based global lead importance, averaged over classes."""
    plt = _agg_pyplot()

    present = [i for i, c in enumerate(imp.class_names) if imp.n_samples.get(c, 0) > 0]
    global_mean = imp.mean[present].mean(axis=0) if present else imp.mean.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(imp.lead_names, global_mean, color="#3B76AF", edgecolor="black", lw=0.5)
    ax.set_xlabel("ECG lead", fontweight="bold")
    ax.set_ylabel(f"Mean |{imp.method}| contribution", fontweight="bold")
    ax.set_title(f"Global lead importance ({imp.method})", fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {n: float(v) for n, v in zip(imp.lead_names, global_mean)}


# ---------------------------------------------------------------------------
# Faithfulness: insertion / deletion (Figure 10)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _accuracy(model, X: torch.Tensor, y: torch.Tensor, batch: int = 64) -> float:
    model.eval()
    correct = 0
    for i in range(0, X.shape[0], batch):
        out = model(X[i : i + batch])
        correct += (out.argmax(1) == y[i : i + batch]).sum().item()
    return correct / max(X.shape[0], 1)


def insertion_deletion(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    attributions: np.ndarray,
    fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00),
    patch_len: int = 100,
) -> Dict[str, object]:
    """Faithfulness curves over attribution-ranked temporal patches.

    Deletion: progressively zero the HIGHEST-attributed patches; accuracy must fall fast.
    Insertion: start from an all-zero record and progressively restore the highest-attributed
    patches; accuracy must rise fast.

    Ranking is done per record over (lead, patch) cells, then aggregated to patches by max,
    so a patch important in any single lead counts. Reported as AUC over the fraction axis;
    `deletion_auc < insertion_auc` is the condition the tests assert.
    """
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)
    B, C, T = X.shape
    n_patch = T // patch_len

    a = np.abs(np.asarray(attributions))                      # (B, C, T)
    a = a[..., : n_patch * patch_len].reshape(B, C, n_patch, patch_len)
    patch_score = a.mean(axis=-1).max(axis=1)                 # (B, n_patch)
    order = np.argsort(-patch_score, axis=1)                  # most important first

    ins_acc: List[float] = []
    del_acc: List[float] = []
    for frac in fractions:
        k = max(1, int(round(frac * n_patch)))
        keep = torch.zeros(B, n_patch, dtype=torch.bool, device=device)
        rows = torch.arange(B, device=device).unsqueeze(1)
        cols = torch.as_tensor(order[:, :k].copy(), device=device)
        keep[rows, cols] = True
        mask = keep.repeat_interleave(patch_len, dim=1)[:, :T].unsqueeze(1)  # (B,1,T)

        ins_acc.append(_accuracy(model, X * mask, y))
        del_acc.append(_accuracy(model, X * (~mask), y))

    fr = np.asarray(fractions, dtype=float)
    return {
        "fractions": fr.tolist(),
        "insertion_accuracy": ins_acc,
        "deletion_accuracy": del_acc,
        "insertion_auc": float(np.trapezoid(ins_acc, fr) / (fr[-1] - fr[0])),
        "deletion_auc": float(np.trapezoid(del_acc, fr) / (fr[-1] - fr[0])),
        "baseline_accuracy": _accuracy(model, X, y),
        "n_records": int(B),
        "n_patches": int(n_patch),
    }


def plot_faithfulness(res: Dict[str, object], path: Path) -> None:
    plt = _agg_pyplot()

    fr = np.asarray(res["fractions"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fr, res["insertion_accuracy"], "o-")
    axes[0].set_title(
        f"Insertion (AUC={res['insertion_auc']:.3f})", fontweight="bold"
    )
    axes[0].set_xlabel("% highest-attribution patches inserted")
    axes[1].plot(fr, res["deletion_accuracy"], "o-", color="crimson")
    axes[1].set_title(
        f"Deletion (AUC={res['deletion_auc']:.3f})", fontweight="bold"
    )
    axes[1].set_xlabel("% highest-attribution patches removed")
    for a in axes:
        a.set_ylabel("Classification accuracy")
        a.axhline(res["baseline_accuracy"], ls="--", lw=0.8, color="grey",
                  label="full-signal accuracy")
        a.grid(alpha=0.3)
        a.legend(fontsize="small")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# t-SNE (Figure 9)
# ---------------------------------------------------------------------------
def tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
    max_samples: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if n > max_samples:
        sel = rng.choice(n, size=max_samples, replace=False)
        embeddings, labels = embeddings[sel], labels[sel]
    X = StandardScaler().fit_transform(embeddings)
    perp = float(min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0)))
    Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(X)
    return Z, labels


def plot_tsne(Z: np.ndarray, labels: np.ndarray, path: Path, title: str,
              class_names: Sequence[str] = CLASS_NAMES) -> None:
    plt = _agg_pyplot()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(class_names):
        m = labels == i
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.65, color=cmap(i % 10), label=name)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title(title, fontweight="bold")
    ax.legend(markerscale=2.5, fontsize="small", title="Class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Clinical concordance (Tables 9 and 10)
# ---------------------------------------------------------------------------
# Leads a cardiologist would expect to matter, per AHA/ACC and ESC criteria. Used to check
# the model's learned lead ranking against clinical expectation. Concordance is declared
# when the model's top-3 leads intersect the expected set -- this is a sanity check on the
# explanation, not evidence of clinical validity, and is reported as such.
CLINICALLY_EXPECTED_LEADS: Dict[str, List[str]] = {
    "AFIB": ["II", "V1"],
    "CD": ["V1", "V2"],
    "SB": ["II", "aVF"],
    "ST": ["V3", "V4", "V5"],
    "SVT": ["II", "V1"],
    "NSR": ["II", "aVF"],
    "OTHER": ["V1", "II"],
}

CLINICAL_CRITERIA: Dict[str, str] = {
    "AFIB": "Absent or disorganised P waves, irregularly irregular RR intervals (ESC 2020).",
    "CD": "QRS > 120 ms, RSR' in V1-V3 (RBBB) or broad slurred R in I/aVL (LBBB) "
          "(AHA/ACCF/HRS Part IV).",
    "SB": "Regular P waves with normal axis, rate < 60 bpm, prolonged RR (ACC/AHA/HRS 2018).",
    "ST": "ST elevation >= 1 mm in two contiguous leads, or T inversion / depression in "
          "precordial leads (Fourth Universal Definition of MI).",
    "SVT": "Narrow-QRS tachycardia > 100 bpm, retrograde or absent P waves, regular RR "
           "(ACC/AHA/HRS 2015).",
    "NSR": "Regular P-QRS-T, PR 120-200 ms, QRS < 120 ms, rate 60-100 bpm (AHA/ACCF/HRS).",
    "OTHER": "Mixed morphology not meeting criteria for the primary classes.",
}


def clinical_concordance(imp: LeadImportance, k: int = 3) -> List[Dict[str, object]]:
    top = imp.top_k(k)
    rows = []
    for cname in imp.class_names:
        model_leads = top.get(cname, [])
        expected = CLINICALLY_EXPECTED_LEADS.get(cname, [])
        overlap = [l for l in model_leads if l in expected]
        rows.append(
            {
                "class": cname,
                "n_records": imp.n_samples.get(cname, 0),
                "model_top_leads": model_leads,
                "clinically_expected_leads": expected,
                "overlap": overlap,
                "concordant": bool(overlap) if imp.n_samples.get(cname, 0) > 0 else None,
                "clinical_criterion": CLINICAL_CRITERIA.get(cname, ""),
            }
        )
    return rows


__all__ = [
    "ViTGradCAM", "upsample_cam", "plot_gradcam_12lead",
    "integrated_gradients", "gradient_shap",
    "LeadImportance", "per_class_lead_importance",
    "plot_lead_importance", "plot_global_lead_importance",
    "insertion_deletion", "plot_faithfulness",
    "tsne_embeddings", "plot_tsne",
    "clinical_concordance", "CLINICALLY_EXPECTED_LEADS", "CLINICAL_CRITERIA",
]

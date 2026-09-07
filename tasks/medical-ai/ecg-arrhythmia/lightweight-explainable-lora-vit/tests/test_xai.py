"""Explainability: Grad-CAM, attribution aggregation, and faithfulness.

The faithfulness tests are the ones that matter. A saliency map can look physiological and
still be unrelated to the model's decision; insertion/deletion is what separates the two.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgvit.config import (CLASS_NAMES, LEAD_ORDER, LoRAConfig, ModelConfig,  # noqa: E402
                           N_LEADS)
from ecgvit.model import build_model  # noqa: E402
from ecgvit.xai import (ViTGradCAM, clinical_concordance, insertion_deletion,  # noqa: E402
                        per_class_lead_importance, tsne_embeddings, upsample_cam)

pytestmark = pytest.mark.torch

SMALL = ModelConfig(seq_len=1000, patch_len=100, embed_dim=32, depth=2, num_heads=4)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m, _ = build_model(SMALL, LoRAConfig(rank=4))
    return m.eval()


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def test_gradcam_shape_and_range(model):
    x = torch.randn(1, N_LEADS, SMALL.seq_len)
    with ViTGradCAM(model) as cam_fn:
        cam, pred, probs = cam_fn(x)
    assert cam.shape == (SMALL.seq_len // SMALL.patch_len,), "CLS token must be dropped"
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    assert np.isclose(cam.max(), 1.0), "CAM must be min-max normalised"
    assert 0 <= pred < len(CLASS_NAMES)
    assert probs.shape == (len(CLASS_NAMES),)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_gradcam_is_not_uniform(model):
    """A constant CAM means the hooks captured nothing useful."""
    x = torch.randn(1, N_LEADS, SMALL.seq_len)
    with ViTGradCAM(model) as cam_fn:
        cam, _, _ = cam_fn(x)
    assert cam.std() > 1e-6, "CAM is constant; check the hooked layer"


def test_gradcam_target_class_changes_the_map(model):
    x = torch.randn(1, N_LEADS, SMALL.seq_len)
    with ViTGradCAM(model) as cam_fn:
        a, _, _ = cam_fn(x, class_idx=0)
        b, _, _ = cam_fn(x, class_idx=3)
    assert not np.allclose(a, b), "CAM is identical for two classes; it is class-agnostic"


def test_gradcam_releases_its_hooks(model):
    """Leaked hooks accumulate across a long XAI sweep and silently corrupt later maps."""
    before = len(model.blocks[-1].norm1._forward_hooks)
    with ViTGradCAM(model):
        during = len(model.blocks[-1].norm1._forward_hooks)
    after = len(model.blocks[-1].norm1._forward_hooks)
    assert during == before + 1
    assert after == before


def test_gradcam_restores_training_mode(model):
    model.train()
    try:
        x = torch.randn(1, N_LEADS, SMALL.seq_len)
        with ViTGradCAM(model) as cam_fn:
            cam_fn(x)
        assert model.training is True
    finally:
        model.eval()


def test_gradcam_rejects_a_batch(model):
    with ViTGradCAM(model) as cam_fn:
        with pytest.raises(ValueError):
            cam_fn(torch.randn(4, N_LEADS, SMALL.seq_len))


def test_upsample_cam_maps_patches_to_samples():
    cam = np.array([0.0, 0.5, 1.0, 0.25])
    out = upsample_cam(cam, patch_len=100, n_samples=400)
    assert out.shape == (400,)
    assert out[0] == 0.0 and out[150] == 0.5 and out[250] == 1.0
    short = upsample_cam(cam, patch_len=100, n_samples=350)
    assert short.shape == (350,)


# ---------------------------------------------------------------------------
# Attribution aggregation (equation 19)
# ---------------------------------------------------------------------------
def test_lead_importance_implements_equation_19():
    rng = np.random.default_rng(0)
    # Lead 6 (V1) is made dominant for class 1; the aggregation must recover that.
    maps = []
    for _ in range(20):
        a = rng.normal(0, 0.01, size=(N_LEADS, 500))
        a[6] += rng.normal(0, 0.5, size=500)
        maps.append(a)
    imp = per_class_lead_importance({1: maps}, "test", CLASS_NAMES, n_bootstrap=200)
    assert imp.mean.shape == (len(CLASS_NAMES), N_LEADS)
    assert int(np.argmax(imp.mean[1])) == 6
    assert imp.n_samples[CLASS_NAMES[1]] == 20
    assert np.allclose(imp.mean[0], 0.0), "a class with no records must stay at zero"


def test_lead_importance_uses_absolute_values():
    """Signed attributions cancel. A lead with strong negative contribution is still an
    important lead, so the aggregation must take |A| before averaging over time."""
    maps = [np.concatenate([np.full((1, 100), -5.0), np.zeros((11, 100))], axis=0)]
    imp = per_class_lead_importance({0: maps}, "t", CLASS_NAMES, n_bootstrap=10)
    assert imp.mean[0, 0] == pytest.approx(5.0)


def test_lead_importance_confidence_intervals_bracket_the_mean():
    rng = np.random.default_rng(1)
    maps = [rng.normal(0, 1, size=(N_LEADS, 200)) for _ in range(30)]
    imp = per_class_lead_importance({2: maps}, "t", CLASS_NAMES, n_bootstrap=500)
    ci = imp.ci_low[2] <= imp.mean[2] + 1e-9
    assert ci.all()
    assert (imp.ci_high[2] >= imp.mean[2] - 1e-9).all()
    assert (imp.ci_high[2] > imp.ci_low[2]).all(), "CI has zero width for 30 records"


def test_lead_importance_rows_are_csv_ready():
    maps = [np.random.default_rng(2).normal(0, 1, (N_LEADS, 100)) for _ in range(5)]
    imp = per_class_lead_importance({0: maps}, "IG", CLASS_NAMES, n_bootstrap=50)
    rows = imp.to_rows()
    assert len(rows) == len(CLASS_NAMES) * N_LEADS
    required = {"method", "class", "lead", "mean_abs_attribution",
                "ci95_low", "ci95_high", "rank_within_class"}
    assert required <= set(rows[0])
    ranks = sorted(r["rank_within_class"] for r in rows if r["class"] == CLASS_NAMES[0])
    assert ranks == list(range(1, N_LEADS + 1)), "ranks must be a permutation of 1..12"


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------
def test_insertion_deletion_output_contract(model):
    torch.manual_seed(3)
    X = torch.randn(8, N_LEADS, SMALL.seq_len)
    y = torch.randint(0, len(CLASS_NAMES), (8,))
    attr = np.random.default_rng(0).normal(0, 1, (8, N_LEADS, SMALL.seq_len))
    res = insertion_deletion(model, X, y, attr, (0.1, 0.5, 1.0), SMALL.patch_len)

    assert set(res) >= {"fractions", "insertion_accuracy", "deletion_accuracy",
                        "insertion_auc", "deletion_auc", "baseline_accuracy"}
    assert len(res["insertion_accuracy"]) == 3
    assert all(0.0 <= v <= 1.0 for v in res["insertion_accuracy"])
    assert all(0.0 <= v <= 1.0 for v in res["deletion_accuracy"])
    assert res["n_patches"] == SMALL.seq_len // SMALL.patch_len


def _all_zero_accuracy(model, X, y) -> float:
    """Accuracy on an all-zero input, computed the same way `insertion_deletion` does
    (integer count / n) so the two are exactly comparable rather than off by a float32
    rounding step."""
    with torch.no_grad():
        out = model(torch.zeros_like(X))
    return int((out.argmax(1) == y).sum()) / X.shape[0]


def test_insertion_at_full_fraction_recovers_the_baseline(model):
    torch.manual_seed(4)
    X = torch.randn(6, N_LEADS, SMALL.seq_len)
    y = torch.randint(0, len(CLASS_NAMES), (6,))
    attr = np.random.default_rng(1).normal(0, 1, (6, N_LEADS, SMALL.seq_len))
    res = insertion_deletion(model, X, y, attr, (0.5, 1.0), SMALL.patch_len)
    assert res["insertion_accuracy"][-1] == pytest.approx(res["baseline_accuracy"])
    assert res["deletion_accuracy"][-1] == pytest.approx(
        _all_zero_accuracy(model, X, y), abs=1e-12
    ), "deleting 100% of patches must leave an all-zero record"


class _PatchEnergyClassifier(torch.nn.Module):
    """A model whose decision is, by construction, driven by specific temporal patches.

    An untrained ViT is useless for testing the faithfulness *metric*: it predicts a near
    constant class regardless of input, so insertion and deletion both sit flat at chance
    and the comparison is vacuous. This stand-in classifies by which half of the record
    carries the energy, so a correct attribution provably moves the curves and an incorrect
    one provably does not. The ViT's own faithfulness is graded in the artefacts tier,
    after training.
    """

    def __init__(self, patch_len: int, n_patch: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.n_patch = n_patch
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        e = x.reshape(B, C, self.n_patch, self.patch_len).pow(2).mean(dim=(1, 3))
        half = self.n_patch // 2
        left, right = e[:, :half].sum(1), e[:, half:].sum(1)
        silent = (left + right) < 1e-6
        logits = torch.stack([left, right, torch.full_like(left, 1e-3)], dim=1)
        logits[silent] = torch.tensor([0.0, 0.0, 1.0])  # class 2 = "nothing here"
        return logits


def test_faithfulness_detects_a_genuinely_informative_attribution():
    """A correct attribution must make deletion hurt faster than insertion helps."""
    T, P = 1000, 100
    n_patch, n = T // P, 24
    clf = _PatchEnergyClassifier(P, n_patch).eval()

    rng = np.random.default_rng(2)
    torch.manual_seed(5)
    X = torch.zeros(n, N_LEADS, T)
    attr = np.zeros((n, N_LEADS, T))
    for i in range(n):
        side = i % 2
        hot = rng.choice(np.arange(0, 5) if side == 0 else np.arange(5, 10),
                         size=2, replace=False)
        for h in hot:
            sl = slice(h * P, (h + 1) * P)
            X[i, :, sl] = torch.randn(N_LEADS, P) * 3.0
            attr[i, :, sl] = 1.0            # attribute exactly the informative patches
    with torch.no_grad():
        y = clf(X).argmax(1)

    res = insertion_deletion(clf, X, y, attr, (0.1, 0.2, 0.5, 1.0), P)
    assert res["deletion_auc"] < res["insertion_auc"], (
        f"deletion AUC {res['deletion_auc']:.3f} is not below insertion AUC "
        f"{res['insertion_auc']:.3f}; the attribution is not faithful"
    )


def test_faithfulness_rejects_an_uninformative_attribution():
    """The counterpart: an attribution that points at empty patches must NOT pass. Without
    this, the metric could be satisfied by any input at all."""
    T, P = 1000, 100
    n_patch, n = T // P, 24
    clf = _PatchEnergyClassifier(P, n_patch).eval()

    rng = np.random.default_rng(3)
    torch.manual_seed(6)
    X = torch.zeros(n, N_LEADS, T)
    attr = np.zeros((n, N_LEADS, T))
    for i in range(n):
        side = i % 2
        pool = np.arange(0, 5) if side == 0 else np.arange(5, 10)
        hot = rng.choice(pool, size=2, replace=False)
        for h in hot:
            X[i, :, h * P : (h + 1) * P] = torch.randn(N_LEADS, P) * 3.0
        cold = [p for p in range(n_patch) if p not in hot]
        for c in cold:                       # attribute the SILENT patches instead
            attr[i, :, c * P : (c + 1) * P] = 1.0
    with torch.no_grad():
        y = clf(X).argmax(1)

    res = insertion_deletion(clf, X, y, attr, (0.1, 0.2, 0.5, 1.0), P)
    assert res["deletion_auc"] >= res["insertion_auc"], (
        "an attribution pointing at empty signal passed the faithfulness check; "
        "the metric is not discriminating"
    )


# ---------------------------------------------------------------------------
# t-SNE and clinical concordance
# ---------------------------------------------------------------------------
def test_tsne_shapes_and_subsampling():
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 1, (300, 32))
    lab = rng.integers(0, len(CLASS_NAMES), 300)
    Z, y = tsne_embeddings(emb, lab, perplexity=15, max_samples=120, seed=0)
    assert Z.shape == (120, 2) and y.shape == (120,)


def test_tsne_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    emb = rng.normal(0, 1, (80, 16))
    lab = rng.integers(0, 3, 80)
    a, _ = tsne_embeddings(emb, lab, perplexity=10, max_samples=80, seed=7)
    b, _ = tsne_embeddings(emb, lab, perplexity=10, max_samples=80, seed=7)
    assert np.allclose(a, b)


def test_clinical_concordance_reports_every_class_with_a_criterion():
    maps = {i: [np.random.default_rng(i).normal(0, 1, (N_LEADS, 100))] for i in range(7)}
    imp = per_class_lead_importance(maps, "IG", CLASS_NAMES, n_bootstrap=20)
    rows = clinical_concordance(imp, k=3)
    assert len(rows) == len(CLASS_NAMES)
    for r in rows:
        assert len(r["model_top_leads"]) == 3
        assert all(l in LEAD_ORDER for l in r["model_top_leads"])
        assert r["clinical_criterion"], f"{r['class']} has no cited criterion"
        assert set(r["clinically_expected_leads"]) <= set(LEAD_ORDER)


def test_concordance_is_none_for_a_class_with_no_records():
    imp = per_class_lead_importance({}, "IG", CLASS_NAMES, n_bootstrap=10)
    rows = clinical_concordance(imp)
    assert all(r["concordant"] is None for r in rows), (
        "a class with zero records must report concordance as unknown, not as False "
        "and certainly not as True"
    )

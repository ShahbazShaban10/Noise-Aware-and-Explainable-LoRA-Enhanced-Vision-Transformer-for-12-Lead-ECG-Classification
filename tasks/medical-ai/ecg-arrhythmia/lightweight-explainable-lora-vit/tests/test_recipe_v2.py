"""Tests for the recipe-v2 training changes and k-fold cross-validation.

Each test here pins a specific finding from `docs/RESULTS_REVIEW.md`, so a regression
reintroduces a defect that has already cost a full training run once:

* selection on val balanced accuracy kept epoch 120 of 150 while val loss bottomed at
  epoch 40, and made the whole LoRA stage a no-op (0 discordant pairs out of 1,534);
* duplicating a 184-record class 5.6x left it over-predicted (precision 0.468 against
  recall 0.595);
* a single 70/15/15 split carries about +/-3 percentage points of noise on balanced
  accuracy, so two configurations cannot be ranked from one split.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "solution" / "src"))

from ecgvit.config import (LoRAConfig, ModelConfig, TrainConfig,  # noqa: E402
                           RECIPES, apply_recipe, reset_active_class_names,
                           set_active_class_names)
from ecgvit.data import Record, assign_fold, balance_indices, class_priors, fold_split  # noqa: E402

torch = pytest.importorskip("torch")
nn = torch.nn

from ecgvit.train import (LogitAdjustedLoss, ModelEMA, build_criterion,  # noqa: E402
                          build_optimizer, run_training)


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------
def test_paper_recipe_changes_nothing():
    """The published configuration must survive the addition of a second recipe."""
    train, lora = TrainConfig(), LoRAConfig()
    apply_recipe("paper", train, lora)
    assert train.select_metric == "val_balanced_acc"
    assert train.early_stop_patience == 0
    assert train.ema_decay == 0.0
    assert train.class_weighting == "none"
    assert train.augment_strength == "basic"
    assert train.lora_lr is None
    assert train.clean_train_eval_every == 0
    assert train.temperature_scaling is False
    assert lora.train_head is False
    assert (train.pretrain_epochs, train.epochs) == (150, 30)


def test_v2_recipe_sets_the_whole_bundle():
    train, lora = TrainConfig(), LoRAConfig()
    apply_recipe("v2", train, lora)
    # Corrected after reading the full 150-epoch curve: val_loss bottoms at epoch 40 while
    # accuracy keeps improving to 120, so selecting on it costs 3.9 points of balanced
    # accuracy. macro AUC is threshold-free and tracks discrimination; the calibration
    # problem val_loss was detecting is handled by temperature scaling instead.
    assert train.select_metric == "val_macro_auc"
    assert train.early_stop_patience == 25
    assert train.clean_train_eval_every == 5
    assert train.temperature_scaling is True
    assert train.ema_decay == 0.999
    assert train.class_weighting == "logit_adjust"
    assert train.augment_strength == "physio"
    assert train.lora_lr == pytest.approx(1e-3)
    assert lora.train_head is True
    # The epoch budget is unchanged: early stopping shortens the run, the schedule is not
    # silently retuned as well.
    assert (train.pretrain_epochs, train.epochs) == (150, 30)


def test_unknown_recipe_and_unknown_field_are_hard_errors():
    with pytest.raises(KeyError):
        apply_recipe("nope", TrainConfig(), LoRAConfig())
    RECIPES["_bad"] = {"train": {"not_a_field": 1}}
    try:
        with pytest.raises(AttributeError):
            apply_recipe("_bad", TrainConfig(), LoRAConfig())
    finally:
        del RECIPES["_bad"]


# ---------------------------------------------------------------------------
# long-tail handling in the loss rather than the sampler
# ---------------------------------------------------------------------------
def test_logit_adjustment_matches_the_published_formula():
    priors = np.array([0.5, 0.3, 0.2])
    loss = LogitAdjustedLoss(priors, tau=1.0)
    assert torch.allclose(
        loss.adjustment, torch.log(torch.tensor(priors, dtype=torch.float32)), atol=1e-6
    )
    # tau = 0 must be exactly plain cross-entropy, so the knob has a true no-op setting.
    logits = torch.randn(8, 3)
    target = torch.randint(0, 3, (8,))
    plain = nn.CrossEntropyLoss()(logits, target)
    assert LogitAdjustedLoss(priors, tau=0.0)(logits, target) == pytest.approx(
        float(plain), abs=1e-6
    )


def test_logit_adjustment_penalises_the_majority_class_at_train_time_only():
    """The correction lives in the loss; inference sees unadjusted logits.

    A model that always predicts the majority class should be punished more under logit
    adjustment than under plain cross-entropy -- that is the entire mechanism.
    """
    priors = np.array([0.9, 0.05, 0.05])
    logits = torch.tensor([[3.0, 0.0, 0.0]] * 4)
    minority = torch.tensor([1, 1, 2, 2])
    plain = float(nn.CrossEntropyLoss()(logits, minority))
    adjusted = float(LogitAdjustedLoss(priors, tau=1.0)(logits, minority))
    assert adjusted > plain


def test_inverse_frequency_weights_are_mean_one_and_capped():
    cfg = TrainConfig(class_weighting="inverse_freq", label_smoothing=0.0)
    # A prior as extreme as OTHER's (184 of 8,719 training records) is exactly the case
    # where an unclipped inverse-frequency weight takes over the gradient.
    priors = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    crit = build_criterion(cfg, priors)
    w = crit.weight.numpy()
    assert w.max() <= 10.0 + 1e-6
    assert np.all(w > 0)
    assert np.argmax(w) == len(priors) - 1        # rarest class gets the largest weight


def test_class_weighting_requires_priors_and_rejects_unknown_values():
    with pytest.raises(ValueError, match="priors"):
        build_criterion(TrainConfig(class_weighting="logit_adjust"))
    with pytest.raises(ValueError, match="unknown class_weighting"):
        build_criterion(TrainConfig(class_weighting="sqrt_inv"), np.array([0.5, 0.5]))


def test_default_criterion_is_plain_cross_entropy():
    crit = build_criterion(TrainConfig())
    assert isinstance(crit, nn.CrossEntropyLoss)
    assert crit.weight is None


def _records(counts):
    out, i = [], 0
    for idx, (label, n) in enumerate(counts.items()):
        for _ in range(n):
            out.append(Record(f"R{i:05d}", Path("x.hea"), Path("x.mat"),
                              500.0, 5000, 12, (), label, idx, "train"))
            i += 1
    return out


def test_class_priors_sum_to_one_and_follow_the_counts():
    recs = _records({"A": 60, "B": 30, "C": 10})
    p = class_priors(recs, 3)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]
    assert p[0] == pytest.approx(0.6)


def test_class_priors_floors_an_empty_class_so_the_log_stays_finite():
    p = class_priors(_records({"A": 10}), 3)
    assert np.all(np.isfinite(np.log(p)))
    assert p[1] > 0 and p[2] > 0


def test_no_balancing_strategy_uses_every_record_exactly_once():
    """The counterpart of logit adjustment: no duplication at all.

    On the completed run OTHER was duplicated 5.603x -- 5.6 gradient steps over the same
    184 records -- and ended with precision 0.468 against recall 0.595.
    """
    # The balance report is keyed by the ACTIVE class names, which follow the resolution
    # order -- canonical_v2 calls the ectopy class VE, not OTHER. Set them explicitly here
    # so the coupling is visible rather than inherited from whatever ran last.
    set_active_class_names(["NSR", "AFIB", "VE"])
    try:
        recs = _records({"NSR": 100, "AFIB": 40, "VE": 5})
        spec = {"augmentation": {"min_unique_samples_per_class": 50,
                                 "max_duplication_factor": 8,
                                 "target": "median_class_count"}}
        idxs, report = balance_indices(recs, spec, strategy="none")
    finally:
        reset_active_class_names()
    assert sorted(idxs) == list(range(len(recs)))
    assert report.strategy == "none_natural_distribution"
    assert all(f == 1.0 for f in report.duplication_factor.values())
    # And it does NOT trip the min-unique guard, because it never inflates anything: the
    # guard exists to stop a 5-record class being multiplied into parity.
    assert report.unique_support["VE"] == 5 == report.support["VE"]


# ---------------------------------------------------------------------------
# weight EMA
# ---------------------------------------------------------------------------
def test_ema_converges_to_stationary_weights():
    model = nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.5)
    ema = ModelEMA(model, decay=0.9)
    for _ in range(500):
        ema.update(model)
    assert torch.allclose(ema.shadow["weight"], model.weight, atol=1e-5)
    assert torch.allclose(ema.shadow["bias"], model.bias, atol=1e-5)


def test_ema_lags_a_moving_weight_and_the_swap_round_trips():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.99)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(10.0)
    for _ in range(5):
        ema.update(model)
    assert float(ema.shadow["weight"].mean()) < 10.0     # averaged, not copied

    live = {k: v.clone() for k, v in model.state_dict().items()}
    backup = ema.copy_to(model)
    assert not torch.allclose(model.weight, live["weight"])
    model.load_state_dict(backup, strict=True)
    for k, v in live.items():
        assert torch.allclose(model.state_dict()[k], v), k


def test_ema_rejects_a_degenerate_decay():
    model = nn.Linear(2, 2)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ModelEMA(model, decay=bad)


# ---------------------------------------------------------------------------
# checkpoint selection and early stopping
# ---------------------------------------------------------------------------
class _ConstantModel(nn.Module):
    """Output independent of the input and of training, so val metrics never move.

    That makes the selection and early-stopping logic testable without depending on
    whether a real model happens to improve.
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))
        self.register_buffer("logits", torch.tensor([2.0, 0.5, 0.1][:n_classes]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(x.shape[0], -1) + 0.0 * self.unused


def _batches(n_batches=3, bs=4, n_classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(bs, 4, generator=g),
         torch.randint(0, n_classes, (bs,), generator=g),
         torch.arange(bs))
        for _ in range(n_batches)
    ]


def _cfg(**kw):
    base = dict(pretrain_epochs=1, epochs=1, batch_size=4, warmup_epochs=0,
                mixup_alpha=0.0, mixup_prob=0.0, label_smoothing=0.0, amp=False,
                num_workers=0, min_epochs=3, seed=0)
    base.update(kw)
    return TrainConfig(**base)


def test_early_stopping_fires_after_patience_and_not_before_min_epochs(tmp_path):
    """With a frozen validation metric, the stop epoch is arithmetic, not luck."""
    data = _batches()
    hist = run_training(
        _ConstantModel(), data, data, torch.device("cpu"),
        _cfg(select_metric="val_loss", early_stop_patience=2, min_epochs=5),
        n_epochs=50, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
    )
    assert hist.stopped_early is True
    assert hist.best_epoch == 1                     # nothing ever improved on epoch 1
    assert hist.epochs_run == 5                     # min_epochs gate, then patience is met
    assert hist.epochs_planned == 50
    assert len(hist.epochs) == 5


def test_zero_patience_disables_early_stopping(tmp_path):
    data = _batches()
    hist = run_training(
        _ConstantModel(), data, data, torch.device("cpu"),
        _cfg(select_metric="val_loss", early_stop_patience=0),
        n_epochs=6, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
    )
    assert hist.stopped_early is False
    assert hist.epochs_run == 6


@pytest.mark.parametrize(
    "metric,pick",
    [("val_loss", "min"), ("val_balanced_acc", "max"), ("val_macro_auc", "max")],
)
def test_checkpoint_is_the_best_epoch_under_the_declared_metric(tmp_path, metric, pick):
    """The regression test for the finding that sank the LoRA stage.

    Stage 2 of the published run saved its epoch-1 adapters because val balanced accuracy
    peaked there, while val loss went on improving to epoch 17 -- so the saved model was
    the frozen backbone, and McNemar found 0 discordant pairs out of 1,534. Whatever
    metric is declared, the checkpoint on disk must be the epoch that optimised it.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    data = _batches(n_batches=4, seed=1)
    val = _batches(n_batches=2, seed=2)
    ckpt = tmp_path / "m.pt"
    hist = run_training(
        model, data, val, torch.device("cpu"),
        _cfg(select_metric=metric, early_stop_patience=0, lr=1e-2),
        n_epochs=8, stage="t", checkpoint_path=ckpt, n_classes=3,
    )
    series = [getattr(e, metric) for e in hist.epochs]
    want = (int(np.argmin(series)) if pick == "min" else int(np.argmax(series))) + 1
    assert hist.best_epoch == want, f"{metric}: history says {series}"

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["epoch"] == want
    assert saved["select_metric"] == metric
    assert saved["select_value"] == pytest.approx(series[want - 1], abs=1e-9)


def test_unknown_select_metric_is_rejected(tmp_path):
    data = _batches()
    with pytest.raises(ValueError, match="unknown select_metric"):
        run_training(
            _ConstantModel(), data, data, torch.device("cpu"),
            _cfg(select_metric="val_f1"),
            n_epochs=2, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
        )


def test_ema_checkpoint_differs_from_the_live_weights(tmp_path):
    """With EMA on, what lands on disk must be the average, not the last iterate."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    data = _batches(n_batches=4, seed=3)
    ckpt = tmp_path / "m.pt"
    run_training(
        model, data, data, torch.device("cpu"),
        _cfg(select_metric="val_loss", ema_decay=0.9, lr=1e-1, early_stop_patience=0),
        n_epochs=4, stage="t", checkpoint_path=ckpt, n_classes=3,
    )
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["ema_decay"] == 0.9
    live = model.state_dict()
    assert any(
        not torch.allclose(saved["state_dict"][k], live[k]) for k in live
    ), "EMA checkpoint is identical to the live weights"
    # Training must continue from the trajectory, not from its own average.
    assert all(torch.isfinite(v).all() for v in live.values())


def test_adapter_learning_rate_override():
    model = nn.Linear(4, 3)
    cfg = TrainConfig(lr=1e-4, lora_lr=1e-3)
    assert build_optimizer(model, cfg).param_groups[0]["lr"] == pytest.approx(1e-4)
    assert build_optimizer(model, cfg, lr=cfg.lora_lr).param_groups[0]["lr"] == \
        pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------
SALT = "ecgvit-chapman-v1"


def test_fold_assignment_is_deterministic_and_in_range():
    ids = [f"JS{i:05d}" for i in range(2000)]
    first = [assign_fold(r, SALT, 5) for r in ids]
    assert first == [assign_fold(r, SALT, 5) for r in ids]
    assert set(first) == {0, 1, 2, 3, 4}


def test_folds_are_roughly_balanced():
    ids = [f"JS{i:05d}" for i in range(10000)]
    counts = np.bincount([assign_fold(r, SALT, 5) for r in ids], minlength=5)
    assert counts.min() > 0.9 * 2000 and counts.max() < 1.1 * 2000, counts


def test_fold_index_rejects_degenerate_k():
    for bad in (0, 1, -3):
        with pytest.raises(ValueError):
            assign_fold("JS00001", SALT, bad)


def test_every_fold_is_test_once_and_validation_once():
    k = 5
    maps = [fold_split(f, k) for f in range(k)]
    for f in range(k):
        assert sum(m[f] == "test" for m in maps) == 1
        assert sum(m[f] == "val" for m in maps) == 1
    for f, m in enumerate(maps):
        assert m[f] == "test"
        assert m[(f + 1) % k] == "val"
        # Validation is never drawn from the test fold: selection must not see the fold it
        # is scored on.
        assert m[f] != m[(f + 1) % k]
        assert sorted(m) == list(range(k))


def test_cross_validation_tests_every_record_exactly_once():
    ids = [f"JS{i:05d}" for i in range(3000)]
    k = 5
    tested = {}
    for f in range(k):
        mapping = fold_split(f, k)
        for r in ids:
            if mapping[assign_fold(r, SALT, k)] == "test":
                tested[r] = tested.get(r, 0) + 1
    assert set(tested) == set(ids)
    assert set(tested.values()) == {1}


def test_fold_split_rejects_an_out_of_range_fold():
    for bad in (-1, 5, 99):
        with pytest.raises(ValueError):
            fold_split(bad, 5)


# ---------------------------------------------------------------------------
# tokeniser stride ablation
# ---------------------------------------------------------------------------
def test_reference_geometry_is_untouched_and_the_ablation_is_exact():
    """patch_len 100 must still reproduce Table 8 to the digit.

    The stride-25 ablation (patch_len 50) exists because a conduction disturbance is
    defined by QRS crossing 120 ms and the reference tokeniser's 100 ms sub-patches cannot
    resolve that. It necessarily changes the parameter count, so the count is pinned here
    rather than left to be noticed when Table 8 stops matching.
    """
    from ecgvit.model import build_model

    ref, ref_acct = build_model(ModelConfig(patch_len=100), LoRAConfig(enabled=False))
    _, ref_lora = build_model(ModelConfig(patch_len=100), LoRAConfig())
    assert ref.n_patches == 50
    assert ref_acct.total_parameters == 1_648_839
    assert ref_lora.total_parameters == 1_779_911
    assert ref_lora.trainable_parameters == 131_072

    fine, fine_acct = build_model(ModelConfig(patch_len=50), LoRAConfig(enabled=False))
    _, fine_lora = build_model(ModelConfig(patch_len=50), LoRAConfig())
    assert fine.n_patches == 100
    # conv1 loses 12*64*25 weights; the positional embedding gains 50*128.
    assert fine_acct.total_parameters == 1_648_839 - 12 * 64 * 25 + 50 * 128
    assert fine_acct.total_parameters == 1_636_039
    # The adapters live in the encoder blocks, which the stride does not touch.
    assert fine_lora.trainable_parameters == 131_072

    x = torch.randn(2, 12, 5000)
    assert fine(x).shape == (2, ModelConfig().n_classes)


def test_training_the_lora_head_costs_exactly_the_head():
    """--train-head moves the reduction from 92.05% to 92.00%, and that is 903 parameters."""
    from ecgvit.model import build_model

    _, frozen = build_model(ModelConfig(), LoRAConfig(train_head=False))
    _, trained = build_model(ModelConfig(), LoRAConfig(train_head=True))
    head = ModelConfig().embed_dim * ModelConfig().n_classes + ModelConfig().n_classes
    assert head == 903
    assert trained.trainable_parameters - frozen.trainable_parameters == head
    assert trained.trainable_reduction_pct == pytest.approx(92.0, abs=0.01)


# ---------------------------------------------------------------------------
# physiologically motivated augmentation
# ---------------------------------------------------------------------------
def _dataset(label="CD", strength="physio", seed=0):
    from ecgvit.data import ECGDataset

    return ECGDataset([], augment=True, augment_strength=strength, rng_seed=seed)


def test_physio_augmentation_preserves_geometry_and_stays_finite():
    ds = _dataset()
    x = np.random.default_rng(0).standard_normal((12, 5000)).astype(np.float32)
    for _ in range(40):
        y = ds._augment_physio(x, "CD")
        assert y.shape == (12, 5000)
        assert y.dtype == np.float32
        assert np.isfinite(y).all()


def test_time_warp_is_suppressed_for_rate_defined_classes():
    """Warping a bradycardia strip changes its heart rate, which IS the label.

    A 55 bpm recording stretched by 1.15x is 63 bpm and no longer bradycardic. Detected
    here by correlation: a warped copy decorrelates from the original far more than the
    amplitude and noise transforms do, so a rate-defined class must never show that.
    """
    ds = _dataset()
    t = np.linspace(0, 10, 5000, dtype=np.float32)
    x = np.tile(np.sin(2 * np.pi * 1.2 * t), (12, 1)).astype(np.float32)

    def worst_corr(label, n=120):
        d = _dataset(seed=7)
        out = []
        for _ in range(n):
            y = d._augment_physio(x, label)
            a, b = x.ravel(), y.ravel()
            if a.std() == 0 or b.std() == 0:
                continue
            out.append(abs(float(np.corrcoef(a, b)[0, 1])))
        return min(out)

    # SB is rate-defined; CD is a morphology finding where warping is legitimate.
    assert "SB" in ds.rate_defined_classes
    assert "CD" not in ds.rate_defined_classes
    assert worst_corr("CD") < worst_corr("SB")


def test_lead_dropout_can_zero_leads_and_does_not_touch_them_all():
    ds = _dataset(seed=3)
    x = np.ones((12, 5000), dtype=np.float32)
    saw_dropout = False
    for _ in range(200):
        y = ds._augment_physio(x, "CD")
        zeroed = int(np.sum(np.all(np.abs(y) < 1e-6, axis=1)))
        assert zeroed <= 2, "lead dropout must never remove more than two leads"
        saw_dropout |= zeroed > 0
    assert saw_dropout, "lead dropout never fired in 200 draws"


def test_basic_and_physio_are_different_transforms():
    ds_b, ds_p = _dataset(strength="basic", seed=1), _dataset(strength="physio", seed=1)
    x = np.random.default_rng(1).standard_normal((12, 5000)).astype(np.float32)
    assert not np.allclose(ds_b._augment(x), ds_p._augment_physio(x, "CD"))


def test_unknown_augment_strength_is_rejected():
    from ecgvit.data import ECGDataset

    with pytest.raises(ValueError, match="augment_strength"):
        ECGDataset([], augment=True, augment_strength="aggressive")


# ---------------------------------------------------------------------------
# measuring fit honestly, and calibrating afterwards
# ---------------------------------------------------------------------------
class _LogitPassthrough(nn.Module):
    """Returns its input as logits, so a test controls the logits exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.0 * self.unused


def _overconfident(n=400, k=3, margin=8.0, wrong_frac=0.25, seed=0):
    """Confident predictions that are wrong a quarter of the time.

    This is the shape of the observed run: validation ACCURACY holds up while validation
    NLL climbs, because the errors that remain are made with high confidence.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, k, (n,), generator=g)
    shown = y.clone()
    flip = torch.rand(n, generator=g) < wrong_frac
    shown[flip] = (y[flip] + 1) % k
    return torch.nn.functional.one_hot(shown, k).float() * margin, y


def test_ece_is_zero_for_a_calibrated_model_and_positive_for_a_confident_one():
    from ecgvit.train import expected_calibration_error

    rng = np.random.default_rng(0)
    n = 4000
    # Calibrated: a two-class model that says p and is right with probability p.
    p = rng.uniform(0.5, 1.0, n)
    y = (rng.random(n) > p).astype(int)          # 1 = the model is wrong
    probs = np.stack([p, 1 - p], axis=1)
    assert expected_calibration_error(probs, y) < 0.03

    # Over-confident: always says 0.99, right only 70% of the time.
    p2 = np.full(n, 0.99)
    y2 = (rng.random(n) > 0.70).astype(int)
    probs2 = np.stack([p2, 1 - p2], axis=1)
    assert expected_calibration_error(probs2, y2) > 0.25


def test_temperature_scaling_softens_an_overconfident_model_without_moving_accuracy():
    """The fix for rising validation loss that costs nothing.

    Dividing every logit by one positive scalar is monotone, so the argmax cannot move.
    That is why calibration belongs here and not in the checkpoint-selection rule: selecting
    on val_loss to chase the same problem gave up 3.9 points of balanced accuracy on the
    completed run.
    """
    from ecgvit.train import fit_temperature

    logits, y = _overconfident()
    loader = [(logits[i:i + 64], y[i:i + 64], torch.arange(64))
              for i in range(0, len(y), 64)]
    out = fit_temperature(_LogitPassthrough(), loader, torch.device("cpu"))

    assert out["temperature"] > 1.0, "an over-confident model must be softened, not sharpened"
    assert out["val_nll_after"] < out["val_nll_before"]
    assert out["val_ece_after"] < out["val_ece_before"]
    assert out["n_val_records"] == len(y)
    # Accuracy is unchanged by construction; fit_temperature raises if it ever is not.
    assert out["val_accuracy_unchanged"] == pytest.approx(0.75, abs=0.05)


def test_temperature_scaling_leaves_a_calibrated_model_alone():
    from ecgvit.train import fit_temperature

    g = torch.Generator().manual_seed(1)
    logits = torch.randn(600, 4, generator=g)
    y = logits.argmax(1)
    # Labels drawn from the model's own distribution: already well calibrated.
    y = torch.multinomial(logits.softmax(1), 1, generator=g).squeeze(1)
    loader = [(logits[i:i + 64], y[i:i + 64], torch.arange(64))
              for i in range(0, len(y), 64)]
    out = fit_temperature(_LogitPassthrough(), loader, torch.device("cpu"))
    assert 0.7 < out["temperature"] < 1.4, out["temperature"]


def test_clean_train_metrics_are_recorded_on_the_declared_schedule(tmp_path):
    """The running train_acc cannot answer 'is it overfitting?'; this can.

    On the completed run train_acc sat BELOW val_acc in 137 of 150 epochs, because it is
    measured on the balanced, augmented split under MixUp. The generalisation gap has to be
    measured in eval mode on the un-augmented training split or it means nothing.
    """
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    data = _batches(n_batches=3, seed=5)
    hist = run_training(
        model, data, data, torch.device("cpu"),
        _cfg(select_metric="val_macro_auc", early_stop_patience=0,
             clean_train_eval_every=3),
        n_epochs=7, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
        clean_train_loader=data,
    )
    measured = [e.epoch for e in hist.epochs if e.generalisation_gap == e.generalisation_gap]
    # First and last epoch always, plus every third.
    assert measured == [1, 3, 6, 7], measured
    for e in hist.epochs:
        if e.epoch in measured:
            assert 0.0 <= e.clean_train_acc <= 1.0
            assert e.generalisation_gap == pytest.approx(e.clean_train_acc - e.val_acc)
        else:
            assert e.clean_train_acc != e.clean_train_acc      # NaN


def test_clean_train_eval_is_off_by_default(tmp_path):
    data = _batches(seed=6)
    hist = run_training(
        _ConstantModel(), data, data, torch.device("cpu"),
        _cfg(select_metric="val_loss", early_stop_patience=0),
        n_epochs=3, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
        clean_train_loader=data,
    )
    assert all(e.generalisation_gap != e.generalisation_gap for e in hist.epochs)


def test_macro_auc_is_always_recorded_not_only_when_selected(tmp_path):
    """A curve you did not record is a curve you cannot go back and read."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    data = _batches(n_batches=3, seed=7)
    hist = run_training(
        model, data, data, torch.device("cpu"),
        _cfg(select_metric="val_balanced_acc", early_stop_patience=0),
        n_epochs=3, stage="t", checkpoint_path=tmp_path / "m.pt", n_classes=3,
    )
    for e in hist.epochs:
        assert 0.0 <= e.val_macro_auc <= 1.0, e.val_macro_auc

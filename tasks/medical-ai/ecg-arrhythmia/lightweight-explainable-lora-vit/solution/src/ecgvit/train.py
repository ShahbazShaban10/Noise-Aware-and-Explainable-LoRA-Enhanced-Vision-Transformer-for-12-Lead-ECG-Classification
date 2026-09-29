"""Two-stage training (manuscript Table 2 and Table 8).

    Stage 1  pretrain   full fine-tune, no adapters, 150 epochs  -> 1,648,839 params
    Stage 2  adapt      inject LoRA r=8, freeze everything else,
                        30 epochs                                -> 131,072 trainable

Stage 2 loads stage 1's weights. Adapting a *randomly initialised* frozen backbone with a
frozen random head is a different and much weaker experiment; `train_lora` refuses to start
without a base checkpoint unless `--allow-random-backbone` is passed explicitly.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import CLASS_NAMES, LoRAConfig, ModelConfig, TrainConfig
from .lora import count_parameters, freeze_backbone, inject_lora
from .model import LoRAViT

log = logging.getLogger("ecgvit.train")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> Dict[str, object]:
    info: Dict[str, object] = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        cap = f"sm_{props.major}{props.minor}"
        info.update(
            {
                "gpu_name": props.name,
                "capability": cap,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_runtime": torch.version.cuda,
                "compiled_architectures": torch.cuda.get_arch_list(),
            }
        )
        arch_ok = any(cap in a for a in torch.cuda.get_arch_list())
        info["architecture_supported_by_build"] = arch_ok
        if not arch_ok:
            log.warning(
                "This torch build (%s, CUDA %s) has no kernels for %s (%s). Arch list: %s. "
                "On Blackwell (RTX 50-series, sm_120) install a cu128 or newer wheel.",
                torch.__version__, torch.version.cuda, cap, props.name,
                torch.cuda.get_arch_list(),
            )
    return info


# ---------------------------------------------------------------------------
# Loss / augmentation
# ---------------------------------------------------------------------------
class MixUp:
    """Equations (17)-(18): x~ = lam*x_a + (1-lam)*x_b, loss mixed with the same lam."""

    def __init__(self, alpha: float = 0.2, prob: float = 0.5, seed: int = 42) -> None:
        self.alpha = alpha
        self.prob = prob
        self._rng = np.random.default_rng(seed)

    def maybe_apply(self, x: torch.Tensor, y: torch.Tensor):
        if self.alpha <= 0 or self._rng.random() > self.prob:
            return x, y, y, 1.0
        lam = float(self._rng.beta(self.alpha, self.alpha))
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    @staticmethod
    def loss(criterion, pred, y_a, y_b, lam: float) -> torch.Tensor:
        if lam == 1.0:
            return criterion(pred, y_a)
        return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class ModelEMA:
    """Exponential moving average of the model weights.

    Averaged weights sit nearer the centre of the loss basin than any single iterate, which
    is worth roughly a point of balanced accuracy here and, more usefully, makes the
    epoch-to-epoch validation curve stable enough that early stopping is not reacting to
    noise. The EMA weights are what get evaluated and checkpointed when enabled.

    Frozen parameters are tracked too: they never change, so their average is themselves,
    and copying them keeps the shadow state a complete, loadable state_dict.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.buffers: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if not v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        # Warm up the decay so the first few hundred steps are not dominated by the random
        # initialisation: d_t = min(decay, (1+t)/(10+t)).
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        sd = model.state_dict()
        for k, shadow in self.shadow.items():
            shadow.mul_(d).add_(sd[k].detach().float(), alpha=1.0 - d)
        for k in self.buffers:
            self.buffers[k] = sd[k].detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out = {k: v.clone() for k, v in self.shadow.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Load the averaged weights into `model`, returning the weights it replaced."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(
            {k: v.to(dtype=backup[k].dtype) for k, v in self.state_dict().items()},
            strict=True,
        )
        return backup


class LogitAdjustedLoss(nn.Module):
    """Cross-entropy with the class prior added to the logits during training.

    Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021: adding
    tau * log(pi_y) to the logit of class y before the softmax makes the argmax of the
    UNADJUSTED logits at test time the Bayes-optimal balanced-error classifier. The
    correction lives entirely in the loss, so no record is ever duplicated and no gradient
    step is spent on a copy the model has already seen.

    tau = 0 recovers plain cross-entropy; tau = 1 fully compensates the prior.
    """

    def __init__(self, priors, tau: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        p = torch.as_tensor(priors, dtype=torch.float32)
        self.register_buffer("adjustment", tau * torch.log(p.clamp_min(1e-12)))
        self.tau = float(tau)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce(logits + self.adjustment.to(logits.device), target)


def build_criterion(cfg: TrainConfig, priors=None) -> nn.Module:
    """The training loss for a recipe. Validation always uses plain cross-entropy."""
    if cfg.class_weighting == "logit_adjust":
        if priors is None:
            raise ValueError("class_weighting='logit_adjust' needs the training class priors")
        return LogitAdjustedLoss(priors, cfg.logit_adjust_tau, cfg.label_smoothing)
    if cfg.class_weighting == "inverse_freq":
        if priors is None:
            raise ValueError("class_weighting='inverse_freq' needs the training class priors")
        p = np.asarray(priors, dtype=np.float64)
        w = (1.0 / np.maximum(p, 1e-12))
        w = w / w.mean()                      # mean-1 so the loss scale is unchanged
        w = np.clip(w, 0.0, 10.0)             # cap: an unclipped weight on a 184-record
        return nn.CrossEntropyLoss(          # class dominates the gradient entirely
            weight=torch.tensor(w, dtype=torch.float32),
            label_smoothing=cfg.label_smoothing,
        )
    if cfg.class_weighting != "none":
        raise ValueError(f"unknown class_weighting {cfg.class_weighting!r}")
    return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


def build_optimizer(
    model: nn.Module, cfg: TrainConfig, lr: Optional[float] = None
) -> torch.optim.Optimizer:
    """AdamW with no weight decay on biases and norms (Table 2).

    `lr` overrides `cfg.lr`; stage 2 passes `cfg.lora_lr` through it. LoRA's B is
    zero-initialised, so an adapter starts with no output at all and needs a larger step
    than a backbone weight that is already near its optimum.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or ".norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    if not decay and not no_decay:
        raise RuntimeError("no trainable parameters; check the freeze configuration")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr if lr is None else float(lr),
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, cfg: TrainConfig, n_epochs: int):
    """Cosine annealing with linear warm-up (Table 2)."""
    warmup = max(0, min(cfg.warmup_epochs, n_epochs - 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup))
    if warmup == 0:
        return cosine
    warm = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup])


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_balanced_acc: float
    lr: float
    seconds: float
    val_macro_auc: float = float("nan")
    # Measured in EVAL mode on the un-augmented, un-duplicated training split. `train_loss`
    # and `train_acc` above are the running figures from the augmented, MixUp'd, balanced
    # loader and are not comparable to anything; these are.
    clean_train_loss: float = float("nan")
    clean_train_acc: float = float("nan")
    clean_train_balanced_acc: float = float("nan")
    generalisation_gap: float = float("nan")   # clean_train_acc - val_acc


#: How to compare two values of each selection metric, and its neutral starting point.
#: `val_loss` is the recipe-v2 default: on the completed 150-epoch run `val_balanced_acc`
#: peaked at epoch 120 while `val_loss` bottomed at epoch 40 and then rose 10%, so
#: selecting on balanced accuracy kept a materially worse-calibrated checkpoint.
SELECT_METRICS: Dict[str, Tuple[str, float]] = {
    "val_loss": ("min", float("inf")),
    "val_balanced_acc": ("max", -float("inf")),
    "val_macro_auc": ("max", -float("inf")),
}


@dataclass
class TrainHistory:
    stage: str
    epochs: List[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_balanced_acc: float = -1.0
    select_metric: str = "val_balanced_acc"
    best_select_value: float = float("nan")
    stopped_early: bool = False
    epochs_run: int = 0
    epochs_planned: int = 0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "best_epoch": self.best_epoch,
            "best_val_balanced_acc": self.best_val_balanced_acc,
            "select_metric": self.select_metric,
            "best_select_value": self.best_select_value,
            "stopped_early": self.stopped_early,
            "epochs_run": self.epochs_run,
            "epochs_planned": self.epochs_planned,
            "epochs": [vars(e) for e in self.epochs],
        }


@torch.no_grad()
def _evaluate_loader(
    model, loader, device, n_classes: int, want_auc: bool = False
) -> Tuple[float, float, float, float]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    model.eval()
    criterion = nn.CrossEntropyLoss()
    tot_loss, correct, n = 0.0, 0, 0
    ys, ps, probs = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        out = model(x)
        tot_loss += criterion(out, y).item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
        if want_auc:
            probs.append(out.float().softmax(1).cpu().numpy())
    if n == 0:
        return float("nan"), 0.0, 0.0, float("nan")
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    auc = float("nan")
    if want_auc:
        pr = np.concatenate(probs)
        present = np.unique(y_all)
        # One-vs-rest macro AUC over the classes actually present. A class with no
        # validation records has no ROC curve at all; averaging a placeholder in would be a
        # fabricated number, so it is left out and the average says so by its support.
        if present.size >= 2:
            try:
                aucs = [
                    roc_auc_score((y_all == c).astype(int), pr[:, c])
                    for c in present if 0 < (y_all == c).sum() < len(y_all)
                ]
                auc = float(np.mean(aucs)) if aucs else float("nan")
            except ValueError:
                auc = float("nan")
    return tot_loss / n, correct / n, float(balanced_accuracy_score(y_all, p_all)), auc


def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
    n_epochs: int,
    stage: str,
    checkpoint_path: Path,
    n_classes: int = len(CLASS_NAMES),
    class_priors=None,
    class_names: Optional[Sequence[str]] = None,
    lr: Optional[float] = None,
    clean_train_loader=None,
) -> TrainHistory:
    model.to(device)
    criterion = build_criterion(cfg, class_priors).to(device)
    optimizer = build_optimizer(model, cfg, lr=lr)
    scheduler = build_scheduler(optimizer, cfg, n_epochs)
    mixup = MixUp(cfg.mixup_alpha, cfg.mixup_prob, seed=cfg.seed)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    if cfg.select_metric not in SELECT_METRICS:
        raise ValueError(
            f"unknown select_metric {cfg.select_metric!r}; "
            f"available: {sorted(SELECT_METRICS)}"
        )
    direction, best_value = SELECT_METRICS[cfg.select_metric]
    # Always computed, not only when it is the selection metric: a curve you did not record
    # is a curve you cannot go back and read, and macro AUC is the one statistic that
    # separates "learning to discriminate" from "learning to be confident".
    want_auc = True

    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    history = TrainHistory(
        stage=stage, select_metric=cfg.select_metric, epochs_planned=n_epochs
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(class_names) if class_names is not None else list(CLASS_NAMES)
    since_improved = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, correct, n = 0.0, 0, 0

        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            xm, y_a, y_b, lam = mixup.maybe_apply(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(xm)
                loss = MixUp.loss(criterion, out, y_a, y_b, lam)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip_norm
                )
                optimizer.step()

            if ema is not None:
                ema.update(model)

            tot_loss += loss.item() * x.size(0)
            # Accuracy against the true labels, not the mixed targets.
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

        scheduler.step()

        # With EMA on, the averaged weights are the model: validate and checkpoint those,
        # not the last iterate. Restore the live weights afterwards so training continues
        # from the trajectory rather than from its own average.
        backup = ema.copy_to(model) if ema is not None else None
        try:
            vl, va, vba, vauc = _evaluate_loader(
                model, val_loader, device, n_classes, want_auc=want_auc
            )
            selected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        finally:
            if backup is not None:
                model.load_state_dict(backup, strict=True)

        ctl = cta = ctb = gap = float("nan")
        if clean_train_loader is not None and cfg.clean_train_eval_every > 0 and (
            epoch % cfg.clean_train_eval_every == 0 or epoch == 1 or epoch == n_epochs
        ):
            backup2 = ema.copy_to(model) if ema is not None else None
            try:
                ctl, cta, ctb, _ = _evaluate_loader(
                    model, clean_train_loader, device, n_classes, want_auc=False
                )
            finally:
                if backup2 is not None:
                    model.load_state_dict(backup2, strict=True)
            gap = cta - va

        rec = EpochRecord(
            epoch=epoch,
            train_loss=tot_loss / max(n, 1),
            train_acc=correct / max(n, 1),
            val_loss=vl,
            val_acc=va,
            val_balanced_acc=vba,
            lr=optimizer.param_groups[0]["lr"],
            seconds=round(time.time() - t0, 2),
            val_macro_auc=vauc,
            clean_train_loss=ctl,
            clean_train_acc=cta,
            clean_train_balanced_acc=ctb,
            generalisation_gap=gap,
        )
        history.epochs.append(rec)
        history.epochs_run = epoch
        log.info(
            "[%s] epoch %3d/%d  train_loss %.4f  train_acc %.4f  val_loss %.4f  "
            "val_acc %.4f  val_bacc %.4f  val_auc %.4f  lr %.2e  (%.1fs)",
            stage, epoch, n_epochs, rec.train_loss, rec.train_acc,
            rec.val_loss, rec.val_acc, rec.val_balanced_acc, rec.val_macro_auc,
            rec.lr, rec.seconds,
        )
        if gap == gap:
            log.info(
                "[%s]   clean train (eval mode, no augmentation): acc %.4f  loss %.4f  "
                "-> generalisation gap %+.4f%s",
                stage, cta, ctl, gap,
                "  <-- MEMORISING" if gap > 0.10 else "",
            )

        current = {"val_loss": vl, "val_balanced_acc": vba, "val_macro_auc": vauc}[
            cfg.select_metric
        ]
        improved = (
            current == current                                   # NaN never improves
            and (current < best_value if direction == "min" else current > best_value)
        )
        if improved:
            best_value = current
            history.best_select_value = float(current)
            history.best_val_balanced_acc = vba
            history.best_epoch = epoch
            since_improved = 0
            torch.save(
                {
                    "state_dict": selected_state,
                    "stage": stage,
                    "epoch": epoch,
                    "val_balanced_acc": vba,
                    "val_loss": vl,
                    "val_macro_auc": vauc,
                    "select_metric": cfg.select_metric,
                    "select_value": float(current),
                    "ema_decay": cfg.ema_decay,
                    "class_names": names,
                },
                checkpoint_path,
            )
        else:
            since_improved += 1

        if (
            cfg.early_stop_patience > 0
            and epoch >= max(cfg.min_epochs, cfg.warmup_epochs + 1)
            and since_improved >= cfg.early_stop_patience
        ):
            history.stopped_early = True
            log.info(
                "[%s] early stop at epoch %d: %s has not improved on %.6f for %d epochs "
                "(best was epoch %d). Remaining %d epochs skipped.",
                stage, epoch, cfg.select_metric, best_value, since_improved,
                history.best_epoch, n_epochs - epoch,
            )
            break

    if history.best_epoch < 0:
        raise RuntimeError(f"stage {stage} completed without ever writing a checkpoint")
    return history


# ---------------------------------------------------------------------------
# Stage entry points
# ---------------------------------------------------------------------------
def train_pretrain(
    train_loader, val_loader, device, model_cfg: ModelConfig, cfg: TrainConfig,
    checkpoint_path: Path, class_priors=None, class_names: Optional[Sequence[str]] = None,
    clean_train_loader=None,
) -> Tuple[LoRAViT, TrainHistory, dict]:
    """Stage 1: full fine-tune, no adapters."""
    set_seed(cfg.seed)
    model = LoRAViT(model_cfg)
    acct = count_parameters(model, LoRAConfig(enabled=False))
    log.info("stage 1 (pretrain): %d trainable parameters", acct.trainable_parameters)
    hist = run_training(
        model, train_loader, val_loader, device, cfg,
        n_epochs=cfg.pretrain_epochs, stage="pretrain",
        checkpoint_path=checkpoint_path, n_classes=model_cfg.n_classes,
        class_priors=class_priors, class_names=class_names,
        clean_train_loader=clean_train_loader,
    )
    state = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(state)
    return model, hist, acct.to_dict()


def train_lora(
    train_loader, val_loader, device, model_cfg: ModelConfig, lora_cfg: LoRAConfig,
    cfg: TrainConfig, base_checkpoint: Optional[Path], checkpoint_path: Path,
    allow_random_backbone: bool = False, class_priors=None,
    class_names: Optional[Sequence[str]] = None, clean_train_loader=None,
) -> Tuple[LoRAViT, TrainHistory, dict]:
    """Stage 2: inject LoRA into the pretrained backbone, freeze, adapt."""
    set_seed(cfg.seed + 1)
    model = LoRAViT(model_cfg)

    if base_checkpoint is not None and Path(base_checkpoint).is_file():
        ckpt = torch.load(base_checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True), None
        log.info("stage 2: loaded pretrained backbone from %s", base_checkpoint)
    elif allow_random_backbone:
        log.warning(
            "stage 2: no base checkpoint -- adapting a RANDOMLY INITIALISED backbone with a "
            "frozen random head. Results are not comparable to the manuscript."
        )
    else:
        raise FileNotFoundError(
            f"stage 2 needs the stage-1 checkpoint ({base_checkpoint}). Run the pretrain "
            "stage first, or pass --allow-random-backbone if you deliberately want to "
            "adapt an untrained backbone (this is not the manuscript's experiment)."
        )

    inject_lora(model, lora_cfg)
    if lora_cfg.freeze_backbone:
        freeze_backbone(model, lora_cfg)
    acct = count_parameters(model, lora_cfg)
    log.info(
        "stage 2 (LoRA r=%d): %d trainable of %d total (%.2f%% reduction vs full fine-tune)",
        lora_cfg.rank, acct.trainable_parameters, acct.total_parameters,
        acct.trainable_reduction_pct,
    )
    if cfg.lora_lr is not None:
        log.info(
            "stage 2: adapter LR %.1e (backbone LR is %.1e). LoRA's B is zero-initialised, "
            "so an adapter starts producing nothing and needs a larger step than a weight "
            "already near its optimum.", cfg.lora_lr, cfg.lr,
        )
    hist = run_training(
        model, train_loader, val_loader, device, cfg,
        n_epochs=cfg.epochs, stage="lora",
        checkpoint_path=checkpoint_path, n_classes=model_cfg.n_classes,
        class_priors=class_priors, class_names=class_names, lr=cfg.lora_lr,
        clean_train_loader=clean_train_loader,
    )
    state = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(state)
    return model, hist, acct.to_dict()


__all__ = [
    "set_seed", "resolve_device", "describe_device", "MixUp", "ModelEMA",
    "LogitAdjustedLoss", "build_criterion", "build_optimizer", "build_scheduler",
    "run_training", "train_pretrain", "train_lora", "TrainHistory", "EpochRecord",
    "SELECT_METRICS", "fit_temperature", "expected_calibration_error",
]


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins (Guo et al., 2017, eq. 3).

    Mean over bins of |accuracy - mean confidence|, weighted by bin occupancy. 0 is a
    perfectly calibrated model; a model that says "90% sure" and is right 90% of the time
    scores 0 whatever its accuracy.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    hit = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()
def _collect_logits(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    xs, ys = [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        xs.append(model(x).float().cpu())
        ys.append(batch[1].cpu())
    return torch.cat(xs), torch.cat(ys)


def fit_temperature(model, val_loader, device, max_iter: int = 200) -> Dict[str, float]:
    """Fit a single scalar T minimising validation NLL of softmax(logits / T).

    This is the right place to fix the rising-validation-loss problem, and the only place
    that fixes it without costing accuracy: dividing every logit by one positive scalar is
    a monotone transform, so the argmax -- and therefore accuracy, balanced accuracy, the
    confusion matrix and every AUC -- is mathematically unchanged. T > 1 means the network
    was over-confident and is being softened.

    Returns the fitted temperature and NLL/ECE before and after, so the improvement is
    reported rather than asserted.
    """
    logits, labels = _collect_logits(model, val_loader, device)
    nll = nn.CrossEntropyLoss()

    log_t = torch.zeros(1, requires_grad=True)          # optimise log T, so T stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    before_nll = float(nll(logits, labels))
    opt.step(closure)
    t = float(log_t.exp().item())
    after_nll = float(nll(logits / t, labels))

    p0 = logits.softmax(1).numpy()
    p1 = (logits / t).softmax(1).numpy()
    y = labels.numpy()
    acc0 = float((p0.argmax(1) == y).mean())
    acc1 = float((p1.argmax(1) == y).mean())
    if abs(acc0 - acc1) > 1e-9:
        raise RuntimeError(
            f"temperature scaling changed accuracy ({acc0} -> {acc1}); a positive scalar "
            "divide cannot do that, so something is wrong with the fit"
        )
    return {
        "temperature": round(t, 6),
        "val_nll_before": round(before_nll, 6),
        "val_nll_after": round(after_nll, 6),
        "val_ece_before": round(expected_calibration_error(p0, y), 6),
        "val_ece_after": round(expected_calibration_error(p1, y), 6),
        "val_accuracy_unchanged": round(acc0, 6),
        "n_val_records": int(len(y)),
        "note": (
            "Fitted on the validation split only. Dividing logits by a positive scalar is "
            "monotone, so accuracy, balanced accuracy, the confusion matrix and every AUC "
            "are unchanged by construction; only the probabilities move. T > 1 means the "
            "network was over-confident."
        ),
    }

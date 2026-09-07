"""Configuration objects for the LoRA-ViT ECG pipeline.

Every default here traces to a specific claim in the manuscript; the `source` comment on
each field says where. Nothing is tuned silently.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical geometry. These are properties of the corpus, not hyperparameters.
# ---------------------------------------------------------------------------
LEAD_ORDER: tuple = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
N_LEADS: int = 12
FS_HZ: float = 500.0  # manuscript eq. (3): T = 10 * fs
WINDOW_S: float = 10.0
N_SAMPLES: int = int(WINDOW_S * FS_HZ)  # 5000

CLASS_NAMES: tuple = ("NSR", "AFIB", "SB", "ST", "SVT", "CD", "OTHER")
N_CLASSES: int = len(CLASS_NAMES)

# The class *names* are a property of the resolution order, not of the corpus: the
# canonical_v2 taxonomy renames OTHER -> VE because that bucket is the ventricular-ectopy
# family and never was a residue. CLASS_NAMES above stays the historical default so every
# published run reproduces byte-for-byte; the active list is set from the loaded ClassMap.
#
# Read it through active_class_names(). A module-level `from .config import CLASS_NAMES`
# binds at import time and would not see the update.
_ACTIVE_CLASS_NAMES: List[str] = list(CLASS_NAMES)


def active_class_names() -> tuple:
    """The class names of the resolution order currently loaded."""
    return tuple(_ACTIVE_CLASS_NAMES)


def set_active_class_names(names) -> None:
    """Called by `labels.get_class_map`; not intended to be called directly."""
    global _ACTIVE_CLASS_NAMES
    names = [str(n) for n in names]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate class names: {names}")
    if not names:
        raise ValueError("class name list is empty")
    _ACTIVE_CLASS_NAMES = names


def reset_active_class_names() -> None:
    set_active_class_names(CLASS_NAMES)


@dataclass
class PreprocessConfig:
    """Manuscript section 3.1, equations (1) and (2)."""

    bandpass_low_hz: float = 0.5      # source: eq. (1), 0.5-40 Hz passband
    bandpass_high_hz: float = 40.0    # source: eq. (1)
    bandpass_order: int = 4           # source: sec. 3.1, "fourth-order Butterworth"
    notch_freq_hz: float = 50.0       # source: sec. 3.1, powerline interference
    notch_q: float = 30.0             # source: sec. 3.1, "quality factor (Q-factor) of 30"
    zscore_eps: float = 1e-8          # source: eq. (2), epsilon for numerical stability
    apply_notch: bool = True
    # If a header omits fs, assume 500 Hz but RECORD that we did so (sec. 3.1 explicitly
    # asks for this assumption to be marked rather than left implicit).
    assumed_fs_hz: float = 500.0


@dataclass
class ModelConfig:
    """Manuscript section 3.3, equations (4)-(13), and Figure 2."""

    n_classes: int = N_CLASSES
    seq_len: int = N_SAMPLES
    in_chans: int = N_LEADS
    patch_len: int = 100              # source: N = 50 patches over 5000 samples
    embed_dim: int = 128              # source: eq. (6), D = 128
    depth: int = 8                    # source: Fig. 2, "repeated eight times"
    num_heads: int = 8                # source: sec. 3.3, "H = 8 attention heads"
    mlp_ratio: float = 4.0
    drop_rate: float = 0.1            # source: eq. (7), Dropout p = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1       # source: Table 2, "stochastic depth + dropout"
    use_cls_token: bool = True        # source: eq. (7), learnable CLS token
    pre_norm: bool = True             # source: sec. 3, "pre-norm layer normalisation"


@dataclass
class LoRAConfig:
    """Manuscript section 3.4, equations (14) and (15), Table 8."""

    enabled: bool = True
    rank: int = 8                     # source: Tables 4, 5 and 8: "LoRA (r=8)"
    alpha: int = 16                   # scaling alpha/r = 2
    dropout: float = 0.0
    # source: sec. 3.4 -- "all Query, Key and Value projections W_q, W_k and W_v, to
    # attention output projection W_o, and to both fully connected layers W_1 and W_2"
    target_modules: tuple = ("qkv", "proj", "fc1", "fc2")
    freeze_backbone: bool = True      # source: sec. 3.4, base weights frozen
    # The classification head is small and task-specific; the manuscript's parameter count
    # (131,072 trainable) corresponds to LoRA matrices ONLY, so the head stays frozen too.
    # Set to True to also train the head; the reported reduction then drops accordingly.
    train_head: bool = False
    train_norms: bool = False


@dataclass
class TrainConfig:
    """Manuscript Table 2 and Table 8.

    Table 8 reports 150 epochs for the no-LoRA variant and 30 for the LoRA variant, with
    1,648,839 total parameters in the former and 131,072 *trainable* out of 1,779,911 in
    the latter. 1,779,911 - 1,648,839 = 131,072 exactly, so the LoRA variant is not trained
    from scratch: it is the *same* backbone, pretrained, then frozen and adapted. Hence the
    two stages below. Training LoRA from a random init with a frozen random head would be
    an entirely different (and far weaker) experiment.
    """

    pretrain_epochs: int = 150        # source: Table 8, stage 1 (full fine-tune, no LoRA)
    epochs: int = 30                  # source: Table 8, stage 2 (LoRA adaptation, r=8)
    batch_size: int = 64              # source: Table 2
    lr: float = 1e-4                  # source: Table 2, initial LR 1e-4
    weight_decay: float = 0.05
    optimizer: str = "adamw"          # source: Table 2
    scheduler: str = "cosine_warmup"  # source: Table 2, "cosine annealing with warm-up"
    warmup_epochs: int = 5
    mixup_alpha: float = 0.2          # source: Table 2, MixUp alpha = 0.2; eqs. (17)-(18)
    mixup_prob: float = 0.5
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    amp: bool = True                  # bf16 autocast on CUDA
    seed: int = 42

    # ---- recipe knobs ----------------------------------------------------
    # Defaults below reproduce the manuscript exactly. `--recipe v2` overrides them as a
    # bundle (see `apply_recipe`); nothing here is tuned silently.
    early_stop_patience: int = 0      # 0 disables; the paper schedule is fixed-length
    min_epochs: int = 20              # never stop before this, whatever the patience
    # Which validation statistic picks the checkpoint.
    #
    #   val_loss          minimum NLL. Detects miscalibration, but on this corpus it bottoms
    #                     at epoch 40 while accuracy keeps improving to 120, so selecting on
    #                     it costs ~3.9 points of balanced accuracy. Use temperature scaling
    #                     for calibration instead.
    #   val_balanced_acc  the published choice. Correct target, but a noisy, thresholded
    #                     statistic on ~1,490 validation records.
    #   val_macro_auc     threshold-free and rank-based: measures discrimination without
    #                     being dragged by confidence drift. The recipe-v2 default.
    select_metric: str = "val_balanced_acc"   # val_loss | val_balanced_acc | val_macro_auc
    # Evaluate the TRAINING split in eval mode -- no augmentation, no MixUp, natural class
    # distribution -- every N epochs. 0 disables. This is the only honest measure of fit:
    # the running `train_acc` is computed on the balanced/oversampled split under MixUp
    # against hard labels, so it is systematically LOWER than validation accuracy and says
    # nothing about memorisation. Without this, "is it overfitting?" cannot be answered.
    clean_train_eval_every: int = 0
    # Post-hoc temperature scaling (Guo et al., 2017), fitted on validation after training.
    # Divides the logits by a single scalar T: it cannot change the argmax, so accuracy is
    # untouched by construction, and it is the standard remedy for rising validation NLL
    # alongside flat-or-improving validation accuracy.
    temperature_scaling: bool = False
    # Exponential moving average of weights; 0 disables. Evaluated and checkpointed instead
    # of the raw weights when enabled.
    ema_decay: float = 0.0
    # Long-tail handling. "none" keeps the duplication-based balancing in split_spec.yaml.
    # "logit_adjust" (Menon et al., 2021) instead adds tau*log(prior) to the logits during
    # training only, and trains on the NATURAL distribution -- no duplicated records at all.
    class_weighting: str = "none"     # none | inverse_freq | logit_adjust
    logit_adjust_tau: float = 1.0
    # Adapter learning rate for stage 2. None -> use `lr`. LoRA's B is zero-initialised, so
    # the backbone LR is roughly an order of magnitude too small for adapters.
    lora_lr: Optional[float] = None
    # "basic"  = the 3-way jitter used for duplication (noise | scale | roll)
    # "physio" = lead dropout, per-lead amplitude scaling, baseline wander, time warp.
    #            Time warping is suppressed for the rate-defined classes, where it would
    #            change the very quantity that defines the label.
    augment_strength: str = "basic"   # basic | physio
    cv_folds: int = 0                 # 0 = single deterministic split; 5 = 5-fold CV


@dataclass
class XAIConfig:
    """Manuscript sections 4.2 and 4.3."""

    gradcam_target: str = "blocks.-1.norm1"   # last encoder block pre-norm
    ig_steps: int = 50                        # source: sec. 4.2
    ig_samples: int = 256                     # records used for the per-class lead ranking
    ig_confidence: float = 0.95               # source: Fig. 7, 95% CI
    shap_background: int = 64
    shap_samples: int = 128
    faithfulness_fractions: tuple = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
    tsne_perplexity: float = 30.0
    tsne_samples: int = 2000
    bootstrap_iterations: int = 1000          # source: Table 5, DeLong with 1000 bootstraps
    n_example_records: int = 8                # per-class Grad-CAM / SHAP example figures


@dataclass
class PipelineConfig:
    chapman_root: Optional[Path] = None
    output_dir: Path = Path("outputs")
    data_dir: Optional[Path] = None
    resolution_order: str = "clinical_specificity"
    recipe: str = "paper"
    #: Cross-validation round currently being built, or None for the ordinary split.
    #: Set by `cv`; the number of folds lives in TrainConfig.cv_folds.
    cv_fold: Optional[int] = None
    device: str = "auto"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)

    # ---- environment -----------------------------------------------------
    @classmethod
    def from_env(cls, **overrides: Any) -> "PipelineConfig":
        root = os.environ.get("CHAPMAN_ROOT")
        cfg = cls(
            chapman_root=Path(root) if root else None,
            output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs")),
            data_dir=Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else None,
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def resolve_data_dir(self) -> Path:
        """Locate `environment/data/`, whether installed or run from the source tree."""
        if self.data_dir is not None:
            return self.data_dir
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "environment" / "data"
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            "Could not locate environment/data/. Set DATA_DIR explicitly."
        )

    def to_dict(self) -> Dict[str, Any]:
        def encode(o: Any) -> Any:
            if dataclasses.is_dataclass(o):
                return {k: encode(v) for k, v in dataclasses.asdict(o).items()}
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o

        return {k: encode(v) for k, v in dataclasses.asdict(self).items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------
#: Named bundles of training settings. "paper" is the manuscript's configuration and is the
#: default, so every published figure keeps reproducing. "v2" is the corrected recipe; each
#: entry traces to a specific observation in docs/RESULTS_REVIEW.md.
RECIPES: Dict[str, Dict[str, Any]] = {
    "paper": {},
    "v2": {
        "train": {
            # Selection metric, corrected after reading the full 150-epoch curve.
            #
            # val_loss bottoms at epoch 40 and rises ~10% after -- but validation ACCURACY
            # and BALANCED accuracy keep improving to epoch 120 (0.7329 -> 0.7718). Those
            # are not in conflict: the network is getting more confident, so its confident
            # errors cost more NLL while its ranking keeps improving. Selecting on val_loss
            # would pick epoch 40 and give up 3.9 points of balanced accuracy.
            #
            # macro AUC is threshold-free and rank-based, so it measures discrimination
            # without being dragged by confidence drift. The calibration problem val_loss
            # was detecting is real and is fixed where it belongs -- post-hoc temperature
            # scaling, which cannot move the argmax.
            "select_metric": "val_macro_auc",
            "early_stop_patience": 25,
            "min_epochs": 40,
            "clean_train_eval_every": 5,
            "temperature_scaling": True,
            "ema_decay": 0.999,
            # OTHER was duplicated 5.6x -- 5.6 gradient steps over the same 184 records --
            # and ended with precision 0.468 against recall 0.595, i.e. over-predicted.
            # Logit adjustment fixes the prior without ever repeating a record.
            "class_weighting": "logit_adjust",
            "logit_adjust_tau": 1.0,
            "augment_strength": "physio",
            # Stage 2 saved its epoch-1 adapters (val balanced acc peaked there) while val
            # loss kept improving to epoch 17; train loss was still falling at epoch 30.
            "lora_lr": 1e-3,
        },
        "lora": {
            # 903 parameters. With the head frozen and the features already optimal for it,
            # the adapters have almost no gradient signal to act on.
            "train_head": True,
        },
    },
}


def apply_recipe(name: str, train: TrainConfig, lora: LoRAConfig) -> None:
    """Apply a named recipe in place. Unknown field names are a hard error, not a no-op."""
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name!r}; available: {sorted(RECIPES)}")
    for section, target in (("train", train), ("lora", lora)):
        for key, value in RECIPES[name].get(section, {}).items():
            if not hasattr(target, key):
                raise AttributeError(
                    f"recipe {name!r} sets {section}.{key}, which does not exist"
                )
            setattr(target, key, value)


__all__ = [
    "LEAD_ORDER", "N_LEADS", "FS_HZ", "WINDOW_S", "N_SAMPLES",
    "CLASS_NAMES", "N_CLASSES", "active_class_names", "set_active_class_names",
    "reset_active_class_names",
    "PreprocessConfig", "ModelConfig", "LoRAConfig", "TrainConfig", "XAIConfig",
    "PipelineConfig", "RECIPES", "apply_recipe",
]

"""Command-line entry points.

    python -m ecgvit.cli verify-data      geometry + label sanity check, no training
    python -m ecgvit.cli index            build and write the record index + cohort report
    python -m ecgvit.cli train            stage 1 (pretrain) + stage 2 (LoRA adapt)
    python -m ecgvit.cli evaluate         metrics, confusion matrix, PR curves
    python -m ecgvit.cli explain          Grad-CAM, IG, SHAP, faithfulness, t-SNE
    python -m ecgvit.cli stats            McNemar and DeLong, LoRA vs no-LoRA
    python -m ecgvit.cli run-all          everything, in order (this is what solve.sh calls)
    python -m ecgvit.cli cv               k-fold cross-validation, mean +/- std per metric
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (CLASS_NAMES, RECIPES, LoRAConfig, ModelConfig, PipelineConfig,
                     TrainConfig, active_class_names, apply_recipe)
from .labels import get_class_map

log = logging.getLogger("ecgvit")


def _available_resolution_orders() -> List[str]:
    """Read the orders from the shipped class map so the CLI cannot drift from the data."""
    try:
        cfg = PipelineConfig.from_env()
        spec = json.loads((cfg.resolve_data_dir() / "class_map_7.json").read_text())
        return list(spec["resolution_orders"])
    except Exception:  # noqa: BLE001 - fall back rather than break --help
        return ["conduction_first", "rhythm_first", "paper_reported"]


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    log.info("wrote %s", path)


def _wilson_ci(p: float, n: int, z: float = 1.96):
    """Wilson score interval for a proportion.

    Wilson rather than normal-approximation: at n = 37 and p = 0.59 the normal interval is
    already unreliable, and at p near 0 or 1 it produces bounds outside [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = min(max(float(p), 0.0), 1.0)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def _cfg_from_args(args) -> PipelineConfig:
    cfg = PipelineConfig.from_env()
    if getattr(args, "chapman_root", None):
        cfg.chapman_root = Path(args.chapman_root)
    if getattr(args, "output_dir", None):
        cfg.output_dir = Path(args.output_dir)
    if getattr(args, "data_dir", None):
        cfg.data_dir = Path(args.data_dir)
    if getattr(args, "resolution_order", None):
        cfg.resolution_order = args.resolution_order
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "seed", None) is not None:
        cfg.train.seed = args.seed
    if getattr(args, "pretrain_epochs", None) is not None:
        cfg.train.pretrain_epochs = args.pretrain_epochs
    if getattr(args, "lora_epochs", None) is not None:
        cfg.train.epochs = args.lora_epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.train.batch_size = args.batch_size
    if getattr(args, "lora_rank", None) is not None:
        cfg.lora.rank = args.lora_rank
    if getattr(args, "num_workers", None) is not None:
        cfg.train.num_workers = args.num_workers
    if getattr(args, "patch_len", None) is not None:
        cfg.model.patch_len = args.patch_len

    # The recipe is applied BEFORE the individual overrides below, so an explicit flag
    # always wins over the bundle it belongs to.
    if getattr(args, "recipe", None):
        cfg.recipe = args.recipe
    apply_recipe(cfg.recipe, cfg.train, cfg.lora)

    if getattr(args, "select_metric", None):
        cfg.train.select_metric = args.select_metric
    if getattr(args, "early_stop_patience", None) is not None:
        cfg.train.early_stop_patience = args.early_stop_patience
    if getattr(args, "ema_decay", None) is not None:
        cfg.train.ema_decay = args.ema_decay
    if getattr(args, "class_weighting", None):
        cfg.train.class_weighting = args.class_weighting
    if getattr(args, "lora_lr", None) is not None:
        cfg.train.lora_lr = args.lora_lr
    if getattr(args, "augment_strength", None):
        cfg.train.augment_strength = args.augment_strength
    if getattr(args, "train_head", False):
        cfg.lora.train_head = True
    if getattr(args, "clean_train_eval_every", None) is not None:
        cfg.train.clean_train_eval_every = args.clean_train_eval_every
    if getattr(args, "temperature_scaling", False):
        cfg.train.temperature_scaling = True
    if getattr(args, "cv_folds", None) is not None:
        cfg.train.cv_folds = args.cv_folds
    if getattr(args, "cv_fold", None) is not None:
        cfg.cv_fold = args.cv_fold

    # Class names follow the resolution order (canonical_v2 renames OTHER -> VE), so the
    # model must be sized from the map rather than from the module default.
    try:
        cmap = get_class_map(cfg.resolve_data_dir(), cfg.resolution_order)
        cfg.model.n_classes = len(cmap.classes)
    except FileNotFoundError:
        pass

    if cfg.chapman_root is None:
        raise SystemExit(
            "CHAPMAN_ROOT is not set and --chapman-root was not passed.\n"
            "The ECG corpus is not vendored in this repository; see "
            "environment/data/README.md for the download link, then:\n"
            "  export CHAPMAN_ROOT=/path/to/WFDB_ChapmanShaoxing"
        )
    return cfg


# ---------------------------------------------------------------------------
# verify-data
# ---------------------------------------------------------------------------
def cmd_verify_data(args) -> int:
    import yaml

    from .data import build_index

    cfg = _cfg_from_args(args)
    data_dir = cfg.resolve_data_dir()
    spec = yaml.safe_load((data_dir / "dataset.yaml").read_text())
    exp = spec["primary"]["expected"]

    log.info("scanning %s", cfg.chapman_root)
    records, report, cmap = build_index(
        cfg.chapman_root, data_dir, cfg.resolution_order, strict_geometry=True
    )

    problems: List[str] = []
    if report.n_indexed < int(exp["min_records"]):
        problems.append(
            f"indexed {report.n_indexed} records, expected at least {exp['min_records']}"
        )

    fs_counts = Counter(r.fs_hz for r in records)
    n_counts = Counter(r.n_samples for r in records)
    want_fs = float(exp["sampling_frequency_hz"])
    off_fs = sum(v for k, v in fs_counts.items() if float(k) != want_fs)
    if off_fs:
        problems.append(f"{off_fs} records are not sampled at {want_fs} Hz: {dict(fs_counts)}")

    names = list(cmap.classes)
    present = {c for c in report.label_counts if report.label_counts[c] > 0}
    missing = [c for c in names if c not in present]
    if missing:
        problems.append(f"classes with zero records under this mapping: {missing}")

    summary = {
        "chapman_root": str(cfg.chapman_root),
        "resolution_order": cfg.resolution_order,
        "index": report.to_dict(),
        "sampling_frequencies": {str(k): v for k, v in fs_counts.items()},
        "samples_per_record": {str(k): v for k, v in n_counts.most_common(10)},
        "class_map_fingerprint": cmap.fingerprint(),
        "problems": problems,
        "ok": not problems,
    }
    _write_json(summary, cfg.output_dir / "data_verification.json")

    print("\n=== Cohort ===")
    print(f"  headers found      : {report.n_headers}")
    print(f"  usable records     : {report.n_indexed}")
    print(f"  rejected           : missing signal {report.n_missing_signal}, "
          f"bad geometry {report.n_bad_geometry}, no #Dx {report.n_no_dx}")
    print(f"\n=== Class distribution ({cfg.resolution_order}) ===")
    total = sum(report.label_counts.values())
    for c in names:
        n = report.label_counts.get(c, 0)
        pct = 100.0 * n / total if total else 0.0
        flag = "   <-- too few to train or evaluate meaningfully" if 0 < n < 50 else ""
        print(f"  {c:6s} {n:6d}  ({pct:5.2f}%){flag}")
    print("\n=== Split sizes ===")
    for s, counter in report.split_counts.items():
        print(f"  {s:5s} {sum(counter.values()):6d}  {dict(counter)}")

    if report.unknown_codes:
        print(f"\n  {len(report.unknown_codes)} SNOMED codes not in the vocabulary "
              f"(these fall through to OTHER): "
              f"{list(report.unknown_codes)[:10]}")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: corpus conforms to dataset.yaml.")
    return 0


# ---------------------------------------------------------------------------
# label-audit
# ---------------------------------------------------------------------------
def cmd_label_audit(args) -> int:
    """Compare every resolution order on the real corpus, and say which classes are
    actually evaluable.

    The distinction this command exists to make: a class can be *trainable* (you can
    oversample 41 records into a balanced training split) while not being *evaluable*
    (9 test records cannot support a per-class metric anyone should read). Augmentation
    changes the first and can do nothing about the second.
    """
    import json as _json
    from collections import Counter, defaultdict

    from .data import assign_split, load_split_spec
    from .labels import ClassMap, load_snomed_vocabulary, parse_dx_codes

    cfg = _cfg_from_args(args)
    data_dir = cfg.resolve_data_dir()
    spec = _json.loads((data_dir / "class_map_7.json").read_text())
    vocab = load_snomed_vocabulary(data_dir)
    split_spec = load_split_spec(data_dir)
    salt = split_spec["hash"]["salt"]
    hexc = int(split_spec["hash"]["digest_hex_chars"])
    props = {k: float(v) for k, v in split_spec["proportions"].items()}
    min_test = int(spec.get("evaluability", {})
                   .get("min_unique_test_records_for_reporting", 50))

    # Scan the corpus ONCE; resolution is then pure computation over the parsed codes.
    log.info("scanning %s", cfg.chapman_root)
    records = []
    for hea in sorted(Path(cfg.chapman_root).rglob("*.hea")):
        if not (hea.with_suffix(".mat").is_file() or hea.with_suffix(".dat").is_file()):
            continue
        codes = parse_dx_codes(hea)
        if codes:
            records.append((hea.stem, codes))
    if not records:
        raise SystemExit(f"no usable records under {cfg.chapman_root}")
    log.info("%d records with a #Dx field", len(records))

    # How many records carry a code from each class AT ALL, ignoring precedence?
    # This is the diagnostic: if CD's "carries" count is large but its resolved count is
    # tiny, the class is not rare -- it is being absorbed by whatever wins first.
    probe = ClassMap(spec, resolution_order=spec["default_resolution_order"], vocabulary=vocab)
    names = list(probe.classes)
    carries = Counter()
    for _, codes in records:
        cs = set(codes)
        for cls in names:
            if cls == probe.fallback:
                continue
            if cs & set(probe.class_to_codes[cls]):
                carries[cls] += 1

    results = {}
    order_classes = {}
    for order_name in spec["resolution_orders"]:
        cmap = ClassMap(spec, resolution_order=order_name, vocabulary=vocab)
        order_classes[order_name] = list(cmap.classes)
        per_split = {s: Counter() for s in ("train", "val", "test")}
        total = Counter()
        n_unassigned = 0
        for rid, codes in records:
            res = cmap.resolve(codes, rid, set(vocab))
            if not res.is_assigned:
                # Orders with a null fallback drop what they cannot claim. Counting the
                # drops here is the whole point of the audit: it is the cost of the
                # taxonomy, and it belongs next to the class counts, not buried.
                n_unassigned += 1
                continue
            total[res.label] += 1
            per_split[assign_split(rid, salt, props, hexc)][res.label] += 1
        results[order_name] = {
            "classes": list(cmap.classes),
            "fallback_class": cmap.fallback,
            "n_unassigned": n_unassigned,
            "total": dict(total),
            "train": dict(per_split["train"]),
            "val": dict(per_split["val"]),
            "test": dict(per_split["test"]),
            "not_evaluable": sorted(
                c for c in cmap.classes if per_split["test"].get(c, 0) < min_test
            ),
        }

    # ---- report ----------------------------------------------------------
    print(f"\n{len(records)} records with a #Dx field under {cfg.chapman_root}\n")
    print("Records CARRYING at least one code of each class, ignoring precedence:")
    for c in names:
        if c == probe.fallback:
            continue
        print(f"  {c:6s} {carries[c]:6d}")
    print("\n(These sum to more than the corpus: records are multi-label. The gap between")
    print(" a class's 'carries' count and its resolved count below is what precedence costs")
    print(" that class.)\n")

    for order_name, r in results.items():
        order = spec["resolution_orders"][order_name]
        tail = (f" > {r['fallback_class']}" if r["fallback_class"]
                else f"  (no fallback: {r['n_unassigned']} records excluded)")
        print("=" * 78)
        print(f"{order_name}   ({' > '.join(order)}{tail})")
        print("=" * 78)
        print(f"  {'class':7s} {'total':>7s} {'train':>7s} {'val':>6s} {'test':>6s} "
              f"{'% corpus':>9s}   evaluable?")
        for c in r["classes"]:
            t = r['total'].get(c, 0)
            te = r['test'].get(c, 0)
            flag = "yes" if te >= min_test else f"NO  (test n={te})"
            denom = max(sum(r['total'].values()), 1)
            print(f"  {c:7s} {t:7d} {r['train'].get(c,0):7d} {r['val'].get(c,0):6d} "
                  f"{te:6d} {100.0*t/denom:8.2f}%   {flag}")
        if r["not_evaluable"]:
            print(f"\n  -> not reportable as per-class metrics: {r['not_evaluable']}")
        else:
            print(f"\n  -> every class has >= {min_test} unique test records")
        print()

    _write_json(
        {"n_records": len(records), "min_unique_test_records_for_reporting": min_test,
         "carries_any_code_of_class": dict(carries), "by_resolution_order": results},
        cfg.output_dir / "label_audit.json",
    )

    print("A class is TRAINABLE if you can oversample it; it is EVALUABLE only if the test")
    print("split holds enough unique records. Augmenting the training split cannot widen a")
    print("test class, so 'not reportable' above is not fixable by balancing.")
    return 0


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def cmd_index(args) -> int:
    from .data import build_index, write_index_csv

    cfg = _cfg_from_args(args)
    records, report, cmap = build_index(
        cfg.chapman_root, cfg.resolve_data_dir(), cfg.resolution_order
    )
    write_index_csv(records, cfg.output_dir / "record_index.csv")
    _write_json(report.to_dict(), cfg.output_dir / "index_report.json")
    _write_json(cmap.fingerprint(), cfg.output_dir / "labels" / "label_index.json")
    log.info("indexed %d records", len(records))
    return 0


# ---------------------------------------------------------------------------
# shared loader construction
# ---------------------------------------------------------------------------
def _write_preprocessing_report(cfg: PipelineConfig, records) -> None:
    """Record what denoising was actually applied, and its measured response.

    The point is auditability: the report states the achieved attenuation at each probe
    frequency, not merely that a filter was configured. A run whose notch silently fell
    back to single-pass, or was skipped because the corpus was not 500 Hz, says so here.
    """
    import dataclasses

    from .data import read_signal
    from .preprocess import (bandpass_response_db, notch_response_db,
                             preprocess_signal)

    pc = cfg.preprocess
    fs_values = sorted({r.fs_hz for r in records})
    fs = float(fs_values[0]) if fs_values else pc.assumed_fs_hz

    probes = [0.05, 0.15, 0.5, 1.0, 5.0, 10.0, 20.0, 35.0, 40.0, 50.0, 60.0, 100.0]
    probes = [p for p in probes if p < 0.5 * fs]

    sample_reports = []
    rng = np.random.default_rng(cfg.train.seed)
    for i in rng.choice(len(records), size=min(25, len(records)), replace=False):
        rec = records[int(i)]
        try:
            sig, rec_fs = read_signal(rec)
            _, rep = preprocess_signal(sig, fs=rec_fs, cfg=pc)
            sample_reports.append(rep.to_dict())
        except Exception as exc:  # noqa: BLE001
            log.warning("preprocessing probe failed for %s: %s", rec.record_id, exc)

    def _all(field, default=False):
        vals = [r[field] for r in sample_reports]
        return bool(vals) and all(v == (default or True) for v in vals)

    report = {
        "config": dataclasses.asdict(pc),
        "corpus_sampling_frequencies_hz": fs_values,
        "probe_frequencies_hz": probes,
        "bandpass_response_db_zerophase": {
            str(h): round(float(d), 3)
            for h, d in zip(probes, bandpass_response_db(fs, pc, probes))
        },
        "notch_response_db_zerophase": (
            {
                str(h): round(float(d), 3)
                for h, d in zip([45.0, 49.0, 50.0, 51.0, 55.0],
                                notch_response_db(fs, pc, [45.0, 49.0, 50.0, 51.0, 55.0]))
            }
            if pc.apply_notch and pc.notch_freq_hz < 0.5 * fs
            else None
        ),
        "n_records_probed": len(sample_reports),
        "bandpass_applied_on_all_probes": _all("bandpass_applied"),
        "notch_applied_on_all_probes": _all("notch_applied"),
        "zerophase_on_all_probes": _all("zerophase"),
        "records_padded": sum(1 for r in sample_reports if r["padded_samples"] > 0),
        "records_cropped": sum(1 for r in sample_reports if r["cropped_samples"] > 0),
        "records_with_assumed_fs": sum(1 for r in sample_reports if r["fs_assumed"]),
        "records_with_flat_leads": sum(1 for r in sample_reports if r["flat_leads"]),
        "records_with_nan_leads": sum(1 for r in sample_reports if r["nan_leads"]),
        "note": (
            "Response values are |H(f)|^2 in dB, matching the zero-phase (filtfilt) "
            "application. 'probed' counts refer to the sampled subset, not the whole "
            "corpus; padding and cropping for every record are logged in index_report."
        ),
    }
    _write_json(report, cfg.output_dir / "preprocessing_report.json")



def _build_loaders(cfg: PipelineConfig, balance_train: bool = True, cv_fold=None):
    """Build train/val/test loaders. `cv_fold` defaults to cfg.cv_fold."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from .data import (balance_indices, build_index, class_priors, load_split_spec,
                       make_torch_dataset, split_records, write_index_csv)

    cv_fold = cfg.cv_fold if cv_fold is None else cv_fold
    data_dir = cfg.resolve_data_dir()
    records, report, cmap = build_index(
        cfg.chapman_root, data_dir, cfg.resolution_order,
        cv_fold=cv_fold, n_folds=cfg.train.cv_folds,
    )
    write_index_csv(records, cfg.output_dir / "record_index.csv")
    _write_json(report.to_dict(), cfg.output_dir / "index_report.json")
    _write_json(cmap.fingerprint(), cfg.output_dir / "labels" / "label_index.json")

    by_split = split_records(records)
    spec = load_split_spec(data_dir)
    cache = cfg.output_dir / "cache"

    _write_preprocessing_report(cfg, records)

    train_records = by_split["train"]
    balance_report = None
    if balance_train:
        # When the loss corrects the prior (logit adjustment / inverse-frequency weights),
        # duplicating records would correct it a second time and would spend gradient steps
        # re-showing the model copies it has already seen. One mechanism, not two.
        strategy = "none" if cfg.train.class_weighting != "none" else None
        idxs, balance_report = balance_indices(
            train_records, spec, rng_seed=cfg.train.seed, strategy=strategy,
        )
        train_records_expanded = [train_records[i] for i in idxs]
    else:
        train_records_expanded = train_records

    ds_train = make_torch_dataset(
        train_records_expanded, cfg.preprocess, cache, augment=balance_train,
        rng_seed=cfg.train.seed, augment_strength=cfg.train.augment_strength,
    )
    # The training split as the model will be JUDGED on it, not as it is trained on it:
    # no augmentation, no MixUp, no duplication, natural class distribution. Comparing this
    # against validation is the only way to see memorisation.
    ds_clean_train = make_torch_dataset(train_records, cfg.preprocess, cache, augment=False)
    ds_val = make_torch_dataset(by_split["val"], cfg.preprocess, cache, augment=False)
    ds_test = make_torch_dataset(by_split["test"], cfg.preprocess, cache, augment=False)

    pin = torch.cuda.is_available()
    common = dict(num_workers=cfg.train.num_workers, pin_memory=pin,
                  persistent_workers=cfg.train.num_workers > 0)
    loaders = {
        "train": DataLoader(ds_train, batch_size=cfg.train.batch_size, shuffle=True,
                            drop_last=False, **common),
        "clean_train": DataLoader(ds_clean_train, batch_size=cfg.train.batch_size,
                                  shuffle=False, **common),
        "val": DataLoader(ds_val, batch_size=cfg.train.batch_size, shuffle=False, **common),
        "test": DataLoader(ds_test, batch_size=cfg.train.batch_size, shuffle=False, **common),
    }
    meta = {
        "records": {k: [r.record_id for r in v] for k, v in by_split.items()},
        "train_expanded_ids": [r.record_id for r in train_records_expanded],
        "balance": balance_report.to_dict() if balance_report else None,
        "index_report": report.to_dict(),
        "class_names": list(cmap.classes),
        "class_priors": class_priors(train_records, len(cmap.classes)).tolist(),
        "cv_fold": cv_fold,
        "n_folds": cfg.train.cv_folds,
    }
    return loaders, meta, by_split


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args) -> int:
    from .train import (describe_device, resolve_device, set_seed, train_lora,
                        train_pretrain)

    cfg = _cfg_from_args(args)
    _write_json(cfg.to_dict(), cfg.output_dir / "config.json")
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.device)
    dev_info = describe_device(device)
    _write_json(dev_info, cfg.output_dir / "device.json")
    log.info("device: %s", dev_info)

    loaders, meta, _ = _build_loaders(cfg)
    _write_json(meta["balance"], cfg.output_dir / "balance_report.json")

    ckpt_dir = cfg.output_dir / "model"
    histories = []

    base_ckpt = ckpt_dir / "base_no_lora.pt"
    if args.skip_pretrain and base_ckpt.is_file():
        log.info("reusing existing stage-1 checkpoint %s", base_ckpt)
        base_acct = json.loads((cfg.output_dir / "param_efficiency.json").read_text())["no_lora"]
    else:
        _, hist1, base_acct = train_pretrain(
            loaders["train"], loaders["val"], device, cfg.model, cfg.train, base_ckpt,
            class_priors=meta["class_priors"], class_names=meta["class_names"],
            clean_train_loader=loaders["clean_train"],
        )
        histories.append(hist1.to_dict())

    _, hist2, lora_acct = train_lora(
        loaders["train"], loaders["val"], device, cfg.model, cfg.lora, cfg.train,
        base_checkpoint=base_ckpt, checkpoint_path=ckpt_dir / "model.pt",
        allow_random_backbone=args.allow_random_backbone,
        class_priors=meta["class_priors"], class_names=meta["class_names"],
        clean_train_loader=loaders["clean_train"],
    )
    histories.append(hist2.to_dict())

    _write_json(histories, cfg.output_dir / "training_history.json")
    _write_json(
        {
            "no_lora": base_acct,
            "lora": lora_acct,
            # The tokeniser stride changes the parameter counts, so an artefact from an
            # ablation has to say which geometry produced it -- otherwise a reader compares
            # 1,636,039 against Table 8's 1,648,839 and concludes the run is wrong.
            "model_config": {
                "patch_len": cfg.model.patch_len,
                "n_patches": cfg.model.seq_len // cfg.model.patch_len,
                "embed_dim": cfg.model.embed_dim,
                "depth": cfg.model.depth,
                "num_heads": cfg.model.num_heads,
                "n_classes": cfg.model.n_classes,
                "is_reference_geometry": cfg.model.patch_len == 100,
            },
            "recipe": cfg.recipe,
            "resolution_order": cfg.resolution_order,
            **lora_acct,  # flattened for schema checks
        },
        cfg.output_dir / "param_efficiency.json",
    )

    if cfg.train.temperature_scaling:
        from .train import fit_temperature

        model = _load_model(cfg, device, with_lora=True)
        calib = fit_temperature(model, loaders["val"], device)
        calib["fitted_on"] = "val"
        calib["checkpoint"] = str(ckpt_dir / "model.pt")
        _write_json(calib, cfg.output_dir / "calibration.json")
        log.info(
            "temperature scaling: T = %.4f  val NLL %.4f -> %.4f  val ECE %.4f -> %.4f  "
            "(accuracy unchanged by construction)",
            calib["temperature"], calib["val_nll_before"], calib["val_nll_after"],
            calib["val_ece_before"], calib["val_ece_after"],
        )

    from .evaluate import plot_training_curves

    plot_training_curves(histories, cfg.output_dir / "figures" / "training_curves.png")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _load_model(cfg: PipelineConfig, device, with_lora: bool = True):
    import torch

    from .lora import count_parameters, freeze_backbone, inject_lora
    from .model import LoRAViT

    model = LoRAViT(cfg.model)
    path = cfg.output_dir / "model" / ("model.pt" if with_lora else "base_no_lora.pt")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}. Run `train` first.")
    if with_lora:
        inject_lora(model, cfg.lora)
        if cfg.lora.freeze_backbone:
            freeze_backbone(model, cfg.lora)
    state = torch.load(path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def cmd_evaluate(args) -> int:
    from .evaluate import (compute_metrics, plot_confusion_matrix,
                           plot_precision_recall, predict, write_predictions_csv)
    from .train import resolve_device

    cfg = _cfg_from_args(args)
    names = active_class_names()
    device = resolve_device(cfg.device)
    loaders, meta, by_split = _build_loaders(cfg, balance_train=False)
    model = _load_model(cfg, device, with_lora=True)

    test_ids = meta["records"]["test"]
    pred = predict(model, loaders["test"], device, record_ids=test_ids)

    balance = None
    bp = cfg.output_dir / "balance_report.json"
    if bp.is_file():
        balance = json.loads(bp.read_text())
    detail = None
    if balance:
        detail = {
            c: {
                "train_unique_support": balance["unique_support"].get(c, 0),
                "train_duplication_factor": balance["duplication_factor"].get(c, 0.0),
            }
            for c in names
        }

    metrics = compute_metrics(pred, names, support_detail=detail)
    metrics["split"] = "test"
    metrics["checkpoint"] = str(cfg.output_dir / "model" / "model.pt")
    metrics["resolution_order"] = cfg.resolution_order
    metrics["recipe"] = cfg.recipe
    metrics["patch_len"] = cfg.model.patch_len

    # ---- evaluability gate ------------------------------------------------
    # class_map_7.json sets min_unique_test_records_for_reporting, but until now that gate
    # only ran inside `label-audit`, so metrics.json reported a point estimate for any
    # class however small its test split was. A class with 37 test records has a recall
    # 95% CI about 30 percentage points wide; printing "0.595" for it without that interval
    # is the number a reader will quote. The gate now travels with the metrics.
    spec_map = json.loads((cfg.resolve_data_dir() / "class_map_7.json").read_text())
    min_test = int(spec_map.get("evaluability", {})
                   .get("min_unique_test_records_for_reporting", 50))
    not_evaluable = []
    for c in names:
        entry = metrics["per_class"].get(c)
        if entry is None:
            continue
        n = int(entry.get("support", 0))
        entry["evaluable"] = n >= min_test
        entry["recall_ci95"] = list(_wilson_ci(entry["recall"], n))
        entry["precision_ci95"] = list(
            _wilson_ci(entry["precision"], int(round(entry["recall"] * n / entry["precision"])))
            if entry["precision"] > 0 else (0.0, 0.0)
        )
        if not entry["evaluable"]:
            not_evaluable.append(c)
    metrics["evaluability"] = {
        "min_unique_test_records_for_reporting": min_test,
        "not_evaluable": not_evaluable,
        "note": (
            "Per-class metrics for the classes listed above rest on fewer than "
            f"{min_test} unique test records. Report them as intervals (recall_ci95, "
            "precision_ci95, both Wilson) or not at all. Augmenting the training split "
            "cannot widen a test class."
        ),
    }
    if not_evaluable:
        log.warning(
            "per-class metrics for %s rest on < %d unique test records and are reported "
            "with Wilson intervals; see metrics.evaluability",
            ", ".join(not_evaluable), min_test,
        )

    _write_json(metrics, cfg.output_dir / "metrics.json")
    np.save(cfg.output_dir / "confusion_matrix.npy",
            np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    write_predictions_csv(pred, cfg.output_dir / "predictions.csv")
    if pred.embeddings is not None:
        np.savez_compressed(
            cfg.output_dir / "embeddings.npz",
            embeddings=pred.embeddings, labels=pred.y_true,
        )
    plot_confusion_matrix(
        metrics["confusion_matrix"], names,
        cfg.output_dir / "figures" / "confusion_matrix.png",
        "Confusion matrix - LoRA-ViT (Chapman test set)",
    )
    aps = plot_precision_recall(
        pred, names, cfg.output_dir / "figures" / "precision_recall.png"
    )
    _write_json(aps, cfg.output_dir / "average_precision.json")

    print(f"\noverall accuracy   {metrics['overall_accuracy']:.4f}")
    print(f"balanced accuracy  {metrics['balanced_accuracy']:.4f}")
    print(f"macro F1           {metrics['macro_f1']:.4f}")
    print(f"macro AUC          {metrics['macro_auc']:.4f}")
    print(f"inference          {metrics['inference_ms_per_sample']:.3f} ms/sample")
    if not_evaluable:
        print(f"\nNOT REPORTABLE (< {min_test} unique test records): "
              f"{', '.join(not_evaluable)}")
        for c in not_evaluable:
            e = metrics["per_class"][c]
            lo, hi = e["recall_ci95"]
            print(f"  {c:6s} recall {e['recall']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
                  f"(n={e['support']})")
    return 0


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------
def cmd_explain(args) -> int:
    import torch

    from .evaluate import predict
    from .train import resolve_device
    from .xai import (ViTGradCAM, clinical_concordance, gradient_shap,
                      insertion_deletion, integrated_gradients,
                      per_class_lead_importance, plot_faithfulness,
                      plot_global_lead_importance, plot_gradcam_12lead,
                      plot_lead_importance, plot_tsne, tsne_embeddings,
                      upsample_cam)

    cfg = _cfg_from_args(args)
    names = active_class_names()
    device = resolve_device(cfg.device)
    loaders, meta, by_split = _build_loaders(cfg, balance_train=False)
    model = _load_model(cfg, device, with_lora=True)
    xai_dir = cfg.output_dir / "xai"
    fig_dir = cfg.output_dir / "figures" / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect a bounded, class-stratified sample of the test split.
    test_ds = loaders["test"].dataset
    n_take = min(cfg.xai.ig_samples, len(test_ds))
    rng = np.random.default_rng(cfg.train.seed)
    take = rng.choice(len(test_ds), size=n_take, replace=False)
    X = torch.stack([test_ds[int(i)][0] for i in take]).to(device)
    Y = torch.tensor([int(test_ds[int(i)][1]) for i in take], device=device)

    # ---- Grad-CAM: one example per class ---------------------------------
    log.info("Grad-CAM on up to %d example records", cfg.xai.n_example_records)
    gradcam_index = []
    with ViTGradCAM(model) as cam_fn:
        seen: Dict[int, int] = defaultdict(int)
        for j in range(X.shape[0]):
            cls = int(Y[j])
            if seen[cls] >= max(1, cfg.xai.n_example_records // len(names)):
                continue
            seen[cls] += 1
            cam, pred_idx, probs = cam_fn(X[j : j + 1], class_idx=None)
            env = upsample_cam(cam, cfg.model.patch_len, cfg.model.seq_len)
            rid = f"{names[cls]}_{j:04d}"
            np.save(xai_dir / f"gradcam_{rid}.npy", cam)
            plot_gradcam_12lead(
                X[j].cpu().numpy(), env, fig_dir / f"gradcam_{rid}.png",
                f"Grad-CAM | True: {names[cls]} | Pred: {names[pred_idx]} "
                f"| Confidence: {probs[pred_idx]:.3f}",
            )
            gradcam_index.append(
                {"record": rid, "true": names[cls], "pred": names[pred_idx],
                 "confidence": float(probs[pred_idx]), "cam_npy": f"gradcam_{rid}.npy"}
            )
    _write_json(gradcam_index, xai_dir / "gradcam_index.json")

    # ---- Integrated Gradients -------------------------------------------
    log.info("Integrated Gradients on %d records (%d steps)", X.shape[0], cfg.xai.ig_steps)
    ig_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    ig_maps = np.zeros((X.shape[0], cfg.model.in_chans, cfg.model.seq_len), dtype=np.float32)
    for j in range(X.shape[0]):
        a = integrated_gradients(model, X[j : j + 1].clone(), int(Y[j]), cfg.xai.ig_steps)
        ig_maps[j] = a
        ig_by_class[int(Y[j])].append(a)
    ig_imp = per_class_lead_importance(
        ig_by_class, "IntegratedGradients", names,
        cfg.xai.ig_confidence, cfg.xai.bootstrap_iterations, cfg.train.seed,
    )
    _write_rows(ig_imp.to_rows(), xai_dir / "integrated_gradients_lead_importance.csv")
    plot_lead_importance(
        ig_imp, fig_dir / "ig_lead_importance.png",
        "Per-lead importance across arrhythmia classes (Integrated Gradients, 95% CI)",
    )

    # ---- Gradient SHAP --------------------------------------------------
    log.info("Gradient SHAP on %d records", X.shape[0])
    bg_n = min(cfg.xai.shap_background, X.shape[0])
    background = X[rng.choice(X.shape[0], size=bg_n, replace=False)].clone()
    shap_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    for j in range(X.shape[0]):
        a = gradient_shap(model, X[j : j + 1].clone(), int(Y[j]), background,
                          n_samples=cfg.xai.shap_samples)
        shap_by_class[int(Y[j])].append(a)
    shap_imp = per_class_lead_importance(
        shap_by_class, "GradientSHAP", names,
        cfg.xai.ig_confidence, cfg.xai.bootstrap_iterations, cfg.train.seed,
    )
    _write_rows(shap_imp.to_rows(), xai_dir / "shap_lead_importance.csv")
    global_shap = plot_global_lead_importance(
        shap_imp, fig_dir / "shap_global_lead_importance.png"
    )
    _write_json(global_shap, xai_dir / "shap_global_lead_importance.json")

    # ---- Faithfulness ---------------------------------------------------
    log.info("insertion / deletion faithfulness")
    faith = insertion_deletion(
        model, X, Y, ig_maps, cfg.xai.faithfulness_fractions, cfg.model.patch_len
    )
    faith["attribution_method"] = "IntegratedGradients"
    faith["faithful"] = bool(faith["deletion_auc"] < faith["insertion_auc"])
    _write_json(faith, xai_dir / "faithfulness.json")
    plot_faithfulness(faith, fig_dir / "faithfulness.png")

    # ---- t-SNE ----------------------------------------------------------
    log.info("t-SNE of learned embeddings")
    pred = predict(model, loaders["test"], device, record_ids=meta["records"]["test"])
    Z, lab = tsne_embeddings(
        pred.embeddings, pred.y_true, cfg.xai.tsne_perplexity,
        cfg.xai.tsne_samples, cfg.train.seed,
    )
    np.savez_compressed(xai_dir / "tsne.npz", Z=Z, labels=lab)
    plot_tsne(Z, lab, fig_dir / "tsne.png",
              "t-SNE of LoRA-ViT CLS embeddings (Chapman test set)")

    # ---- Clinical concordance -------------------------------------------
    conc = {
        "integrated_gradients": clinical_concordance(ig_imp),
        "gradient_shap": clinical_concordance(shap_imp),
        "note": (
            "Concordance means the model's top-3 leads intersect the leads a cardiologist "
            "would weight under AHA/ACC and ESC criteria. It is a sanity check on the "
            "explanation, not evidence of clinical validity, and was not adjudicated by a "
            "reader study."
        ),
    }
    _write_json(conc, xai_dir / "clinical_concordance.json")
    return 0


def _write_rows(rows: List[Dict[str, object]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cmd_stats(args) -> int:
    from .evaluate import predict
    from .stats import delong_per_class, mcnemar_test
    from .train import resolve_device

    cfg = _cfg_from_args(args)
    names = active_class_names()
    device = resolve_device(cfg.device)
    loaders, meta, _ = _build_loaders(cfg, balance_train=False)

    lora = _load_model(cfg, device, with_lora=True)
    base = _load_model(cfg, device, with_lora=False)
    ids = meta["records"]["test"]
    p_lora = predict(lora, loaders["test"], device, record_ids=ids, return_embeddings=False)
    p_base = predict(base, loaders["test"], device, record_ids=ids, return_embeddings=False)

    mc = mcnemar_test(p_base.y_true, p_base.y_pred, p_lora.y_pred, "no_lora", "lora")
    _write_json(mc, cfg.output_dir / "stats" / "mcnemar.json")

    dl = delong_per_class(
        p_base.y_true, p_base.probs, p_lora.probs, names,
        cfg.xai.bootstrap_iterations, cfg.train.seed, "no_lora", "lora",
    )
    _write_json(dl, cfg.output_dir / "stats" / "delong_auc.json")

    a_key = f"{mc['model_a']}_correct_{mc['model_b']}_wrong"
    b_key = f"{mc['model_b']}_correct_{mc['model_a']}_wrong"
    print(f"\nMcNemar: {mc['statistic_name']} = {mc['statistic']}, "
          f"p = {mc['p_value']:.4f}  ({mc['method']})")
    print(f"  discordant pairs: {mc['discordant_pairs']} of {mc['n_samples']}  "
          f"({mc['model_a']} right on {mc[a_key]}, {mc['model_b']} on {mc[b_key]})")
    print(f"  {mc['interpretation']}")
    print(f"DeLong macro AUC: no_lora={dl['macro_auc_no_lora']:.4f}, "
          f"lora={dl['macro_auc_lora']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# cv
# ---------------------------------------------------------------------------
def cmd_cv(args) -> int:
    """k-fold cross-validation over the whole corpus, reporting mean +/- std.

    Why this exists: on the single 70/15/15 split the bootstrap 95% CI on balanced accuracy
    is about +/-3 percentage points. Two configurations differing by 2 points cannot be
    ranked from one split -- the difference is inside the noise of which records happened
    to land in the test set. k-fold removes the split from the comparison: every record is
    tested exactly once, and the fold-to-fold standard deviation is a direct estimate of
    how much of any observed difference is split luck.

    Folds come from the same SHA-256 bucketing as the ordinary split, so they are
    deterministic and reproduce across machines. Fold f is the test set and fold (f+1) mod k
    is validation, so model selection never sees the fold it is scored on.
    """
    import copy
    import statistics

    cfg0 = _cfg_from_args(args)
    k = int(args.cv_folds or cfg0.train.cv_folds or 5)
    if k < 2:
        raise SystemExit(f"--cv-folds must be >= 2, got {k}")

    root_out = cfg0.output_dir
    per_fold: List[Dict[str, object]] = []

    for fold in range(k):
        log.info("=" * 70)
        log.info("fold %d/%d", fold + 1, k)
        log.info("=" * 70)
        fold_args = copy.deepcopy(args)
        fold_args.output_dir = str(root_out / "cv" / f"fold_{fold}")
        fold_args.cv_fold = fold
        fold_args.cv_folds = k
        # Each fold gets its own seed so that fold-to-fold spread reflects BOTH the split
        # and the training run, which is the variance a reader actually faces.
        fold_args.seed = (cfg0.train.seed or 42) + fold
        for step, fn in (("train", cmd_train), ("evaluate", cmd_evaluate)):
            rc = fn(fold_args)
            if rc != 0:
                log.error("fold %d failed at %s", fold, step)
                return rc
        m = json.loads((Path(fold_args.output_dir) / "metrics.json").read_text())
        per_fold.append(m)

    # ---- aggregate --------------------------------------------------------
    names = active_class_names()
    scalar_keys = ["overall_accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                   "macro_precision", "macro_recall", "macro_auc"]

    def agg(values: List[float]) -> Dict[str, float]:
        vals = [v for v in values if v == v]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": round(statistics.fmean(vals), 6),
            "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "n": len(vals),
        }

    summary = {
        "n_folds": k,
        "resolution_order": cfg0.resolution_order,
        "recipe": cfg0.recipe,
        "patch_len": cfg0.model.patch_len,
        "class_names": list(names),
        "overall": {key: agg([m[key] for m in per_fold]) for key in scalar_keys},
        "per_class": {
            c: {
                metric: agg([m["per_class"][c][metric] for m in per_fold
                             if c in m["per_class"]])
                for metric in ("precision", "recall", "f1")
            }
            for c in names
        },
        "per_class_auc": {
            c: agg([m["per_class_auc"][c] for m in per_fold if c in m["per_class_auc"]])
            for c in names
        },
        "per_fold": [
            {"fold": i, "n_test": m["n_samples"],
             **{key: m[key] for key in scalar_keys}}
            for i, m in enumerate(per_fold)
        ],
        "total_test_records": sum(m["n_samples"] for m in per_fold),
        "note": (
            "Every record is tested exactly once across the k folds, so total_test_records "
            "equals the indexed corpus size. std is the fold-to-fold standard deviation; a "
            "difference between two configurations smaller than about 2*std is not "
            "distinguishable from split noise."
        ),
    }
    _write_json(summary, root_out / "cv" / "summary.json")

    print(f"\n=== {k}-fold cross-validation "
          f"({cfg0.resolution_order}, recipe={cfg0.recipe}) ===")
    print(f"  {summary['total_test_records']} records tested, each exactly once\n")
    for key in scalar_keys:
        a = summary["overall"][key]
        if a["mean"] is None:
            continue
        print(f"  {key:<20} {a['mean']:.4f} +/- {a['std']:.4f}   "
              f"[{a['min']:.4f}, {a['max']:.4f}]")
    print(f"\n  {'class':<7}{'recall':>18}{'F1':>18}{'AUC':>18}")
    for c in names:
        r = summary["per_class"][c]["recall"]
        f = summary["per_class"][c]["f1"]
        u = summary["per_class_auc"][c]
        if r["mean"] is None:
            continue
        print(f"  {c:<7}{r['mean']:>11.3f} +/-{r['std']:.3f}"
              f"{f['mean']:>11.3f} +/-{f['std']:.3f}"
              f"{u['mean']:>11.3f} +/-{u['std']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------
def cmd_run_all(args) -> int:
    for step, fn in (
        ("verify-data", cmd_verify_data),
        ("train", cmd_train),
        ("evaluate", cmd_evaluate),
        ("explain", cmd_explain),
        ("stats", cmd_stats),
    ):
        log.info("=== %s ===", step)
        rc = fn(args)
        if rc != 0:
            log.error("step %s failed with exit code %d", step, rc)
            return rc
    log.info("all steps completed")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecgvit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chapman-root", default=None, help="corpus root (or CHAPMAN_ROOT)")
    p.add_argument("--output-dir", default=None, help="artefact directory (or OUTPUT_DIR)")
    p.add_argument("--data-dir", default=None, help="environment/data (or DATA_DIR)")
    p.add_argument("--resolution-order", default=None,
                   choices=_available_resolution_orders(),
                   help="how multi-label records are reduced to one class; "
                        "run `label-audit` to compare them on your corpus")
    p.add_argument("--device", default=None, help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--lora-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--patch-len", type=int, default=None,
                   help="tokeniser patch length in samples (100 = the manuscript's 50 "
                        "tokens; 50 = 100 tokens at 50 ms resolution, which changes the "
                        "parameter counts and so is an ablation, not the reference model)")
    p.add_argument("--recipe", default=None, choices=sorted(RECIPES),
                   help="training recipe: 'paper' reproduces the manuscript exactly; "
                        "'v2' applies the corrected bundle (selection on val loss, early "
                        "stopping, weight EMA, logit adjustment instead of duplication, "
                        "physiological augmentation, trainable LoRA head at 1e-3)")
    p.add_argument("--select-metric", default=None,
                   choices=["val_loss", "val_balanced_acc", "val_macro_auc"],
                   help="which validation statistic picks the checkpoint")
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="stop after this many epochs without improvement (0 disables)")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="exponential moving average of weights (0 disables)")
    p.add_argument("--class-weighting", default=None,
                   choices=["none", "inverse_freq", "logit_adjust"],
                   help="'none' keeps duplication-based balancing; the others correct the "
                        "prior in the loss and train on the natural distribution")
    p.add_argument("--augment-strength", default=None, choices=["basic", "physio"])
    p.add_argument("--lora-lr", type=float, default=None,
                   help="stage-2 adapter learning rate (default: same as --lr)")
    p.add_argument("--train-head", action="store_true",
                   help="also train the classifier head during LoRA adaptation")
    p.add_argument("--clean-train-eval-every", type=int, default=None,
                   help="evaluate the un-augmented training split every N epochs and log "
                        "the generalisation gap (0 disables). The running train_acc is "
                        "measured under MixUp on the balanced split and cannot answer "
                        "'is it overfitting?'")
    p.add_argument("--temperature-scaling", action="store_true",
                   help="fit a post-hoc temperature on validation after training; writes "
                        "calibration.json. Cannot change accuracy.")
    p.add_argument("--cv-folds", type=int, default=None,
                   help="number of cross-validation folds for the `cv` command")
    p.add_argument("--cv-fold", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--skip-pretrain", action="store_true",
                   help="reuse an existing stage-1 checkpoint")
    p.add_argument("--allow-random-backbone", action="store_true",
                   help="adapt an untrained backbone (not the manuscript's experiment)")
    p.add_argument("-q", "--quiet", action="store_true")

    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, helptext in (
        ("verify-data", cmd_verify_data, "check corpus geometry and labels"),
        ("label-audit", cmd_label_audit,
         "compare resolution orders and report which classes are evaluable"),
        ("index", cmd_index, "build the record index"),
        ("train", cmd_train, "stage 1 pretrain + stage 2 LoRA adaptation"),
        ("evaluate", cmd_evaluate, "metrics, confusion matrix, PR curves"),
        ("cv", cmd_cv, "k-fold cross-validation, mean +/- std per metric"),
        ("explain", cmd_explain, "Grad-CAM, IG, SHAP, faithfulness, t-SNE"),
        ("stats", cmd_stats, "McNemar and DeLong, LoRA vs no-LoRA"),
        ("run-all", cmd_run_all, "everything, in order"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=fn)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(not args.quiet)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

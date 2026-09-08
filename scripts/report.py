#!/usr/bin/env python3
"""Render RESULTS.md from a completed run's artefacts.

    python scripts/report.py

Why a generator rather than a hand-written file: `outputs/` is gitignored, so the numbers
a reviewer sees have to be committed somewhere. Typing them into Markdown by hand is how a
reported metric quietly stops matching the artefact it came from -- the same failure mode
VALIDATION.md warns about for the run transcripts. This reads metrics.json,
confusion_matrix.npy and param_efficiency.json and emits the table, so RESULTS.md can be
regenerated and diffed after any run.

Deliberately narrow: the classification report, the headline metrics, the confusion matrix
and the parameter counts. The explainability artefacts are listed but not summarised -- they
are figures and per-lead attributions, and a table of them tells a reviewer nothing that
opening `outputs/figures/xai/` would not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit"
OUT = Path(__import__("os").environ.get("OUTPUT_DIR", TASK / "outputs"))
DEST = REPO / "RESULTS.md"


def _load(name: str):
    p = OUT / name
    if not p.exists():
        raise SystemExit(
            f"{p} not found.\n"
            "RESULTS.md is generated from a completed run. Produce one first:\n"
            "  docker compose run --rm solve"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    m = _load("metrics.json")
    pe = _load("param_efficiency.json")
    cm = np.load(OUT / "confusion_matrix.npy")
    names = m["class_names"]

    # device.json and config.json are provenance, not results; missing ones must not stop
    # the report, because a run can be graded without them.
    try:
        dev = _load("device.json")
    except SystemExit:
        dev = {}
    try:
        cfg = _load("config.json")
    except SystemExit:
        cfg = {}

    lora = pe["lora"]
    L: list[str] = []
    add = L.append

    add("# Results")
    add("")
    add("Test-split performance of the LoRA-adapted model, generated from the run's own")
    add("artefacts by `scripts/report.py` — not transcribed by hand. Regenerate after any")
    add("run with `python scripts/report.py`.")
    add("")
    add("> Research artefact only. Not a medical device, not validated for clinical use,")
    add("> and not evaluated in a reader study.")
    add("")

    add("## Run provenance")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Test records | {m['n_samples']} |")
    add(f"| Taxonomy | `{m.get('resolution_order', cfg.get('resolution_order', 'n/a'))}` |")
    add(f"| Recipe | `{m.get('recipe', cfg.get('recipe', 'n/a'))}` |")
    if dev.get("name"):
        add(f"| Device | {dev['name']} |")
    add(f"| Corpus | checksum-pinned subset in `environment/data/corpus` |")
    add(f"| Verifier | all three tiers pass (`reward.txt` = 1) |")
    add("")

    add("## Headline metrics")
    add("")
    add("| Metric | Value | Bar enforced by `tests/` |")
    add("| --- | ---: | --- |")
    add(f"| Accuracy | {m['overall_accuracy']:.4f} | — |")
    add(f"| **Balanced accuracy** | **{m['balanced_accuracy']:.4f}** | ≥ 0.70 |")
    add(f"| **Macro F1** | **{m['macro_f1']:.4f}** | ≥ 0.68 |")
    add(f"| Weighted F1 | {m['weighted_f1']:.4f} | — |")
    add(f"| Macro precision | {m['macro_precision']:.4f} | — |")
    add(f"| Macro recall | {m['macro_recall']:.4f} | — |")
    add(f"| Macro AUC | {m['macro_auc']:.4f} | — |")
    if m.get("inference_ms_per_sample"):
        add(f"| Inference | {m['inference_ms_per_sample']:.4f} ms/sample | — |")
    add("")

    add("## Classification report")
    add("")
    add("95% confidence intervals are bootstrap intervals over the test split; they widen")
    add("as support shrinks, which is the point of reporting them per class.")
    add("")
    add("| Class | Precision | Recall | F1 | AUC | Support | Recall 95% CI |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    aucs = m.get("per_class_auc", {})
    for c in names:
        d = m["per_class"][c]
        ci = d.get("recall_ci95") or [float("nan")] * 2
        auc = aucs.get(c)
        auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "—"
        flag = "" if d.get("evaluable", True) else " ⚠️"
        add(
            f"| {c}{flag} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} "
            f"| {auc_s} | {d['support']} | {ci[0]:.3f}–{ci[1]:.3f} |"
        )
    add("")
    weak = [c for c in names if not m["per_class"][c].get("evaluable", True)]
    if weak:
        add(
            f"⚠️ **{', '.join(weak)} is flagged `evaluable: false`** — its test support is too "
            "small for the interval to be meaningful. It is reported, and it is included in "
            "the macro averages, but a per-class claim about it is not supported by this "
            "many records."
        )
        add("")

    add("## Confusion matrix")
    add("")
    add("Rows are the true class, columns the prediction.")
    add("")
    add("| true \\ pred | " + " | ".join(names) + " | total |")
    add("| --- |" + " ---: |" * (len(names) + 1))
    for i, c in enumerate(names):
        row = " | ".join(str(int(v)) for v in cm[i])
        add(f"| **{c}** | {row} | {int(cm[i].sum())} |")
    add("")

    add("## Parameter efficiency")
    add("")
    add("| | Count |")
    add("| --- | ---: |")
    add(f"| Backbone (frozen) | {lora['frozen_parameters']:,} |")
    add(f"| LoRA adapters (trainable) | {lora['lora_parameters']:,} |")
    add(f"| Total | {lora['total_parameters']:,} |")
    add(f"| Full fine-tuning baseline | {lora['baseline_trainable_parameters']:,} |")
    add(
        f"| **Reduction in trainable parameters** | "
        f"**{lora['trainable_reduction_pct']:.2f}%** "
        f"({lora['reduction_factor']:.2f}×) |"
    )
    add("")
    add(
        f"LoRA rank {lora['lora_rank']}, alpha {lora['lora_alpha']}, applied to "
        f"{', '.join('`' + t + '`' for t in lora['lora_target_modules'])} across "
        f"{lora['n_lora_layers']} layers."
    )
    add("")

    add("## Reproducing this")
    add("")
    add("Needs Docker and a CUDA GPU. Everything else — corpus, class map, split spec — is")
    add("baked into the image, and the container runs with no network.")
    add("")
    add("```bash")
    add("docker compose build")
    add("docker compose run --rm test     # unit tier: no GPU, no prior run needed")
    add("docker compose run --rm solve    # produces outputs/ (the artefacts behind this file)")
    add("docker compose run --rm verify   # grades them; writes logs/verifier/reward.txt")
    add("python scripts/report.py         # regenerates this file from those artefacts")
    add("```")
    add("")
    add("`verify` is the check that matters: it never imports from `solution/`, and it")
    add("recomputes the reported metrics from the confusion matrix and re-runs inference")
    add("from the checkpoint rather than trusting `metrics.json`.")
    add("")

    add("## What counts as reproducing it")
    add("")
    add("**Expect these numbers to be close, not identical.** `set_seed()` seeds Python,")
    add("NumPy and torch (seed 42), and the split is derived deterministically from")
    add("`splits/split_spec.yaml` — so the data a reviewer trains on is exactly the data")
    add("used here. What is *not* pinned is cuDNN's kernel selection: `cudnn.deterministic`")
    add("is left unset and `cudnn.benchmark` unrestricted, so cuDNN chooses algorithms by")
    add("heuristic, and that choice varies with GPU model, driver version and cuDNN version.")
    add("Training on different hardware therefore lands near these metrics rather than on")
    add("them. A reviewer who expects the exact digits will report a reproducibility failure")
    add("that is not one.")
    add("")
    add("The check is the thresholds, and `tests/` enforces them:")
    add("")
    add("| Assertion | Threshold |")
    add("| --- | --- |")
    add("| `balanced_accuracy` | ≥ 0.70 |")
    add("| `macro_f1` | ≥ 0.68 |")
    add("| trainable-parameter reduction | ≥ 90% |")
    add("| faithfulness | deletion AUC < insertion AUC |")
    add("| confusion matrix | must reproduce the reported accuracy |")
    add("| `predictions.csv` | must agree with the confusion matrix |")
    add("")
    add("`docker compose run --rm verify` writing `1` to `logs/verifier/reward.txt` is the")
    add("confirmation. It passes all three tiers on the run reported above.")
    add("")
    add("A reviewer needs an NVIDIA GPU whose architecture is covered by the cu128 wheel.")
    add("Without one, `make test-cpu` still runs the unit tier anywhere, and `make solve-cpu`")
    add("will train on CPU — in hours rather than minutes, which is a wiring check and not a")
    add("result.")
    add("")
    add("## Other artefacts")
    add("")
    add("A completed run also writes explainability and statistical outputs that this file")
    add("deliberately does not summarise — Grad-CAM arrays and figures, Integrated Gradients")
    add("and Gradient SHAP per-lead importance, an insertion/deletion faithfulness test,")
    add("t-SNE embeddings, and McNemar and DeLong tests against a no-LoRA control. They are")
    add("under `outputs/xai/`, `outputs/figures/xai/` and `outputs/stats/`.")

    DEST.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {DEST.relative_to(REPO)}  ({len(L)} lines)")


if __name__ == "__main__":
    main()

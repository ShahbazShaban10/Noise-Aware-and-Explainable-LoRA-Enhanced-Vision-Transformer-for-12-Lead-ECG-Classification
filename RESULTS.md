# Results

Test-split performance of the LoRA-adapted model, generated from the run's own
artefacts by `scripts/report.py` — not transcribed by hand. Regenerate after any
run with `python scripts/report.py`.

> Research artefact only. Not a medical device, not validated for clinical use,
> and not evaluated in a reader study.

## Run provenance

| | |
| --- | --- |
| Test records | 671 |
| Taxonomy | `canonical_v2` |
| Recipe | `paper` |
| Corpus | checksum-pinned subset in `environment/data/corpus` |
| Verifier | all three tiers pass (`reward.txt` = 1) |

## Headline metrics

| Metric | Value | Bar enforced by `tests/` |
| --- | ---: | --- |
| Accuracy | 0.7705 | — |
| **Balanced accuracy** | **0.7438** | ≥ 0.70 |
| **Macro F1** | **0.7433** | ≥ 0.68 |
| Weighted F1 | 0.7670 | — |
| Macro precision | 0.7495 | — |
| Macro recall | 0.7438 | — |
| Macro AUC | 0.9507 | — |
| Inference | 0.8137 ms/sample | — |

## Classification report

95% confidence intervals are bootstrap intervals over the test split; they widen
as support shrinks, which is the point of reporting them per class.

| Class | Precision | Recall | F1 | AUC | Support | Recall 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| NSR | 0.7438 | 0.8182 | 0.7792 | 0.9742 | 110 | 0.736–0.879 |
| AFIB | 0.7077 | 0.8364 | 0.7667 | 0.9652 | 110 | 0.756–0.894 |
| SB | 0.8291 | 0.8818 | 0.8546 | 0.9873 | 110 | 0.808–0.930 |
| ST | 0.8349 | 0.8273 | 0.8311 | 0.9826 | 110 | 0.746–0.887 |
| SVT | 0.8659 | 0.7978 | 0.8304 | 0.9759 | 89 | 0.703–0.868 |
| CD | 0.7317 | 0.5455 | 0.6250 | 0.8887 | 110 | 0.452–0.635 |
| VE ⚠️ | 0.5333 | 0.5000 | 0.5161 | 0.8811 | 32 | 0.336–0.664 |

⚠️ **VE is flagged `evaluable: false`** — its test support is too small for the interval to be meaningful. It is reported, and it is included in the macro averages, but a per-class claim about it is not supported by this many records.

## Confusion matrix

Rows are the true class, columns the prediction.

| true \ pred | NSR | AFIB | SB | ST | SVT | CD | VE | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **NSR** | 90 | 0 | 9 | 8 | 0 | 3 | 0 | 110 |
| **AFIB** | 1 | 92 | 1 | 0 | 3 | 6 | 7 | 110 |
| **SB** | 10 | 0 | 97 | 0 | 0 | 1 | 2 | 110 |
| **ST** | 11 | 6 | 0 | 91 | 0 | 2 | 0 | 110 |
| **SVT** | 3 | 5 | 0 | 4 | 71 | 6 | 0 | 89 |
| **CD** | 5 | 20 | 8 | 6 | 6 | 60 | 5 | 110 |
| **VE** | 1 | 7 | 2 | 0 | 2 | 4 | 16 | 32 |

## Parameter efficiency

| | Count |
| --- | ---: |
| Backbone (frozen) | 1,648,839 |
| LoRA adapters (trainable) | 131,072 |
| Total | 1,779,911 |
| Full fine-tuning baseline | 1,648,839 |
| **Reduction in trainable parameters** | **92.05%** (12.58×) |

LoRA rank 8, alpha 16, applied to `qkv`, `proj`, `fc1`, `fc2` across 32 layers.

## Reproducing this

Needs Docker and a CUDA GPU. Everything else corpus, class map, split spec is
baked into the image, and the container runs with no network.

```bash
docker compose build
docker compose run --rm test     # unit tier: no GPU, no prior run needed
docker compose run --rm solve    # produces outputs/ (the artefacts behind this file)
docker compose run --rm verify   # grades them; writes logs/verifier/reward.txt
python scripts/report.py         # regenerates this file from those artefacts
```

`verify` is the check that matters: it never imports from `solution/`, and it
recomputes the reported metrics from the confusion matrix and re-runs inference
from the checkpoint rather than trusting `metrics.json`.

## What counts as reproducing it

**Expect these numbers to be close, not identical.** `set_seed()` seeds Python,
NumPy and torch (seed 42), and the split is derived deterministically from
`splits/split_spec.yaml` so the data a reviewer trains on is exactly the data
used here. What is *not* pinned is cuDNN's kernel selection: `cudnn.deterministic`
is left unset and `cudnn.benchmark` unrestricted, so cuDNN chooses algorithms by
heuristic, and that choice varies with GPU model, driver version and cuDNN version.
Training on different hardware therefore lands near these metrics rather than on
them. A reviewer who expects the exact digits will report a reproducibility failure
that is not one.

The check is the thresholds, and `tests/` enforces them:

| Assertion | Threshold |
| --- | --- |
| `balanced_accuracy` | ≥ 0.70 |
| `macro_f1` | ≥ 0.68 |
| trainable-parameter reduction | ≥ 90% |
| faithfulness | deletion AUC < insertion AUC |
| confusion matrix | must reproduce the reported accuracy |
| `predictions.csv` | must agree with the confusion matrix |

`docker compose run --rm verify` writing `1` to `logs/verifier/reward.txt` is the
confirmation. It passes all three tiers on the run reported above.

A reviewer needs an NVIDIA GPU whose architecture is covered by the cu128 wheel.
Without one, `make test-cpu` still runs the unit tier anywhere, and `make solve-cpu`
will train on CPU — in hours rather than minutes, which is a wiring check and not a
result.

## Other artefacts

A completed run also writes explainability and statistical outputs that this file
deliberately does not summarise — Grad-CAM arrays and figures, Integrated Gradients
and Gradient SHAP per-lead importance, an insertion/deletion faithfulness test,
t-SNE embeddings, and McNemar and DeLong tests against a no-LoRA control. They are
under `outputs/xai/`, `outputs/figures/xai/` and `outputs/stats/`.

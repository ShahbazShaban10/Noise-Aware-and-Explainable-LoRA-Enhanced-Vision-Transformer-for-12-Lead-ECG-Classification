# Results

Test-split performance of the LoRA-adapted model, generated from the run's own
artefacts by `scripts/report.py` — not transcribed by hand. Regenerate after any
run with `python scripts/report.py`.

> Research artefact only. Not a medical device, not validated for clinical use,
> and not evaluated in a reader study.

## Run provenance

| | |
| --- | --- |
| Test records | 1532 |
| Taxonomy | `canonical_v2` |
| Recipe | `paper` |
| Device | NVIDIA GeForce RTX 5070 Laptop GPU |
| Corpus | Chapman-Shaoxing, 10,247 records (fetched and checksum-verified by `fetch_dataset.sh`) |
| Verifier | all three tiers pass (`reward.txt` = 1) |

## Headline metrics

| Metric | Value | Bar enforced by `tests/` |
| --- | ---: | --- |
| Accuracy | 0.8146 | — |
| **Balanced accuracy** | **0.7404** | ≥ 0.70 |
| **Macro F1** | **0.7318** | ≥ 0.68 |
| Weighted F1 | 0.8140 | — |
| Macro precision | 0.7296 | — |
| Macro recall | 0.7404 | — |
| Macro AUC | 0.9551 | — |
| Inference | 0.3302 ms/sample | — |

## Classification report

95% confidence intervals are bootstrap intervals over the test split; they widen
as support shrinks, which is the point of reporting them per class.

| Class | Precision | Recall | F1 | AUC | Support | Recall 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| NSR | 0.7965 | 0.8566 | 0.8255 | 0.9735 | 265 | 0.809–0.894 |
| AFIB | 0.7712 | 0.8018 | 0.7862 | 0.9774 | 227 | 0.745–0.848 |
| SB | 0.9151 | 0.9117 | 0.9134 | 0.9832 | 532 | 0.884–0.933 |
| ST | 0.8632 | 0.8394 | 0.8512 | 0.9851 | 218 | 0.785–0.882 |
| SVT | 0.7614 | 0.7528 | 0.7571 | 0.9851 | 89 | 0.654–0.831 |
| CD | 0.6519 | 0.5207 | 0.5789 | 0.8863 | 169 | 0.446–0.595 |
| VE ⚠️ | 0.3478 | 0.5000 | 0.4103 | 0.8949 | 32 | 0.336–0.664 |

⚠️ **VE is flagged `evaluable: false`** — its test support is too small for the interval to be meaningful. It is reported, and it is included in the macro averages, but a per-class claim about it is not supported by this many records.

## Confusion matrix

Rows are the true class, columns the prediction.

| true \ pred | NSR | AFIB | SB | ST | SVT | CD | VE | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **NSR** | 227 | 1 | 18 | 12 | 0 | 6 | 1 | 265 |
| **AFIB** | 3 | 182 | 2 | 5 | 6 | 13 | 16 | 227 |
| **SB** | 30 | 1 | 485 | 0 | 0 | 15 | 1 | 532 |
| **ST** | 16 | 6 | 1 | 183 | 3 | 5 | 4 | 218 |
| **SVT** | 3 | 8 | 0 | 3 | 67 | 8 | 0 | 89 |
| **CD** | 5 | 30 | 21 | 8 | 9 | 88 | 8 | 169 |
| **VE** | 1 | 8 | 3 | 1 | 3 | 0 | 16 | 32 |

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

Needs Docker and a CUDA GPU. Everything else — corpus, class map, split spec — is
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
`splits/split_spec.yaml` — so the data a reviewer trains on is exactly the data
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
| faithfulness | deletion AUC < insertion AUC (balanced-accuracy curves) |
| confusion matrix | must reproduce the reported accuracy |
| `predictions.csv` | must agree with the confusion matrix |

`docker compose run --rm verify` writing `1` to `logs/verifier/reward.txt` is the
confirmation; the provenance table above records the result for this run.

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

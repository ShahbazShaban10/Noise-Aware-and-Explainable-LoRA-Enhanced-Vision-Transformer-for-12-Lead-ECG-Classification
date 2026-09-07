# Parameter-efficient, explainable arrhythmia classification from 12-lead ECG

A 12-lead electrocardiogram corpus in WFDB format is mounted read-only at
`/app/data/corpus`. Each record is a `.hea` header plus a `.mat` signal file, and the
header's `#Dx:` line carries one or more SNOMED CT concept identifiers assigned by
licensed cardiologists. Reference inputs sit alongside it: `/app/data/class_map_7.json`
gives the SNOMED code set for each of seven classes and the order in which to resolve a
record's codes to exactly one of them, `/app/data/splits/split_spec.yaml` specifies the
train/validation/test partition, and `/app/data/snomed_conditions.csv` is the code
vocabulary.

Denoise and window the signals, resolve every record to a single class, and train a
classifier that models all twelve leads jointly. Adapt it with low-rank adapters rather
than full fine-tuning. Then attribute its predictions back to leads and time regions, and
test whether those attributions are causal rather than merely plausible.

Write these to `/app/outputs`: `metrics.json`, `confusion_matrix.npy`,
`predictions.csv`, `param_efficiency.json`, `model/model.pt`,
`model/base_no_lora.pt`, `labels/label_index.json`, `index_report.json`,
`preprocessing_report.json`, `balance_report.json`, `training_history.json`,
`xai/integrated_gradients_lead_importance.csv`, `xai/shap_lead_importance.csv`,
`xai/shap_global_lead_importance.json`, `xai/faithfulness.json`, `xai/gradcam_*.npy`,
`xai/clinical_concordance.json`, `xai/tsne.npz`, `stats/mcnemar.json` and
`stats/delong_auc.json`.

Your code must be importable from `/app` as a package named `ecgvit`, exposing modules
`config`, `labels`, `preprocess`, `model`, `lora`, `xai` and `stats`.

A CUDA GPU is available.

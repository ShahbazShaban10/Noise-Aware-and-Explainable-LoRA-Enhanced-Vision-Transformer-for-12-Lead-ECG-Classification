# Solution explanation

How an expert approaches this task: how to read the data, why this method suits it, how each
graded output is computed, what validation the result needs, and which assumptions carry the
argument.

## Reading the data

The corpus is a set of WFDB records, each a plain-text `.hea` header and a binary `.mat`
signal file. A record holds twelve simultaneous leads in the order *I, II, III, aVR, aVL,
aVF, V1–V6*, sampled at 500 Hz for 5,000 samples — ten seconds — at an ADC gain of 1000
counts per millivolt. Three properties of this format decide most of the engineering.

**The annotation is genuinely multi-label.** The header's `#Dx:` line carries one or more
SNOMED CT concept identifiers, because a single ten-second strip can show sinus bradycardia
*and* a right bundle branch block *and* a premature ventricular contraction at once. A
seven-class single-label task is therefore a deliberate simplification, and the projection
from code set to class must be explicit, ordered and auditable rather than incidental. The
shipped class map gives each class its SNOMED code set together with a named resolution
order: walk the order, take the first class whose code set intersects the record's codes.
Because rhythm and morphology classes coexist, the order — not the code sets — determines
the label of most multi-coded records, and it is the single highest-leverage decision in the
whole pipeline.

**The imbalance is a property of the clinical population, not noise.** Sinus bradycardia
dominates; ventricular ectopy is rare. Plain accuracy is therefore uninformative — a model
predicting the majority class everywhere posts a respectable number — so both the checkpoint
selection metric and the headline result must be balance-aware.

**Malformed records must be rejected, not repaired.** Fewer than twelve leads, a missing
`#Dx:` line, an absent signal file: the correct response is to exclude and count. A pipeline
that pads a nine-lead record to twelve with zeros produces a plausible tensor and a
meaningless label, and nothing downstream reveals it. Rejection counts belong in a report
artefact so the cohort is reconstructible from the outputs alone.

## Why this method

Convolutional and recurrent baselines treat the twelve leads as either independent channels
or a single fused stream. In both cases the cross-lead structure clinicians actually read —
a Q wave in III but not II, concordance across the precordial leads — is available to the
model only implicitly. Self-attention over tokens that each span all twelve leads makes that
structure explicit: every token attends to every other, over time, so a 200 ms window in one
part of the strip can be related to a window elsewhere without the distance penalty a
recurrence imposes.

The tokeniser matters as much as the encoder. A hand-drawn segmentation into beats would
require reliable R-peak detection — itself hard on noisy or arrhythmic strips — and would
make the classifier's performance a function of the detector's failure modes. Instead a
two-stage strided convolution learns the tokenisation: `Conv1d(12→64, k=50, s=50)` — a 100 ms
receptive field, about one QRS width at 500 Hz — then GELU, then `Conv1d(64→128, k=2, s=2)`,
yielding fifty tokens of width 128. All twelve leads enter the first convolution together, so
every token is already a joint-lead object and no later fusion step is needed.

Low-rank adaptation addresses the deployment constraint rather than the accuracy one.
Writing an adapted projection as `y = Wx + (α/r)·BAx + b`, with rank 8 and α = 16 and `B`
zero-initialised, makes the adapted model functionally identical to the frozen backbone at
step zero, so adaptation can only depart from a known state. Injecting adapters into the
`qkv`, `proj`, `fc1` and `fc2` projections of all eight blocks — thirty-two adapted layers —
leaves exactly 131,072 trainable parameters against a 1,648,839-parameter backbone.

That arithmetic settles a design question. Since 1,779,911 − 1,648,839 = 131,072 exactly, the
adapters are additive to a *complete* backbone, so the protocol must be two-stage: fully
fine-tune the backbone, then freeze it and train only the adapters. Adapting a randomly
initialised backbone behind a frozen random head is a different and far weaker experiment
that happens to produce the same parameter count.

## How the outputs are computed

**Denoising** applies, per lead and in order: a fourth-order Butterworth band-pass at
0.5–40 Hz to remove baseline wander below and myoelectric noise above; an IIR notch at 50 Hz
with Q = 30 for powerline interference; per-lead z-scoring; and windowing to exactly 5,000
samples by centre-crop or zero-pad, with every padded or cropped record logged. Filtering is
zero-phase — forward and backward — which preserves the temporal alignment that Grad-CAM
overlays and lead attributions depend on. Zero-phase application squares the magnitude
response, so the effective attenuation at a corner is −6.02 dB rather than −3.01 dB; the
acceptance criteria are written against |H(f)|² for that reason, and a test checking the
single-pass response would silently accept a filter of the wrong order.

**Splitting** is a pure function of the record identifier: SHA-256 of the identifier with a
fixed salt, bucketed to a 70/15/15 partition, stratified within class. Because the assignment
depends on nothing but the identifier, it is identical across machines, stable when records
are added, and reproducible from the specification alone — which is what makes a golden-file
test of the split meaningful. Two rules follow and are graded: split first and balance
second, so validation and test are never resampled or augmented; and never fabricate a class,
so a class with too few unique training records fails loudly instead of being oversampled to
parity. Oversampling forty records into four thousand raises balanced accuracy without adding
diagnostic ability, so the artefacts report `unique_support` beside `support` and the
duplication factor stays visible.

**Training** uses AdamW at learning rate 1e-4 with cosine annealing and linear warm-up, batch
size 64, MixUp α = 0.2, label smoothing 0.1 and gradient clipping at 1.0, for both stages.
Checkpoint selection is on a balance-aware validation metric, never plain accuracy. The
corrected recipe additionally applies logit adjustment in place of duplication-based
balancing, an exponential moving average of the weights, and post-hoc temperature scaling —
which is monotone and therefore improves calibration without altering any prediction, so
discrimination and calibration are reported as the separate quantities they are.

**Explainability** produces four views on the held-out test split, plus an embedding map.
Grad-CAM over the last block's pre-norm activations, CLS token dropped, min-max normalised
and upsampled from fifty patches to five thousand samples, gives a per-class twelve-lead
overlay. Integrated Gradients at fifty steps from a zero baseline gives per-class, per-lead
importances with bootstrap 95% confidence intervals. Gradient SHAP against a background of
real records gives per-class and global lead rankings. Crucially, insertion/deletion
faithfulness tests whether those attributions are causal: patches are ranked by attribution,
and deleting the top-ranked patches must degrade accuracy faster than inserting them recovers
it. Without that test the other three views are pictures, not explanations. A concordance
table compares each class's top-three model leads against the leads AHA/ACC and ESC criteria
emphasise; this is a sanity check on the attribution, explicitly not clinical validation.

**Statistical validation** compares the adapted model against the stage-one backbone on
identical test records: McNemar's test on the paired error pattern, using the exact binomial
form when the discordant count is small and the continuity-corrected χ² otherwise, with the
reported statistic named so the two are never confused; and DeLong's test per class,
one-versus-rest, with 1,000 bootstrap iterations and Holm correction across classes. Where a
statistic is not estimable — a class with perfect AUC in both models has zero bootstrap
variance — the artefact carries `estimable: false` and the reason rather than a placeholder
p-value.

## What validation is needed

Three kinds. *Determinism* is validated by golden files: the resolved class map, the split
assignment of every fixture record, the band-pass magnitude response and a fully preprocessed
reference signal are each committed and compared byte-for-byte or within a stated tolerance.
*Internal consistency* is validated by recomputation: the confusion matrix must reproduce the
reported accuracy, the predictions file must reproduce the confusion matrix, the reduction
percentage must follow arithmetically from the parameter counts, and no record may appear in
both an augmented training set and the test split. *Statistical honesty* is validated by an
evaluability gate: per-class metrics carry Wilson intervals whose width must grow as support
shrinks, and a class with too little support to support an estimate is flagged rather than
reported.

## Which assumptions matter

Six, in descending order of consequence.

1. **The resolution order is a modelling decision, not a detail.** An order that ranks a
   morphology class above a rhythm class relabels thousands of records. An earlier taxonomy in
   this project merged sinus tachycardia with five repolarisation findings on a shared
   acronym prefix, mislabelling roughly 1,094 bradycardic records as tachycardic and
   depressing every downstream number.
2. **The single-label projection is lossy.** The two morphology classes that coexist with
   every rhythm account for most residual error. That ceiling is a property of the task
   formulation, not of the optimiser.
3. **500 Hz is assumed and must be verified.** A resampled record silently changes every
   filter's effective corner.
4. **50 Hz mains is a regional assumption** and would be 60 Hz elsewhere.
5. **Zero-phase filtering squares the response,** as above.
6. **Training is seeded but not bitwise deterministic on GPU.** cuDNN autotuning and
   non-deterministic reductions move metrics in the third decimal place, so a run-to-run
   difference of 0.001 is not a result. The data pipeline, by contrast, is bit-identical
   across platforms, and that is the reproducibility claim this package actually makes.

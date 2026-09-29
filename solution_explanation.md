## Why this solution works

How an expert would do this: How to read the data, why this way is appropriate, how each graded output is calculated, what validation the result needs, and what assumptions carry the argument.

## Loading data

The corpus consists of a set of WFDB records , each a plain-text `.hea header and a binary.mat signal file. A record consists of twelve simultaneous leads in the order *I, II, III, aVR, aVL, aVF, V1–V6*, sampled at 500 Hz for 5,000 samples 10 seconds at an ADC gain of 1,000 counts/mV. Most of the engineering is determined by three properties of this format.

**The annotation is really multi-label.** The `#Dx:` line of the header contains one or more SNOMED CT concept identifiers, since a single ten-second strip may show sinus bradycardia*and* a right bundle branch block *and* a premature ventricular contraction *and* all at the same time. Therefore, a seven-class single-label task is a deliberate simplification, and the projection from code set to class must be deliberate, ordered and auditable rather than incidental. The shipped class map assigns to each class its set of SNOMED codes and a named resolution order: walk the order, taking the first class whose code set intersects the record's codes.
Because rhythm and morphology classes live side by side, the label of most multi-coded records is determined by the order and not the code sets, and it is the single highest-leverage decision in the whole pipeline.
**The imbalance is a property of the clinical population, not noise.** Dominant: Sinus bradycardia. Rare: Ventricular ectopy Thus, plain accuracy is not informative a model predicting the majority class everyplace returns a respectable number so both the checkpoint selection metric and the headline result must be balance-aware.

Invalid records should be rejected, not corrected. Fewer than twelve leads, no `#Dx:` line, no signal file: the right answer is to skip and count. A pipeline that pads a 9-lead record with zeroes up to 12 produces a plausible tensor and a meaningless label and nothing downstream can tell. Rejection counts should be part of a report artifact so that the cohort can be reconstructed from the outputs alone.

## Why this approach

The convolutional and recurrent baselines treat the twelve leads as independent channels, or a single fused stream. In both cases the cross-lead structure clinicians actually read a Q wave in III but not II. Concordance across the precordial leads is available to the model only implicitly. The structure is made explicit by self-attention over tokens spanning all twelve leads: every token attends to every other, over time, so a 200 ms window in one part of the strip can be related to a window elsewhere without the distance penalty a recurrence imposes.

The tokenizer is just as important as the encoder. A hand-drawn beat segmentation would require reliable R-peak detection, which itself is hard on noisy or arrhythmic strips, and make the classifier’s performance dependent on the detector’s failure modes. Instead, a two-stage strided convolution learns the tokenization: `Conv1d(12->64, k=50, s=50)` a 100 ms receptive field, about one QRS width at 500 Hz then GELU, then `Conv1d(64->128, k=2, s=2)`, resulting in fifty tokens of width 128. All twelve leads go into the first convolution together, so every token is already a joint-lead object and no later fusion step is needed.

Low-rank adaptation is for the deployment constraint, not the accuracy constraint.
If we write the adapted projection as `y = Wx + (alpha/r).BAx + b', where rank is 8, alpha = 16 and `B' is zero-initialised, the adapted model is functionally identical to the frozen backbone at step zero, so adaptation can only depart from a known state. The injection of adapters into the qkv, proj, fc1 and fc2 projections of all eight blocks (thirty-two adapted layers in total) leaves exactly 131,072 trainable parameters against a 1,648,839-parameter backbone.

That arithmetic answers a design question. The adapters are additive to a *complete* backbone since 1,779,911 − 1,648,839 = 131,072 exactly, so the protocol must be two-stage: fully fine-tune the backbone, then freeze it and train only the adapters. Another different, and far weaker, experiment that has the same number of parameters, is to adapt a randomly initialized backbone behind a frozen random head.

## How the results are calculated
**Denoising**: per lead, sequentially: a fourth-order Butterworth band-pass at 0.5–40 Hz to remove baseline wander below and myoelectric noise above; an IIR notch at 50 Hz with Q = 30 for powerline interference; z-scoring per lead; and windowing to exactly 5,000 samples by centre-crop or zero-pad, with every padded or cropped record logged. The filtering is zero-phase (forward and backward) and maintains the temporal alignment needed for Grad-CAM overlays and lead attributions. Since the zero-phase application squares the magnitude response, the effective attenuation at a corner is -6.02 dB, not -3.01 dB. Therefore, the acceptance criteria are written against |H(f)|^2, and a test against the single-pass response would silently accept a filter of the wrong order.

**Splitting** is a pure function of the record ID: SHA-256 of the ID with a fixed salt, bucketed to a 70/15/15 partition, stratified within class. Since the assignment depends on nothing other than the identifier, it is the same across machines, is stable when records are added, and is reproducible from the specification alone—which is what makes a golden-file test of the split meaningful. There are two rules that follow, and are graded: split first and balance second, so validation and test are never resampled or augmented; and never make up a class, so a class with too few unique training records fails loudly instead of being oversampled to parity. Oversampling forty records into four thousand increases balanced accuracy, but does not add diagnostic ability, so the artifacts report `unique_support` next to `support` and the duplication factor remains visible.
**Training** is performed with AdamW with learning rate 1e-4 with cosine annealing and linear warm-up, batch size 64, MixUp $\alpha$ = 0.2, label smoothing 0.1 and gradient clipping at 1.0 for both stages.
The choice of checkpoint is made using a balance-aware validation metric and not simply accuracy. The corrected recipe also uses logit adjustment instead of duplication-based balancing, an exponential moving average of the weights, and post-hoc temperature scaling which is monotone and thus improves calibration without changing any prediction, so discrimination and calibration are reported as the separate quantities they are.

**Explainability** generates 4 views on the held-out test split and an embedding map.
Grad-CAM over pre-norm activations of the last block, CLS token dropped, min-max normalized and upsampled from fifty patches to five thousand samples. Per-class twelve-lead overlay. Integrated Gradients (zero baseline, 50 steps) Per-class, per-lead importances with bootstrap 95% confidence intervals. Gradient SHAP over a background of real records gives per-class and global lead rankings. Most importantly, insertion/deletion faithfulness tests whether those attributions are causal: patches are ranked by attribution, and removing the top-ranked patches must degrade accuracy faster than inserting them recovers it. Without that test, the other three views are pictures, not explanations. A concordance table compares the top-three model leads for each class with the leads emphasized by AHA/ACC and ESC criteria. This is a sanity check on the attribution, explicitly not a clinical validation.

Three types. *Determinism* is verified by golden files: the resolved class map, the split assignment of each fixture record, the band-pass magnitude response and a fully preprocessed reference signal are each committed and compared byte-for-byte or within a stated tolerance.
*Internal consistency* is checked by recomputing: the confusion matrix must reproduce the reported accuracy, the predictions file must reproduce the confusion matrix, the reduction percentage must follow arithmetically from the parameter counts, and no record may appear in both an augmented training set and the test split.

Descending, sixth in order of consequence.

1. **Resolution order is a modeling issue, not a detail.** Moving a morphology class above a rhythm class means rebranding thousands of records. In a previous taxonomy within this project, sinus tachycardia was bundled with five repolarization findings under a common acronym prefix, incorrectly tagging about 1094 bradycardic records as tachycardic, and depressing every downstream number.
2. **Lossy single-label projection**. Most residual error comes from the two morphology classes that occur with each rhythm. The ceiling is a property of the task formulation, not of the optimizer.
3. 500 Hz is assumed and should be checked.** The effective corner of each filter silently changes a resampled record.
4. **50Hz mains is a regional assumption** and would be 60Hz in other regions.
5. **Zero-phase filtering squares the response.** Same as above.
6. **Training is seeded but not bitwise deterministic on GPU.** cuDNN autotuning and
   non-deterministic reductions move metrics in the third decimal place, so a run-to-run
   difference of 0.001 is not a result. The data pipeline, by contrast, is bit-identical
   across platforms, and that is the reproducibility claim this package actually makes.

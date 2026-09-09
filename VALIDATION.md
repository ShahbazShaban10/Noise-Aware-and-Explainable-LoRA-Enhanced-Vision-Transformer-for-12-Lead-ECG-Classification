# Validation

Terminal output of the two required local runs.

> **This file is a template. Nothing below has been filled in.** Both transcripts must be
> pasted verbatim from runs on your own machine including the `mean` line from each, and
> including the run's date, host and GPU. Do not summarise them, do not retype them, and do
> not fill them in before the runs have actually produced those numbers.
>
> `scripts/checks.sh` runs both and captures the output to `logs/nop.txt` and
> `logs/oracle.txt`; paste from those files.

## Environment

| | |
| --- | --- |
| Date | *(fill in)* |
| Host / OS | *(fill in)* |
| GPU / driver | *(fill in)* |
| Docker | `docker --version` → *(fill in)* |
| Harbor | `harbor --version` → *(fill in)* |
| Python | *(fill in)* |
| Corpus | records under `environment/data/corpus`, manifest verified: *(fill in)* |

## Check 1 — NOP

Proves the tests are real: a do-nothing agent must score zero. Run this one first it is
fast, needs no GPU time, and fails for structural reasons that would waste an Oracle run.

```
$ harbor run -p tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit -a nop

(paste logs/nop.txt here)
```

**Result: mean = _____**  (required: 0.000)

## Check 2 — Oracle

Proves the task is solvable and the reference solution passes marking in full.

```
$ harbor run -p tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit -a oracle

(paste logs/oracle.txt here)
```

**Result: mean = _____**  (required: 1.000)

## Environment build

```
$ docker build -t ecgvit:1.0.0 -f tasks/.../environment/Dockerfile tasks/.../

(paste the tail of the build here, including the input-presence check it prints)
```

## Notes

Record anything that had to be changed to reach these numbers, and why. If a threshold was
moved, say what it was, what it became, and what measurement justified the new value — a bar
set from a claim rather than from a measurement is the most common reason an Oracle run does
not return 1.000.

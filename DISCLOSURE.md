# Disclosure of AI assistance

**Draft. The recipient, date and signature below are not yet filled in; complete them before
sending this to the assessor.**

---

To: *(Hurix contact)*
From: Shahbaz Ahmad Khanday
Date: *(fill in)*
Re: `Noise-Aware-and-Explainable-LoRA-Enhanced-Vision-Transformer-for-12-Lead-ECG-Classification`

## Why I am writing

The submission brief asks for a signed statement that the work is entirely my own and that no AI or LLM tool was used at any stage. I cannot sign that statement as written, because on 8 September 2026 I used an AI coding assistant (Claude, via Claude Code) on part of this repository. I am disclosing it rather than signing something inaccurate, and I would like a ruling on whether the assistance described below is permitted.

The assistance was confined to packaging and documentation. It did not touch the method, the
model, the tests, or the results. The boundary is precise and is verifiable from the git
history, which I have not altered — every affected commit carries a
`Co-Authored-By: Claude Opus 5` trailer, and those trailers are public on GitHub.

## What was AI-assisted

Four commits, all on 8 September 2026, all after the scientific work was complete:

| Commit | Subject | Files | Lines |
| --- | --- | --- | ---: |
| `b440674` | Raise the local run batch size to 64 | `scripts/run_local.sh` | +1 −1 |
| `cc9a86b` | Make every service runnable without a GPU, and grade without a prior solve | 12 files | +544 −66 |
| `ccc5888` | Publish the test-split results, generated from the artefacts rather than typed | 3 files | +294 |
| `70a29f9` | Say what reproducing this actually means | 3 files | +75 |

Totalling **914 insertions and 67 deletions across 15 files**. In detail:

**Containerisation** — `docker-compose.yml` (rewritten), `docker-compose.gpu.yml`, `.env`,
`.env.cpu`, `scripts/docker/entrypoint.sh`, `.dockerignore`,
`tasks/**/environment/.dockerignore`, `tasks/**/environment/requirements-cpu.txt`, and build
arguments added to `tasks/**/environment/Dockerfile`. This work made the container services
runnable without a GPU, allowed the verifier to run without a preceding training run, added
a CPU-only image variant, and allowed an external corpus to be mounted.

**Documentation and reporting** — `RESULTS.md`, `scripts/report.py` (which generates it),
and sections of `README.md`. Also `Makefile` targets and `.gitattributes` line-ending rules.

**One clarification on `b440674`.** That commit's content — changing `BATCH_SIZE` from 8 to
64 in `scripts/run_local.sh` — was my own edit, already staged in my working tree before the
session began. The assistant only created the commit. The co-authorship trailer therefore
overstates its involvement in that one change, and I would rather state that than let the
trailer imply otherwise.

## What was not AI-assisted

The following are entirely my own work and predate the session. The git diff across all four
commits is **empty** for every path listed here:

- `tasks/**/solution/src/ecgvit/` — all 11 modules: the joint-lead transformer, the LoRA
  implementation, preprocessing, label resolution, training, evaluation, the explainability
  layer and the statistical tests
- `tasks/**/solution/solve.sh` — the reference pipeline entry point
- `tasks/**/tests/` — all 280 tests, the golden files and the fixture generator
- `tasks/**/instruction.md` and `task.toml`
- `tasks/**/environment/data/` — the SNOMED vocabulary, class map, split specification and
  the checksum-pinned corpus subset
- `solution_explanation.md`
- `ecgvit_standalone (1).ipynb` and the training run recorded in it

Verifiable with:

```bash
git diff --stat e0ff372..70a29f9 -- \
  tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit/solution/src \
  tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit/tests
# returns nothing
```

## Provenance of the reported metrics

`RESULTS.md` is new, but **the numbers in it are not**. They come from a training run I had
already performed, whose artefacts had been written to disk before the session. That run had
been terminated by a disk-full condition on my machine; the assistant recovered the completed
artefacts from the stopped container and wrote the script that formats them. It did not train
the model, did not compute any metric, and did not alter `metrics.json`,
`confusion_matrix.npy` or any other artefact. Every figure in `RESULTS.md` is derived
mechanically from those files and can be regenerated with `python scripts/report.py`.

The assistant also performed operational work on my machine that produced no code: diagnosing
why Docker Desktop would not start, clearing caches, restarting services, and running the
existing test suite.

## What I am asking

Whether assistance confined to containerisation and documentation, with the method,
implementation, tests and results untouched, is permitted under the brief. If it is not, I
will rebuild the affected files without assistance and resubmit; the commits are isolated and
can be reverted without affecting the task, the solution or the tests.

I have not removed the co-authorship trailers, and I do not intend to.

Signed: ______________________   Date: ______________

Shahbaz Ahmad Khanday

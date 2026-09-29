# Declaration

## My own work

The research behind the project titled "Noise-Aware-and-Explainable-LoRA-Enhanced-Vision-Transformer-for-12-Lead-ECG-Classification" is my own: the literature survey and the gap it identifies (a lightweight, explainable solution for arrhythmia classification from 12-lead ECG), the problem formulation, the model architecture (the joint-lead vision transformer and its LoRA adaptation), the experimental design, the analysis and the manuscript. The manuscript is available at: <https://drive.google.com/drive/folders/1Y4pnFJuANI2m4doxnaOA5icwbDIyNbqB?usp=drive_link>

The reference implementation under `tasks/**/solution/src/ecgvit/`, the test suite under `tasks/**/tests/`, the task statement, the class map and split specification, `solution_explanation.md` and the notebook `ecgvit_standalone (1).ipynb` were written by me, except for the one change listed below.

## Use of AI

I used an AI coding assistant, Claude (Anthropic), through Claude Code, to build the Git project around that work. Specifically, Claude:

- **Containerisation:** wrote or rewrote `docker-compose.yml`, `docker-compose.gpu.yml`, `.env`, `.env.cpu`, `scripts/docker/entrypoint.sh`, the `.dockerignore` files, `requirements-cpu.txt`, and build arguments in the `Dockerfile`, so the services run with or without a GPU.
- **Dataset download:** wrote `environment/data/fetch_dataset.sh`, which downloads the Chapman-Shaoxing cohort from PhysioNet at image build time and verifies every file against PhysioNet's published checksums.
- **Documentation and reporting:** wrote `scripts/report.py`, which generates `RESULTS.md` from a run's own artefacts, and wrote sections of `README.md`, `environment/data/README.md` and `dataset.yaml`, plus `Makefile` targets and `.gitattributes` rules.
- **One change to the solution:** in `solution/src/ecgvit/xai.py`, changed the insertion/deletion faithfulness curves to use balanced accuracy instead of plain accuracy, with the matching test helper in `tests/test_xai.py`. On the full, class-imbalanced test split, plain accuracy let a blank signal score about 29% by predicting the majority class, which made the faithfulness check tie. This change affects only the faithfulness figure, not the classifier or its metrics.
- **Running and validating:** ran the pipeline and the test harness in Docker, and confirmed that the results in `RESULTS.md` reproduce.

Claude did not design the method, choose the architecture or the experiments, or write the manuscript. Every reported metric comes from training runs of my model on the data, computed by my evaluation code.

Signed: Shahbaz Ahmad Khanday
Date: 29-09-2026

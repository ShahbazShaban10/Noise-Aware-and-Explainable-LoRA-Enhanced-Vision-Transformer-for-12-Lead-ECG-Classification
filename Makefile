# Convenience targets. Every one of these is a single docker or harbor command; the
# Makefile exists so the exact invocation is recorded rather than remembered.
#
# The *-cpu variants build and run the CPU-only image on a host with no NVIDIA GPU or no
# NVIDIA Container Toolkit. They are a wiring check, not the graded path -- see .env.cpu.

TASK  := tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit
IMAGE := ecgvit:1.0.0

# The GPU overlay comes from COMPOSE_FILE in .env, which Compose loads by itself.
COMPOSE     := docker compose
# --env-file replaces .env wholesale, so the GPU overlay has to be left out with an
# explicit -f rather than by unsetting COMPOSE_FILE.
COMPOSE_CPU := docker compose --env-file .env.cpu -f docker-compose.yml

.PHONY: help gpu-check build solve verify test shell smoke \
        build-cpu solve-cpu verify-cpu test-cpu shell-cpu \
        oracle nop checks clean

help:
	@echo "gpu-check  prove the host can pass a GPU into a container at all"
	@echo "build      build the environment image (CUDA 12.8)"
	@echo "solve      run the reference pipeline (needs a GPU and the corpus)"
	@echo "smoke      2 + 1 epochs; wiring check, metrics are meaningless"
	@echo "verify     grade the artefacts in \$$(TASK)/outputs"
	@echo "test       unit tier only -- no GPU, no prior run needed"
	@echo "shell      interactive shell in the image"
	@echo ""
	@echo "build-cpu / solve-cpu / verify-cpu / test-cpu / shell-cpu"
	@echo "           the same, on the CPU-only image (ecgvit:1.0.0-cpu)"
	@echo ""
	@echo "oracle     harbor run -a oracle   (mean must be 1.000)"
	@echo "nop        harbor run -a nop      (mean must be 0.000)"
	@echo "checks     both of the above, output captured to logs/"

# Run this before anything else when solve fails with "no CUDA device is visible". If it
# does not print your GPU the problem is Docker or the toolkit, not this repository.
gpu-check:
	docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

# --- GPU (default) ----------------------------------------------------------
# Built through Compose rather than `docker build` so the build arguments that select the
# CUDA wheel travel with it. The context is still $(TASK)/environment, exactly as Harbor
# sets it, so a hand build and a Harbor build cannot diverge.
build:
	$(COMPOSE) build

solve:
	$(COMPOSE) run --rm solve

smoke:
	SMOKE=1 $(COMPOSE) run --rm solve

verify:
	$(COMPOSE) run --rm verify

test:
	$(COMPOSE) run --rm test

shell:
	$(COMPOSE) run --rm shell

# --- CPU --------------------------------------------------------------------
build-cpu:
	$(COMPOSE_CPU) build

solve-cpu:
	$(COMPOSE_CPU) run --rm solve

verify-cpu:
	$(COMPOSE_CPU) run --rm verify

test-cpu:
	$(COMPOSE_CPU) run --rm test

shell-cpu:
	$(COMPOSE_CPU) run --rm shell

# --- Harbor -----------------------------------------------------------------
oracle:
	harbor run -p $(TASK) -a oracle

nop:
	harbor run -p $(TASK) -a nop

checks:
	@mkdir -p logs
	harbor run -p $(TASK) -a nop    2>&1 | tee logs/nop.txt
	harbor run -p $(TASK) -a oracle 2>&1 | tee logs/oracle.txt
	@echo "Paste both transcripts into VALIDATION.md"

clean:
	rm -rf $(TASK)/outputs $(TASK)/cache logs

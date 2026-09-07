# Convenience targets. Every one of these is a single docker or harbor command; the
# Makefile exists so the exact invocation is recorded rather than remembered.

TASK := tasks/medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit
IMAGE := ecgvit:1.0.0

.PHONY: help build solve verify shell oracle nop checks clean

help:
	@echo "build    build the environment image"
	@echo "solve    run the reference pipeline (needs a GPU and the corpus)"
	@echo "verify   grade the artefacts in \$$(TASK)/outputs"
	@echo "shell    interactive shell in the image"
	@echo "oracle   harbor run -a oracle   (mean must be 1.000)"
	@echo "nop      harbor run -a nop      (mean must be 0.000)"
	@echo "checks   both of the above, output captured to logs/"

build:
	docker build -t $(IMAGE) -f $(TASK)/environment/Dockerfile $(TASK)

solve:
	docker compose run --rm solve

verify:
	docker compose run --rm verify

shell:
	docker compose run --rm shell

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

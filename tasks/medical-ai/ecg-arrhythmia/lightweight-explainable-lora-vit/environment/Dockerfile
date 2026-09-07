# Environment image for the LoRA-ViT ECG task.
#
# What goes in this image: the runtime, the pinned dependencies, and every input the task
# needs. What must NEVER go in: solution/ and tests/. Harbor uploads solution/ to
# /solution at run time and only for the oracle agent, and copies tests/ to /tests after
# the agent phase. Baking either one in would hand the reference implementation to every
# agent -- including the do-nothing agent, which would then score above zero.
#
# CUDA 12.8 is the floor: the RTX 50-series (Blackwell) is compute capability sm_120 and
# no earlier CUDA toolkit emits kernels for it.
#
# Build (from the task directory):
#   docker build -t ecgvit:1.0.0 -f environment/Dockerfile .

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="ecgvit"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Parameter-efficient explainable 12-lead ECG arrhythmia classification"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- OS packages, version-pinned -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11=3.11.* \
        python3.11-venv=3.11.* \
        python3.11-dev=3.11.* \
        python3-pip \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

RUN python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"

WORKDIR /app

# --- Python dependencies (separate layers so the heavy torch layer caches) ---
COPY environment/requirements-cu128.txt /tmp/requirements-cu128.txt
RUN python -m pip install -r /tmp/requirements-cu128.txt

COPY environment/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Fail the build rather than ship an image whose torch cannot drive the target GPU.
RUN python - <<'PY'
import torch
archs = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch list:", archs)
assert any("sm_120" in a or "sm_90" in a for a in archs), (
    f"this torch build targets {archs} and cannot run on Blackwell (sm_120); "
    "pin a cu128 or newer wheel in requirements-cu128.txt"
)
PY

# --- Task inputs ------------------------------------------------------------
# environment/data/ carries the vendored corpus subset, the SNOMED vocabulary, the class
# map and the split spec. The task runs with no network, so this is the only source of
# data and nothing can drift under it.
COPY environment/data/ /app/data/
COPY instruction.md /app/instruction.md

# The corpus is an input, not something to be edited.
RUN chmod -R a-w /app/data/corpus 2>/dev/null || true

RUN mkdir -p /app/outputs

# Verify at build time that the inputs the verifier expects are actually present, so a
# broken image fails here rather than as a mysterious zero reward hours later.
RUN test -d /app/data/corpus \
    && test -f /app/data/class_map_7.json \
    && test -f /app/data/splits/split_spec.yaml \
    && echo "inputs present: $(find /app/data/corpus -name '*.hea' | wc -l) records"

# Runs as root: Harbor mounts /logs into the container and the verifier writes the reward
# there. Dropping to an unprivileged user is a common cause of an unwritable reward file.
CMD ["sleep", "infinity"]

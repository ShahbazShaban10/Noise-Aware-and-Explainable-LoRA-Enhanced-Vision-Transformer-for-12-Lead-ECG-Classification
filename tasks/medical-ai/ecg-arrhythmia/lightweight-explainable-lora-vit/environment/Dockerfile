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
# Build context is the ENVIRONMENT directory, not the task directory. Harbor sets the
# compose build context to environment/ (docker.py -> ComposeInfraEnvVars.context_dir,
# consumed by docker-compose-build.yaml), so every COPY path below is relative to
# environment/. Build it by hand the same way:
#   docker build -t ecgvit:1.0.0 environment/

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
# Python 3.10 is Ubuntu 22.04's own interpreter, from main. An earlier version of this
# file asked for python3.11, which jammy does not package at all -- it would have needed
# the deadsnakes PPA, i.e. an external apt repository fetched at build time, which is
# exactly the "changing web resource" the environment is supposed to avoid. Every pinned
# dependency supports 3.10 and solution/pyproject.toml declares requires-python >=3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

RUN python --version && python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
RUN python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"

WORKDIR /app

# --- Python dependencies (separate layers so the heavy torch layer caches) ---
# COPY sources are relative to environment/. A path like `environment/requirements.txt`
# or `instruction.md` resolves outside the build context and fails the build under
# Harbor, even though it works with a hand-rolled `docker build -f ... <task dir>`.
COPY requirements-cu128.txt /tmp/requirements-cu128.txt
RUN python -m pip install -r /tmp/requirements-cu128.txt

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Fail the build rather than ship an image whose torch cannot drive the target GPU.
#
# This checks torch.version.cuda, NOT torch.cuda.get_arch_list(). On torch 2.8 the
# latter begins `if not is_available(): return []` -- it needs a live device, and
# docker build steps run without --gpus, so it returns an empty list on a perfectly
# good image. The CUDA build version is device-free and is what actually separates the
# cu128 wheel from the default PyPI one that carries no sm_120 kernels. The arch list
# itself is verified at RUN time by solution/solve.sh, where a GPU is attached and the
# check means something.
RUN python - <<'TORCHCHECK'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
assert torch.version.cuda is not None, (
    "a CPU-only torch wheel is installed; requirements-cu128.txt must be installed "
    "from the cu128 index"
)
major, minor = (int(x) for x in torch.version.cuda.split(".")[:2])
assert (major, minor) >= (12, 8), (
    f"torch is built against CUDA {torch.version.cuda}; Blackwell (sm_120) needs 12.8 "
    "or newer. Check the --index-url in environment/requirements-cu128.txt"
)
try:                      # informational: populated only when a device is visible
    print("arch flags:", torch._C._cuda_getArchFlags())
except Exception as exc:
    print(f"arch flags unavailable at build time ({type(exc).__name__}); "
          "solve.sh verifies them at run time")
TORCHCHECK

# --- Task inputs ------------------------------------------------------------
# environment/data/ carries the vendored corpus subset, the SNOMED vocabulary, the class
# map and the split spec. The task runs with no network, so this is the only source of
# data and nothing can drift under it.
COPY data/ /app/data/
# instruction.md is deliberately NOT copied: it sits above the build context, and
# Harbor hands the instruction to the agent as its prompt rather than as a file.
# Nothing in solution/ or tests/ reads /app/instruction.md.

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

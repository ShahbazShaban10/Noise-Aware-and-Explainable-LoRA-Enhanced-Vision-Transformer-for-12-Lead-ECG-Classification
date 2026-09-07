"""Shared fixtures and tier gating.

Three tiers:

  unit       no corpus, no trained model. Runs everywhere. Uses the seeded synthetic
             fixtures in `tests/fixtures/`.
  artifacts  grades what a completed run wrote to $OUTPUT_DIR.
  corpus     needs the real Chapman-Shaoxing corpus at $CHAPMAN_ROOT.

`test.sh` decides which tiers are required. A skipped tier is reported as skipped, never as
a pass -- a grading harness that reads "1 passed" when 20 tests were silently skipped is
worse than one that fails.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parent.parent
SOLUTION_SRC = TASK_DIR / "solution" / "src"
DATA_DIR = Path(os.environ.get("DATA_DIR") or (TASK_DIR / "environment" / "data"))

# The submission under test is whatever provides `ecgvit`. Prefer an installed package
# (an agent may have installed its own); fall back to the reference solution's source tree.
if str(SOLUTION_SRC) not in sys.path:
    sys.path.insert(0, str(SOLUTION_SRC))


def pytest_configure(config: pytest.Config) -> None:
    for marker, desc in (
        ("corpus", "requires the Chapman-Shaoxing corpus (CHAPMAN_ROOT)"),
        ("artifacts", "grades artefacts from a completed run (OUTPUT_DIR)"),
        ("slow", "takes more than a few seconds"),
        ("torch", "requires PyTorch"),
    ):
        config.addinivalue_line("markers", f"{marker}: {desc}")


def pytest_collection_modifyitems(config, items) -> None:
    have_corpus = bool(os.environ.get("CHAPMAN_ROOT")) and Path(
        os.environ.get("CHAPMAN_ROOT", "/nonexistent")
    ).is_dir()
    out = os.environ.get("OUTPUT_DIR")
    have_artifacts = bool(out) and (Path(out) / "metrics.json").is_file()

    try:
        import torch  # noqa: F401

        have_torch = True
    except ImportError:
        have_torch = False

    skip_corpus = pytest.mark.skip(
        reason="CHAPMAN_ROOT is not set or does not exist; corpus tier not run"
    )
    skip_artifacts = pytest.mark.skip(
        reason="OUTPUT_DIR/metrics.json not found; run solve.sh first"
    )
    skip_torch = pytest.mark.skip(reason="PyTorch is not installed")

    for item in items:
        if "corpus" in item.keywords and not have_corpus:
            item.add_marker(skip_corpus)
        if "artifacts" in item.keywords and not have_artifacts:
            item.add_marker(skip_artifacts)
        if "torch" in item.keywords and not have_torch:
            item.add_marker(skip_torch)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def task_dir() -> Path:
    return TASK_DIR


@pytest.fixture(scope="session")
def data_dir() -> Path:
    assert DATA_DIR.is_dir(), f"environment/data not found at {DATA_DIR}"
    return DATA_DIR


@pytest.fixture(scope="session")
def expected_dir() -> Path:
    return Path(__file__).parent / "expected"


@pytest.fixture(scope="session")
def output_dir() -> Path:
    out = os.environ.get("OUTPUT_DIR")
    if not out:
        pytest.skip("OUTPUT_DIR not set")
    return Path(out)


@pytest.fixture(scope="session")
def chapman_root() -> Path:
    root = os.environ.get("CHAPMAN_ROOT")
    if not root or not Path(root).is_dir():
        pytest.skip("CHAPMAN_ROOT not set or missing")
    return Path(root)


# ---------------------------------------------------------------------------
# Synthetic fixture corpus (generated once per session into a tmp dir)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixture_corpus(tmp_path_factory) -> Path:
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    import make_fixtures

    out = tmp_path_factory.mktemp("fixture_corpus")
    make_fixtures.generate(out)
    return out


@pytest.fixture(scope="session")
def fixture_labels() -> dict:
    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    import make_fixtures

    idx, out = 0, {}
    for label in sorted(make_fixtures.CLASS_FIXTURES):
        for _ in range(int(make_fixtures.CLASS_FIXTURES[label]["n"])):
            out[f"FX{idx:05d}"] = label
            idx += 1
    return out


@pytest.fixture(scope="session")
def metrics(output_dir: Path) -> dict:
    p = output_dir / "metrics.json"
    if not p.is_file():
        pytest.skip(f"{p} not found")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def small_model():
    """A tiny LoRA-ViT, for tests that only need structure and gradients."""
    pytest.importorskip("torch")
    from ecgvit.config import LoRAConfig, ModelConfig
    from ecgvit.model import build_model

    m_cfg = ModelConfig(seq_len=1000, patch_len=100, embed_dim=32, depth=2, num_heads=4)
    l_cfg = LoRAConfig(rank=4, alpha=8)
    return build_model(m_cfg, l_cfg) + (m_cfg, l_cfg)

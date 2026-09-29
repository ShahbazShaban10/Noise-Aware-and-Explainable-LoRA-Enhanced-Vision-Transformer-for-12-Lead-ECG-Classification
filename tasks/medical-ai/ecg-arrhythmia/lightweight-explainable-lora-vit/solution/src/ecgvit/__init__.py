"""ecgvit -- Noise-aware, explainable LoRA-enhanced Vision Transformer for 12-lead ECG.

Reference implementation for the task
`medical-ai/ecg-arrhythmia/lightweight-explainable-lora-vit`.

Public surface used by the test suite:

    from ecgvit.config     import PipelineConfig, ModelConfig, LoRAConfig, CLASS_NAMES
    from ecgvit.labels     import ClassMap, parse_dx_codes, get_class_map
    from ecgvit.preprocess import preprocess_signal, bandpass_response_db, window
    from ecgvit.model      import build_model, LoRAViT
    from ecgvit.lora       import LoRALinear, count_parameters
    from ecgvit.xai        import ViTGradCAM, integrated_gradients, insertion_deletion
    from ecgvit.stats      import mcnemar_test, delong_per_class

Submodules are imported lazily: `import ecgvit.labels` must work without torch installed,
so that label and preprocessing tests can run in a minimal environment.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (  # noqa: F401
    CLASS_NAMES,
    FS_HZ,
    LEAD_ORDER,
    N_CLASSES,
    N_LEADS,
    N_SAMPLES,
    LoRAConfig,
    ModelConfig,
    PipelineConfig,
    PreprocessConfig,
    TrainConfig,
    XAIConfig,
)

__all__ = [
    "__version__",
    "CLASS_NAMES", "LEAD_ORDER", "N_LEADS", "N_CLASSES", "N_SAMPLES", "FS_HZ",
    "PipelineConfig", "PreprocessConfig", "ModelConfig", "LoRAConfig",
    "TrainConfig", "XAIConfig",
]

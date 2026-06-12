"""tls2dseg.config — typed configuration models + YAML loader.

Public surface:

- :class:`RunConfig` — top-level frozen pydantic-settings model. Load via
  :func:`tls2dseg.config.loader.load_config` (see ``loader.py``); do NOT
  instantiate directly.
- 10 sub-models (``IOConfig``, ``RuntimeConfig``, ``PromptConfig``,
  ``PreprocessingConfig``, ``ProjectionConfig``, ``SlicingConfig``,
  ``InferenceConfig``, ``D3DExtractionConfig``, ``FusionConfig``,
  ``LoggingConfig``) — re-exported for type-annotation use in downstream
  modules (Phase 4+ engines).
- ``InferenceConfig`` is an ``Annotated`` discriminated union (D-C-02).
  Concrete inference sub-models: ``InferenceSharedConfig``,
  ``GroundedSAM2Config``, ``GroundedSAM2HFConfig``.

Phase 3 ground floor (CFG-01..03 + MODE-01 + MODE-06). Every field carries
a ``tag`` annotation in ``{primary, tuning, pipings, other}`` for Phase 8
DOC-02 schema-doc generation.
"""

from __future__ import annotations

from tls2dseg.config.models import (
    D3DExtractionConfig,
    FusionConfig,
    GroundedSAM2Config,
    GroundedSAM2HFConfig,
    InferenceConfig,
    InferenceSharedConfig,
    IOConfig,
    LoggingConfig,
    PreprocessingConfig,
    ProjectionConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
    SlicingConfig,
)

__all__ = [
    "D3DExtractionConfig",
    "FusionConfig",
    "GroundedSAM2Config",
    "GroundedSAM2HFConfig",
    "IOConfig",
    "InferenceConfig",
    "InferenceSharedConfig",
    "LoggingConfig",
    "PreprocessingConfig",
    "ProjectionConfig",
    "PromptConfig",
    "RunConfig",
    "RuntimeConfig",
    "SlicingConfig",
]

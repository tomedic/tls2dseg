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

Phase 3 ground floor (CFG-01..03 + MODE-01 + MODE-06). Every field carries
a ``tag`` annotation in ``{primary, tuning, pipings, other}`` for Phase 8
DOC-02 schema-doc generation.
"""

from __future__ import annotations

from tls2dseg.config.models import (
    D3DExtractionConfig,
    FusionConfig,
    InferenceConfig,
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
    "IOConfig",
    "InferenceConfig",
    "LoggingConfig",
    "PreprocessingConfig",
    "ProjectionConfig",
    "PromptConfig",
    "RunConfig",
    "RuntimeConfig",
    "SlicingConfig",
]

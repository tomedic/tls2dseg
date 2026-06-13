"""Engine sub-package public surface — Protocols + dict registry + builders.

Phase 4 plan 01 Task 2 (ENG-01, ENG-02). Re-exports the three Protocol
contracts and declares the name->constructor dict registries with lazy-import
builder functions.

Design decisions:
- D-A-01: dict registry (``name -> type``) in this ``__init__.py``. No
  setuptools entry-points (premature; all impls are in-tree).
- D-A-05: concrete engine classes are imported LAZILY inside the builder
  functions (not at module level) — importing ``tls2dseg.engines`` never
  triggers torch/sam2/transformers/pchandler/pc2img/pyvips loading. This is
  the key invariant for the ``--dry-run`` CLI and tier_a collection.
- Unknown ``name`` raises a clear ``ValueError`` listing registered names.
- Registries are populated by later Phase 4 plans (02: projection engine,
  03: inference + fusion engines).

Closest analog: ``tls2dseg.runtime.__init__`` (re-export surface pattern).
"""

from __future__ import annotations

import logging

from tls2dseg.engines.protocols import FusionEngine, InferenceEngine, ProjectionEngine

logger = logging.getLogger("tls2dseg.engines")

# ---------------------------------------------------------------------------
# Dict registries — name -> constructor callable.
# Populated by concrete engine modules (Phase 4 plans 02 / 03).
# Concrete classes are NOT imported here at module level (D-A-05 heavy-import
# guard); each builder function imports lazily inside its own body.
# ---------------------------------------------------------------------------

PROJECTION_ENGINES: dict[str, type] = {}
"""Registry mapping ``projection.type`` config strings to engine constructors."""

INFERENCE_ENGINES: dict[str, type] = {}
"""Registry mapping ``inference.type`` config strings to engine constructors."""

FUSION_ENGINES: dict[str, type] = {}
"""Registry mapping ``fusion.type`` config strings to engine constructors."""


def _populate_projection_registry() -> None:
    """Lazily populate PROJECTION_ENGINES with the built-in spherical engine.

    Called once at module load.  Heavy deps (pyvips, pc2img, pchandler) are
    NOT imported here — only the class reference is stored.  The class itself
    confines heavy imports to its ``project()`` method body (D-A-05).
    """
    from tls2dseg.engines.projection.spherical import SphericalProjectionEngine

    PROJECTION_ENGINES["spherical"] = SphericalProjectionEngine


_populate_projection_registry()


def _populate_inference_registry() -> None:
    """Lazily populate INFERENCE_ENGINES with both built-in SAM2 engines.

    Called once at module load.  Heavy deps (torch, sam2, transformers) are
    NOT imported here — only the class references are stored.  Each class
    confines heavy imports to its ``__init__`` method body (D-A-05).
    """
    from tls2dseg.engines.inference.grounded_sam2 import GroundedSAM2Engine
    from tls2dseg.engines.inference.grounded_sam2_hf import GroundedSAM2HFEngine

    INFERENCE_ENGINES["grounded_sam2"] = GroundedSAM2Engine
    INFERENCE_ENGINES["grounded_sam2_hf"] = GroundedSAM2HFEngine


_populate_inference_registry()


def _populate_fusion_registry() -> None:
    """Lazily populate FUSION_ENGINES with the built-in graph-cluster engine.

    Called once at module load.  Heavy deps (scipy, igraph, leidenalg) are
    NOT imported here — only the class reference is stored.  The class itself
    confines heavy imports to its ``fuse()`` method helpers (D-A-05).
    """
    from tls2dseg.engines.fusion.graph import GraphClusterFusionEngine

    FUSION_ENGINES["graph_cluster"] = GraphClusterFusionEngine


_populate_fusion_registry()


def build_projection_engine(name: str, *args: object, **kwargs: object) -> ProjectionEngine:
    """Construct a projection engine by registry name.

    Parameters
    ----------
    name :
        Registry key (e.g. ``"spherical"``). Must be present in
        ``PROJECTION_ENGINES``; unknown names raise ``ValueError``.
    *args, **kwargs :
        Forwarded to the engine constructor (model/hardware config).

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in PROJECTION_ENGINES:
        registered = list(PROJECTION_ENGINES)
        raise ValueError(
            f"Unknown projection engine {name!r}. Registered: {registered}. "
            "Add the engine to PROJECTION_ENGINES in engines/__init__.py."
        )
    cls = PROJECTION_ENGINES[name]
    return cls(*args, **kwargs)


def build_inference_engine(name: str, *args: object, **kwargs: object) -> InferenceEngine:
    """Construct an inference engine by registry name.

    Parameters
    ----------
    name :
        Registry key (e.g. ``"grounded_sam2"`` or ``"grounded_sam2_hf"``).
        Must be present in ``INFERENCE_ENGINES``; unknown names raise
        ``ValueError``.
    *args, **kwargs :
        Forwarded to the engine constructor (model/hardware config).

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in INFERENCE_ENGINES:
        registered = list(INFERENCE_ENGINES)
        raise ValueError(
            f"Unknown inference engine {name!r}. Registered: {registered}. "
            "Add the engine to INFERENCE_ENGINES in engines/__init__.py."
        )
    cls = INFERENCE_ENGINES[name]
    return cls(*args, **kwargs)


def build_fusion_engine(name: str, *args: object, **kwargs: object) -> FusionEngine:
    """Construct a fusion engine by registry name.

    Parameters
    ----------
    name :
        Registry key (e.g. ``"graph_cluster"``). Must be present in
        ``FUSION_ENGINES``; unknown names raise ``ValueError``.
    *args, **kwargs :
        Forwarded to the engine constructor (model/hardware config).

    Raises
    ------
    ValueError
        If ``name`` is not in the registry.
    """
    if name not in FUSION_ENGINES:
        registered = list(FUSION_ENGINES)
        raise ValueError(
            f"Unknown fusion engine {name!r}. Registered: {registered}. "
            "Add the engine to FUSION_ENGINES in engines/__init__.py."
        )
    cls = FUSION_ENGINES[name]
    return cls(*args, **kwargs)


__all__ = [
    "FUSION_ENGINES",
    "INFERENCE_ENGINES",
    "PROJECTION_ENGINES",
    "FusionEngine",
    "InferenceEngine",
    "ProjectionEngine",
    "build_fusion_engine",
    "build_inference_engine",
    "build_projection_engine",
]

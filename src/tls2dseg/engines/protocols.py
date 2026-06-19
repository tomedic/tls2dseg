"""Engine Protocol contracts — ProjectionEngine, InferenceEngine, FusionEngine.

Phase 4 plan 01 Task 2 (ENG-01, ENG-02). Three ``@runtime_checkable`` Protocols
that form the stable seam between ``pipeline/run.py`` and concrete engine impls.
These contracts NEVER leak SAM2 types, numpy-shape specifics, or pchandler APIs.

Design decisions:
- D-A-01: ``typing.Protocol`` (``@runtime_checkable``) + dict registry (see
  ``engines/__init__.py``). No setuptools entry-points.
- D-A-02 (LOCKED P4->P6): ``InferenceEngine.detect(image, *, request:
  InferenceRequest)`` — stable frozen request object; signature never changes.
- D-A-05: single-method Protocols; models load in engine constructors (eager);
  heavy imports stay INSIDE ``__init__`` bodies (not at module level here).
- ENG-01: each Protocol exposes at most 5 public methods. Currently 1 each.

Closest analog: ``tls2dseg.runtime.capability`` (frozen dataclass conventions)
— no existing Protocol file in the codebase; this is the first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from tls2dseg.types import Detections2D, FusionInput, FusionResult, InferenceRequest, ProjectionResult

logger = logging.getLogger("tls2dseg.engines.protocols")


@runtime_checkable
class ProjectionEngine(Protocol):
    """Engine contract for spherical projection of a point cloud scan.

    Engine = configured projection parameters + pc2img rasterization method.
    Heavy imports (pyvips, pc2img, pchandler, sklearn) live ONLY inside
    ``__init__`` so that importing this module never triggers them (D-A-05).

    Concrete impl: ``engines/projection/spherical.py``
    ``SphericalProjectionEngine`` (Phase 4 plan 02).
    """

    def project(
        self,
        pcd: object,
        *,
        features: list[str],
        resolution: tuple[int, int],
    ) -> list[ProjectionResult]:
        """Project a single scan point cloud to a list of per-feature images.

        Parameters
        ----------
        pcd :
            ``PointCloudData`` instance (typed as ``object`` to avoid pchandler
            import at Protocol definition time — callers pass the real type).
        features :
            List of scalar-field names to render (e.g. ``["intensity"]``).
        resolution :
            ``(height, width)`` target image resolution in pixels.

        Returns
        -------
        list[ProjectionResult]
            One ``ProjectionResult`` per entry in ``features``.
        """
        ...

    def set_output_dir_images(self, path: Path) -> None:
        """Set the directory where spherical feature PNGs are written.

        Callers (e.g. stage1) invoke this after ``make_output_folders`` resolves
        the run-specific image output directory.  Must be called before the first
        ``project()`` so the directory is included in every projection's params.

        Parameters
        ----------
        path :
            Filesystem path (``pathlib.Path`` or equivalent) for PNG output.
        """
        ...

    def set_pcd_path(self, path: Path) -> None:
        """Set the source-cloud path used for per-scan PNG naming.

        Callers invoke this once per scan before ``project()`` so each scan's
        intermediate PNGs are named by its own stem rather than a fallback.

        Parameters
        ----------
        path :
            Path to the current scan's source file (e.g. ``scan_0.e57``).
        """
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Engine contract for 2D inference (Grounded-DINO + SAM2 segmentation).

    Engine = loaded models on the GPU. Request = the job for this image.
    Heavy imports (torch, sam2, transformers) live ONLY inside ``__init__``
    so that importing this module never triggers model loading (D-A-05).

    D-A-02 (LOCKED P4->P6): ``detect(image, *, request: InferenceRequest)``.
    The Protocol method signature NEVER changes between Phase 4 and Phase 6 —
    only what is inside ``InferenceRequest`` grows (per-group settings in P6).

    Concrete impls: ``GroundedSAM2Engine`` / ``GroundedSAM2HFEngine``
    (Phase 4 plan 03).
    """

    def detect(
        self,
        image: np.ndarray,
        *,
        request: InferenceRequest,
    ) -> Detections2D:
        """Run 2D detection + segmentation on a single image.

        Parameters
        ----------
        image :
            Input image as a numpy array (H, W) or (H, W, C) float32.
        request :
            Frozen job spec carrying text prompt + all tuning-tagged inference
            and slicing settings (D-A-04). Construct from YAML config via the
            orchestrator before calling.

        Returns
        -------
        Detections2D
            Typed aggregate of per-detection boxes, masks, class names/ids,
            confidences, and mask labels.
        """
        ...


@runtime_checkable
class FusionEngine(Protocol):
    """Engine contract for cross-scan graph clustering / fusion (stage 2).

    Engine = graph algorithm configuration. Input = all per-scan Detections3D.
    Heavy imports (scipy, igraph, leidenalg) live ONLY inside ``__init__`` so
    that importing this module never triggers them (D-A-05).

    D-A-06: ``fuse(FusionInput) -> FusionResult`` owns the ENTIRE stage-2
    pipeline — sparse connectivity -> edge weights -> support-outlier removal
    -> graph clustering -> cluster ids -> small-cluster removal. The engine
    does NOT write files; the orchestrator applies labels.

    Concrete impl: ``GraphClusterFusionEngine`` (Phase 4 plan 03).
    """

    def fuse(self, fusion_input: FusionInput) -> FusionResult:
        """Run cross-scan fusion on the complete collection of 3D detections.

        Parameters
        ----------
        fusion_input :
            Frozen aggregate wrapping all per-scan ``Detections3D`` collections
            plus scan-id metadata.

        Returns
        -------
        FusionResult
            Per-detection cluster IDs plus a ``kept_mask`` boolean array
            indicating which detections survived all outlier-removal passes.
        """
        ...


__all__ = [
    "FusionEngine",
    "InferenceEngine",
    "ProjectionEngine",
]

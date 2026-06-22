"""Shared type aggregates for tls2dseg — engine-Protocol-facing contracts.

Phase 4 plan 01 (ENG-01..04). Populates the Phase-2 stub with the typed
request/result aggregates used by all three engine Protocols:

* ``ProjectionResult`` — per-scan 2D image result (D-B-01: data-only, no
  lift() method, no FoV data packed in).
* ``InferenceRequest`` — per-call frozen job spec for
  ``InferenceEngine.detect()``; carries the tuning-tagged config fields
  (D-A-02/D-A-04). Signature LOCKED P4→P6 (D-A-02).
* ``Detections2D`` — typed aggregate of the six dict keys currently produced
  by inference post-processing; NOT frozen because masks list is mutated
  in-place during detection (RESEARCH Pitfall 4).
* ``FusionInput`` / ``FusionResult`` — stage-2 fusion I/O aggregates
  (D-A-06: FusionEngine owns the full stage-2 pipeline).
* ``Detections3D`` — re-exported from ``detections_3d.py`` via lazy
  ``__getattr__`` to avoid pulling pchandler into this module's scope at
  import time. ``detections_3d.py`` imports pchandler at module level
  (pre-Phase-5 state); the lazy re-export keeps ``types.py`` importable in
  ``tier_a`` (no pchandler) while still satisfying
  ``from tls2dseg.types import Detections3D`` when pchandler IS available.

Import-cycle rule (RESEARCH Pitfall 2/4): this module imports ONLY numpy,
stdlib, and a lazy ref to Detections3D. It MUST NOT import from
``engines/`` or ``preprocessing/``. Heavy deps (torch, sam2, transformers,
pc2img, pchandler) must NOT be imported at module level here.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    # TYPE_CHECKING-only import for annotations; never executed at runtime.

    from tls2dseg.detections_3d import Detections3D

logger = logging.getLogger("tls2dseg.types")


def __getattr__(name: str) -> object:
    """Lazy re-export for symbols that require heavy deps (pchandler).

    ``Detections3D`` is defined in ``detections_3d.py`` which imports
    pchandler at module level. Using module ``__getattr__`` defers that
    import until the caller actually requests ``Detections3D``, keeping
    this module importable under ``tier_a`` (no pchandler installed).
    """
    if name == "Detections3D":
        from tls2dseg.detections_3d import Detections3D

        return Detections3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclasses.dataclass(frozen=True)
class ProjectionResult:
    """Per-feature spherical projection output — one per requested feature.

    D-B-01: data-only container; NO lift() method; NO FoV data packed in.
    Back-projection reads ``pcd.fov`` directly (ordering guarantee: lifting
    runs before the post-inference un-rotation, while pcd.fov still matches).

    Fields
    ------
    feature_name : str
        Name of the scalar field (e.g. ``"intensity"``, ``"range"``).
    image : np.ndarray
        2-D float32 rasterised spherical image; shape (H, W).
    path : Path
        Filesystem path where the image was / will be written.
    """

    feature_name: str
    image: np.ndarray
    path: Path
    d_azim_rad: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Per-call frozen job spec for ``InferenceEngine.detect()``.

    D-A-02 (LOCKED P4→P6): carries the text prompt + all tuning-tagged
    inference/slicing fields. Phase 6's multi-zoom grouper fills the *same*
    object with per-group settings; the Protocol method signature never
    changes.

    D-A-04: these are the ``tuning``-tagged config fields (per-call).
    Engine constructor args (``pipings``/``other``-tagged) live on the
    engine instance, NOT here.

    ``slots=True`` for performance — ``InferenceRequest`` is constructed
    once per ``detect()`` call (potentially N * scan-count per run).

    ``thread_workers`` is sourced from ``runtime.n_workers`` (no separate
    YAML field); ``empty_slice_removal_threshold`` is sourced from
    ``inference.slicing.empty_slice_removal_threshold``.
    """

    text_prompt: str
    box_threshold: float
    text_threshold: float
    slicing_enabled: bool
    slice_width_height: tuple[int, int]
    overlap_width_height: tuple[int, int]
    iou_threshold: float
    overlap_filter_strategy: str
    large_object_removal_threshold: float
    partial_detection_edge_touching_threshold: int
    thread_workers: int
    empty_slice_removal_threshold: float
    resize_factor: float = 1.0
    is_full_image_pass: bool = False
    pass_class_names: tuple[str, ...] = ()


class ClassMetadata(TypedDict):
    """Per-class metadata populated by the multi-zoom dispatcher."""

    resize_factor: float
    was_tiled: bool
    tile_size_px: int | None
    grouped_with: tuple[str, ...]


@dataclasses.dataclass
class Detections2D:
    """Typed aggregate for per-image 2D detections.

    Formalises the six dict keys currently produced by inference
    post-processing (``detections_2d.py`` dict-based format).

    NOT frozen — the ``masks`` list is mutated in-place during detection
    post-processing (RESEARCH Pitfall 4; matches current dict-based usage
    where masks are appended / filtered after the initial detection pass).

    Fields
    ------
    masks : list
        Per-detection sparse masks; each entry is a (M, 2) int32 array of
        (row, col) coordinates.
    input_boxes : np.ndarray
        (N, 4) float32 bounding boxes in xyxy format.
    confidences : np.ndarray
        (N,) float32 detection confidence scores.
    class_names : list[str]
        Length-N list of class-name strings.
    class_ids : np.ndarray
        (N,) int32 class ID array.
    mask_labels : list[str]
        Length-N list of display labels (e.g. ``"tree 0.85"``).
    """

    masks: list
    input_boxes: np.ndarray
    confidences: np.ndarray
    class_names: list[str]
    class_ids: np.ndarray
    mask_labels: list[str]


@dataclasses.dataclass(frozen=True)
class FusionInput:
    """Input aggregate for ``FusionEngine.fuse()`` — stage-2 pipeline entry.

    D-A-06: ``FusionEngine.fuse()`` owns the entire stage-2 pipeline from
    sparse connectivity through cluster removal; this dataclass is its sole
    input.

    Fields
    ------
    detections_list : list
        Per-scan 3D detection collections (``Detections3D`` instances); one
        entry per registered scan. Typed as ``list`` here to avoid importing
        pchandler at this module's load time; callers should pass
        ``list[Detections3D]`` instances.
    scan_ids : np.ndarray
        (N_scans,) int / float array of numeric scan identifiers corresponding
        1-to-1 with ``detections_list``.
    """

    detections_list: list
    scan_ids: np.ndarray


@dataclasses.dataclass(frozen=True)
class FusionResult:
    """Output of ``FusionEngine.fuse()`` — post-clustering per-detection labels.

    D-A-06: the engine does NOT write files; the orchestrator applies labels.

    Fields
    ------
    cluster_ids : np.ndarray
        (N,) int32 cluster-id per detection (across all scans, flat). Matches
        the flat ordering of detections in ``FusionInput.detections_list``.
    kept_mask : np.ndarray
        (N,) bool; True where a detection survived all outlier-removal and
        small-cluster-removal passes.
    """

    cluster_ids: np.ndarray
    kept_mask: np.ndarray


__all__ = [
    "ClassMetadata",
    "Detections2D",
    "Detections3D",
    "FusionInput",
    "FusionResult",
    "InferenceRequest",
    "ProjectionResult",
]

"""Deterministic FakeEngine stand-ins for Protocol conformance tests.

Phase 4 plan 01 Task 3 (ENG-04, D-D-02). One FakeEngine per Protocol — these
are minimal, deterministic dataclass-based stand-ins that:

* Conform to the corresponding ``@runtime_checkable`` Protocol (structurally)
  via ``isinstance`` check.
* Return typed, non-empty results suitable for pipeline piping tests.
* Import ONLY numpy + stdlib + tls2dseg.types — NO torch / sam2 / transformers /
  pchandler / pc2img at module level OR inside ``__init__``.

This tier_a constraint is what makes the conformance test in
``test_engines_protocols.py`` tier_a safe (no heavy-dep venv needed).

Patterns reference: ``PATTERNS.md`` "FakeEngines" section + D-D-02 spec;
``test_cpu_smoke.py`` lines 307-343 (``_FakePointCloud``/``_FakeDetections``
style evolved into proper ``@dataclasses.dataclass`` conforming to Protocol).
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("tls2dseg.tests.unit.fakes")


@dataclasses.dataclass
class FakeProjectionEngine:
    """Deterministic stand-in for ``ProjectionEngine`` (D-D-02).

    Returns one tiny ``ProjectionResult`` per requested feature — a small
    all-zeros image with path ``Path("fake")``. Constructor takes an optional
    ``image_size`` for test variation.

    No pchandler/pc2img/torch imports — tier_a safe.
    """

    image_size: tuple[int, int] = (2, 2)

    def project(
        self,
        pcd: object,
        *,
        features: list[str],
        resolution: tuple[int, int],
    ) -> list:
        """Return one ProjectionResult per requested feature (all-zeros image)."""
        from tls2dseg.types import ProjectionResult

        h, w = self.image_size
        return [
            ProjectionResult(
                feature_name=feature,
                image=np.zeros((h, w), dtype=np.float32),
                path=Path("fake"),
            )
            for feature in features
        ]


@dataclasses.dataclass
class FakeInferenceEngine:
    """Deterministic stand-in for ``InferenceEngine`` (D-D-02).

    Returns a fixed non-empty ``Detections2D`` on every ``detect()`` call:
    * One tiny top-left bounding box.
    * One sparse mask (single pixel at (0, 0)).
    * Class ``"fake_object"`` with confidence 0.99.

    Parametrised by ``n_detections`` for test variation (default 1).
    No torch/sam2/transformers imports — tier_a safe.
    """

    n_detections: int = 1

    def detect(self, image: np.ndarray, *, request: object) -> object:
        """Return a fixed Detections2D with ``n_detections`` fake detections."""
        from tls2dseg.types import Detections2D

        n = self.n_detections
        # Tiny top-left bounding boxes: [0, 0, 1, 1] repeated n times
        input_boxes = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (n, 1))
        # One-pixel sparse masks: each mask is a (1, 2) int32 array with the
        # top-left pixel coordinate (0, 0)
        masks = [np.array([[0, 0]], dtype=np.int32) for _ in range(n)]
        confidences = np.full(n, 0.99, dtype=np.float32)
        class_names = ["fake_object"] * n
        class_ids = np.ones(n, dtype=np.int32)
        mask_labels = [f"fake_object {0.99:.2f}"] * n

        return Detections2D(
            masks=masks,
            input_boxes=input_boxes,
            confidences=confidences,
            class_names=class_names,
            class_ids=class_ids,
            mask_labels=mask_labels,
        )


@dataclasses.dataclass
class FakeFusionEngine:
    """Deterministic stand-in for ``FusionEngine`` (D-D-02).

    Returns a ``FusionResult`` with all detections placed in cluster 0 and
    all marked as kept. Handles empty input gracefully (no detections -> 0-length arrays).

    No igraph/leidenalg/scipy imports — tier_a safe.
    """

    def fuse(self, fusion_input: object) -> object:
        """Return a FusionResult with all detections in cluster 0."""
        from tls2dseg.types import FusionResult

        # Count total detections across all scans in the fusion input
        detections_list = getattr(fusion_input, "detections_list", [])
        total = 0
        for d3d in detections_list:
            # d3d may be a Detections3D or any object with pcd_ids
            arr = getattr(d3d, "pcd_ids", np.array([]))
            total += len(arr)

        cluster_ids = np.zeros(total, dtype=np.int32)
        kept_mask = np.ones(total, dtype=bool)

        return FusionResult(cluster_ids=cluster_ids, kept_mask=kept_mask)


__all__ = [
    "FakeFusionEngine",
    "FakeInferenceEngine",
    "FakeProjectionEngine",
]

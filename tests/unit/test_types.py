"""types.py contract lock — Phase 4 plan 01 Task 1.

Phase 4 plan 01 Task 1. Locks the contract of ``tls2dseg.types``:

* Six aggregates are importable: ``ProjectionResult``, ``InferenceRequest``,
  ``Detections2D``, ``FusionInput``, ``FusionResult``, ``Detections3D``.
* Import does NOT pull ``torch``, ``sam2``, ``transformers``, or ``pc2img``
  into ``sys.modules``.
* ``InferenceRequest`` is frozen: assigning a field raises
  ``dataclasses.FrozenInstanceError``.

Most tests are ``tier_a`` (no pchandler/pc2img/torch). The ``Detections3D``
re-export and full-six-aggregate tests are ``tier_b_light`` because
``detections_3d.py`` imports pchandler at module level (pre-Phase-5 state);
they skip cleanly in cloud CI via the module-level guard below.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# pchandler availability probe — used to conditionally skip tier_b_light
# tests that depend on Detections3D (which requires pchandler at import time
# via detections_3d.py's module-level import). This guard is intentionally
# broad (catches AttributeError from numpy 2.0 compat issues too).
# ---------------------------------------------------------------------------
try:
    import pchandler

    _pchandler_available = True
except Exception:
    _pchandler_available = False


@pytest.mark.tier_a
def test_types_import_five_aggregates_tier_a() -> None:
    """Five aggregates (excluding Detections3D) import from ``tls2dseg.types`` in tier_a."""
    from tls2dseg.types import (
        Detections2D,
        FusionInput,
        FusionResult,
        InferenceRequest,
        ProjectionResult,
    )


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _pchandler_available, reason="pchandler not importable — tier_b_light only")
def test_types_import_all_six_aggregates() -> None:
    """All six aggregates import from ``tls2dseg.types`` (requires pchandler for Detections3D)."""
    from tls2dseg.types import (
        Detections2D,
        Detections3D,
        FusionInput,
        FusionResult,
        InferenceRequest,
        ProjectionResult,
    )


@pytest.mark.tier_a
def test_types_import_no_heavy_dep_leak() -> None:
    """Importing tls2dseg.types does NOT pull torch/sam2/transformers/pc2img."""
    before = set(sys.modules)
    import tls2dseg.types

    after = set(sys.modules)
    leaked = {"torch", "sam2", "transformers", "pc2img"} & (after - before)
    assert not leaked, (
        f"tls2dseg.types leaked heavy deps into sys.modules: {leaked}. "
        "Heavy deps must only be imported inside engine __init__ methods (D-A-05)."
    )


@pytest.mark.tier_a
def test_inference_request_is_frozen_dataclass() -> None:
    """``InferenceRequest`` is a frozen stdlib dataclass (FrozenInstanceError on mutation)."""
    from tls2dseg.types import InferenceRequest

    assert dataclasses.is_dataclass(InferenceRequest), (
        "InferenceRequest must be a dataclass (not pydantic) per RESEARCH §3.2"
    )
    assert InferenceRequest.__dataclass_params__.frozen, (
        "InferenceRequest must be frozen=True so per-call objects are immutable (D-A-04)"
    )

    req = InferenceRequest(
        text_prompt="tree.",
        box_threshold=0.10,
        text_threshold=0.10,
        slicing_enabled=False,
        slice_width_height=(640, 640),
        overlap_width_height=(100, 100),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.30,
        partial_detection_edge_touching_threshold=5,
        thread_workers=4,
        empty_slice_removal_threshold=0.95,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.text_prompt = "changed"  # type: ignore[misc]


@pytest.mark.tier_a
def test_inference_request_field_set() -> None:
    """``InferenceRequest`` exposes the exact RESEARCH §3.2 field set."""
    from tls2dseg.types import InferenceRequest

    expected_fields = {
        "text_prompt",
        "box_threshold",
        "text_threshold",
        "slicing_enabled",
        "slice_width_height",
        "overlap_width_height",
        "iou_threshold",
        "overlap_filter_strategy",
        "large_object_removal_threshold",
        "partial_detection_edge_touching_threshold",
        "thread_workers",
        "empty_slice_removal_threshold",
    }
    actual_fields = {f.name for f in dataclasses.fields(InferenceRequest)}
    assert actual_fields == expected_fields, (
        f"InferenceRequest fields drifted from RESEARCH §3.2 spec: {actual_fields} != {expected_fields}"
    )


@pytest.mark.tier_a
def test_projection_result_is_frozen_dataclass() -> None:
    """``ProjectionResult`` is a frozen dataclass with feature_name/image/path fields."""
    from pathlib import Path

    from tls2dseg.types import ProjectionResult

    assert dataclasses.is_dataclass(ProjectionResult)
    assert ProjectionResult.__dataclass_params__.frozen

    field_names = {f.name for f in dataclasses.fields(ProjectionResult)}
    assert "feature_name" in field_names
    assert "image" in field_names
    assert "path" in field_names

    pr = ProjectionResult(
        feature_name="intensity",
        image=np.zeros((2, 2), dtype=np.float32),
        path=Path("test.png"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pr.feature_name = "changed"  # type: ignore[misc]


@pytest.mark.tier_a
def test_detections2d_is_not_frozen() -> None:
    """``Detections2D`` is a mutable dataclass (masks list mutated in place per Pitfall 4)."""
    from tls2dseg.types import Detections2D

    assert dataclasses.is_dataclass(Detections2D)
    # Must NOT be frozen — masks list is mutated in-place
    assert not Detections2D.__dataclass_params__.frozen, (
        "Detections2D must NOT be frozen — masks list is mutated in place (RESEARCH Pitfall 4)"
    )

    field_names = {f.name for f in dataclasses.fields(Detections2D)}
    expected = {"masks", "input_boxes", "confidences", "class_names", "class_ids", "mask_labels"}
    assert expected.issubset(field_names), f"Detections2D missing fields: {expected - field_names}"


@pytest.mark.tier_a
def test_fusion_input_and_result_are_frozen() -> None:
    """``FusionInput`` and ``FusionResult`` are frozen dataclasses."""
    from tls2dseg.types import FusionInput, FusionResult

    assert dataclasses.is_dataclass(FusionInput)
    assert FusionInput.__dataclass_params__.frozen

    assert dataclasses.is_dataclass(FusionResult)
    assert FusionResult.__dataclass_params__.frozen


@pytest.mark.tier_b_light
@pytest.mark.skipif(not _pchandler_available, reason="pchandler not importable — tier_b_light only")
def test_detections3d_re_export() -> None:
    """``Detections3D`` is importable from ``tls2dseg.types`` (re-export from detections_3d.py).

    Marked tier_b_light — detections_3d.py imports pchandler at module level
    (pre-Phase-5 state); this test skips in cloud CI (tier_a sandbox).
    """
    from tls2dseg.types import Detections3D

    assert dataclasses.is_dataclass(Detections3D)
    field_names = {f.name for f in dataclasses.fields(Detections3D)}
    # Check the key fields from detections_3d.py lines 25-36
    assert "pcd_ids" in field_names
    assert "instances" in field_names
    assert "classes" in field_names
    assert "centroids" in field_names

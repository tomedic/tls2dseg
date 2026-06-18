"""Engine Protocol conformance tests — Phase 4 plan 01 Task 3 (ENG-04, TEST-10).

Phase 4 plan 01 Task 3. Locks the engine Protocol contracts by asserting that
FakeEngines structurally conform AND can be called successfully:

* ``isinstance(FakeProjectionEngine(), ProjectionEngine)`` is True AND
  ``project()`` returns ``list[ProjectionResult]``.
* ``isinstance(FakeInferenceEngine(), InferenceEngine)`` is True AND
  ``detect()`` returns ``Detections2D`` with at least 1 detection.
* ``isinstance(FakeFusionEngine(), FusionEngine)`` is True AND
  ``fuse()`` returns ``FusionResult``.
* ``build_inference_engine`` / ``build_projection_engine`` / ``build_fusion_engine``
  exist and are callable (registry builder smoke check).

ENG-04 hard rule: FakeEngines live in the SAME plan as the Protocol. If
the Fake is awkward to write, redesign the Protocol.

Pitfall 3 guard: ``isinstance`` alone is insufficient — we MUST call the method
and assert the concrete return type.

Marked ``tier_a``: imports ONLY stdlib + numpy + tls2dseg.engines + tls2dseg.types +
tests.unit.fakes. No torch/sam2/transformers/pchandler/pc2img at import time.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tls2dseg.engines import (
    build_fusion_engine,
    build_inference_engine,
    build_projection_engine,
)
from tls2dseg.engines.protocols import FusionEngine, InferenceEngine, ProjectionEngine
from tls2dseg.types import Detections2D, FusionInput, FusionResult, InferenceRequest, ProjectionResult


@pytest.mark.tier_a
def test_fake_projection_engine_conforms_to_protocol() -> None:
    """FakeProjectionEngine isinstance-conforms AND project() returns list[ProjectionResult].

    Pitfall 3 (PATTERNS.md): isinstance alone is insufficient. We call
    project() and assert the concrete return type.
    """
    from tests.unit.fakes import FakeProjectionEngine

    engine = FakeProjectionEngine()

    # isinstance check (D-A-01: @runtime_checkable)
    assert isinstance(engine, ProjectionEngine), (
        "FakeProjectionEngine must satisfy isinstance(engine, ProjectionEngine)"
    )

    # Call the method and assert concrete return type (Pitfall 3)
    results = engine.project(object(), features=["intensity"], resolution=(2, 2))
    assert isinstance(results, list), f"project() must return list, got {type(results)}"
    assert len(results) == 1, f"Expected 1 result for 1 feature, got {len(results)}"
    assert isinstance(results[0], ProjectionResult), f"Each element must be ProjectionResult, got {type(results[0])}"


@pytest.mark.tier_a
def test_fake_inference_engine_conforms_to_protocol() -> None:
    """FakeInferenceEngine isinstance-conforms AND detect() returns Detections2D with >=1 detection.

    Pitfall 3 (PATTERNS.md): isinstance alone is insufficient. We call
    detect() and assert the concrete return type + non-empty detections.
    """
    from tests.unit.fakes import FakeInferenceEngine

    engine = FakeInferenceEngine()

    # isinstance check
    assert isinstance(engine, InferenceEngine), "FakeInferenceEngine must satisfy isinstance(engine, InferenceEngine)"

    # Build a minimal valid InferenceRequest inline
    request = InferenceRequest(
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

    # Call detect() and assert concrete return type (Pitfall 3)
    image = np.zeros((4, 4), dtype=np.float32)
    result = engine.detect(image, request=request)
    assert isinstance(result, Detections2D), f"detect() must return Detections2D, got {type(result)}"
    assert dataclasses.is_dataclass(result), "Detections2D must be a dataclass"
    # Non-empty: FakeInferenceEngine returns >= 1 detection (D-D-02)
    assert len(result.class_names) >= 1, (
        f"FakeInferenceEngine must return >= 1 detection, got {len(result.class_names)}"
    )


@pytest.mark.tier_a
def test_fake_fusion_engine_conforms_to_protocol() -> None:
    """FakeFusionEngine isinstance-conforms AND fuse() returns FusionResult.

    Pitfall 3 (PATTERNS.md): isinstance alone is insufficient. We call
    fuse() and assert the concrete return type.
    """
    from tests.unit.fakes import FakeFusionEngine

    engine = FakeFusionEngine()

    # isinstance check
    assert isinstance(engine, FusionEngine), "FakeFusionEngine must satisfy isinstance(engine, FusionEngine)"

    # Build a minimal FusionInput with an empty list (no real pchandler needed)
    fusion_input = FusionInput(
        detections_list=[],
        scan_ids=np.array([], dtype=np.int32),
    )

    # Call fuse() and assert concrete return type (Pitfall 3)
    result = engine.fuse(fusion_input)
    assert isinstance(result, FusionResult), f"fuse() must return FusionResult, got {type(result)}"
    assert dataclasses.is_dataclass(result), "FusionResult must be a dataclass"
    assert hasattr(result, "cluster_ids"), "FusionResult must have cluster_ids"
    assert hasattr(result, "kept_mask"), "FusionResult must have kept_mask"


@pytest.mark.tier_a
def test_registry_builders_are_callable() -> None:
    """build_inference/projection/fusion_engine exist and are callable.

    Registry smoke check — the builders should exist and raise a clear
    ValueError for an unknown engine name (not AttributeError or ImportError).
    The registries are empty until Phase 4 plan 02/03 populates them.
    """
    with pytest.raises(ValueError, match="Unknown"):
        build_inference_engine("nonexistent_engine")

    with pytest.raises(ValueError, match="Unknown"):
        build_projection_engine("nonexistent_engine")

    with pytest.raises(ValueError, match="Unknown"):
        build_fusion_engine("nonexistent_engine")

"""FakeEngine piping test — end-to-end wiring with Fake inference and real lift.

Phase 4 plan 05 Task 4 (TEST-10, D-D-06 half 1). Exercises the pipeline wiring:

    FakeProjectionEngine.project() -> FakeInferenceEngine.detect() -> lift_masks_to_pcd()

using a synthetic ``SyntheticPcd`` fixture (provided by ``tests/conftest.py``)
that has the required ``spherical_coordinates``, ``fov``, and ``scalar_fields``
attributes needed by ``lift_masks_to_pcd``.

Key constraints (D-D-06 binding rule):
- MUST NOT import or instantiate GroundedSAM2Engine or GroundedSAM2HFEngine.
- MUST NOT load any torch model.
- The FakeInferenceEngine returns deterministic non-empty detections so the
  downstream lift path is exercised (not silently skipped due to empty output).

Marked ``tier_b_light`` per D-D-06 — this is the piping-level wiring test that
gives tier_b_light real, non-trivial content. Uses no real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.unit.fakes import FakeInferenceEngine, FakeProjectionEngine
from tls2dseg.lifting.masks_to_pcd import lift_masks_to_pcd
from tls2dseg.types import Detections2D


@pytest.mark.tier_b_light
def test_fake_engine_piping_writes_instances_and_classes(
    synthetic_pcd: object,
) -> None:
    """FakeProjectionEngine + FakeInferenceEngine + lift_masks_to_pcd writes scalar fields.

    Wires the three-stage pipeline path:
    1. ``FakeProjectionEngine.project()`` produces a tiny fake 2D image.
    2. ``FakeInferenceEngine.detect()`` returns one deterministic detection.
    3. ``lift_masks_to_pcd()`` back-projects the detection masks onto the
       synthetic point cloud.

    Asserts:
    - ``pcd.scalar_fields["instances"]`` and ``pcd.scalar_fields["classes"]``
      exist after the lift (verifies wiring reaches the scalar-field write).
    - Neither array is None and both have length equal to the number of points.
    - No real GroundedSAM2 or HF engine is instantiated; no torch model loaded.

    This test is the D-D-06 half-1 deliverable: a tier_b_light test that runs
    real data types through Fake inference with no real model run.
    """
    proj_engine = FakeProjectionEngine(image_size=(4, 4))
    inf_engine = FakeInferenceEngine(n_detections=1)

    # Stage 1: project the synthetic pcd to a tiny fake image
    projection_results = proj_engine.project(
        synthetic_pcd,
        features=["intensity"],
        resolution=(4, 4),
    )
    assert len(projection_results) == 1, "FakeProjectionEngine must return one result per feature"
    fake_image = projection_results[0].image
    assert fake_image.shape == (4, 4), f"Expected (4,4) fake image, got {fake_image.shape}"

    # Stage 2: run fake inference on the 2D image
    # FakeInferenceEngine.detect() ignores ``request`` — pass a simple sentinel.
    import types as _types

    request = _types.SimpleNamespace(text_prompt="fake_object")
    detections: Detections2D = inf_engine.detect(fake_image, request=request)
    assert isinstance(detections, Detections2D), "FakeInferenceEngine must return Detections2D"
    assert len(detections.class_names) == 1, "FakeInferenceEngine(n_detections=1) must return 1 detection"

    # Stage 3: build a 2D mask from the fake detection (single pixel at (0,0))
    # The FakeInferenceEngine returns masks as list of (M, 2) int32 arrays of pixel indices.
    image_height, image_width = fake_image.shape
    instance_mask = np.zeros((image_height, image_width), dtype=np.int32)
    semantic_mask = np.zeros((image_height, image_width), dtype=np.int32)

    for det_idx, mask_pixels in enumerate(detections.masks):
        # mask_pixels is (M, 2) int32 array with [row, col] pairs
        if len(mask_pixels) > 0:
            rows = np.clip(mask_pixels[:, 0], 0, image_height - 1)
            cols = np.clip(mask_pixels[:, 1], 0, image_width - 1)
            instance_mask[rows, cols] = det_idx + 1  # 1-indexed instances
            semantic_mask[rows, cols] = detections.class_ids[det_idx]

    # Stage 4: lift the masks back onto the 3D point cloud
    lift_masks_to_pcd(synthetic_pcd, instance_mask, semantic_mask)

    # Assert scalar fields were written
    sf = synthetic_pcd.scalar_fields
    n_points = synthetic_pcd.xyz.shape[0]

    assert "instances" in sf, "lift_masks_to_pcd must write 'instances' scalar field"
    assert "classes" in sf, "lift_masks_to_pcd must write 'classes' scalar field"

    instances_arr = sf["instances"]
    classes_arr = sf["classes"]

    assert instances_arr is not None, "'instances' scalar field must not be None"
    assert classes_arr is not None, "'classes' scalar field must not be None"
    assert len(instances_arr) == n_points, f"'instances' must have length {n_points}, got {len(instances_arr)}"
    assert len(classes_arr) == n_points, f"'classes' must have length {n_points}, got {len(classes_arr)}"


@pytest.mark.tier_b_light
def test_fake_engine_piping_no_real_engine_imported() -> None:
    """Confirm no real GroundedSAM2 / HF engine is imported in this module.

    Guards D-D-06 binding rule: no real model is instantiated by tier_b_light
    piping tests. This test is a static guard — it verifies the module's
    namespace never imports the real engine classes.

    Since the import is done at module load time, a simple name check on the
    module dict suffices.
    """
    import sys

    this_module = sys.modules[__name__]
    # Real engine class names must NOT appear in this module's namespace
    forbidden = {"GroundedSAM2Engine", "GroundedSAM2HFEngine"}
    module_names = set(dir(this_module))
    present = forbidden & module_names
    assert not present, (
        f"D-D-06 binding rule violated: real engine class(es) {present} "
        "found in test_fake_engine_piping module namespace (no real model "
        "may be imported in tier_b_light piping tests)"
    )

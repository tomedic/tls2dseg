"""Dispatch regression test — run.py selects the engine from config.inference.type.

Phase 4 gap-closure (04-06, CR-01 + integration gap).

Asserts that:
1. ``build_inference_engine`` constructs the engine registered under
   ``inference.type``, not a hardcoded legacy function.
2. A ``FakeInferenceEngine`` injected into ``INFERENCE_ENGINES`` under a test key
   is selected when ``build_inference_engine`` is called with that key.
3. ``detect()`` on the constructed engine returns a valid ``Detections2D`` object.
4. No torch / sam2 / transformers imports occur during dispatch (D-A-05 guard).

This test regression-locks the wiring closed by the Phase 4 gap-closure:
  config.inference.type → INFERENCE_ENGINES registry → engine instance → detect().

Marked ``tier_b_light``: runs with pchandler + pc2img available, but uses a
FakeInferenceEngine so no real model or GPU is needed.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.unit.fakes import FakeInferenceEngine
from tls2dseg.engines import INFERENCE_ENGINES, build_inference_engine
from tls2dseg.types import Detections2D, InferenceRequest

_FAKE_KEY = "__test_fake_inference_engine__"


@pytest.fixture(autouse=False)
def inject_fake_engine(monkeypatch: pytest.MonkeyPatch):
    """Inject FakeInferenceEngine into INFERENCE_ENGINES under a test-only key.

    Removes the entry after the test so the registry is not polluted across
    the session.
    """
    monkeypatch.setitem(INFERENCE_ENGINES, _FAKE_KEY, FakeInferenceEngine)
    yield
    # monkeypatch automatically reverts setitem after the test.


@pytest.mark.tier_b_light
def test_build_inference_engine_selects_registered_class(
    inject_fake_engine: None,
) -> None:
    """build_inference_engine returns an instance of the class registered under name.

    Validates the dispatch path:
        build_inference_engine(_FAKE_KEY) → FakeInferenceEngine instance.

    No torch/sam2/transformers are imported — D-A-05 lazy-import contract holds.
    """
    engine = build_inference_engine(_FAKE_KEY)
    assert isinstance(engine, FakeInferenceEngine), (
        f"build_inference_engine({_FAKE_KEY!r}) must return FakeInferenceEngine, got {type(engine).__name__}"
    )


@pytest.mark.tier_b_light
def test_fake_engine_detect_returns_detections2d(
    inject_fake_engine: None,
) -> None:
    """Engine constructed via the registry calls detect() and returns Detections2D.

    Simulates the run.py dispatch pattern:
        engine = build_inference_engine(cfg.inference.type, ...)
        detections_2d = engine.detect(image, request=inference_request)

    Validates the full wiring path from config-driven dispatch to typed output,
    without instantiating any real model or importing torch/sam2/transformers.
    """
    engine = build_inference_engine(_FAKE_KEY)

    # Synthetic 8x8 float32 image — matches what the spherical projector produces
    image = np.zeros((8, 8), dtype=np.float32)

    # Minimal InferenceRequest with representative tuning values
    request = InferenceRequest(
        text_prompt="tree.pole",
        box_threshold=0.10,
        text_threshold=0.10,
        slicing_enabled=False,
        slice_width_height=(320, 320),
        overlap_width_height=(32, 32),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.30,
        partial_detection_edge_touching_threshold=5,
    )

    detections_2d = engine.detect(image, request=request)

    assert isinstance(detections_2d, Detections2D), (
        f"engine.detect() must return Detections2D, got {type(detections_2d).__name__}"
    )
    assert len(detections_2d.class_names) == 1, "FakeInferenceEngine(n_detections=1) must return 1 detection"
    assert detections_2d.masks is not None, "Detections2D.masks must not be None"
    assert isinstance(detections_2d.input_boxes, np.ndarray), "Detections2D.input_boxes must be np.ndarray"


@pytest.mark.tier_b_light
def test_dispatch_does_not_import_torch_or_sam2(
    inject_fake_engine: None,
) -> None:
    """No torch / sam2 / transformers module is imported by the dispatch path.

    Guards D-A-05: heavy model deps must NOT be imported at registry-lookup or
    engine-construction time when using a FakeInferenceEngine. Verifies the
    lazy-import contract from engines/__init__.py and FakeInferenceEngine.

    This test inspects sys.modules AFTER construction + detect() to confirm
    the fake path never touches heavy deps.
    """
    # Build and call detect() with the fake engine
    engine = build_inference_engine(_FAKE_KEY)
    image = np.zeros((4, 4), dtype=np.float32)
    request = InferenceRequest(
        text_prompt="object",
        box_threshold=0.10,
        text_threshold=0.10,
        slicing_enabled=False,
        slice_width_height=(64, 64),
        overlap_width_height=(8, 8),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.30,
        partial_detection_edge_touching_threshold=5,
    )
    engine.detect(image, request=request)

    # Confirm no heavy deps were pulled in by the fake dispatch path
    forbidden_prefixes = ("torch", "sam2", "transformers")
    loaded_forbidden = [
        mod
        for mod in sys.modules
        if any(mod == prefix or mod.startswith(f"{prefix}.") for prefix in forbidden_prefixes)
    ]
    assert not loaded_forbidden, (
        f"D-A-05 violated: heavy deps imported via FakeInferenceEngine dispatch: "
        f"{loaded_forbidden[:10]!r} (first 10 shown)"
    )


@pytest.mark.tier_b_light
def test_build_inference_engine_unknown_key_raises_value_error(
    inject_fake_engine: None,
) -> None:
    """build_inference_engine raises ValueError for an unregistered engine key.

    Guards the user-facing error path (no silent KeyError from dict lookup).
    """
    with pytest.raises(ValueError, match="Unknown inference engine"):
        build_inference_engine("__definitely_not_registered__")

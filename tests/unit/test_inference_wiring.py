"""Spy test: engine forwards runtime.n_workers and empty_slice_removal_threshold
to SparseMasksInferenceSlicer.

Tier: tier_b_light
Rationale: grounded_sam2.detect() imports torch at the top of the method body.
torch is not available in the tier_a venv, so this test must run in tier_b_light
where torch is installed as a transitive dep. No real model is loaded; the engine
__init__ is monkeypatched to skip model construction, and SparseMasksInferenceSlicer
is replaced with a spy via sys.modules injection.

This test pins the wiring added in 05-08 so it cannot silently regress. It would
have failed against the pre-fix source where thread_workers was hardcoded to 1 and
empty_slice_removal_threshold was hardcoded to 0.95 regardless of the request.
"""

from __future__ import annotations

import sys
import types
from threading import Lock
from unittest.mock import MagicMock

import numpy as np
import pytest


def _inject_fake_supervision(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a minimal fake supervision package satisfying sahi_slicer's imports.

    sahi_slicer.py imports supervision at module level and pulls
    supervision.detection.tools.inference_slicer helpers. We satisfy those
    imports with lightweight stubs so the module can be loaded without the
    real package.
    """
    sv = types.ModuleType("supervision")

    class FakeOverlapFilter:
        NON_MAX_SUPPRESSION = "nms"

    sv.OverlapFilter = FakeOverlapFilter  # type: ignore[attr-defined]

    class FakeDetections:
        def __init__(self, xyxy: object = None, confidence: object = None, class_id: object = None) -> None:
            self.class_id = np.array([1], dtype=np.int32) if class_id is None else class_id
            self.confidence = np.array([0.9], dtype=np.float32) if confidence is None else confidence
            self.xyxy = np.array([[0, 0, 4, 4]], dtype=np.float32) if xyxy is None else xyxy
            self.data: dict = {"sparse_masks": []}

    sv.Detections = FakeDetections  # type: ignore[attr-defined]

    sv_detection = types.ModuleType("supervision.detection")
    sv.detection = sv_detection  # type: ignore[attr-defined]

    sv_detection_tools = types.ModuleType("supervision.detection.tools")
    sv_detection.tools = sv_detection_tools  # type: ignore[attr-defined]

    sv_inference_slicer_mod = types.ModuleType("supervision.detection.tools.inference_slicer")

    class FakeInferenceSlicer:
        pass

    sv_inference_slicer_mod.InferenceSlicer = FakeInferenceSlicer  # type: ignore[attr-defined]
    sv_inference_slicer_mod.crop_image = lambda image, xyxy: image  # type: ignore[attr-defined]
    sv_inference_slicer_mod.move_detections = lambda d, o, r: d  # type: ignore[attr-defined]
    sv_detection_tools.inference_slicer = sv_inference_slicer_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "supervision", sv)
    monkeypatch.setitem(sys.modules, "supervision.detection", sv_detection)
    monkeypatch.setitem(sys.modules, "supervision.detection.tools", sv_detection_tools)
    monkeypatch.setitem(sys.modules, "supervision.detection.tools.inference_slicer", sv_inference_slicer_mod)
    return sv


def _make_request(thread_workers: int, empty_slice_removal_threshold: float) -> object:
    """Build an InferenceRequest with slicing enabled and the given wiring values."""
    from tls2dseg.types import InferenceRequest

    return InferenceRequest(
        text_prompt="tree.",
        box_threshold=0.30,
        text_threshold=0.30,
        slicing_enabled=True,
        slice_width_height=(64, 64),
        overlap_width_height=(8, 8),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.30,
        partial_detection_edge_touching_threshold=5,
        thread_workers=thread_workers,
        empty_slice_removal_threshold=empty_slice_removal_threshold,
    )


def _patch_engine(monkeypatch: pytest.MonkeyPatch, fake_sv: types.ModuleType, spy_slicer_cls: type) -> object:
    """Prepare the engine with a monkeypatched __init__ and SpySlicer.

    detect() does a local import:
      from tls2dseg.engines.inference.sahi_slicer import SparseMasksInferenceSlicer
    We intercept it by placing our spy into the cached sahi_slicer module
    (after ensuring sahi_slicer has been imported with fake supervision in sys.modules).
    """
    # Remove any cached sahi_slicer import so it re-imports with our fake supervision
    monkeypatch.delitem(sys.modules, "tls2dseg.engines.inference.sahi_slicer", raising=False)

    # Import sahi_slicer NOW (fake supervision is already in sys.modules from the caller)
    import tls2dseg.engines.inference.sahi_slicer as sahi_mod

    # Replace SparseMasksInferenceSlicer on the loaded module with our spy
    monkeypatch.setattr(sahi_mod, "SparseMasksInferenceSlicer", spy_slicer_cls)

    # Import engine module
    from tls2dseg.engines.inference import grounded_sam2 as eng_mod

    # Patch engine __init__ to skip heavy model loading
    def _fake_init(self: object, **kwargs: object) -> None:
        self._device = "cpu"  # type: ignore[attr-defined]
        self._sam_lock = Lock()  # type: ignore[attr-defined]
        self._gdino_processor = MagicMock()  # type: ignore[attr-defined]
        self._gdino_model = MagicMock()  # type: ignore[attr-defined]
        self._sam2_predictor = MagicMock()  # type: ignore[attr-defined]
        self._sam_box_prompt_batch_size = 4  # type: ignore[attr-defined]

    monkeypatch.setattr(eng_mod.GroundedSAM2Engine, "__init__", _fake_init)
    monkeypatch.setattr(eng_mod, "split_class_keys", lambda _: ["tree"])

    engine = eng_mod.GroundedSAM2Engine(  # type: ignore[call-arg]
        object_detection_model_id="fake",
        sam2_checkpoint="fake.pt",
        sam2_model_config="fake.yaml",
        sam_box_prompt_batch_size=4,
        device="cpu",
    )
    return engine


@pytest.mark.tier_b_light
def test_grounded_sam2_engine_forwards_thread_workers_to_slicer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GroundedSAM2Engine passes request.thread_workers to SparseMasksInferenceSlicer.

    Pre-fix this would have failed: thread_workers was hardcoded to 1 regardless
    of the request value.
    """
    fake_sv = _inject_fake_supervision(monkeypatch)

    slicer_kwargs: dict = {}

    class SpySlicer:
        def __init__(self, **kwargs: object) -> None:
            slicer_kwargs.update(kwargs)

        def __call__(self, image: np.ndarray) -> object:
            return fake_sv.Detections()  # type: ignore[attr-defined]

    engine = _patch_engine(monkeypatch, fake_sv, SpySlicer)

    N = 7  # != 1 (pre-fix hardcode), != 12 (runtime default)
    request = _make_request(thread_workers=N, empty_slice_removal_threshold=0.95)
    image = np.zeros((64, 64), dtype=np.float32)

    engine.detect(image, request=request)  # type: ignore[arg-type]

    assert "thread_workers" in slicer_kwargs, "SparseMasksInferenceSlicer was not constructed with thread_workers"
    assert slicer_kwargs["thread_workers"] == N, (
        f"Expected thread_workers={N} (from request), got {slicer_kwargs['thread_workers']}. "
        "Pre-fix this would have been 1 (hardcoded)."
    )


@pytest.mark.tier_b_light
def test_grounded_sam2_engine_forwards_empty_slice_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GroundedSAM2Engine passes request.empty_slice_removal_threshold to the callback.

    The threshold is passed via slice_inference_parameters into the partial callback;
    we capture it from the callback partial's keywords when the slicer is constructed.

    Pre-fix this would have been 0.95 hardcoded regardless of the request.
    """
    fake_sv = _inject_fake_supervision(monkeypatch)

    captured_threshold: list[float] = []

    class SpySlicerThreshold:
        def __init__(self, callback: object = None, **kwargs: object) -> None:
            if hasattr(callback, "keywords"):
                params = callback.keywords.get("slice_inference_parameters", {})  # type: ignore[union-attr]
                t = params.get("empty_slice_removal_threshold")
                if t is not None:
                    captured_threshold.append(float(t))

        def __call__(self, image: np.ndarray) -> object:
            return fake_sv.Detections()  # type: ignore[attr-defined]

    engine = _patch_engine(monkeypatch, fake_sv, SpySlicerThreshold)

    THRESH = 0.75  # != 0.95 (pre-fix hardcode)
    request = _make_request(thread_workers=4, empty_slice_removal_threshold=THRESH)
    image = np.zeros((64, 64), dtype=np.float32)

    engine.detect(image, request=request)  # type: ignore[arg-type]

    assert len(captured_threshold) == 1, (
        "empty_slice_removal_threshold was not captured from slice_inference_parameters"
    )
    assert captured_threshold[0] == THRESH, (
        f"Expected empty_slice_removal_threshold={THRESH}, got {captured_threshold[0]}. "
        "Pre-fix this would have been 0.95 hardcoded."
    )

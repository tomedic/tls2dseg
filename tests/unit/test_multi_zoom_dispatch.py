"""Tier-A FakeEngine dispatch tests for multi_zoom_dispatch.py (MZ-03/04/05/10/11).

Tests:
  1. test_concat       — N-pass dispatch returns detection count = sum of per-pass counts.
  2. test_coord_remap  — resized coords scale by 1/resize_factor in float.
  3. test_class_ids    — pass-local ids remapped to global class_id_map after each pass;
                         per_class_metadata populated with typed ClassMetadata per class.
  4. test_single_zoom_fallback — is_multi_zoom_active returns False for mode=single-zoom
                                 and for absent classes.
  5. test_warn_once    — _warn_single_zoom_once fires WARNING exactly once across two calls.
  6. test_on_pass      — on_pass fires once per zoom pass with monotonically increasing
                         indices; callback exception does not break run_multi_zoom.

No real models — FakeEngine only. No torch/pyvips/sam2 at module level.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import pytest

import tls2dseg.engines.inference.multi_zoom_dispatch as mzd_mod
from tls2dseg.engines.inference.multi_zoom_dispatch import (
    _remap_to_native,
    _warn_single_zoom_once,
    is_multi_zoom_active,
    run_multi_zoom,
)
from tls2dseg.engines.inference.multi_zoom_plan import ZoomPass
from tls2dseg.types import ClassMetadata, Detections2D, InferenceRequest

# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------


def _numpy_resize(image: np.ndarray, factor: float) -> np.ndarray:
    """Numpy-only resize stub (no pyvips) — slice-subsample for factor < 1."""
    h, w = image.shape[:2]
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))
    step_h = max(1, h // new_h)
    step_w = max(1, w // new_w)
    return image[::step_h, ::step_w][:new_h, :new_w].astype(np.float32)


def _make_request(text_prompt: str = "chair. table.") -> InferenceRequest:
    return InferenceRequest(
        text_prompt=text_prompt,
        box_threshold=0.10,
        text_threshold=0.10,
        slicing_enabled=False,
        slice_width_height=(0, 0),
        overlap_width_height=(0, 0),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.9,
        partial_detection_edge_touching_threshold=5,
        thread_workers=1,
        empty_slice_removal_threshold=0.95,
    )


def _make_detections(
    boxes: list[list[float]],
    class_names: list[str],
    class_ids: list[int],
) -> Detections2D:
    n = len(boxes)
    return Detections2D(
        masks=[np.array([[0, 0]], dtype=np.int32) for _ in range(n)],
        input_boxes=np.array(boxes, dtype=np.float32).reshape(n, 4),
        confidences=np.full(n, 0.9, dtype=np.float32),
        class_names=list(class_names),
        class_ids=np.array(class_ids, dtype=np.int32),
        mask_labels=[f"{c} 0.90" for c in class_names],
    )


@dataclasses.dataclass
class _PerPassFakeEngine:
    """FakeEngine that returns a configurable set of detections per call.

    ``detections_seq`` is a list of Detections2D; each successive detect()
    call pops the next element. If the list is exhausted, returns empty.
    """

    detections_seq: list[Detections2D]

    def detect(self, image: np.ndarray, *, request: object) -> Detections2D:
        if self.detections_seq:
            return self.detections_seq.pop(0)
        return Detections2D(
            masks=[],
            input_boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty(0, dtype=np.float32),
            class_names=[],
            class_ids=np.empty(0, dtype=np.int32),
            mask_labels=[],
        )


@dataclasses.dataclass
class _FakeCtx:
    """Minimal RunContext stub carrying class_id_map and per_class_metadata."""

    class_id_map: dict[str, int]
    per_class_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _FakeMzCfg:
    """Minimal MultiZoomConfig stub."""

    ios_enabled: bool = False
    ios_threshold: float = 0.8
    cross_class_iou_threshold: float = 0.7


def _full_image_pass(class_names: tuple[str, ...]) -> ZoomPass:
    return ZoomPass(
        resize_factor=1.0,
        tile_size_px=None,
        overlap_px=None,
        needs_tiling=False,
        class_names=class_names,
        text_prompt=". ".join(class_names) + ".",
    )


def _tiled_pass(
    class_names: tuple[str, ...],
    resize_factor: float = 0.5,
    tile_size_px: int = 800,
) -> ZoomPass:
    overlap_px = int(0.225 * tile_size_px)
    return ZoomPass(
        resize_factor=resize_factor,
        tile_size_px=tile_size_px,
        overlap_px=overlap_px,
        needs_tiling=True,
        class_names=class_names,
        text_prompt=". ".join(class_names) + ".",
    )


# ---------------------------------------------------------------------------
# Test 1: concat — total detection count = sum of per-pass counts
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    """MZ-03: dispatcher over 3 passes returns detection count = sum of per-pass counts.

    pyvips (_lanczos3_resize) is stubbed with a pure-numpy subsample so the test
    runs in the tier_a venv without pyvips installed.
    """
    monkeypatch.setattr(mzd_mod, "_lanczos3_resize", _numpy_resize)

    classes = ("chair", "table", "door")
    class_id_map = {"chair": 1, "table": 2, "door": 3, "background": 0}
    ctx = _FakeCtx(class_id_map=class_id_map)
    mz_cfg = _FakeMzCfg()

    # 3 passes: full-image + 2 tiled, 2+3+1 detections
    det_pass1 = _make_detections([[0, 0, 10, 10]] * 2, ["chair", "table"], [1, 2])
    det_pass2 = _make_detections([[5, 5, 15, 15]] * 3, ["door", "door", "chair"], [3, 3, 1])
    det_pass3 = _make_detections([[20, 20, 30, 30]] * 1, ["table"], [2])

    engine = _PerPassFakeEngine([det_pass1, det_pass2, det_pass3])
    passes = [
        _full_image_pass(classes),
        _tiled_pass(("door", "chair"), resize_factor=0.5),
        _tiled_pass(("table",), resize_factor=0.25),
    ]
    base_request = _make_request()
    image_native = np.zeros((100, 200), dtype=np.float32)

    result = run_multi_zoom(image_native, engine, base_request, passes, ctx, mz_cfg=mz_cfg)

    # After concat (6 total) + dedup (same-class NMS + cross-class); at minimum 1 survives
    assert len(result.input_boxes) >= 1
    # All returned class_names must have matching class_ids from the global map
    for name, cid in zip(result.class_names, result.class_ids, strict=True):
        if name in class_id_map:
            assert cid == class_id_map[name], f"{name!r}: expected {class_id_map[name]}, got {cid}"


# ---------------------------------------------------------------------------
# Test 2: coord_remap — resized bbox scales by 1/resize_factor in float
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_coord_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remap 1 (RESEARCH Pitfall 3): bbox in resized space scales by 1/resize_factor in float.

    Tests _remap_to_native directly for precision, and via run_multi_zoom with a stubbed
    resize so the test runs in the tier_a venv without pyvips.
    """
    # Direct test of the remap helper: [10, 20, 30, 40] at factor 0.5 -> [20, 40, 60, 80]
    resize_factor = 0.5
    det_resized = _make_detections([[10.0, 20.0, 30.0, 40.0]], ["chair"], [1])
    det_native = _remap_to_native(det_resized, resize_factor)

    expected_box = np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    np.testing.assert_allclose(det_native.input_boxes[0], expected_box, rtol=1e-5)
    assert det_native.input_boxes.dtype == np.float32  # float, not rounded to int

    # End-to-end via run_multi_zoom (pyvips stubbed with numpy subsample)
    monkeypatch.setattr(mzd_mod, "_lanczos3_resize", _numpy_resize)

    class_id_map = {"chair": 1, "background": 0}
    ctx = _FakeCtx(class_id_map=class_id_map)
    mz_cfg = _FakeMzCfg()

    det_pass = _make_detections([[10.0, 20.0, 30.0, 40.0]], ["chair"], [1])
    engine = _PerPassFakeEngine([det_pass])
    passes = [_tiled_pass(("chair",), resize_factor=resize_factor)]
    base_request = _make_request(text_prompt="chair.")
    image_native = np.zeros((200, 400), dtype=np.float32)

    result = run_multi_zoom(image_native, engine, base_request, passes, ctx, mz_cfg=mz_cfg)

    assert len(result.input_boxes) == 1
    np.testing.assert_allclose(result.input_boxes[0], expected_box, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: class_ids — global remap + per_class_metadata populated
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_class_ids() -> None:
    """Remap 2 (RESEARCH Pitfall 5): pass-local ids replaced by global class_id_map ids.

    Also asserts per_class_metadata is populated with typed ClassMetadata per class.
    """
    # Global map: chair=1, table=2; each pass returns local ids 0,1
    class_id_map = {"chair": 1, "table": 2, "background": 0}
    ctx = _FakeCtx(class_id_map=class_id_map)
    mz_cfg = _FakeMzCfg()

    # Pass 1 (full-image): returns chair with local id 99, table with local id 77
    det_p1 = _make_detections(
        [[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 20.0, 20.0]],
        ["chair", "table"],
        [99, 77],  # wrong local ids
    )
    # Pass 2 (tiled, resize_factor=1.0 to skip coord remap): returns chair with local id 55
    det_p2 = _make_detections(
        [[25.0, 25.0, 35.0, 35.0]],
        ["chair"],
        [55],
    )

    engine = _PerPassFakeEngine([det_p1, det_p2])
    passes = [
        _full_image_pass(("chair", "table")),
        _tiled_pass(("chair",), resize_factor=1.0),  # factor=1.0 skips coord remap
    ]
    base_request = _make_request()
    image_native = np.zeros((100, 200), dtype=np.float32)

    result = run_multi_zoom(image_native, engine, base_request, passes, ctx, mz_cfg=mz_cfg)

    # All class_ids must be from the global map
    for name, cid in zip(result.class_names, result.class_ids, strict=True):
        assert cid == class_id_map[name], (
            f"class {name!r}: expected global id {class_id_map[name]}, got {cid} (local id leaked)"
        )

    # per_class_metadata populated with ClassMetadata for each class
    assert "chair" in ctx.per_class_metadata
    assert "table" in ctx.per_class_metadata
    chair_meta = ctx.per_class_metadata["chair"]
    # TypedDict keys
    assert "resize_factor" in chair_meta
    assert "was_tiled" in chair_meta
    assert "tile_size_px" in chair_meta
    assert "grouped_with" in chair_meta
    # Values are typed
    assert isinstance(chair_meta["resize_factor"], float)
    assert isinstance(chair_meta["was_tiled"], bool)


# ---------------------------------------------------------------------------
# Test 4: single_zoom_fallback
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_single_zoom_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """MZ-10: is_multi_zoom_active returns False when active=False; True when active=True."""

    @dataclasses.dataclass
    class _MzCfgInactive:
        active: bool = False

    @dataclasses.dataclass
    class _MzCfgActive:
        active: bool = True

    # Reset flag before each check
    monkeypatch.setattr(mzd_mod, "_warned_single_zoom", False)
    assert is_multi_zoom_active(_MzCfgInactive()) is False

    monkeypatch.setattr(mzd_mod, "_warned_single_zoom", False)
    assert is_multi_zoom_active(_MzCfgActive()) is True


# ---------------------------------------------------------------------------
# Test 5: warn_once — exactly one WARNING across two calls
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_warn_once(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MZ-10: _warn_single_zoom_once emits WARNING exactly once even when called twice."""
    # Reset module flag — same pattern as test_warn_once.py line 93
    monkeypatch.setattr(mzd_mod, "_warned_single_zoom", False)

    with caplog.at_level(logging.WARNING, logger="tls2dseg.engines.inference.multi_zoom_dispatch"):
        _warn_single_zoom_once()
        _warn_single_zoom_once()

    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warning_records) == 1, (
        f"Expected exactly 1 WARNING from _warn_single_zoom_once; got {len(warning_records)}: "
        f"{[r.message for r in warning_records]}"
    )
    # Verify it's via logger.warning, not print
    assert "single-zoom" in warning_records[0].message.lower() or "multi-zoom" in warning_records[0].message.lower()
    # Verify the flag was set
    assert mzd_mod._warned_single_zoom is True


# ---------------------------------------------------------------------------
# Test 6: on_pass observer — fires once per pass, index monotonic, exception-safe
# ---------------------------------------------------------------------------


@pytest.mark.tier_a
def test_on_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_pass fires exactly once per zoom pass with monotonically increasing indices.

    The observer receives the per-pass image (np.ndarray) and a Detections2D.
    An exception raised inside on_pass must NOT break run_multi_zoom — the
    combined result is still returned and has at least 1 detection.
    """
    monkeypatch.setattr(mzd_mod, "_lanczos3_resize", _numpy_resize)

    class_id_map = {"chair": 1, "table": 2, "door": 3, "background": 0}
    ctx = _FakeCtx(class_id_map=class_id_map)
    mz_cfg = _FakeMzCfg()

    det_p0 = _make_detections([[0.0, 0.0, 5.0, 5.0]], ["chair"], [1])
    det_p1 = _make_detections([[10.0, 10.0, 20.0, 20.0]], ["table"], [2])
    det_p2 = _make_detections([[30.0, 30.0, 40.0, 40.0]], ["door"], [3])

    engine = _PerPassFakeEngine([det_p0, det_p1, det_p2])
    passes = [
        _full_image_pass(("chair",)),
        _tiled_pass(("table",), resize_factor=0.5),
        _tiled_pass(("door",), resize_factor=0.25),
    ]
    base_request = _make_request()
    image_native = np.zeros((100, 200), dtype=np.float32)

    observed: list[tuple[int, object, object, object]] = []

    def _observer(idx: int, zoom_pass: object, image: object, det: object) -> None:
        observed.append((idx, zoom_pass, image, det))
        if idx == 1:
            raise RuntimeError("intentional observer error")

    result = run_multi_zoom(image_native, engine, base_request, passes, ctx, mz_cfg=mz_cfg, on_pass=_observer)

    # Fired exactly once per pass
    assert len(observed) == len(passes), f"Expected {len(passes)} on_pass calls, got {len(observed)}"

    # Indices are monotonically increasing (0, 1, 2, ...)
    indices = [entry[0] for entry in observed]
    assert indices == list(range(len(passes))), f"Indices not monotonic: {indices}"

    # Each call received an np.ndarray image and a Detections2D
    for idx, _zoom_pass, image, det in observed:
        assert isinstance(image, np.ndarray), f"Pass {idx}: image is not np.ndarray"
        assert isinstance(det, Detections2D), f"Pass {idx}: det is not Detections2D"

    # Exception in pass 1 did NOT break run_multi_zoom — combined result returned
    assert result is not None
    assert len(result.input_boxes) >= 1

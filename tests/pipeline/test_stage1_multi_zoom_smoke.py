"""End-to-end multi-zoom FakeEngine smoke — tier_b_light.

Drives the real run_stage1 path in multi-zoom mode using FakeEngines so
no real model or GPU is needed.  Heavy pchandler operations (load_e57,
toggle_socs2prcs, etc.) are monkeypatched with minimal stubs; the multi-zoom
dispatch (is_multi_zoom_active → compute_zoom_passes → run_multi_zoom) runs
for real.

Two scenarios:
- Multi-zoom config: confirms the dispatcher fires and ctx.per_class_metadata
  is populated after a complete stage-1 run.
- Single-zoom config: confirms the legacy detect() path fires and the
  warn-once fallback message is emitted exactly once per process.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_N_SCANS = 1


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic point-cloud stub (no pchandler)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeScalarField:
    """Minimal scalar-field wrapper matching pchandler's .data attribute."""

    def __init__(self, arr: np.ndarray) -> None:
        self.data = arr


class _FakePcd:
    """Duck-typed pcd stub with the attributes stage1 touches."""

    def __init__(self, n: int = 20, rng_seed: int = 42) -> None:
        rng = np.random.RandomState(rng_seed)
        x = rng.uniform(0.5, 5.0, n).astype(np.float32)
        y = rng.uniform(-0.3, 0.3, n).astype(np.float32)
        z = rng.uniform(0.1, 0.5, n).astype(np.float32)
        self.xyz = np.column_stack([x, y, z])
        sph = np.zeros((n, 3), dtype=np.float32)
        sph[:, 0] = np.sqrt(x**2 + y**2 + z**2)  # range
        sph[:, 1] = np.arctan2(np.sqrt(x**2 + y**2), z)
        sph[:, 2] = -np.arctan2(y, x)
        self.spherical_coordinates = sph
        instances = np.ones(n, dtype=np.int32)
        classes = np.ones(n, dtype=np.int32)
        self.scalar_fields: dict = {
            "instances": _FakeScalarField(instances),
            "classes": _FakeScalarField(classes),
        }

    def copy(self) -> _FakePcd:
        new = _FakePcd.__new__(_FakePcd)
        new.xyz = self.xyz.copy()
        new.spherical_coordinates = self.spherical_coordinates.copy()
        new.scalar_fields = {k: _FakeScalarField(v.data.copy()) for k, v in self.scalar_fields.items()}
        return new


# ─────────────────────────────────────────────────────────────────────────────
# Fake Detections3D stub for clean_pcd_instances_and_get_detections3d return
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _FakeD3D:
    pcd_ids: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Config + context builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_cfg_ctx(tmp_path: Path, *, multi_zoom: bool) -> tuple:
    """Return (RunConfig, RunContext) for a single-scan multi-view run."""
    from tls2dseg.config import RunConfig
    from tls2dseg.runtime import build_context
    from tls2dseg.runtime.capability import Runtime

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "scan_0.e57").touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base_cfg: dict = {
        "mode": "multi-view",
        "io": {
            "input_path": str(input_dir),
            "output_dir": str(output_dir),
            "resume_from_checkpoint": False,
            "save_intermediate": False,
        },
        "prompt": {"text": "tree. pole."},
        "preprocessing": {"output_resolution_m": 0.05},
        "projection": {"features": ["intensity"]},
        "inference": {
            "type": "grounded_sam2",
            "sam2_checkpoint": "/tmp/fake_sam2.pt",
        },
        "d3d_extraction": {},
        "fusion": {},
        "runtime": {"device": "cpu"},
    }
    if multi_zoom:
        base_cfg["prompt"]["sizes_m"] = [5.0, 0.1]
        base_cfg["inference"]["multi_zoom"] = {"active": True}
    else:
        base_cfg["inference"]["multi_zoom"] = {"active": False}

    cfg_dict = base_cfg
    cfg = RunConfig(**cfg_dict)

    runtime = Runtime(
        cuml_available=False,
        torch_cuda_available=False,
        sam2_available=False,
        libvips_available=False,
        numpy_version="1.26.0",
        torch_version="2.7.0",
        cuml_version=None,
    )
    ctx = build_context(cfg, runtime)
    return cfg, ctx


# ─────────────────────────────────────────────────────────────────────────────
# Multi-zoom-aware FakeInferenceEngine variant
# ─────────────────────────────────────────────────────────────────────────────


class _MZFakeEngine:
    """FakeInferenceEngine that returns per-pass class names from request.pass_class_names."""

    def detect(self, image: np.ndarray, *, request: object) -> object:
        from tls2dseg.types import Detections2D

        pass_names = list(getattr(request, "pass_class_names", ()) or ("tree",))
        n = len(pass_names)
        input_boxes = np.tile(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (n, 1))
        masks = [np.array([[0, 0]], dtype=np.int32) for _ in range(n)]
        confidences = np.full(n, 0.80, dtype=np.float32)
        class_ids = np.arange(1, n + 1, dtype=np.int32)
        mask_labels = [f"{name} 0.80" for name in pass_names]
        return Detections2D(
            masks=masks,
            input_boxes=input_boxes,
            confidences=confidences,
            class_names=pass_names,
            class_ids=class_ids,
            mask_labels=mask_labels,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────


def _patch_heavy_stage1(monkeypatch: pytest.MonkeyPatch, fake_pcd: _FakePcd) -> None:
    """Patch pchandler and pchandler-dependent calls in stage1 to no-ops / stubs.

    Because stage1 uses heavy imports *inside* the function body, we patch the
    source modules (pchandler.data_io, tls2dseg.preprocessing.cleanup, etc.)
    so the function-level ``from X import Y`` picks up the patched version.
    """
    # Patch pchandler.data_io.load_e57 at the source so stage1's
    # ``from pchandler.data_io import load_e57`` picks it up.
    import pchandler.data_io as pchandler_data_io

    import tls2dseg.detections_3d as detections_3d
    import tls2dseg.engines.inference.shared as shared_mod
    import tls2dseg.lifting.masks_to_pcd as masks_to_pcd_mod
    import tls2dseg.preprocessing.cleanup as cleanup
    import tls2dseg.utils_main as utils_main

    monkeypatch.setattr(pchandler_data_io, "load_e57", lambda *a, **kw: fake_pcd)

    # pchandler.geometry.transforms.toggle_socs2prcs → return pcd unchanged
    import pchandler.geometry.transforms as pchandler_transforms

    monkeypatch.setattr(pchandler_transforms, "toggle_socs2prcs", lambda pcd: pcd)

    # filter_pcd_roi_range → no-op
    monkeypatch.setattr(cleanup, "filter_pcd_roi_range", lambda pcd, params: None)

    # subsample_pcd_to_output_resolution → return pcd unchanged
    monkeypatch.setattr(cleanup, "subsample_pcd_to_output_resolution", lambda pcd, params: pcd)

    # remove_unclassified_points → return pcd unchanged
    monkeypatch.setattr(cleanup, "remove_unclassified_points", lambda pcd, params: pcd)

    # remove_small_instances → return pcd unchanged
    monkeypatch.setattr(cleanup, "remove_small_instances", lambda pcd, min_pts: pcd)

    # apply_robust_sor_filter → no-op
    monkeypatch.setattr(cleanup, "apply_robust_sor_filter", lambda pcd, **kw: None)

    # assure_common_global_shift → return (pcd, zeros)
    monkeypatch.setattr(
        utils_main,
        "assure_common_global_shift",
        lambda pcd, shift, idx: (pcd, np.zeros(3, dtype=np.float64)),
    )

    # get_per_mask_depth_parallel → no-op
    monkeypatch.setattr(shared_mod, "get_per_mask_depth_parallel", lambda results, scan_images, n_jobs: None)

    # get_instance_and_semantic_mask_with_confidence → return stub zero masks
    def _fake_get_masks(results: dict, text_prompt: str, image_hw: tuple) -> tuple:
        h, w = image_hw
        inst = np.zeros((h, w), dtype=np.int32)
        sem = np.zeros((h, w), dtype=np.int32)
        conf = np.zeros((h, w), dtype=np.float32)
        return inst, sem, conf, {}

    monkeypatch.setattr(shared_mod, "get_instance_and_semantic_mask_with_confidence", _fake_get_masks)

    # lift_masks_to_pcd, lift_mask_to_pcd → no-op
    monkeypatch.setattr(masks_to_pcd_mod, "lift_masks_to_pcd", lambda pcd, inst, sem: None)
    monkeypatch.setattr(masks_to_pcd_mod, "lift_mask_to_pcd", lambda pcd, mask, mask_name: None)

    # clean_pcd_instances_and_get_detections3d → return stub D3D + same pcd
    monkeypatch.setattr(
        detections_3d,
        "clean_pcd_instances_and_get_detections3d",
        lambda pcd, pcd_id, d3d_params, pcp_params: (_FakeD3D(pcd_ids=np.zeros(1, dtype=np.int32)), pcd),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_run_stage1_multi_zoom_dispatches_and_populates_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_stage1 in multi-zoom mode calls run_multi_zoom and populates ctx.per_class_metadata.

    Wires real run_stage1 with a multi-zoom config (tree:5m / pole:0.1m),
    FakeProjectionEngine (d_azim_rad=1e-3), and _MZFakeEngine (per-pass class
    names from request.pass_class_names).  Heavy pchandler calls are stubbed.

    Asserts:
    - run_stage1 completes without error.
    - ctx.per_class_metadata is populated for at least one class.
    - The Stage1Result contains at least one d3d entry.
    - No real GroundedSAM2Engine is instantiated.
    """
    from tests.unit.fakes import FakeProjectionEngine
    from tls2dseg.pipeline.stage1 import run_stage1
    from tls2dseg.types import InferenceRequest

    cfg, ctx = _make_cfg_ctx(tmp_path, multi_zoom=True)
    fake_pcd = _FakePcd()
    _patch_heavy_stage1(monkeypatch, fake_pcd)

    proj_engine = FakeProjectionEngine(image_size=(8, 8))
    inf_engine = _MZFakeEngine()
    inference_request = InferenceRequest(
        text_prompt="tree. pole.",
        box_threshold=0.1,
        text_threshold=0.1,
        slicing_enabled=False,
        slice_width_height=(0, 0),
        overlap_width_height=(0, 0),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.9,
        partial_detection_edge_touching_threshold=5,
        thread_workers=1,
        empty_slice_removal_threshold=0.0,
    )

    # Reset the warn-once flag so the multi-zoom branch doesn't fire the warning
    import tls2dseg.engines.inference.multi_zoom_dispatch as _mzd

    monkeypatch.setattr(_mzd, "_warned_single_zoom", False)

    result = run_stage1(cfg, ctx, proj_engine, inf_engine, inference_request)

    assert result is not None, "run_stage1 must return a Stage1Result"
    assert len(result.d3d_collection) >= 1, "multi-view run must produce at least one d3d entry"
    assert len(ctx.per_class_metadata) >= 1, (
        "run_multi_zoom must populate ctx.per_class_metadata for at least one class"
    )
    assert "GroundedSAM2Engine" not in dir(sys.modules[__name__]), (
        "D-D-06: no real engine may be imported in tier_b_light tests"
    )


@pytest.mark.tier_b_light
def test_run_stage1_single_zoom_fallback_and_warn_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """run_stage1 in single-zoom mode uses the legacy detect() path and fires warn-once.

    Wires real run_stage1 with mode=single-zoom config and a plain
    FakeInferenceEngine (one detection per call).  Asserts:
    - run_stage1 completes without error.
    - ctx.per_class_metadata is NOT populated (single-zoom never calls dispatcher).
    - Exactly one WARNING containing 'single-zoom fallback' appears in the log.
    """
    import logging

    from tests.unit.fakes import FakeInferenceEngine, FakeProjectionEngine
    from tls2dseg.pipeline.stage1 import run_stage1
    from tls2dseg.types import InferenceRequest

    cfg, ctx = _make_cfg_ctx(tmp_path, multi_zoom=False)
    fake_pcd = _FakePcd()
    _patch_heavy_stage1(monkeypatch, fake_pcd)

    proj_engine = FakeProjectionEngine(image_size=(8, 8))
    inf_engine = FakeInferenceEngine(n_detections=1)
    inference_request = InferenceRequest(
        text_prompt="tree. pole.",
        box_threshold=0.1,
        text_threshold=0.1,
        slicing_enabled=False,
        slice_width_height=(0, 0),
        overlap_width_height=(0, 0),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.9,
        partial_detection_edge_touching_threshold=5,
        thread_workers=1,
        empty_slice_removal_threshold=0.0,
    )

    # Reset warn-once flag so warn fires fresh in this test
    import tls2dseg.engines.inference.multi_zoom_dispatch as _mzd

    monkeypatch.setattr(_mzd, "_warned_single_zoom", False)

    with caplog.at_level(logging.WARNING, logger="tls2dseg.engines.inference.multi_zoom_dispatch"):
        result = run_stage1(cfg, ctx, proj_engine, inf_engine, inference_request)

    assert result is not None, "run_stage1 must return a Stage1Result in single-zoom mode"
    assert len(result.d3d_collection) >= 1, "single-zoom run must produce at least one d3d entry"
    assert len(ctx.per_class_metadata) == 0, "single-zoom fallback must NOT populate ctx.per_class_metadata"

    warn_records = [r for r in caplog.records if "single-zoom" in r.message and r.levelno == logging.WARNING]
    assert len(warn_records) == 1, (
        f"warn-once must fire exactly one WARNING; got {len(warn_records)}: {[r.message for r in warn_records]}"
    )


@pytest.mark.tier_b_light
def test_multi_zoom_smoke_no_real_engine_imported() -> None:
    """No real GroundedSAM2Engine is imported in this test module (D-D-06 guard)."""
    this_module = sys.modules[__name__]
    forbidden = {"GroundedSAM2Engine", "GroundedSAM2HFEngine"}
    present = forbidden & set(dir(this_module))
    assert not present, (
        f"D-D-06 binding rule violated: real engine class(es) {present} found in test_multi_zoom_smoke module namespace"
    )

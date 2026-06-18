"""Real run_stage1-driven single-view resume test — fake engines, no real model.

Phase 05 plan 05 Task 3 (ORC-03, MODE-04, TEST-12, TEST-15).

Drives the REAL ``run_stage1`` function (no monkeypatching of stage1 or
run_stage1) with pre-seeded checkpoint pkl files and FakeEngines to verify that:

1. Full single-view resume: with n_features==2 and all checkpoint pkls present,
   ``run_stage1`` returns early without calling inference (zero detect calls).
   This test would FAIL against pre-fix code (CR-01 broke have_processed_all for
   n_features>1 in single-view mode).

2. Corrupt pkl: a corrupt checkpoint forces recompute of that scan only; the
   resulting collection has no None slot for the recomputed scan.

Marked ``tier_b_light``: driving the real resume path requires pchandler
(``PointCloudData`` / ``Detections3D`` for pkl round-trip) and the real per-scan
loop. ``tier_a`` structurally cannot cover this — the existing checkpoint smoke
test monkeypatches ``Pipeline.stage1`` wholesale, bypassing the real index
arithmetic and ``load_previously_saved_inference_results_if_any``.
"""

from __future__ import annotations

import pickle
import types as _types
from pathlib import Path

import numpy as np
import pytest
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_N_SCANS = 2


def _make_cfg_ctx(tmp_path: Path) -> tuple:
    """Build a minimal single-view RunConfig + RunContext pointing at tmp_path."""
    from tls2dseg.config import RunConfig
    from tls2dseg.runtime import build_context
    from tls2dseg.runtime.capability import Runtime

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(_N_SCANS):
        (input_dir / f"scan_{i}.e57").touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg_dict = yaml.safe_load(f"""\
mode: single-view
io:
  input_path: {input_dir}
  output_dir: {output_dir}
  resume_from_checkpoint: true
  save_intermediate: false
prompt:
  text: fake_object
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity, range]
inference:
  type: grounded_sam2
  sam2_checkpoint: /tmp/fake_sam2.pt
d3d_extraction: {{}}
fusion: {{}}
runtime:
  device: cpu
""")
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


def _make_inference_request() -> object:
    """Build a minimal InferenceRequest matching the smoke test's config."""
    from tls2dseg.types import InferenceRequest

    return InferenceRequest(
        text_prompt="fake_object",
        box_threshold=0.3,
        text_threshold=0.3,
        slicing_enabled=False,
        slice_width_height=(320, 320),
        overlap_width_height=(20, 20),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.8,
        partial_detection_edge_touching_threshold=10,
    )


def _make_minimal_pcd() -> object:
    """Return a minimal PointCloudData suitable for pkl round-trip."""
    from pchandler.geometry import PointCloudData

    xyz = np.zeros((5, 3), dtype=np.float32)
    pcd = PointCloudData(xyz=xyz)
    return pcd


def _make_minimal_d3d(pcd_id: int) -> object:
    """Return a minimal Detections3D for pkl round-trip."""
    from tls2dseg.detections_3d import Detections3D

    return Detections3D(
        pcd_ids=np.array([pcd_id], dtype=np.int32),
        instances=np.array([1], dtype=np.int32),
        classes=np.array([1], dtype=np.int32),
        confidences=np.array([0.9], dtype=np.float32),
        point_counts=np.array([5], dtype=np.int32),
        centroids=np.zeros((1, 3), dtype=np.float32),
        bboxes=np.zeros((1, 6), dtype=np.float32),
        bboxes_type="aabb",
        centroid_type="mean",
        preprocessing_applied=False,
    )


def _seed_checkpoints(stage1_dir: Path, n_scans: int, n_features: int) -> None:
    """Write valid pkl pairs for all scans using single-view base id convention."""
    stage1_dir.mkdir(parents=True, exist_ok=True)
    for scan_idx in range(1, n_scans + 1):
        # single-view base id = (scan_idx - 1) * n_features + 1
        base_id = (scan_idx - 1) * n_features + 1
        pcd = _make_minimal_pcd()
        d3d = _make_minimal_d3d(pcd_id=base_id - 1)
        with open(stage1_dir / f"pcd_ij_{base_id}.pkl", "wb") as f:
            pickle.dump(pcd, f)
        with open(stage1_dir / f"d3d_ij_{base_id}.pkl", "wb") as f:
            pickle.dump(d3d, f)


# ─────────────────────────────────────────────────────────────────────────────
# Spy wrapper for FakeInferenceEngine
# ─────────────────────────────────────────────────────────────────────────────


class _SpyInferenceEngine:
    """Wraps FakeInferenceEngine and counts detect() calls."""

    def __init__(self) -> None:
        from tests.unit.fakes import FakeInferenceEngine

        self._inner = FakeInferenceEngine(n_detections=1)
        self.detect_call_count = 0

    def detect(self, image: np.ndarray, *, request: object) -> object:
        self.detect_call_count += 1
        return self._inner.detect(image, request=request)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_single_view_full_resume_skips_inference(tmp_path: Path) -> None:
    """Single-view full-resume with n_features==2 fires zero detect() calls.

    Pre-seeds stage1_dir with valid pkl pairs at base ids (1, 3) for 2 scans
    with n_features==2 (i.e. base_id = (scan-1)*2+1).  After CR-01 fix,
    have_processed_all == True for single-view compares len(pcd_map)==n_scans,
    so the full-resume early-return fires.

    This test WOULD FAIL against pre-fix code: the old have_processed_all
    compared len(pcd_map)==n_scans*n_features (2==4 is False), so the
    early-return never triggered and every scan ran full inference.
    """
    from tests.unit.fakes import FakeProjectionEngine

    cfg, ctx = _make_cfg_ctx(tmp_path)
    n_features = len(cfg.projection.features)  # 2

    _seed_checkpoints(ctx.stage1_dir, _N_SCANS, n_features)

    spy_engine = _SpyInferenceEngine()
    proj_engine = FakeProjectionEngine(image_size=(4, 4))
    request = _make_inference_request()

    from tls2dseg.pipeline.stage1 import run_stage1

    result = run_stage1(cfg, ctx, proj_engine, spy_engine, request)

    assert spy_engine.detect_call_count == 0, (
        f"Full single-view resume must call detect() 0 times; got {spy_engine.detect_call_count}. "
        "CR-01 fix: have_processed_all must compare == n_scans in single-view mode."
    )
    assert result.n_scans == _N_SCANS
    assert len(result.d3d_collection) == _N_SCANS


@pytest.mark.tier_b_light
def test_single_view_corrupt_pkl_recomputes_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt pcd pkl for one scan triggers recompute of that scan only.

    Positive path: scan 1 has valid pkls — loaded from disk (no inference).
    Negative path: scan 2 has a corrupt pcd pkl — falls back to inference.

    After recompute, the collection has no None slot for scan 2.

    Monkeypatches ``load_e57``, the preprocessing chain, and the single-view
    processing chain so the recompute path completes without real .e57 files
    or GPU engines.
    """
    from tests.unit.fakes import FakeProjectionEngine

    cfg, ctx = _make_cfg_ctx(tmp_path)
    n_features = len(cfg.projection.features)  # 2

    # Seed valid checkpoints for scan 1 only
    _seed_checkpoints(ctx.stage1_dir, _N_SCANS, n_features)

    # Corrupt the pcd pkl for scan 2
    scan2_base_id = (2 - 1) * n_features + 1  # = 3
    corrupt_pcd_path = ctx.stage1_dir / f"pcd_ij_{scan2_base_id}.pkl"
    corrupt_pcd_path.write_bytes(b"THIS IS NOT VALID PICKLE DATA")

    # Monkeypatch load_e57 to return a minimal PointCloudData
    pcd_stub = _make_minimal_pcd()

    import pchandler.data_io as _pchandler_io

    monkeypatch.setattr(_pchandler_io, "load_e57", lambda *a, **kw: pcd_stub)

    # Monkeypatch the heavy preprocessing/inference chain so the recompute path
    # completes without real models or GPU:
    import tls2dseg.preprocessing.cleanup as _cleanup

    monkeypatch.setattr(_cleanup, "filter_pcd_roi_range", lambda *a, **kw: None)
    monkeypatch.setattr(_cleanup, "subsample_pcd_to_output_resolution", lambda pcd, *a, **kw: pcd)
    monkeypatch.setattr(_cleanup, "remove_unclassified_points", lambda pcd, *a, **kw: pcd)
    monkeypatch.setattr(_cleanup, "remove_small_instances", lambda pcd, *a, **kw: pcd)

    import tls2dseg.utils_main as _utils_main

    def _stub_global_shift(pcd, shift, pid):
        return pcd, shift

    monkeypatch.setattr(_utils_main, "assure_common_global_shift", _stub_global_shift)

    import tls2dseg.lifting.masks_to_pcd as _lift

    monkeypatch.setattr(_lift, "lift_masks_to_pcd", lambda *a, **kw: None)
    monkeypatch.setattr(_lift, "lift_mask_to_pcd", lambda *a, **kw: None)

    import tls2dseg.engines.inference.shared as _shared

    monkeypatch.setattr(
        _shared,
        "get_per_mask_depth_parallel",
        lambda results, *a, **kw: None,
    )
    monkeypatch.setattr(
        _shared,
        "get_instance_and_semantic_mask_with_confidence",
        lambda results, text_prompt, image_hw: (
            np.zeros(image_hw, dtype=np.int32),
            np.zeros(image_hw, dtype=np.int32),
            np.zeros(image_hw, dtype=np.float32),
            {"fake_object": 1},
        ),
    )

    import tls2dseg.preprocessing.nms_combine as _nms

    def _stub_nms(scan_d2d_list, **kw):
        return scan_d2d_list[0]

    monkeypatch.setattr(_nms, "nms_combine_detections", _stub_nms)

    import tls2dseg.pc_preprocessing as _pcpreproc

    monkeypatch.setattr(_pcpreproc, "save_segmented_pcd", lambda *a, **kw: None)

    import pchandler.geometry.transforms as _transforms

    monkeypatch.setattr(_transforms, "toggle_socs2prcs", lambda pcd, *a, **kw: pcd)

    import tls2dseg.preprocessing.cleanup as _cleanup2

    monkeypatch.setattr(_cleanup2, "apply_robust_sor_filter", lambda *a, **kw: None)

    # Monkeypatch clean_pcd_instances_and_get_detections3d to return a minimal result
    def _stub_clean(pcd, pcd_id, d3d_params, pcp_params):
        d3d = _make_minimal_d3d(pcd_id=pcd_id)
        pcd.scalar_fields["instances"] = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        return d3d, pcd

    import tls2dseg.detections_3d as _d3d_mod

    monkeypatch.setattr(_d3d_mod, "clean_pcd_instances_and_get_detections3d", _stub_clean)

    spy_engine = _SpyInferenceEngine()
    proj_engine = FakeProjectionEngine(image_size=(4, 4))
    request = _make_inference_request()

    from tls2dseg.pipeline.stage1 import run_stage1

    result = run_stage1(cfg, ctx, proj_engine, spy_engine, request)

    # Scan 1 was fully cached — no inference
    # Scan 2 had a corrupt pcd pkl — inference was called (once per feature)
    assert spy_engine.detect_call_count == n_features, (
        f"Corrupt pkl for scan 2 must trigger inference ({n_features} calls for {n_features} features); "
        f"got {spy_engine.detect_call_count}."
    )
    assert result.n_scans == _N_SCANS
    assert len(result.d3d_collection) == _N_SCANS

    # No None slot in the collection for either scan
    assert result.pcd_collection.seg_pcds[0] is not None, "Scan 1 slot must be populated from cache"
    assert result.pcd_collection.seg_pcds[1] is not None, "Scan 2 slot must be populated after recompute"

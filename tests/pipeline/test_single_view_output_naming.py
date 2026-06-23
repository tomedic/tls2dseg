"""Single-view output naming regression test — real run_stage1, fake engines.

Tier: tier_b_light — driving the real run_stage1 per-scan loop requires pchandler
(PointCloudData / build_context).  tier_a cannot cover the real per-scan naming
path for the same reason as test_stage1_resume.py and test_feature_images_saved.py:
the smoke tests monkeypatch stage1 wholesale, bypassing the real loop.

Two tier_b_light pure-path functions (test_save_segmented_pcd_stem_naming and
test_save_segmented_pcd_legacy_stem_from_dir) assert the output_stem rename logic
in save_segmented_pcd.  They would qualify for tier_a by their logic alone, but
pc_preprocessing.py imports hdbscan at module level, so any import of it requires
the tier_b_light venv — tier_a cannot cover them.

All tests in this module are tier_b_light.  The two pure-path helpers
(test_save_segmented_pcd_stem_naming / legacy) are tier_b_light because
pc_preprocessing imports hdbscan at module level — not present in tier_a venv.
The real run_stage1 tests require pchandler.  tier_a cannot cover either group.

The tests assert all three single-view output fixes introduced in 260619-n0n:

(a) BUG1 — per-scan projection engine receives the correct scan stem (no "unknown"):
    Verified via a spy wrapper that records every set_pcd_path() call; stems for
    scan_0 and scan_1 must differ and neither may contain "unknown".

(b) BUG2 — per-feature segmented PCDs reach intermediate/segmented_point_clouds
    when save_intermediate is on:
    Verified via a monkeypatched save_segmented_pcd_ij that records (stem, feature)
    pairs; must see N_SCANS * N_FEATURES distinct entries named by scan stems.

(c) BUG3 — results/ receives one PLY per scan named by scan stem:
    Verified via a monkeypatched save_segmented_pcd that records the output_stem
    arg; must see N_SCANS distinct calls, each named by a scan stem, not the
    folder name.

Design choice: save_segmented_pcd_ij and save_segmented_pcd are monkeypatched to
record call args rather than writing real PLYs, because the fakes supply a minimal
5-point PointCloudData that lacks the real scalar-field schema save_ply expects.
The spy set_pcd_path records the engine-side naming path independently of PLY I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_N_SCANS = 2
_N_FEATURES = 2  # intensity, range


# ─────────────────────────────────────────────────────────────────────────────
# tier_a — pure-path unit test for save_segmented_pcd output_stem param
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_save_segmented_pcd_stem_naming(tmp_path: Path) -> None:
    """save_segmented_pcd with output_stem produces <stem>_segmented.ply.

    Pure-path assertion: verifies the filename logic without real PLY I/O.
    Marked tier_b_light (not tier_a) because pc_preprocessing imports hdbscan
    at module level — that dep is absent from the tier_a venv.
    """
    import json
    from unittest.mock import patch

    from tls2dseg.pc_preprocessing import save_segmented_pcd

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    data_dir = tmp_path / "input_folder"

    saved_paths: list[Path] = []

    def _fake_save_ply(path, pcd, **kw):
        saved_paths.append(Path(path))

    with (
        patch("tls2dseg.pc_preprocessing.save_ply", side_effect=_fake_save_ply),
        patch.object(json, "dump", return_value=None),
    ):
        save_segmented_pcd(data_dir, out_dir, object(), {"tree": 1}, output_stem="scan_0")

    assert len(saved_paths) == 1
    assert saved_paths[0].name == "scan_0_segmented.ply", (
        f"Expected 'scan_0_segmented.ply', got {saved_paths[0].name!r}"
    )
    assert "unknown" not in saved_paths[0].name


@pytest.mark.tier_b_light
def test_save_segmented_pcd_legacy_stem_from_dir(tmp_path: Path) -> None:
    """save_segmented_pcd without output_stem uses data_dir.name (legacy behaviour).

    Confirms the None default preserves the existing multi-view naming contract.
    Marked tier_b_light for the same reason as test_save_segmented_pcd_stem_naming.
    """
    import json
    from unittest.mock import patch

    from tls2dseg.pc_preprocessing import save_segmented_pcd

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    data_dir = tmp_path / "my_scan_folder"

    saved_paths: list[Path] = []

    def _fake_save_ply(path, pcd, **kw):
        saved_paths.append(Path(path))

    with (
        patch("tls2dseg.pc_preprocessing.save_ply", side_effect=_fake_save_ply),
        patch.object(json, "dump", return_value=None),
    ):
        save_segmented_pcd(data_dir, out_dir, object(), {"tree": 1})

    assert len(saved_paths) == 1
    assert saved_paths[0].name == "my_scan_folder_segmented.ply"


# ─────────────────────────────────────────────────────────────────────────────
# Shared harness
# ─────────────────────────────────────────────────────────────────────────────


def _make_cfg_ctx(tmp_path: Path, save_intermediate: bool = True) -> tuple:
    """Build a minimal single-view RunConfig + RunContext with 2 fake scans."""
    from tls2dseg.config import RunConfig
    from tls2dseg.runtime import build_context
    from tls2dseg.runtime.capability import Runtime

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(_N_SCANS):
        (input_dir / f"scan_{i}.e57").touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    save_flag = "true" if save_intermediate else "false"
    cfg_dict = yaml.safe_load(f"""\
mode: single-view
io:
  input_path: {input_dir}
  output_dir: {output_dir}
  resume_from_checkpoint: false
  save_intermediate: {save_flag}
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
        thread_workers=1,
        empty_slice_removal_threshold=0.95,
    )


def _make_minimal_pcd() -> object:
    from pchandler.geometry import PointCloudData

    xyz = np.zeros((5, 3), dtype=np.float32)
    return PointCloudData(xyz=xyz)


def _make_minimal_d3d(pcd_id: int) -> object:
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


def _apply_recompute_monkeypatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy/GPU helpers so the recompute path completes without real models."""
    pcd_stub = _make_minimal_pcd()

    import pchandler.data_io as _pchandler_io

    monkeypatch.setattr(_pchandler_io, "load_e57", lambda *a, **kw: pcd_stub)

    import tls2dseg.preprocessing.cleanup as _cleanup

    monkeypatch.setattr(_cleanup, "filter_pcd_roi_range", lambda *a, **kw: None)
    monkeypatch.setattr(_cleanup, "subsample_pcd_to_output_resolution", lambda pcd, *a, **kw: pcd)
    monkeypatch.setattr(_cleanup, "remove_unclassified_points", lambda pcd, *a, **kw: pcd)
    monkeypatch.setattr(_cleanup, "remove_small_instances", lambda pcd, *a, **kw: pcd)
    monkeypatch.setattr(_cleanup, "apply_robust_sor_filter", lambda *a, **kw: None)

    import tls2dseg.utils_main as _utils_main

    monkeypatch.setattr(_utils_main, "assure_common_global_shift", lambda pcd, shift, pid: (pcd, shift))

    import tls2dseg.lifting.masks_to_pcd as _lift

    monkeypatch.setattr(_lift, "lift_masks_to_pcd", lambda *a, **kw: None)
    monkeypatch.setattr(_lift, "lift_mask_to_pcd", lambda *a, **kw: None)

    import tls2dseg.engines.inference.shared as _shared

    monkeypatch.setattr(_shared, "get_per_mask_depth_parallel", lambda results, *a, **kw: None)
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

    monkeypatch.setattr(_nms, "nms_combine_detections", lambda scan_d2d_list, **kw: scan_d2d_list[0])

    import pchandler.geometry.transforms as _transforms

    monkeypatch.setattr(_transforms, "toggle_socs2prcs", lambda pcd, *a, **kw: pcd)

    def _stub_clean(pcd, pcd_id, d3d_params, pcp_params):
        d3d = _make_minimal_d3d(pcd_id=pcd_id)
        pcd.scalar_fields["instances"] = np.array([1, 1, 1, 1, 1], dtype=np.int32)
        return d3d, pcd

    import tls2dseg.detections_3d as _d3d_mod

    monkeypatch.setattr(_d3d_mod, "clean_pcd_instances_and_get_detections3d", _stub_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Spy projection engine — records set_pcd_path calls
# ─────────────────────────────────────────────────────────────────────────────


class _SpyProjectionEngine:
    """FakeProjectionEngine that records every set_pcd_path() call."""

    def __init__(self) -> None:
        from tests.unit.fakes import FakeProjectionEngine

        self._inner = FakeProjectionEngine(image_size=(4, 4))
        self.pcd_paths_seen: list[Path] = []

    def project(
        self,
        pcd: object,
        *,
        features: list[str],
        resolution: tuple[int, int],
        skip_image_reduction: bool = False,
    ) -> list:
        return self._inner.project(
            pcd, features=features, resolution=resolution, skip_image_reduction=skip_image_reduction
        )

    def set_output_dir_images(self, path: Path) -> None:
        self._inner.set_output_dir_images(path)

    def set_pcd_path(self, path: Path) -> None:
        self.pcd_paths_seen.append(path)
        self._inner.set_pcd_path(path)


class _SpyInferenceEngine:
    """FakeInferenceEngine that counts detect() calls."""

    def __init__(self) -> None:
        from tests.unit.fakes import FakeInferenceEngine

        self._inner = FakeInferenceEngine(n_detections=1)
        self.detect_call_count = 0

    def detect(self, image: object, *, request: object) -> object:
        self.detect_call_count += 1
        return self._inner.detect(image, request=request)


# ─────────────────────────────────────────────────────────────────────────────
# tier_b_light — real run_stage1 single-view naming tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_single_view_per_scan_distinct_stems_no_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) BUG1: set_pcd_path is called per scan with distinct, non-'unknown' stems.

    Drives the real run_stage1 in single-view mode with 2 fake scans (scan_0,
    scan_1) and 2 features (intensity, range).  The spy projection engine records
    every set_pcd_path() call; asserts:
    - Exactly N_SCANS calls were made (one per scan, not per (scan, feature)).
    - The two stems differ.
    - Neither stem contains the substring 'unknown'.

    This test WOULD FAIL against pre-fix code (before set_pcd_path was added to
    the engine and wired in the per-scan loop).
    """
    _apply_recompute_monkeypatches(monkeypatch)

    # Stub save_segmented_pcd so the final PLY write does not attempt real I/O
    import tls2dseg.pc_preprocessing as _pcpreproc

    monkeypatch.setattr(_pcpreproc, "save_segmented_pcd", lambda *a, **kw: None)

    cfg, ctx = _make_cfg_ctx(tmp_path, save_intermediate=False)
    spy_proj = _SpyProjectionEngine()
    spy_infer = _SpyInferenceEngine()
    request = _make_inference_request()

    from tls2dseg.pipeline.stage1 import run_stage1

    result = run_stage1(cfg, ctx, spy_proj, spy_infer, request)

    assert result.n_scans == _N_SCANS

    # set_pcd_path must be called once per scan
    assert len(spy_proj.pcd_paths_seen) == _N_SCANS, (
        f"Expected {_N_SCANS} set_pcd_path calls (one per scan); "
        f"got {len(spy_proj.pcd_paths_seen)}: {spy_proj.pcd_paths_seen}"
    )

    stems = [p.stem for p in spy_proj.pcd_paths_seen]
    assert len(set(stems)) == _N_SCANS, f"Expected {_N_SCANS} distinct stems; got {stems!r}"
    for stem in stems:
        assert "unknown" not in stem, f"Stem {stem!r} contains 'unknown' — BUG1 not fixed for this scan"


@pytest.mark.tier_b_light
def test_single_view_per_feature_intermediate_dumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) BUG2: per-feature segmented PCDs are dumped when save_intermediate is on.

    Asserts that save_segmented_pcd_ij is called N_SCANS * N_FEATURES times when
    save_intermediate=true, and that the recorded pcd_path_pathlib.stem values
    match the two scan stems (scan_0 and scan_1).

    This test WOULD FAIL against pre-fix code (the single-view branch had no
    save_segmented_pcd_ij call).
    """
    _apply_recompute_monkeypatches(monkeypatch)

    # Intercept save_segmented_pcd_ij — record (stem, feature) pairs
    ij_calls: list[tuple[str, str]] = []

    def _spy_save_segmented_pcd_ij(pcd_path_pathlib, pcd, inference_params, class_id_map, image_j):
        feature_name = image_j[0]
        ij_calls.append((pcd_path_pathlib.stem, feature_name))

    import tls2dseg.pc_preprocessing as _pcpreproc

    monkeypatch.setattr(_pcpreproc, "save_segmented_pcd_ij", _spy_save_segmented_pcd_ij)
    monkeypatch.setattr(_pcpreproc, "save_segmented_pcd", lambda *a, **kw: None)

    cfg, ctx = _make_cfg_ctx(tmp_path, save_intermediate=True)
    spy_proj = _SpyProjectionEngine()
    spy_infer = _SpyInferenceEngine()
    request = _make_inference_request()

    from tls2dseg.pipeline.stage1 import run_stage1

    run_stage1(cfg, ctx, spy_proj, spy_infer, request)

    assert len(ij_calls) == _N_SCANS * _N_FEATURES, (
        f"Expected {_N_SCANS * _N_FEATURES} save_segmented_pcd_ij calls "
        f"(N_SCANS={_N_SCANS} x N_FEATURES={_N_FEATURES}); got {len(ij_calls)}: {ij_calls}"
    )

    stems_seen = {stem for stem, _ in ij_calls}
    assert len(stems_seen) == _N_SCANS, f"Expected {_N_SCANS} distinct scan stems in ij calls; got {stems_seen!r}"
    for stem in stems_seen:
        assert "unknown" not in stem, f"Intermediate dump stem {stem!r} contains 'unknown' — BUG1 not fixed"

    features_seen = {feature for _, feature in ij_calls}
    assert features_seen == {"intensity", "range"}, f"Expected features {{intensity, range}}; got {features_seen!r}"


@pytest.mark.tier_b_light
def test_single_view_result_ply_per_scan_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) BUG3: results/ receives one PLY per scan named by scan stem (not folder).

    Asserts that save_segmented_pcd is called once per scan with output_stem equal
    to the scan file stem (scan_0, scan_1 — not the input folder name).

    This test WOULD FAIL against pre-fix code (save_segmented_pcd received
    data_folder_path with no output_stem, collapsing all scans to one folder-named
    file).
    """
    _apply_recompute_monkeypatches(monkeypatch)

    # Intercept save_segmented_pcd — record output_stem args
    result_calls: list[tuple] = []

    def _spy_save_segmented_pcd(data_dir, out_dir, pcd, class_id_map, output_stem=None):
        result_calls.append((data_dir, out_dir, output_stem))

    import tls2dseg.pc_preprocessing as _pcpreproc

    monkeypatch.setattr(_pcpreproc, "save_segmented_pcd", _spy_save_segmented_pcd)

    cfg, ctx = _make_cfg_ctx(tmp_path, save_intermediate=False)
    spy_proj = _SpyProjectionEngine()
    spy_infer = _SpyInferenceEngine()
    request = _make_inference_request()

    from tls2dseg.pipeline.stage1 import run_stage1

    run_stage1(cfg, ctx, spy_proj, spy_infer, request)

    assert len(result_calls) == _N_SCANS, (
        f"Expected {_N_SCANS} save_segmented_pcd calls (one per scan); got {len(result_calls)}: {result_calls}"
    )

    stems_seen = [output_stem for _, _, output_stem in result_calls]
    assert all(s is not None for s in stems_seen), f"output_stem must be set for single-view; got {stems_seen!r}"
    assert len(set(stems_seen)) == _N_SCANS, f"Expected {_N_SCANS} distinct output stems; got {stems_seen!r}"
    for stem in stems_seen:
        assert "unknown" not in stem, f"Result PLY stem {stem!r} contains 'unknown'"
    # stems must be scan file stems, not the input folder name
    input_folder_name = str(tmp_path / "input")
    for stem in stems_seen:
        assert stem != Path(input_folder_name).name, f"output_stem {stem!r} is the folder name — BUG3 not fixed"
    assert set(stems_seen) == {"scan_0", "scan_1"}, f"Expected stems {{scan_0, scan_1}}; got {set(stems_seen)!r}"

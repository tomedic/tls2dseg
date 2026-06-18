"""CR-05 regression: feature images are saved when intermediate saving is enabled.

Tier: tier_b_light — needs pchandler/pc2img available (SphericalProjectionEngine
imports pc2img inside project(); set_output_dir_images is tls2dseg-owned and
tested here without needing a real cloud).

Two complementary assertions:

1. ``test_set_output_dir_images_populates_params`` — unit assertion on
   SphericalProjectionEngine itself: after calling set_output_dir_images(path),
   the key is present in ``self._params`` (the dict project() shallow-copies),
   so every subsequent project() call will include output_dir_images.  This test
   is sufficient to pin the CR-05 fix mechanically; it does NOT require a real
   scan because it never calls project().

2. ``test_stage1_calls_set_output_dir_images_when_saving_enabled`` — spy test
   that patches the heavy pchandler/torch imports inside run_stage1 and verifies
   that, when save_intermediate=true, stage1 calls
   projection_engine.set_output_dir_images(...) with a path under
   intermediate/images/.  This test would FAIL against pre-fix stage1 (where
   make_output_folders received a throwaway dict that was never connected to the
   engine).

Note: tier_a cannot cover the real-projection PNG write path because
``pc2img_run`` (called inside SphericalProjectionEngine.project()) requires an
actual PointCloudData from pchandler.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — engine setter populates params (pure engine unit, tier_b_light)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_set_output_dir_images_populates_params(tmp_path: Path) -> None:
    """set_output_dir_images mutates self._params so project() will see the key.

    SphericalProjectionEngine.project() does ``params = dict(self._params)``
    (shallow copy) at each call, so setting output_dir_images on self._params
    before the first project() call makes every projection write PNGs.
    """
    from tls2dseg.engines.projection.spherical import SphericalProjectionEngine

    params: dict = {
        "features": ["intensity"],
        "image_width": 512,
        "scan_resolution": 0.036,
        "rotate_pcd": False,
        "rasterization_method": "nanconv",
    }
    engine = SphericalProjectionEngine(params, pcd_path=None)

    assert "output_dir_images" not in engine._params, (
        "output_dir_images must NOT be present before set_output_dir_images is called"
    )

    save_dir = tmp_path / "intermediate" / "images"
    save_dir.mkdir(parents=True)
    engine.set_output_dir_images(save_dir)

    assert "output_dir_images" in engine._params, "set_output_dir_images must add output_dir_images to self._params"
    assert engine._params["output_dir_images"] == save_dir, (
        f"Expected {save_dir!r}, got {engine._params['output_dir_images']!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — stage1 calls set_output_dir_images when save_intermediate=true
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _SpyProjectionEngine:
    """Projection engine spy that records set_output_dir_images calls."""

    received_path: Path | None = dataclasses.field(default=None, init=False)

    def set_output_dir_images(self, path: Path) -> None:
        self.received_path = path

    def project(self, pcd: object, *, features: list[str], resolution: tuple[int, int]) -> list:
        return []


def _make_save_enabled_cfg_ctx(tmp_path: Path) -> tuple:
    """Build cfg + ctx with save_intermediate=true (dump_json_results=true)."""
    import yaml

    from tls2dseg.config import RunConfig
    from tls2dseg.runtime import build_context
    from tls2dseg.runtime.capability import Runtime

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "scan_0.e57").touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg_dict = yaml.safe_load(f"""\
mode: single-view
io:
  input_path: {input_dir}
  output_dir: {output_dir}
  resume_from_checkpoint: false
  save_intermediate: true
prompt:
  text: fake_object
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity]
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


@pytest.mark.tier_b_light
def test_stage1_calls_set_output_dir_images_when_saving_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stage1 calls set_output_dir_images(intermediate/images/) when save_intermediate=true.

    Drives run_stage1 with the heavy pchandler/torch imports mocked out so the
    test runs in the tier_b_light environment (pchandler available but torch/sam2
    not required).  The spy records whether set_output_dir_images was called with
    a path whose name is 'images' and whose parent ends with 'intermediate'.

    This test FAILS against pre-fix stage1 (where make_output_folders received a
    throwaway dict — the engine's set_output_dir_images was never called).
    """
    # Patch the heavy imports that run_stage1 pulls in at function-body level
    # Stub out torch (not available in tier_b_light venv by design)
    import sys
    import types

    import tls2dseg.pipeline.stage1 as _stage1_mod

    fake_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Stub pchandler.data_io.load_e57 — stage1 imports it but our spy test
    # returns early before the per-scan loop calls it.
    fake_pchandler = MagicMock()
    fake_pchandler_data_io = MagicMock()
    fake_pchandler_geometry = MagicMock()
    fake_pchandler_geometry_transforms = MagicMock()
    monkeypatch.setitem(sys.modules, "pchandler", fake_pchandler)
    monkeypatch.setitem(sys.modules, "pchandler.data_io", fake_pchandler_data_io)
    monkeypatch.setitem(sys.modules, "pchandler.geometry", fake_pchandler_geometry)
    monkeypatch.setitem(sys.modules, "pchandler.geometry.transforms", fake_pchandler_geometry_transforms)

    # Stub the remaining heavy deps stage1 imports inside the function body
    for mod_name in [
        "tls2dseg.detections_3d",
        "tls2dseg.engines.inference.shared",
        "tls2dseg.lifting.masks_to_pcd",
        "tls2dseg.pc_preprocessing",
        "tls2dseg.pcd_collection",
        "tls2dseg.preprocessing.cleanup",
        "tls2dseg.preprocessing.nms_combine",
    ]:
        monkeypatch.setitem(sys.modules, mod_name, MagicMock())

    # SegPCDCollection constructor is called before the per-scan loop — stub it
    fake_seg_pcd_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(sys.modules["tls2dseg.pcd_collection"], "SegPCDCollection", fake_seg_pcd_cls)

    # Stub utils_main helpers needed before the per-scan loop
    import tls2dseg.utils_main as _utils_main

    _orig_make_output_folders = _utils_main.make_output_folders

    def _stub_load_previously(*args: Any, **kwargs: Any) -> tuple:
        pcd_coll, d3d_coll = args[1], args[3]
        return pcd_coll, d3d_coll, False

    monkeypatch.setattr(_utils_main, "load_previously_saved_inference_results_if_any", _stub_load_previously)

    cfg, ctx = _make_save_enabled_cfg_ctx(tmp_path)

    spy = _SpyProjectionEngine()

    from tls2dseg.types import InferenceRequest

    fake_request = InferenceRequest(
        text_prompt="fake_object",
        box_threshold=0.10,
        text_threshold=0.10,
        slicing_enabled=False,
        slice_width_height=(320, 320),
        overlap_width_height=(32, 32),
        iou_threshold=0.5,
        overlap_filter_strategy="nms",
        large_object_removal_threshold=0.30,
        partial_detection_edge_touching_threshold=5,
        thread_workers=1,
        empty_slice_removal_threshold=0.95,
    )

    # Intercept the per-scan loop to abort after the wiring section but before
    # actual inference — raise a sentinel once the spy has been called (or if
    # SegPCDCollection is constructed, which happens after the wiring).
    class _DoneEarly(Exception):
        pass

    # Make SegPCDCollection construction raise so we abort after the wiring lines.
    fake_seg_pcd_cls.side_effect = _DoneEarly("done-early")

    import tls2dseg.pipeline.stage1 as stage1_mod

    with pytest.raises(_DoneEarly):
        stage1_mod.run_stage1(cfg, ctx, spy, MagicMock(), fake_request)

    # Verify set_output_dir_images was called with a path under intermediate/images/
    assert spy.received_path is not None, (
        "set_output_dir_images was NOT called — CR-05 regression: stage1 must call "
        "projection_engine.set_output_dir_images(intermediate/images/) when save_intermediate=true"
    )
    assert spy.received_path.name == "images", f"Expected path name 'images', got {spy.received_path.name!r}"
    assert "intermediate" in str(spy.received_path), f"Expected path under 'intermediate/', got {spy.received_path!r}"

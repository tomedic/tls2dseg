"""Regression test: output_resolution_m wired into projection params (CR-03 fix).

Verifies:
1. image_generation_parameters carries 'output_resolution' so the single-zoom
   reduction guard in SphericalProjectionEngine.project() can fire.
2. Single-zoom project() with a coarser-than-native output_resolution calls
   resolve_necessary_image_resolution and reduce_image_resolution.
3. skip_image_reduction=True (multi-zoom) leaves those functions uncalled.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal synthetic PointCloudData for SphericalProjectionEngine.project()
# ---------------------------------------------------------------------------


def _make_pcd() -> object:
    """Return a small PointCloudData suitable for a one-feature projection."""
    from pchandler.geometry import PointCloudData

    rng = np.random.default_rng(0)
    n = 500

    # Points spread over a modest FoV: ~30 deg azimuth, ~10 deg elevation, 5-10 m range
    az = rng.uniform(np.deg2rad(-15.0), np.deg2rad(15.0), n)
    el = rng.uniform(np.deg2rad(-5.0), np.deg2rad(5.0), n)
    r = rng.uniform(5.0, 10.0, n)

    cos_el = np.cos(el)
    x = r * cos_el * np.cos(az)
    y = -r * cos_el * np.sin(az)
    z = r * np.sin(el)
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    tmat = np.eye(4, dtype=np.float64)
    return PointCloudData(xyz=xyz, tmat_socs2prcs=tmat)


# ---------------------------------------------------------------------------
# Params factory (no output_resolution — mirrors the broken pre-fix state)
# ---------------------------------------------------------------------------


def _base_params() -> dict:
    return {
        "features": ["intensity"],
        "rasterization_method": "max",
        "scan_resolution": 0.1,
        "image_width": "scan_resolution",
        "rotate_pcd": False,
    }


# ---------------------------------------------------------------------------
# Test 1: image_generation_parameters carries the output_resolution key
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
def test_pipeline_image_generation_parameters_has_output_resolution() -> None:
    """Pipeline.__init__ must include 'output_resolution' in image_generation_parameters.

    This is the direct regression guard: fails on the old missing-key code.
    Uses only config types (no heavy engines) so it's fast.
    """
    import yaml

    from tls2dseg.config import RunConfig

    cfg_dict = yaml.safe_load("""\
mode: single-view
io:
  input_path: /tmp/fake_input
  output_dir: /tmp/fake_output
prompt:
  text: tree
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity]
inference:
  type: grounded_sam2
  sam2_checkpoint: /tmp/fake_sam2.pt
d3d_extraction: {}
fusion: {}
runtime:
  device: cpu
""")
    cfg = RunConfig(**cfg_dict)

    # Mirror what Pipeline.__init__ builds — without instantiating the full Pipeline
    # (which would pull torch, sam2, pchandler).
    params: dict = {
        "image_width": cfg.projection.image_width,
        "scan_resolution": cfg.projection.scan_resolution,
        "rotate_pcd": cfg.projection.rotate_pcd,
        "rasterization_method": cfg.projection.rasterization_method,
        "features": list(cfg.projection.features),
        "output_resolution": cfg.preprocessing.output_resolution_m,
    }

    assert "output_resolution" in params, (
        "'output_resolution' missing from image_generation_parameters — "
        "CR-03 regression: single-zoom reduction can never fire"
    )
    assert params["output_resolution"] == cfg.preprocessing.output_resolution_m


# ---------------------------------------------------------------------------
# Test 2: single-zoom reduction fires; multi-zoom stays native
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
def test_single_zoom_fires_reduction_multi_zoom_stays_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-zoom project() calls reduction helpers; multi-zoom skips them.

    Strategy:
    - Build a SphericalProjectionEngine with output_resolution set.
    - Monkeypatch pc2img_run to return a dummy image (avoid heavy rasterisation).
    - Monkeypatch resolve_necessary_image_resolution to return 0.5 (reduction needed).
    - Monkeypatch reduce_image_resolution to record calls and halve the image.
    - Run project() with skip_image_reduction=False (single-zoom) — expect reduction called.
    - Run project() with skip_image_reduction=True  (multi-zoom)  — expect reduction NOT called.
    """
    import tls2dseg.pc2img_utils as pc2img_utils
    from tls2dseg.engines.projection.spherical import SphericalProjectionEngine

    pcd = _make_pcd()

    dummy_image = np.zeros((16, 32), dtype=np.float32)

    monkeypatch.setattr(
        pc2img_utils,
        "pc2img_run",
        lambda pcd_arg, pcd_path, params, image_width, image_height: [
            ("intensity", dummy_image.copy(), "no Path - images not saved")
        ],
    )

    resolve_calls: list[float] = []
    reduce_calls: list[bool] = []

    def _fake_resolve(pcd_arg, params, d_azim):  # type: ignore[override]
        resolve_calls.append(0.5)
        return 0.5  # coeff < 1.0 → reduction will proceed

    def _fake_reduce(images, coeff, params, pcd_path):  # type: ignore[override]
        reduce_calls.append(True)
        # Return images unchanged (shape preserved so downstream code is fine)
        return images

    monkeypatch.setattr(pc2img_utils, "resolve_necessary_image_resolution", _fake_resolve)
    monkeypatch.setattr(pc2img_utils, "reduce_image_resolution", _fake_reduce)

    params = _base_params()
    params["output_resolution"] = 0.1  # coarser than native → reduction wanted

    engine = SphericalProjectionEngine(params)

    # Single-zoom: skip_image_reduction=False (default) — reduction must fire
    resolve_calls.clear()
    reduce_calls.clear()
    results_sv = engine.project(pcd, features=["intensity"], resolution=(0, 0), skip_image_reduction=False)
    assert len(results_sv) == 1
    assert len(resolve_calls) == 1, (
        "resolve_necessary_image_resolution must be called once in single-zoom "
        f"(got {len(resolve_calls)} calls) — CR-03: output_resolution key missing?"
    )
    assert len(reduce_calls) == 1, (
        f"reduce_image_resolution must be called once in single-zoom (got {len(reduce_calls)} calls)"
    )

    # Multi-zoom: skip_image_reduction=True — no reduction
    resolve_calls.clear()
    reduce_calls.clear()
    results_mz = engine.project(pcd, features=["intensity"], resolution=(0, 0), skip_image_reduction=True)
    assert len(results_mz) == 1
    assert len(resolve_calls) == 0, (
        "resolve_necessary_image_resolution must NOT be called with skip_image_reduction=True "
        f"(got {len(resolve_calls)} calls) — multi-zoom must stay native"
    )
    assert len(reduce_calls) == 0, (
        f"reduce_image_resolution must NOT be called with skip_image_reduction=True (got {len(reduce_calls)} calls)"
    )

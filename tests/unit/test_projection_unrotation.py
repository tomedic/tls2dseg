"""Projection-time rotations are reported and undone before SOCS→PRCS conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCAN = Path(__file__).parent.parent.parent / "examples" / "data" / "mountains_small" / "Epoch_1.e57"

pytestmark = [
    pytest.mark.tier_b_light,
    pytest.mark.skipif(importlib.util.find_spec("pchandler") is None, reason="requires pchandler"),
    pytest.mark.skipif(importlib.util.find_spec("pc2img") is None, reason="requires pc2img"),
]


def _load():
    from pchandler.data_io import load_e57

    return load_e57(_SCAN, stay_prcs=False, save_prcs_info=True)


def _prcs_xyz(pcd) -> np.ndarray:
    from pchandler.geometry.transforms import toggle_socs2prcs

    toggle_socs2prcs(pcd)
    xyz = pcd.xyz.astype(np.float64)
    if pcd.global_coordinate_shift is not None:
        xyz += pcd.global_coordinate_shift
    return xyz


def _project(pcd, **params) -> np.ndarray:
    from tls2dseg.engines import build_projection_engine

    image_params = {
        "image_width": 400,
        "scan_resolution": 0.3,
        "rotate_pcd": False,
        "rasterization_method": "nanconv",
        "features": ["intensity"],
        "output_resolution": None,
        "flip_upsidedown_scans_deg": None,
        **params,
    }
    engine = build_projection_engine("spherical", image_generation_parameters=image_params, pcd_path=_SCAN)
    results = engine.project(pcd, features=["intensity"], resolution=(0, 0))
    return results[0].socs_rotation


def _assert_unrotation_restores_prcs(pcd, rotation) -> None:
    from tls2dseg.pipeline.stage1 import _unrotate_socs

    reference = _prcs_xyz(_load())
    _unrotate_socs(pcd, rotation)
    assert np.abs(_prcs_xyz(pcd) - reference).max() < 1e-3


def test_no_rotation_reports_identity() -> None:
    assert np.allclose(_project(_load(), rotate_pcd=False), np.eye(4))


def test_z_rotation_is_reported_and_undone() -> None:
    from tls2dseg.pc2img_utils import rotation_matrix_z

    pcd = _load()
    rotation = _project(pcd, rotate_pcd=90.0)
    np.testing.assert_allclose(rotation, rotation_matrix_z(90.0), atol=1e-12)
    _assert_unrotation_restores_prcs(pcd, rotation)


def test_upside_down_flip_only_when_configured(monkeypatch) -> None:
    import tls2dseg.pc2img_utils as pu

    monkeypatch.setattr(pu, "check_was_scanner_upsidedown", lambda _pcd: True)

    np.testing.assert_allclose(_project(_load(), rotate_pcd=30.0), pu.rotation_matrix_z(30.0), atol=1e-12)

    pcd = _load()
    rotation = _project(pcd, rotate_pcd=30.0, flip_upsidedown_scans_deg=90.0)
    np.testing.assert_allclose(rotation, pu.rotation_matrix_z(30.0) @ pu.rotation_matrix_x(90.0), atol=1e-12)
    _assert_unrotation_restores_prcs(pcd, rotation)


def test_flip_skipped_for_upright_scan() -> None:
    assert np.allclose(_project(_load(), flip_upsidedown_scans_deg=90.0), np.eye(4))

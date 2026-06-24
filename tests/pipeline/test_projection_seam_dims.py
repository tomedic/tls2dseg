"""Regression test: seam-straddling cloud sizes canvas from post-rotation FoV.

Verifies that compute_image_dimensions is called AFTER the azimuth-rotation
step in SphericalProjectionEngine.project(), so a cloud whose azimuths straddle
the ±180° seam gets a canvas sized from the tight post-rotation span rather
than the inflated ~360° naive span.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic seam-straddling point cloud fixture
# ---------------------------------------------------------------------------


def _make_seam_pcd() -> object:
    """Return a PointCloudData whose azimuths straddle the ±180° seam.

    Geometry: bulk of points near azimuth +175° to +180°, a small tail near
    -175° to -180°.  The real arc is ~10° wide; the naive horizontal_max -
    horizontal_min reads ~350° (≈ 2π rad) because the seam inflates the span.
    A clear gap covers roughly -170° to +170° (≈ 340°) so rotate_pcd='auto'
    finds it and collapses the span to the tight ~10° arc.

    Realistic ranges (2-5 m) and a modest elevation span (±5°) are used so
    PointCloudData can compute its FOV from xyz.
    """
    from pchandler.geometry import PointCloudData

    rng = np.random.default_rng(42)
    n_bulk = 200
    n_tail = 20

    # --- bulk points near azimuth +175° to +180° ---
    az_bulk = rng.uniform(np.deg2rad(175.0), np.deg2rad(180.0), n_bulk)
    el_bulk = rng.uniform(np.deg2rad(-5.0), np.deg2rad(5.0), n_bulk)
    r_bulk = rng.uniform(2.0, 5.0, n_bulk)

    # --- tail points near azimuth -180° to -175° ---
    az_tail = rng.uniform(np.deg2rad(-180.0), np.deg2rad(-175.0), n_tail)
    el_tail = rng.uniform(np.deg2rad(-5.0), np.deg2rad(5.0), n_tail)
    r_tail = rng.uniform(2.0, 5.0, n_tail)

    az = np.concatenate([az_bulk, az_tail])
    el = np.concatenate([el_bulk, el_tail])
    r = np.concatenate([r_bulk, r_tail])

    # Convert spherical (r, elevation, azimuth) → Cartesian xyz.
    # Azimuth convention: atan2(-y, x) → x = r·cos(el)·cos(az), y = -r·cos(el)·sin(az)
    cos_el = np.cos(el)
    x = r * cos_el * np.cos(az)
    y = -r * cos_el * np.sin(az)
    z = r * np.sin(el)
    xyz = np.column_stack([x, y, z]).astype(np.float64)

    # Identity scanner pose (no upside-down flag — keep flip step a no-op here)
    tmat = np.eye(4, dtype=np.float64)

    return PointCloudData(xyz=xyz, tmat_socs2prcs=tmat)


# ---------------------------------------------------------------------------
# Minimal projection params for the engine
# ---------------------------------------------------------------------------

_SCAN_RES_DEG = 0.1  # fixed so expected pixel counts are deterministic


def _make_params() -> dict:
    return {
        "features": ["intensity"],
        "rasterization_method": "max",
        "scan_resolution": _SCAN_RES_DEG,
        "image_width": "scan_resolution",
        "rotate_pcd": "auto",
        # subsampling knobs used by rotate_pcd_to_azimuth_gap
        "subsampling_fraction": 0.5,
        "bin_width_for_theta_search_deg": 1.0,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.tier_b_light
def test_seam_cloud_canvas_sized_from_post_rotation_fov(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canvas width reflects the tight post-rotation FoV for seam-straddling clouds.

    Strategy (ordering-invariant path): monkeypatch compute_image_dimensions to
    capture the pcd.fov.horizontal_max - horizontal_min it sees when called from
    project().  Assert the captured span is the tight post-rotation value (~10°),
    NOT the inflated pre-rotation ~360° span.
    """
    import tls2dseg.pc2img_utils as pc2img_utils
    from tls2dseg.engines.projection.spherical import SphericalProjectionEngine

    pcd = _make_seam_pcd()

    # Sanity-check: pre-rotation naive span should be close to 2π (seam-inflated).
    pre_span_rad = pcd.fov.horizontal_max - pcd.fov.horizontal_min
    assert pre_span_rad > np.deg2rad(300.0), (
        f"Fixture pre-rotation span {np.degrees(pre_span_rad):.1f}° must be ≥300° "
        "to exercise the seam-inflation bug; fixture geometry is wrong."
    )

    # Capture the azimuth span seen by compute_image_dimensions at call time.
    captured_spans: list[float] = []
    _real_compute = pc2img_utils.compute_image_dimensions

    def _capturing_compute(pcd_arg, params, d_azim, d_elev):  # type: ignore[override]
        span = pcd_arg.fov.horizontal_max - pcd_arg.fov.horizontal_min
        captured_spans.append(span)
        return _real_compute(pcd_arg, params, d_azim, d_elev)

    monkeypatch.setattr(pc2img_utils, "compute_image_dimensions", _capturing_compute)

    # Also patch pc2img_run to avoid heavy rasterisation in this tier_b_light test.
    dummy_image = np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(
        pc2img_utils,
        "pc2img_run",
        lambda pcd_arg, pcd_path, params, image_width, image_height: [
            ("intensity", dummy_image, "no Path - images not saved")
        ],
    )

    params = _make_params()
    engine = SphericalProjectionEngine(params)

    # project() imports compute_image_dimensions fresh from tls2dseg.pc2img_utils
    # inside the method body (D-A-05), so we must also patch the engine's import
    # path to intercept the call.
    import tls2dseg.engines.projection.spherical as sph_mod

    # The engine body does `from tls2dseg.pc2img_utils import compute_image_dimensions`
    # on every call, so we need to patch it on the source module (already done via
    # monkeypatch above) AND ensure the import inside project() picks up the patched
    # version.  Because Python caches modules, patching pc2img_utils.compute_image_dimensions
    # in-place is sufficient — the `from ... import` inside project() binds to the
    # module attribute at call time only when there is no local cache.
    # Confirm by checking sph_mod is loaded (it is — we imported it above).
    _ = sph_mod

    results = engine.project(pcd, features=["intensity"], resolution=(0, 0))

    assert len(results) == 1, "project() must return one ProjectionResult"
    assert len(captured_spans) == 1, (
        f"compute_image_dimensions must be called exactly once; called {len(captured_spans)} times"
    )

    captured_span_deg = np.degrees(captured_spans[0])
    tight_threshold_deg = 30.0  # post-rotation span must be well below this

    assert captured_span_deg < tight_threshold_deg, (
        f"compute_image_dimensions saw azimuth span {captured_span_deg:.1f}°; "
        f"expected < {tight_threshold_deg}° (tight post-rotation FoV). "
        "The reorder may be missing — compute_image_dimensions is running before "
        "resolve_rotate_pcd_parameter."
    )

"""TEST-03 wrappers-only: pure computation helpers from SphericalProjectionEngine tier_a.

Phase 4 plan 05 Task 1. Locks the contract of tls2dseg-owned helpers in
``tls2dseg.engines.projection.spherical`` (ENG-07, D-D-04 wrappers-only rule):

* ``check_was_scanner_upsidedown`` — synthetic 4x4 matrix, upside-down detection.
* ``resolve_scanning_resolution_parameter`` — string and numeric inputs to float.

These helpers live in ``engines/projection/spherical.py`` (moved there in Plan 04-02).
Wrappers-only rule (D-D-04): tests tls2dseg-owned helpers, NOT upstream pchandler
transforms or pc2img rasterization.

Marked ``tier_a`` — pure numpy math, no pchandler/pc2img/torch needed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_identity_tmat() -> np.ndarray:
    """4x4 identity transform matrix — scanner upright."""
    return np.eye(4, dtype=np.float64)


def _make_upsidedown_tmat() -> np.ndarray:
    """4x4 transform where local-Z column points downward (scanner upside-down).

    The ``check_was_scanner_upsidedown`` function reads column 2 (local-z),
    specifically the z-component (row 2). For upside-down: local-z points down,
    so tmat[:3, 2] = [0, 0, -1] gives angle_deg = 180 degrees > 170 threshold.
    """
    tmat = np.eye(4, dtype=np.float64)
    # Set local-Z column to point downward: [0, 0, -1]
    tmat[:3, 2] = [0.0, 0.0, -1.0]
    return tmat


@pytest.mark.tier_a
def test_check_was_scanner_upsidedown_upright() -> None:
    """Upright scanner (local-Z up) returns False."""
    from tls2dseg.engines.projection.spherical import check_was_scanner_upsidedown

    # Use a duck-typed mock with only tmat_socs2prcs attribute
    class _FakePcd:
        tmat_socs2prcs = _make_identity_tmat()

    assert check_was_scanner_upsidedown(_FakePcd()) is False, "Upright scanner (identity tmat) should return False"


@pytest.mark.tier_a
def test_check_was_scanner_upsidedown_inverted() -> None:
    """Upside-down scanner (local-Z down) returns True."""
    from tls2dseg.engines.projection.spherical import check_was_scanner_upsidedown

    class _FakePcd:
        tmat_socs2prcs = _make_upsidedown_tmat()

    assert check_was_scanner_upsidedown(_FakePcd()) is True, (
        "Upside-down scanner (local-Z pointing down) should return True"
    )


@pytest.mark.tier_a
def test_check_was_scanner_upsidedown_threshold() -> None:
    """Scanner tilted exactly 45 degrees is not considered upside-down (default threshold 10 deg)."""
    from tls2dseg.engines.projection.spherical import check_was_scanner_upsidedown

    # Local-Z tilted 45 degrees: cosine = sqrt(2)/2 ≈ 0.707 -> angle = 45 deg < 170
    tmat = np.eye(4, dtype=np.float64)
    tmat[:3, 2] = [0.0, np.sin(np.radians(45)), np.cos(np.radians(45))]

    class _FakePcd:
        tmat_socs2prcs = tmat

    assert check_was_scanner_upsidedown(_FakePcd()) is False, "45-degree tilt is well below the upside-down threshold"


@pytest.mark.tier_a
def test_resolve_scanning_resolution_parameter_numeric_string() -> None:
    """String like '1.0mm@10m' resolves to the expected azimuth/elevation radians tuple."""
    from tls2dseg.engines.projection.spherical import resolve_scanning_resolution_parameter

    # "1.0mm@10m": resolution = 0.001 m / 10 m = 0.0001 rad (approximately, via arctan)
    # The function accepts dict-based image_generation_parameters["scan_resolution"]
    params = {"scan_resolution": "1.0mm@10m"}

    class _FakePcd:
        """Minimal mock with spherical_coordinates for estimate_scanning_resolution fallback."""

        spherical_coordinates = np.zeros((10, 3), dtype=np.float64)

    result = resolve_scanning_resolution_parameter(_FakePcd(), params)
    # "1.0mm@10m" parses to 0.001/10 = 0.0001 rad via arcsin; tuple of 2 floats
    assert isinstance(result, tuple), "Should return a tuple of (azimuth_rad, elevation_rad)"
    assert len(result) == 2, "Should return exactly 2 values"
    d_azim, _d_elev = result
    assert isinstance(d_azim, float), f"azimuth_rad should be float, got {type(d_azim)}"
    assert d_azim > 0.0, "Azimuth resolution should be positive"


@pytest.mark.tier_a
def test_resolve_scanning_resolution_parameter_numeric_degrees() -> None:
    """Numeric float input (degrees) resolves to float radians tuple."""
    from tls2dseg.engines.projection.spherical import resolve_scanning_resolution_parameter

    # Numeric value 0.036 degrees (a common TLS scan resolution)
    params = {"scan_resolution": 0.036}

    class _FakePcd:
        spherical_coordinates = np.zeros((10, 3), dtype=np.float64)

    result = resolve_scanning_resolution_parameter(_FakePcd(), params)
    assert isinstance(result, tuple), "Should return a tuple"
    d_azim, _ = result
    expected_rad = np.radians(0.036)
    assert abs(d_azim - expected_rad) < 1e-9, (
        f"Numeric degrees {0.036} should convert to radians {expected_rad:.6f}, got {d_azim:.6f}"
    )

"""Shared pytest fixtures for the tls2dseg test suite.

Phase 4 plan 05 Task 4 (D-D-06 half 1, VALIDATION Wave 0).

Provides:
- ``fake_projection_engine`` — ``FakeProjectionEngine`` fixture (tier_a safe)
- ``fake_inference_engine`` — ``FakeInferenceEngine`` fixture (tier_a safe)
- ``fake_fusion_engine`` — ``FakeFusionEngine`` fixture (tier_a safe)
- ``synthetic_pcd`` — duck-typed synthetic point cloud with ``spherical_coordinates``,
  ``fov``, and ``scalar_fields`` attributes (tier_b_light: used only in tier_b_light
  piping tests that need realistic 3D geometry for lift_masks_to_pcd).

Design:
- All Fake fixtures import ONLY numpy + stdlib + tls2dseg.types — they are
  tier_a safe (no heavy deps at fixture-call or module-load time).
- ``synthetic_pcd`` uses a duck-typed ``_SyntheticPointCloud`` class so the
  fixture works WITHOUT pchandler installed. This avoids the cudf transitive
  import failure that prevents ``pchandler.geometry.PointCloudData`` from
  loading in the test environment (BUGS-03: np.float_ removed in NumPy 2.0;
  pchandler's geometry.filters.gpu imports cudf at module level).
  The duck-typed class satisfies ``lift_masks_to_pcd``'s interface contract:
  ``pcd.spherical_coordinates``, ``pcd.fov.as_numpy(unit='rad')``, and
  ``pcd.scalar_fields`` with ``__setitem__``/``__getitem__``/``__contains__``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.unit.fakes import FakeFusionEngine, FakeInferenceEngine, FakeProjectionEngine

logger = logging.getLogger("tls2dseg.tests.conftest")


# ─────────────────────────────────────────────────────────────────────────────
# FakeEngine fixtures — tier_a safe (no heavy deps)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_projection_engine() -> FakeProjectionEngine:
    """Return a ``FakeProjectionEngine`` with default (2, 2) image size.

    tier_a safe — imports only numpy + stdlib + tls2dseg.types.
    """
    return FakeProjectionEngine(image_size=(2, 2))


@pytest.fixture
def fake_inference_engine() -> FakeInferenceEngine:
    """Return a ``FakeInferenceEngine`` with 1 deterministic detection.

    tier_a safe — imports only numpy + stdlib + tls2dseg.types.
    """
    return FakeInferenceEngine(n_detections=1)


@pytest.fixture
def fake_fusion_engine() -> FakeFusionEngine:
    """Return a ``FakeFusionEngine`` (all detections in cluster 0).

    tier_a safe — imports only numpy + stdlib + tls2dseg.types.
    """
    return FakeFusionEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic point cloud — duck-typed; no pchandler.geometry required
# ─────────────────────────────────────────────────────────────────────────────


class _FakeFoV:
    """Minimal duck-typed stand-in for ``pchandler.fov.FoV``.

    Only implements ``as_numpy(unit='rad')`` which is all ``lift_masks_to_pcd``
    needs. Returns ``[azimuth_min, elevation_min, azimuth_max, elevation_max]``
    in radians — the same layout as ``pchandler.fov.FoV.as_numpy``.
    """

    def __init__(
        self,
        azimuth_min: float,
        elevation_min: float,
        azimuth_max: float,
        elevation_max: float,
    ) -> None:
        self._values = np.array(
            [azimuth_min, elevation_min, azimuth_max, elevation_max],
            dtype=np.float64,
        )

    def as_numpy(self, unit: str = "rad") -> np.ndarray:
        """Return ``[azimuth_min, elevation_min, azimuth_max, elevation_max]``.

        Only ``unit='rad'`` is implemented (sufficient for ``lift_masks_to_pcd``).
        """
        if unit != "rad":
            raise NotImplementedError(f"_FakeFoV.as_numpy: unit={unit!r} not supported")
        return self._values.copy()


class _ScalarFieldDict(dict):
    """Minimal scalar-field container — just a dict with the same interface.

    ``lift_masks_to_pcd`` calls ``pcd.scalar_fields.__setitem__(name, arr)``
    and later tests check ``"instances" in pcd.scalar_fields`` and
    ``pcd.scalar_fields["instances"]``. A plain dict satisfies all three.
    """


class _SyntheticPointCloud:
    """Duck-typed synthetic point cloud for tier_b_light piping tests.

    Provides the minimal attributes needed by ``lift_masks_to_pcd``:

    - ``.xyz`` — (N, 3) float32 Cartesian coordinates
    - ``.spherical_coordinates`` — (N, 3) float32 array
      (column 0: range, column 1: elevation, column 2: azimuth)
      in the same layout as ``pchandler.geometry.PointCloudData.spherical_coordinates``
    - ``.fov`` — ``_FakeFoV`` with matching azimuth/elevation bounds in radians
    - ``.scalar_fields`` — ``_ScalarFieldDict`` for ``__setitem__``/``__getitem__``/``__contains__``

    No pchandler/pc2img/pyvips/cudf imports. Safe for any test environment.

    Geometry: 20 synthetic points arranged in a small-angle azimuth+elevation
    grid so all points are within the FoV (all valid_indices True in lift).
    """

    def __init__(self, n_points: int = 20, rng_seed: int = 42) -> None:
        rng = np.random.RandomState(rng_seed)

        # Build (N, 3) xyz in a unit hemisphere — scattered in front of scanner
        # (positive x half-space so azimuth is near 0 and elevation is positive).
        x = rng.uniform(0.5, 1.5, n_points).astype(np.float32)
        y = rng.uniform(-0.3, 0.3, n_points).astype(np.float32)
        z = rng.uniform(0.1, 0.5, n_points).astype(np.float32)
        self.xyz = np.column_stack([x, y, z])

        # Compute spherical coordinates exactly as pchandler does:
        #   range = sqrt(x^2 + y^2 + z^2)
        #   elevation = arctan2(sqrt(x^2+y^2), z)  [from z-axis]
        #   azimuth = -arctan2(y, x)
        xy_sq = self.xyz[:, 0] ** 2 + self.xyz[:, 1] ** 2
        sph = np.zeros((n_points, 3), dtype=np.float32)
        sph[:, 0] = np.sqrt(xy_sq + self.xyz[:, 2] ** 2)  # range
        sph[:, 1] = np.arctan2(np.sqrt(xy_sq), self.xyz[:, 2])  # elevation
        sph[:, 2] = -np.arctan2(self.xyz[:, 1], self.xyz[:, 0])  # azimuth
        self.spherical_coordinates = sph

        # FoV = bounding box of the azimuth/elevation distribution (radians)
        azimuth_min = float(sph[:, 2].min())
        azimuth_max = float(sph[:, 2].max())
        elevation_min = float(sph[:, 1].min())
        elevation_max = float(sph[:, 1].max())
        self.fov = _FakeFoV(
            azimuth_min=azimuth_min,
            elevation_min=elevation_min,
            azimuth_max=azimuth_max,
            elevation_max=elevation_max,
        )

        self.scalar_fields: _ScalarFieldDict = _ScalarFieldDict()


@pytest.fixture
def synthetic_pcd() -> _SyntheticPointCloud:
    """Return a duck-typed synthetic point cloud for tier_b_light piping tests.

    Provides ``.xyz``, ``.spherical_coordinates``, ``.fov``, and
    ``.scalar_fields`` — all that ``lift_masks_to_pcd`` needs.

    No pchandler.geometry import required; works in any venv that has numpy.
    """
    return _SyntheticPointCloud(n_points=20, rng_seed=42)

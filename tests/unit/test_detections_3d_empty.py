"""Regression test: all-instances-below-threshold scan returns empty, no crash.

Tier: tier_b_light — imports pchandler.geometry.PointCloudData; skipped in
tier_a (cloud CI) where pchandler is absent. Runs in the local
``nox -s tier_b_light`` session.

CR-06 regression guard. Pre-fix,
``clean_pcd_instances_and_get_detections3d`` called
``PointCloudData.merge_pcd(pcds_clean)`` unconditionally; when
``pcds_clean`` was empty (every instance below ``min_d3d_pcd_point_count``)
that reached ``ScalarFieldManager.merge([])`` →
``set.intersection()`` with no args →
``TypeError: unbound method set.intersection() needs an argument``.

Post-fix, an early-return guard skips ``merge_pcd`` and returns a valid
empty ``Detections3D`` plus a zero-point ``PointCloudData``.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from pchandler.geometry import PointCloudData
except Exception as exc:
    pytest.skip(
        f"tier_b_light test — pchandler.geometry not importable: {exc!r}",
        allow_module_level=True,
    )

from tls2dseg.detections_3d import Detections3D, clean_pcd_instances_and_get_detections3d


@pytest.mark.tier_b_light
def test_all_instances_below_threshold_returns_empty_no_crash() -> None:
    """Every instance has fewer points than min_point_count → no crash, empty result.

    Constructs a PointCloudData with two small instances (10 pts each) and
    sets ``min_d3d_pcd_point_count`` to 50 so both are filtered out.
    Asserts:
    1. No exception raised (pre-fix: TypeError from merge_pcd([])).
    2. Returned Detections3D has zero detections.
    3. Returned cloud is a valid PointCloudData with zero points.
    """
    rng = np.random.RandomState(7)
    # 3 points per instance — the effective threshold is min(min_d3d_pcd_point_count, 4)
    # so 3 pts < 4 means neither instance passes, pcds_clean stays empty
    pts_a = rng.randn(3, 3).astype(np.float32) * 0.1
    pts_b = pts_a + np.array([5.0, 0.0, 0.0], dtype=np.float32)
    pts = np.vstack([pts_a, pts_b]).astype(np.float32)

    instances = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        ]
    )
    classes = np.zeros(6, dtype=np.float32)

    pcd = PointCloudData(
        xyz=pts,
        scalar_fields={"instances": instances, "classes": classes},
    )

    # effective min_npts = min(min_d3d_pcd_point_count, 4) = 4;
    # each instance has 3 pts → all below threshold → pcds_clean stays []
    d3d_parameters = {
        "min_d3d_pcd_point_count": 50,
        "bounding_box_type": "aabb",
        "centroid_type": "mean",
        "preprocess": False,
    }
    pcp_parameters = {
        "keep_confidences": False,
        "output_resolution": 0.01,
    }

    # Pre-fix this raises TypeError via merge_pcd([])
    detections, pcd_clean = clean_pcd_instances_and_get_detections3d(
        pcd, pcd_id=0.0, d3d_parameters=d3d_parameters, pcp_parameters=pcp_parameters
    )

    assert isinstance(detections, Detections3D)
    assert detections.pcd_ids.shape[0] == 0, f"expected 0 detections, got {detections.pcd_ids.shape[0]}"
    assert detections.instances.shape[0] == 0

    assert isinstance(pcd_clean, PointCloudData), f"expected PointCloudData, got {type(pcd_clean)!r}"
    assert pcd_clean.xyz.shape[0] == 0, f"expected zero-point cloud, got {pcd_clean.xyz.shape[0]} points"

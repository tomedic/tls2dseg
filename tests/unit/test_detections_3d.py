"""Unit tests for clean_pcd_instances_and_get_detections3d — BUGS-01 + D-D-05.

Phase 2 regression-locker (02-04) + Phase 4 plan 03 Task 3 (D-D-05).

BUGS-01 (02-04): AABB centroid array overwrite bug — pre-fix code at
``tls2dseg/src/tls2dseg/detections_3d.py`` had the AABB + ``centroid_type=
"bbox_c"`` branch writing ``centroids_d3d = (mx + mn) / 2`` (no ``[i]``
subscript), which overwrites the entire pre-allocated ``(N, 3)`` array with
the last-iteration ``(3,)`` vector.

D-D-05 (04-03): AABB ``pcds_clean`` asymmetry bug — the OBB branch appends
each cleaned pcd_i to pcds_clean, but the AABB branch did not, causing
``PointCloudData.merge_pcd(pcds_clean)`` to crash on an empty list. The fix
adds ``pcds_clean.append(pcd_i)`` in the AABB branch under the same
``preprocess and npts_i > min_npts`` guard as OBB.

Phase 4 plan 03 Task 3 REMOVES the monkeypatch workaround and exercises the
real AABB path — the merge_pcd call must now succeed with a non-empty list.

Marked ``tier_b_light`` — imports ``pchandler.geometry.PointCloudData``.
"""

# Phase 3 D-A4-01: skip the WHOLE module if pchandler can't be imported (cloud
# CI tier_a sandbox installs `pip install -e . --no-deps`; pchandler is absent
# OR fails to import because of numpy 2.0 vs `pchandler.np.float_` carry-over).
# We use a broad try/except instead of pytest.importorskip because the
# pchandler failure mode in this environment is AttributeError (numpy 2.0
# removed np.float_, pchandler still annotates with it), NOT ImportError —
# importorskip only catches ImportError, so a broader pytest.skip is needed
# at module level to make `pytest -m tier_a` collect cleanly across the entire
# tests/unit/ directory. The local `nox -s tier_b_light` session (which
# `pip install -e ../PCHandler --no-deps`s a working PCHandler) will still
# run this regression-locker.
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

from tls2dseg.detections_3d import clean_pcd_instances_and_get_detections3d


@pytest.mark.tier_b_light
def test_aabb_centroid_per_instance_not_overwritten() -> None:
    """Two synthetic AABB clusters → two distinct centroids, not one shared (3,) row.

    Constructs a ``PointCloudData`` with two well-separated point clusters
    (cluster 0 near the origin, cluster 1 offset by ``[10, 0, 0]``) and
    exercises the AABB + ``bbox_c`` branch of
    ``clean_pcd_instances_and_get_detections3d``. The pre-fix bug at
    ``detections_3d.py:165`` (missing ``[i]`` subscript) either raises an
    ``IndexError`` during the trailing ``centroids_d3d[keep_mask]`` filter
    or collapses both centroids onto the last cluster — either failure
    mode is detected by the assertions below.

    With preprocess=False, pcd_i is still appended to pcds_clean (D-D-05 fix
    ensures AABB path is symmetric with OBB), so merge_pcd succeeds.
    This test focuses primarily on the BUGS-01 centroid regression.
    """
    # Deterministic synthetic clusters — RandomState(0) for reproducibility.
    rng = np.random.RandomState(0)
    pts_a = rng.randn(100, 3).astype(np.float32) * 0.5
    pts_b = pts_a + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    pts = np.vstack([pts_a, pts_b]).astype(np.float32)

    # Per-point instance labels: cluster 0 then cluster 1.
    instances = np.concatenate([np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32)])
    classes = np.zeros(200, dtype=np.float32)

    pcd = PointCloudData(
        xyz=pts,
        scalar_fields={"instances": instances, "classes": classes},
    )

    # preprocess=False: skips SOR + DBSCAN steps and skips pcds_clean.append
    # so the legacy merge_pcd([]) issue is not triggered here.
    d3d_parameters = {
        "min_d3d_pcd_point_count": 10,
        "bounding_box_type": "aabb",
        "centroid_type": "bbox_c",
        "preprocess": False,
    }
    pcp_parameters = {
        "keep_confidences": False,
        "output_resolution": 0.01,
    }

    detections, _pcd_clean = clean_pcd_instances_and_get_detections3d(
        pcd, pcd_id=0.0, d3d_parameters=d3d_parameters, pcp_parameters=pcp_parameters
    )

    centroids = detections.centroids
    assert centroids.shape == (2, 3), (
        f"expected (2, 3) centroids, got shape {centroids.shape} — "
        "indicates the AABB+bbox_c branch overwrote the (N,3) array with a (3,) vector"
    )

    # Cluster A and cluster B are separated by [10, 0, 0]. PointCloudData may
    # apply a global coordinate shift internally (to center large coords), so
    # we assert on the RELATIVE separation rather than absolute positions.
    delta = centroids[1] - centroids[0]
    assert np.linalg.norm(delta - np.array([10.0, 0.0, 0.0])) < 1.0, (
        f"expected centroid[1] - centroid[0] ≈ (10, 0, 0); got {delta} — "
        f"centroids={centroids!r}; indicates the AABB+bbox_c branch is "
        "writing to the wrong index (per-instance overwrite bug)."
    )

    # Belt-and-braces: the two centroids must not be identical.
    assert not np.allclose(centroids[0], centroids[1]), (
        f"expected two distinct centroids; both rows equal {centroids[0]!r} — "
        "AABB+bbox_c branch is overwriting the pre-allocated (N,3) array."
    )

    # Also verify bboxes_type is correct
    assert detections.bboxes_type == "aabb", f"Expected bboxes_type='aabb', got {detections.bboxes_type!r}"


@pytest.mark.tier_b_light
def test_aabb_pcds_clean_populated_with_preprocess() -> None:
    """D-D-05: AABB path with preprocess=True populates pcds_clean → merge succeeds.

    This test exercises the D-D-05 bug fix. With preprocess=True and enough
    points, the AABB branch should append cleaned pcd_i to pcds_clean so
    that PointCloudData.merge_pcd(pcds_clean) receives a non-empty list.

    Without the D-D-05 fix this test would crash at merge_pcd with an empty
    list (or an error inside ScalarFieldManager.merge). With the fix, the
    call completes and returns a valid Detections3D.

    No monkeypatch — exercises the real merge_pcd call path.
    """
    rng = np.random.RandomState(42)
    # Two well-separated clusters with enough points to pass preprocess
    pts_a = rng.randn(200, 3).astype(np.float32) * 0.1
    pts_b = pts_a + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    pts = np.vstack([pts_a, pts_b]).astype(np.float32)

    instances = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            np.ones(200, dtype=np.float32),
        ]
    )
    classes = np.zeros(400, dtype=np.float32)

    pcd = PointCloudData(
        xyz=pts,
        scalar_fields={"instances": instances, "classes": classes},
    )

    # preprocess=True + min_d3d_pcd_point_count=10: AABB branch should
    # append pcd_i to pcds_clean when npts_i > min_npts
    d3d_parameters = {
        "min_d3d_pcd_point_count": 10,
        "bounding_box_type": "aabb",
        "centroid_type": "mean",
        "preprocess": True,
    }
    pcp_parameters = {
        "keep_confidences": False,
        "output_resolution": 0.01,
    }

    # This call would crash (empty pcds_clean -> merge_pcd([]) error) without
    # the D-D-05 fix. With the fix it completes successfully.
    detections, pcd_clean = clean_pcd_instances_and_get_detections3d(
        pcd, pcd_id=0.0, d3d_parameters=d3d_parameters, pcp_parameters=pcp_parameters
    )

    assert detections.bboxes_type == "aabb", f"Expected bboxes_type='aabb', got {detections.bboxes_type!r}"
    assert detections.centroids.ndim == 2
    assert detections.centroids.shape[1] == 3
    # merged pcd should be a valid PointCloudData
    assert isinstance(pcd_clean, PointCloudData)

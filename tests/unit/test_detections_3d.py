"""Unit test for BUGS-01 — AABB centroid array overwrite bug.

Phase 2 regression-locker (02-04). Pre-fix code at
``tls2dseg/src/tls2dseg/detections_3d.py`` had the AABB + ``centroid_type=
"bbox_c"`` branch writing ``centroids_d3d = (mx + mn) / 2`` (no ``[i]``
subscript), which overwrites the entire pre-allocated ``(N, 3)`` array with
the last-iteration ``(3,)`` vector. On ``N >= 2`` this either silently
collapses every centroid to the last cluster's centroid, or — because the
trailing ``centroids_d3d = centroids_d3d[keep_mask]`` line indexes a
``(3,)`` array with an ``(N,)`` boolean mask — raises an ``IndexError``.

This test exercises the AABB + ``bbox_c`` branch with two synthetic, well-
separated clusters and asserts the resulting centroids are (a) shape
``(2, 3)`` and (b) distinct, located roughly at the two cluster centers.

The test FAILS on pre-fix code (IndexError or collapsed centroids) and
PASSES once the ``[i]`` subscript is added (Task 3 of 02-04-PLAN.md).

Marked ``tier_a`` per Phase 1 D-17 — lightweight, deterministic, no GPU,
no I/O, no large fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from pchandler.geometry import PointCloudData

from tls2dseg.detections_3d import clean_pcd_instances_and_get_detections3d


@pytest.mark.tier_a
def test_aabb_centroid_per_instance_not_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two synthetic AABB clusters → two distinct centroids, not one shared (3,) row.

    Constructs a ``PointCloudData`` with two well-separated point clusters
    (cluster 0 near the origin, cluster 1 offset by ``[10, 0, 0]``) and
    exercises the AABB + ``bbox_c`` branch of
    ``clean_pcd_instances_and_get_detections3d``. The pre-fix bug at
    ``detections_3d.py:165`` (missing ``[i]`` subscript) either raises an
    ``IndexError`` during the trailing ``centroids_d3d[keep_mask]`` filter
    or collapses both centroids onto the last cluster — either failure
    mode is detected by the assertions below.

    The function unconditionally calls ``PointCloudData.merge_pcd(pcds_clean)``
    after the loop (line 226), but the AABB branch never appends to
    ``pcds_clean`` (only the OBB branch does — a separate pre-existing
    quirk in this file, out of scope for the BUGS-01 surgical fix). To
    isolate the BUGS-01 assertion from that downstream merge step, this
    test monkeypatches ``PointCloudData.merge_pcd`` to a stub that returns
    the input pcd unchanged. Without the monkeypatch, the post-fix call
    would crash inside ``ScalarFieldManager.merge`` on the empty
    ``pcds_clean`` list — which is unrelated to the centroid-overwrite
    bug we're locking.
    """
    # Deterministic synthetic clusters — RandomState(0) for reproducibility.
    rng = np.random.RandomState(0)
    pts_a = rng.randn(100, 3).astype(np.float32) * 0.5
    pts_b = pts_a + np.array([10.0, 0.0, 0.0], dtype=np.float32)
    pts = np.vstack([pts_a, pts_b]).astype(np.float32)

    # Per-point instance labels: cluster 0 then cluster 1. The function uses
    # np.unique() on this field to count detections — two unique IDs → two
    # iterations of the per-instance loop.
    instances = np.concatenate([np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32)])
    # The function reads classes_pcd.data[mask][0] per-instance to populate
    # the Detections3D.classes column — fill with a single class id.
    classes = np.zeros(200, dtype=np.float32)

    pcd = PointCloudData(
        xyz=pts,
        scalar_fields={"instances": instances, "classes": classes},
    )

    # Stub out the trailing PointCloudData.merge_pcd(pcds_clean) call. The
    # AABB branch never appends to pcds_clean (separate quirk — OBB-only
    # bookkeeping), so the live merge_pcd would receive [] and crash inside
    # ScalarFieldManager.merge. We don't care about the merged pcd here; we
    # only assert on the returned Detections3D.centroids.
    monkeypatch.setattr(
        PointCloudData,
        "merge_pcd",
        classmethod(lambda cls, _pcds_clean: pcd),
    )

    # Exercise the buggy code path: AABB bounding box + bbox_c centroid.
    # preprocess=False skips the SOR + DBSCAN steps so the test stays Tier A
    # (no heavy clustering libs touched). keep_confidences=False skips the
    # confidence scalar-field read.
    d3d_parameters = {
        "min_d3d_pcd_point_count": 10,
        "bounding_box_type": "aabb",
        "centroid_type": "bbox_c",
        "preprocess": False,
    }
    pcp_parameters = {
        "keep_confidences": False,
        "output_resolution": 0.01,  # unused when preprocess=False, but read
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
    # we assert on the RELATIVE separation rather than absolute positions —
    # the bug's signature is that both rows collapse to the same point.
    delta = centroids[1] - centroids[0]
    assert np.linalg.norm(delta - np.array([10.0, 0.0, 0.0])) < 1.0, (
        f"expected centroid[1] - centroid[0] ≈ (10, 0, 0); got {delta} — "
        f"centroids={centroids!r}; indicates the AABB+bbox_c branch is "
        "writing to the wrong index (per-instance overwrite bug)."
    )

    # Belt-and-braces: the two centroids must not be identical (the silent-
    # collapse failure mode).
    assert not np.allclose(centroids[0], centroids[1]), (
        f"expected two distinct centroids; both rows equal {centroids[0]!r} — "
        "AABB+bbox_c branch is overwriting the pre-allocated (N,3) array."
    )

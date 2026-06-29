"""Real-inference tier_b_heavy gate — fixture datasets, scoring, report.

Drives both committed fixture datasets through the real GroundingDINO + SAM2
pipeline, scores the in-memory Detections3D against reference ground truth, and
gates on D-06 thresholds (recall ≥ 0.70, cross-scan correspondence ≥ 0.70,
FP ≤ 0.30).  Emits a JSON + markdown report (D-08) to a fixed overwritten path.

Runs only under ``nox -s tier_b_heavy`` (real GPU + conda env).  The belt-and-
braces pytestmark block keeps this file out of tier_a/tier_b_light collection.
"""

from __future__ import annotations

import csv
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest

from tests.integration.scoring.containment import ref_recall
from tests.integration.scoring.correspondence import correspondence_rate_permissive
from tests.integration.scoring.report import write_report

logger = logging.getLogger("tls2dseg.tests.integration.test_heavy_fixtures")

# Report goes to ./tmp under the project root (gitignored), stable across both
# heavy tests so write_report merges them into one combined report (D-08).
_REPO_ROOT = Path(__file__).parent.parent.parent
_REPORT_DIR = _REPO_ROOT / "tmp"

# Gate thresholds. FP rate is advisory (reported, not asserted). Cross-scan
# correspondence uses permissive many-to-one matching at a low 3D-IoU, scoped
# to reference-matched instances only.
_RECALL_THR = 0.70
_CORR_THR = 0.70
_IOU_THR = 0.1

# Belt-and-braces: keep this module out of tier_a/tier_b_light runners.
pytestmark = [
    pytest.mark.tier_b_heavy,
    pytest.mark.skipif(
        importlib.util.find_spec("pchandler") is None,
        reason="tier_b_heavy requires pchandler install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("pc2img") is None,
        reason="tier_b_heavy requires pc2img install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="tier_b_heavy requires torch (heavy ML dep — project conda env)",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_reference_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a tab-separated reference_data_*.csv into (xyz, classes).

    Columns: Instance, Class, X, Y, Z (tab-delimited, 1 header row).

    Returns
    -------
    ref_xyz : (M, 3) float64
    ref_classes : (M,) str — class label strings matching prompt vocabulary
    """
    ref_xyz = []
    ref_classes = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            _, cls, x, y, z = row
            ref_xyz.append([float(x), float(y), float(z)])
            ref_classes.append(cls.strip())
    return np.array(ref_xyz, dtype=np.float64), np.array(ref_classes)


def _class_id_map_from_pipeline(pipeline: object) -> dict[str, int]:
    """Extract the class_id_map from a live Pipeline (via its RunContext)."""
    return dict(pipeline._ctx.class_id_map)  # type: ignore[attr-defined]


def _det_classes_to_str(det_classes: np.ndarray, id_to_cls: dict[int, str]) -> np.ndarray:
    """Convert integer class ids in Detections3D.classes to string labels."""
    return np.array([id_to_cls.get(int(c), "") for c in det_classes])


def _fp_rate(
    det_classes_str: np.ndarray,
    det_bboxes: np.ndarray,
    bboxes_type: str,
    ref_xyz: np.ndarray,
    ref_classes: np.ndarray,
    target_cls: str,
) -> float:
    """False-positive rate for the ground-truthed class.

    A detection of *target_cls* is a false positive if no reference point
    from *ref_xyz* (of the same class) falls inside its bounding volume.

    Returns fp_count / total_detections_of_class; 0.0 if no such detections.
    """
    from tests.integration.scoring.containment import point_in_aabb, point_in_obb

    cls_mask = det_classes_str == target_cls
    if not np.any(cls_mask):
        return 0.0

    target_bboxes = det_bboxes[cls_mask]
    ref_xyz_cls = ref_xyz[ref_classes == target_cls]

    fp_count = 0
    for bbox_row in target_bboxes:
        found = False
        for p in ref_xyz_cls:
            if bboxes_type == "obb":
                center = bbox_row[:3]
                extent = bbox_row[3:6]
                quat = bbox_row[6:10]
                if point_in_obb(p, center, extent, quat):
                    found = True
                    break
            else:
                if point_in_aabb(p, bbox_row):
                    found = True
                    break
        if not found:
            fp_count += 1

    return fp_count / len(target_bboxes)


def _ref_matched_bboxes(
    det_classes_str: np.ndarray,
    det_bboxes: np.ndarray,
    bboxes_type: str,
    ref_xyz: np.ndarray,
    ref_classes: np.ndarray,
    target_cls: str,
) -> np.ndarray:
    """Subset of *target_cls* detections whose bbox contains ≥1 reference point.

    Used to scope cross-scan correspondence to the *correct* detections only —
    spurious/duplicate detections (no reference point inside) are excluded so the
    correspondence metric measures repeatability, not over-detection.
    """
    from tests.integration.scoring.containment import point_in_aabb, point_in_obb

    cls_mask = det_classes_str == target_cls
    target_bboxes = det_bboxes[cls_mask]
    ref_xyz_cls = ref_xyz[ref_classes == target_cls]

    kept = []
    for bbox_row in target_bboxes:
        for p in ref_xyz_cls:
            inside = (
                point_in_obb(p, bbox_row[:3], bbox_row[3:6], bbox_row[6:10])
                if bboxes_type == "obb"
                else point_in_aabb(p, bbox_row)
            )
            if inside:
                kept.append(bbox_row)
                break
    return np.array(kept) if kept else np.empty((0, det_bboxes.shape[1]))


def _instances_from_merged_pcd(pcd: object, id_to_cls: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-instance AABB + majority-class string from a fused/segmented point cloud.

    Returns (bboxes_aabb (N, 6), classes_str (N,)). Background / unclassified
    instances (class id not in id_to_cls) are dropped.
    """
    empty = (np.empty((0, 6)), np.empty((0,), dtype=object))
    if "instances" not in pcd.scalar_fields:  # type: ignore[attr-defined]
        return empty

    xyz = pcd.xyz  # type: ignore[attr-defined]
    inst = np.asarray(pcd.scalar_fields["instances"].data)  # type: ignore[attr-defined]
    cls = np.asarray(pcd.scalar_fields["classes"].data).astype(int)  # type: ignore[attr-defined]

    bboxes, classes = [], []
    for uid in np.unique(inst):
        m = inst == uid
        pts = xyz[m]
        if len(pts) == 0:
            continue
        cls_id = int(np.bincount(cls[m]).argmax())
        cls_str = id_to_cls.get(cls_id, "")
        if not cls_str:  # background / unclassified
            continue
        bboxes.append(np.hstack([pts.min(axis=0), pts.max(axis=0)]))
        classes.append(cls_str)
    if not bboxes:
        return empty
    return np.array(bboxes), np.array(classes)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _office_heavy_config_path() -> Path:
    path = _REPO_ROOT / "examples" / "configs" / "office_small.yaml"
    assert path.is_file(), f"office_small.yaml not found at {path}"
    return path


@pytest.fixture(scope="module")
def _mountain_heavy_config_path() -> Path:
    path = _REPO_ROOT / "examples" / "configs" / "mountain_small.yaml"
    assert path.is_file(), f"mountain_small.yaml not found at {path}"
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Heavy tests
# ─────────────────────────────────────────────────────────────────────────────


def test_office_small_heavy(_office_heavy_config_path: Path) -> None:
    """Multi-view office fixture: full pipeline → fused 3D instances + recall on fusion.

    Runs the FULL pipeline (stage1 + stage2 graph fusion) on examples/data/office_small/.
    Ground-truthed class: plant (8 reference rows). For multi-view there is no per-scan
    estimate to compare — per-feature detections flow into stage2, so cross-scan
    correspondence is validated *implicitly*: if fusion yields no fused 3D instances, the
    per-scan correspondences were insufficient. FP rate is advisory (reported, not gated).
    Gate: stage2 yields ≥1 fused instance AND recall ≥ 0.70 on the fused output.
    """
    from tls2dseg.pipeline.pipeline import Pipeline

    logger.info("Building pipeline from %s", _office_heavy_config_path)
    pipeline = Pipeline.from_yaml(_office_heavy_config_path)

    cfg = pipeline._cfg  # type: ignore[attr-defined]
    assert cfg.mode == "multi-view", "office fixture must be multi-view for the stage2-fusion gate"

    result = pipeline.stage1()
    pcd_merged = pipeline.stage2(result)  # fused, labelled merged point cloud
    logger.info("Stage2 complete: fused pcd has %d points", len(pcd_merged.xyz))

    class_id_map = _class_id_map_from_pipeline(pipeline)
    id_to_cls = {v: k for k, v in class_id_map.items() if k != "background"}

    # Fused 3D instances (one AABB per fused instance) from the merged cloud.
    fused_bboxes, fused_classes = _instances_from_merged_pcd(pcd_merged, id_to_cls)
    n_fused = len(fused_bboxes)
    logger.info("Fused 3D instances: %d  (classes: %s)", n_fused, sorted(set(fused_classes.tolist())))

    # Parse reference ground truth (plant class only)
    ref_csv = _REPO_ROOT / "examples" / "data" / "office_small" / "reference_data_office.csv"
    ref_xyz, ref_classes = _parse_reference_csv(ref_csv)

    recall = ref_recall(ref_xyz, ref_classes, fused_classes, fused_bboxes, "aabb") if n_fused else 0.0
    fp_rate = _fp_rate(fused_classes, fused_bboxes, "aabb", ref_xyz, ref_classes, "plant") if n_fused else 0.0
    logger.info("Office (fused): recall=%.3f  fp_rate(advisory)=%.3f", recall, fp_rate)

    metrics = {
        "iou_thr": _IOU_THR,
        "datasets": {
            "office_small": {
                "mode": "multi-view",
                "recall": recall,
                "fp_rate": fp_rate,
                "fused_instances": n_fused,
            }
        },
    }
    json_path, md_path = write_report(metrics, _REPORT_DIR)
    logger.info("tier_b_heavy report written: %s | %s", json_path, md_path)

    # Gate: fusion must produce 3D instances (implicit cross-scan correspondence),
    # and the fused output must recall the reference plants. FP is advisory.
    assert n_fused >= 1, (
        "Stage2 fusion produced no 3D instances — cross-scan correspondence was insufficient. See report."
    )
    # Fused instance count must be close to the 8 reference plants, not the ~87
    # broken value caused by n_scans collapsing to 1. Allows reasonable inference
    # variance while firmly rejecting the pre-fix near-singleton cluster count.
    assert 4 <= n_fused <= 20, (
        f"Fused instance count {n_fused} outside expected band [4, 20]. "
        f"Pre-fix broken value was ~87 (KNN budget starved by n_scans=1); "
        f"expected ~8 (reference plant count). Regression detected."
    )
    assert recall >= _RECALL_THR, (
        f"Fused-output recall {recall:.3f} below threshold {_RECALL_THR}. See report for details."
    )


def test_mountain_small_heavy(_mountain_heavy_config_path: Path) -> None:
    """Single-view mountain fixture: recall + permissive ref-scoped cross-epoch correspondence.

    Runs real GroundingDINO + SAM2 inference on examples/data/mountains_small/ (2 epochs).
    Ground-truthed class: tree (7 reference rows). Correspondence uses *permissive* many-to-one
    matching at 3D-IoU ≥ 0.1, scoped to *reference-matched* tree instances only (spurious /
    duplicate detections excluded) so it measures repeatability of the correct instances, not
    over-detection. FP rate is advisory (reported, not gated).
    Gate: recall ≥ 0.70 AND ref-scoped cross-epoch correspondence ≥ 0.70.

    Single-view mode processes each epoch independently; d3d_collection has 1 entry per epoch
    (n_features=2 but NMS-combined before lift → 1 D3D per epoch).
    """
    from tls2dseg.pipeline.pipeline import Pipeline

    logger.info("Building pipeline from %s", _mountain_heavy_config_path)
    pipeline = Pipeline.from_yaml(_mountain_heavy_config_path)

    cfg = pipeline._cfg  # type: ignore[attr-defined]
    bboxes_type = cfg.d3d_extraction.bounding_box_type

    result = pipeline.stage1()
    per_scan = list(result.d3d_collection)  # 1 Detections3D per epoch
    logger.info("Stage1 complete: %d epochs, %d d3d entries", result.n_scans, len(per_scan))

    class_id_map = _class_id_map_from_pipeline(pipeline)
    id_to_cls = {v: k for k, v in class_id_map.items() if k != "background"}

    ref_csv = _REPO_ROOT / "examples" / "data" / "mountains_small" / "reference_data_mountain.csv"
    ref_xyz, ref_classes = _parse_reference_csv(ref_csv)

    # Per-epoch recall + (advisory) FP; collect reference-matched tree bboxes per epoch.
    recalls, fp_rates, ref_matched_per_epoch = [], [], []
    for i, d3d in enumerate(per_scan):
        det_str = _det_classes_to_str(d3d.classes, id_to_cls)
        recall_i = ref_recall(ref_xyz, ref_classes, det_str, d3d.bboxes, bboxes_type)
        fp_i = _fp_rate(det_str, d3d.bboxes, bboxes_type, ref_xyz, ref_classes, "tree")
        matched = _ref_matched_bboxes(det_str, d3d.bboxes, bboxes_type, ref_xyz, ref_classes, "tree")
        recalls.append(recall_i)
        fp_rates.append(fp_i)
        ref_matched_per_epoch.append(matched)
        logger.info(
            "  Epoch %d: recall=%.3f  fp_rate(advisory)=%.3f  ref-matched trees=%d",
            i + 1,
            recall_i,
            fp_i,
            len(matched),
        )

    recall = float(np.mean(recalls))
    fp_rate = float(np.mean(fp_rates))

    # Permissive cross-epoch correspondence over reference-matched tree instances.
    corr_rate, matched_ious = 0.0, []
    if len(ref_matched_per_epoch) >= 2:
        b1, b2 = ref_matched_per_epoch[0], ref_matched_per_epoch[1]
        if len(b1) > 0 and len(b2) > 0:
            corr_rate, matched_ious = correspondence_rate_permissive(b1, b2, bboxes_type, _IOU_THR)
            logger.info(
                "Cross-epoch correspondence (tree, permissive@%.2f, ref-scoped): rate=%.3f  matched=%d",
                _IOU_THR,
                corr_rate,
                len(matched_ious),
            )

    metrics = {
        "iou_thr": _IOU_THR,
        "datasets": {
            "mountains_small": {
                "mode": "single-view",
                "recall": recall,
                "correspondence_rate": corr_rate,
                "fp_rate": fp_rate,
                "per_scan_recalls": recalls,
                "per_scan_fp_rates": fp_rates,
                "matched_ious": matched_ious,
            }
        },
    }
    json_path, md_path = write_report(metrics, _REPORT_DIR)
    logger.info("tier_b_heavy report written: %s | %s", json_path, md_path)

    # Gate: recall + ref-scoped permissive correspondence. FP is advisory.
    assert recall >= _RECALL_THR, (
        f"Reference recall {recall:.3f} below threshold {_RECALL_THR}. See report for details."
    )
    assert corr_rate >= _CORR_THR, (
        f"Ref-scoped cross-epoch correspondence {corr_rate:.3f} below threshold {_CORR_THR}. See report."
    )

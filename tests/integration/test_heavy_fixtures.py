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
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.integration.scoring.containment import ref_recall
from tests.integration.scoring.correspondence import correspondence_rate
from tests.integration.scoring.report import write_report

logger = logging.getLogger("tls2dseg.tests.integration.test_heavy_fixtures")

# Report is written to a dedicated temp dir (outside the repo), stable across
# both heavy tests so write_report merges them into one combined report (D-08).
_REPO_ROOT = Path(__file__).parent.parent.parent
_REPORT_DIR = Path(tempfile.gettempdir()) / "tls2dseg_tier_b_heavy"

# Threshold constants (D-06).
_RECALL_THR = 0.70
_CORR_THR = 0.70
_FP_THR = 0.30
_IOU_THR = 0.3

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


def _merge_by_scan(d3d_collection: list, n_scans: int, n_features: int) -> list:
    """Merge per-feature Detections3D entries into one per scan.

    For multi-view with n_features > 1 the collection holds
    n_scans * n_features entries grouped as:
      [scan1_feat0, scan1_feat1, ..., scan2_feat0, scan2_feat1, ...]
    Returns a list of length n_scans with one merged Detections3D per scan.
    """
    from tls2dseg.detections_3d import merge_detections3d

    merged = []
    for scan_idx in range(n_scans):
        start = scan_idx * n_features
        end = start + n_features
        scan_items = [d3d_collection[k] for k in range(start, end) if k < len(d3d_collection)]
        if not scan_items:
            continue
        merged.append(merge_detections3d(scan_items) if len(scan_items) > 1 else scan_items[0])
    return merged


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
    """Multi-view office fixture: recall/correspondence/FP gates + cabinet consistency.

    Runs real GroundingDINO + SAM2 inference on examples/data/office_small/.
    Ground-truthed class: plant (8 reference rows).
    Consistency-only class: cabinet (D-07).
    Thresholds (D-06): recall ≥ 0.70, cross-scan corr ≥ 0.70, FP ≤ 0.30.
    """
    from tls2dseg.pipeline.pipeline import Pipeline

    logger.info("Building pipeline from %s", _office_heavy_config_path)
    pipeline = Pipeline.from_yaml(_office_heavy_config_path)

    cfg = pipeline._cfg  # type: ignore[attr-defined]
    n_features = len(cfg.projection.features)  # 2 (intensity, range)
    bboxes_type = cfg.d3d_extraction.bounding_box_type

    result = pipeline.stage1()
    d3d_collection = result.d3d_collection
    n_scans = result.n_scans
    logger.info("Stage1 complete: %d scans, %d d3d entries", n_scans, len(d3d_collection))

    # Build inverse class map: int -> str (e.g. {1: "plant", 2: "cabinet"})
    class_id_map = _class_id_map_from_pipeline(pipeline)
    id_to_cls = {v: k for k, v in class_id_map.items() if k != "background"}

    # Merge per-feature entries into one Detections3D per scan
    per_scan = _merge_by_scan(d3d_collection, n_scans, n_features)
    logger.info("Merged into %d per-scan collections", len(per_scan))

    # Parse reference ground truth (plant class only)
    ref_csv = _REPO_ROOT / "examples" / "data" / "office_small" / "reference_data_office.csv"
    ref_xyz, ref_classes = _parse_reference_csv(ref_csv)

    # Score each scan independently; aggregate via mean for the final gate
    recalls, fp_rates = [], []
    for i, d3d in enumerate(per_scan):
        det_str = _det_classes_to_str(d3d.classes, id_to_cls)
        recall_i = ref_recall(ref_xyz, ref_classes, det_str, d3d.bboxes, bboxes_type)
        fp_i = _fp_rate(det_str, d3d.bboxes, bboxes_type, ref_xyz, ref_classes, "plant")
        recalls.append(recall_i)
        fp_rates.append(fp_i)
        logger.info("  Scan %d: recall=%.3f  fp_rate=%.3f", i + 1, recall_i, fp_i)

    recall = float(np.mean(recalls))
    fp_rate = float(np.mean(fp_rates))

    # Cross-scan correspondence (3D IoU, Hungarian) on plant detections only
    corr_rate, matched_ious = 0.0, []
    if len(per_scan) >= 2:
        d3d_s1, d3d_s2 = per_scan[0], per_scan[1]
        id_to_cls_local = id_to_cls

        def _filter_cls(d3d: object, cls: str) -> np.ndarray:
            det_str = _det_classes_to_str(d3d.classes, id_to_cls_local)  # type: ignore[arg-type]
            mask = det_str == cls
            return d3d.bboxes[mask]  # type: ignore[attr-defined]

        bboxes_s1 = _filter_cls(d3d_s1, "plant")
        bboxes_s2 = _filter_cls(d3d_s2, "plant")
        if len(bboxes_s1) > 0 and len(bboxes_s2) > 0:
            corr_rate, matched_ious = correspondence_rate(bboxes_s1, bboxes_s2, bboxes_type, _IOU_THR)
            logger.info("Cross-scan correspondence (plant): rate=%.3f  matched=%d", corr_rate, len(matched_ious))

    # Consistency-only class: cabinet (D-07)
    cabinet_counts = []
    cabinet_corr_pairs = 0
    for d3d in per_scan:
        det_str = _det_classes_to_str(d3d.classes, id_to_cls)
        cabinet_counts.append(int(np.sum(det_str == "cabinet")))
    total_cabinet = sum(cabinet_counts)
    logger.info("Cabinet detections per scan: %s  total=%d", cabinet_counts, total_cabinet)

    if len(per_scan) >= 2:
        bboxes_cab1 = _filter_cls(per_scan[0], "cabinet")
        bboxes_cab2 = _filter_cls(per_scan[1], "cabinet")
        if len(bboxes_cab1) > 0 and len(bboxes_cab2) > 0:
            _, cab_ious = correspondence_rate(bboxes_cab1, bboxes_cab2, bboxes_type, _IOU_THR)
            cabinet_corr_pairs = len(cab_ious)

    # Compile and write report (D-08)
    metrics = {
        "iou_thr": _IOU_THR,
        "datasets": {
            "office_small": {
                "recall": recall,
                "correspondence_rate": corr_rate,
                "fp_rate": fp_rate,
                "per_scan_recalls": recalls,
                "per_scan_fp_rates": fp_rates,
                "matched_ious": matched_ious,
                "consistency_only": {
                    "cabinet": {
                        "detected": total_cabinet,
                        "consistent": cabinet_corr_pairs >= 1,
                    }
                },
            }
        },
    }
    json_path, md_path = write_report(metrics, _REPORT_DIR)
    logger.info("tier_b_heavy report written: %s | %s", json_path, md_path)

    # Assertions (D-06)
    assert total_cabinet >= 1, (
        f"D-07: expected ≥1 cabinet detection; got {total_cabinet}. Check prompt or threshold settings."
    )
    assert cabinet_corr_pairs >= 1, f"D-07: expected ≥1 cross-scan cabinet match; got {cabinet_corr_pairs}."
    assert recall >= _RECALL_THR, (
        f"Reference recall {recall:.3f} below threshold {_RECALL_THR}. See report for details."
    )
    assert corr_rate >= _CORR_THR, (
        f"Cross-scan correspondence {corr_rate:.3f} below threshold {_CORR_THR}. See report for details."
    )
    assert fp_rate <= _FP_THR, f"False-positive rate {fp_rate:.3f} above threshold {_FP_THR}. See report for details."


def test_mountain_small_heavy(_mountain_heavy_config_path: Path) -> None:
    """Single-view mountain fixture: recall/correspondence/FP gates + rock consistency.

    Runs real GroundingDINO + SAM2 inference on examples/data/mountains_small/.
    Ground-truthed class: tree (7 reference rows).
    Consistency-only class: rock (D-07).
    Thresholds (D-06): recall ≥ 0.70, cross-scan corr ≥ 0.70, FP ≤ 0.30.

    Note: mountains_small has two epochs (Epoch_1.e57, Epoch_2.e57).
    Single-view mode processes each independently; d3d_collection has 1 entry
    per scan (n_features=2 but NMS-combined before lift → 1 D3D per scan).
    """
    from tls2dseg.pipeline.pipeline import Pipeline

    logger.info("Building pipeline from %s", _mountain_heavy_config_path)
    pipeline = Pipeline.from_yaml(_mountain_heavy_config_path)

    cfg = pipeline._cfg  # type: ignore[attr-defined]
    bboxes_type = cfg.d3d_extraction.bounding_box_type

    result = pipeline.stage1()
    d3d_collection = result.d3d_collection
    n_scans = result.n_scans
    logger.info("Stage1 complete: %d scans, %d d3d entries", n_scans, len(d3d_collection))

    # Single-view: 1 Detections3D per scan (NMS-combined before lift)
    per_scan = list(d3d_collection)

    # Build inverse class map: int -> str
    class_id_map = _class_id_map_from_pipeline(pipeline)
    id_to_cls = {v: k for k, v in class_id_map.items() if k != "background"}

    # Parse reference ground truth (tree class only)
    ref_csv = _REPO_ROOT / "examples" / "data" / "mountains_small" / "reference_data_mountain.csv"
    ref_xyz, ref_classes = _parse_reference_csv(ref_csv)

    # Score each scan independently
    recalls, fp_rates = [], []
    for i, d3d in enumerate(per_scan):
        det_str = _det_classes_to_str(d3d.classes, id_to_cls)
        recall_i = ref_recall(ref_xyz, ref_classes, det_str, d3d.bboxes, bboxes_type)
        fp_i = _fp_rate(det_str, d3d.bboxes, bboxes_type, ref_xyz, ref_classes, "tree")
        recalls.append(recall_i)
        fp_rates.append(fp_i)
        logger.info("  Scan %d: recall=%.3f  fp_rate=%.3f", i + 1, recall_i, fp_i)

    recall = float(np.mean(recalls))
    fp_rate = float(np.mean(fp_rates))

    # Cross-epoch correspondence (3D IoU, Hungarian) on tree detections
    corr_rate, matched_ious = 0.0, []
    id_to_cls_local = id_to_cls

    def _filter_cls(d3d: object, cls: str) -> np.ndarray:
        det_str = _det_classes_to_str(d3d.classes, id_to_cls_local)  # type: ignore[arg-type]
        mask = det_str == cls
        return d3d.bboxes[mask]  # type: ignore[attr-defined]

    if len(per_scan) >= 2:
        bboxes_s1 = _filter_cls(per_scan[0], "tree")
        bboxes_s2 = _filter_cls(per_scan[1], "tree")
        if len(bboxes_s1) > 0 and len(bboxes_s2) > 0:
            corr_rate, matched_ious = correspondence_rate(bboxes_s1, bboxes_s2, bboxes_type, _IOU_THR)
            logger.info("Cross-epoch correspondence (tree): rate=%.3f  matched=%d", corr_rate, len(matched_ious))

    # Consistency-only class: rock (D-07)
    rock_counts = []
    for d3d in per_scan:
        det_str = _det_classes_to_str(d3d.classes, id_to_cls)
        rock_counts.append(int(np.sum(det_str == "rock")))
    total_rock = sum(rock_counts)
    logger.info("Rock detections per scan: %s  total=%d", rock_counts, total_rock)

    rock_corr_pairs = 0
    if len(per_scan) >= 2:
        bboxes_b1 = _filter_cls(per_scan[0], "rock")
        bboxes_b2 = _filter_cls(per_scan[1], "rock")
        if len(bboxes_b1) > 0 and len(bboxes_b2) > 0:
            _, b_ious = correspondence_rate(bboxes_b1, bboxes_b2, bboxes_type, _IOU_THR)
            rock_corr_pairs = len(b_ious)

    # Compile and write report — appended to existing datasets (D-08)
    metrics = {
        "iou_thr": _IOU_THR,
        "datasets": {
            "mountains_small": {
                "recall": recall,
                "correspondence_rate": corr_rate,
                "fp_rate": fp_rate,
                "per_scan_recalls": recalls,
                "per_scan_fp_rates": fp_rates,
                "matched_ious": matched_ious,
                "consistency_only": {
                    "rock": {
                        "detected": total_rock,
                        "consistent": rock_corr_pairs >= 1,
                    }
                },
            }
        },
    }
    json_path, md_path = write_report(metrics, _REPORT_DIR)
    logger.info("tier_b_heavy report written: %s | %s", json_path, md_path)

    # Assertions (D-06)
    assert total_rock >= 1, f"D-07: expected ≥1 rock detection; got {total_rock}. Check prompt or threshold settings."
    assert rock_corr_pairs >= 1, f"D-07: expected ≥1 cross-epoch rock match; got {rock_corr_pairs}."
    assert recall >= _RECALL_THR, (
        f"Reference recall {recall:.3f} below threshold {_RECALL_THR}. See report for details."
    )
    assert corr_rate >= _CORR_THR, (
        f"Cross-epoch correspondence {corr_rate:.3f} below threshold {_CORR_THR}. See report for details."
    )
    assert fp_rate <= _FP_THR, f"False-positive rate {fp_rate:.3f} above threshold {_FP_THR}. See report for details."

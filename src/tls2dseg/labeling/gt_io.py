"""Ground-truth persistence: 2D (instance PNG + COCO JSON + metadata) and 3D (labeled PLY).

Label values in the 3D ``classes`` / ``instances`` fields:

* ``>= 1`` — class id (``classes``) / instance id (``instances``)
* ``0``    — background (labeled, no object)
* ``-1``   — point outside the projection's field of view
* ``-2``   — ignore: outside every labeled region (only when regions are drawn)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("tls2dseg.labeling.gt_io")

INSTANCES_PNG = "gt_instances.png"
COCO_JSON = "gt_coco.json"
META_JSON = "gt_meta.json"

LABEL_VALUES = {
    "-2": "ignore (outside labeled regions)",
    "-1": "outside projection field of view",
    "0": "background",
}


@dataclass
class BoxAnnotation:
    instance_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    mask_source: str = "none"  # none | sam2 | sam2_edited | manual
    mask_feature: str | None = None


def class_id_map(class_names: list[str]) -> dict[str, int]:
    """Class name → id (1-based in the given order, 0 = background), as in the pipeline."""
    return {name: i + 1 for i, name in enumerate(class_names)}


# ─────────────────────────────────────────────────────────────────────────────
# COCO RLE from sparse pixel indices (no dense per-instance masks needed)
# ─────────────────────────────────────────────────────────────────────────────


def _rle_counts_to_string(counts: list[int]) -> str:
    """COCO compressed-RLE string (port of pycocotools ``rleToString``)."""
    out = []
    for i, count in enumerate(counts):
        x = int(count)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def sparse_to_coco_rle(rows: np.ndarray, cols: np.ndarray, h: int, w: int) -> dict:
    """Compressed COCO RLE of the mask given by pixel ``rows``/``cols`` in an ``h`` x ``w`` image."""
    idx = np.unique(np.asarray(cols, dtype=np.int64) * h + np.asarray(rows, dtype=np.int64))
    total = h * w
    if idx.size == 0:
        counts = [total]
    else:
        breaks = np.nonzero(np.diff(idx) != 1)[0]
        starts = idx[np.r_[0, breaks + 1]]
        ends = idx[np.r_[breaks, idx.size - 1]] + 1
        prev_ends = np.r_[0, ends[:-1]]
        counts = np.empty(2 * starts.size + 1, dtype=np.int64)
        counts[0:-1:2] = starts - prev_ends
        counts[1::2] = ends - starts
        counts[-1] = total - ends[-1]
        counts = counts.tolist()
        if counts[-1] == 0:
            counts.pop()
    return {"size": [int(h), int(w)], "counts": _rle_counts_to_string(counts)}


def instance_pixels(instance_img: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Instance id → (rows, cols) of its pixels, for every non-zero id in ``instance_img``."""
    rows, cols = np.nonzero(instance_img)
    ids = instance_img[rows, cols].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    rows, cols, ids = rows[order], cols[order], ids[order]
    uniq, first = np.unique(ids, return_index=True)
    bounds = np.r_[first, ids.size]
    return {int(u): (rows[bounds[k] : bounds[k + 1]], cols[bounds[k] : bounds[k + 1]]) for k, u in enumerate(uniq)}


def regions_mask(regions: list[tuple[float, float, float, float]], img_hw: tuple[int, int]) -> np.ndarray:
    """int32 (H, W) image: 1 inside any xyxy region rectangle, 0 elsewhere."""
    h, w = img_hw
    mask = np.zeros((h, w), dtype=np.int32)
    for x0, y0, x1, y1 in regions:
        c0, c1 = int(np.clip(np.floor(min(x0, x1)), 0, w)), int(np.clip(np.ceil(max(x0, x1)), 0, w))
        r0, r1 = int(np.clip(np.floor(min(y0, y1)), 0, h)), int(np.clip(np.ceil(max(y0, y1)), 0, h))
        mask[r0:r1, c0:c1] = 1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 2D save / load
# ─────────────────────────────────────────────────────────────────────────────


def build_coco(boxes: list[BoxAnnotation], instance_img: np.ndarray, class_names: list[str], image_file: str) -> dict:
    """COCO dict with one image; ``bbox`` is the drawn box, ``segmentation`` the (edited) mask."""
    h, w = instance_img.shape
    ids = class_id_map(class_names)
    pixels = instance_pixels(instance_img)
    annotations = []
    for n, b in enumerate(sorted(boxes, key=lambda b: b.instance_id), start=1):
        x0, y0, x1, y1 = b.box_xyxy
        ann: dict[str, Any] = {
            "id": n,
            "image_id": 1,
            "category_id": ids[b.class_name],
            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            "iscrowd": 0,
            "attributes": {
                "instance_id": b.instance_id,
                "mask_source": b.mask_source,
                "mask_feature": b.mask_feature,
            },
        }
        if b.instance_id in pixels:
            rows, cols = pixels[b.instance_id]
            ann["segmentation"] = sparse_to_coco_rle(rows, cols, h, w)
            ann["area"] = int(rows.size)
        else:
            ann["area"] = float((x1 - x0) * (y1 - y0))
        annotations.append(ann)

    orphans = sorted(set(pixels) - {b.instance_id for b in boxes})
    if orphans:
        logger.warning("Mask pixels without a box are not saved (instance ids %s)", orphans)

    return {
        "images": [{"id": 1, "file_name": image_file, "height": int(h), "width": int(w)}],
        "categories": [{"id": i, "name": name} for name, i in ids.items()],
        "annotations": annotations,
    }


def save_2d(
    gt_dir: Path,
    meta: dict,
    class_names: list[str],
    boxes: list[BoxAnnotation],
    instance_img: np.ndarray,
    regions: list[tuple[float, float, float, float]],
) -> None:
    """Write ``gt_instances.png`` (uint16), ``gt_coco.json`` and ``gt_meta.json`` into ``gt_dir``."""
    import cv2

    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    if instance_img.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("More than 65535 instance ids are not supported")
    if not cv2.imwrite(str(gt_dir / INSTANCES_PNG), instance_img.astype(np.uint16)):
        raise OSError(f"Could not write {gt_dir / INSTANCES_PNG}")

    coco = build_coco(boxes, instance_img, class_names, image_file=INSTANCES_PNG)
    (gt_dir / COCO_JSON).write_text(json.dumps(coco), encoding="utf-8")

    full_meta = {
        **meta,
        "class_names": list(class_names),
        "class_ids": class_id_map(class_names),
        "label_values": LABEL_VALUES,
        "image_hw": list(instance_img.shape),
        "regions_xyxy": [list(map(float, r)) for r in regions],
        "boxes": [asdict(b) for b in boxes],
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (gt_dir / META_JSON).write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
    logger.info("Saved 2D ground truth (%d boxes) to %s", len(boxes), gt_dir)


def load_2d(gt_dir: Path) -> tuple[dict, list[BoxAnnotation], np.ndarray, list[tuple[float, float, float, float]]]:
    """Read back what :func:`save_2d` wrote: (meta, boxes, instance image, regions)."""
    import cv2

    gt_dir = Path(gt_dir)
    meta = json.loads((gt_dir / META_JSON).read_text(encoding="utf-8"))
    instance_img = cv2.imread(str(gt_dir / INSTANCES_PNG), cv2.IMREAD_UNCHANGED)
    if instance_img is None:
        raise OSError(f"Could not read {gt_dir / INSTANCES_PNG}")
    boxes = [
        BoxAnnotation(
            instance_id=int(b["instance_id"]),
            class_name=b["class_name"],
            box_xyxy=tuple(float(v) for v in b["box_xyxy"]),  # type: ignore[arg-type]
            mask_source=b.get("mask_source", "none"),
            mask_feature=b.get("mask_feature"),
        )
        for b in meta.get("boxes", [])
    ]
    regions = [tuple(float(v) for v in r) for r in meta.get("regions_xyxy", [])]
    return meta, boxes, instance_img, regions  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# 3D export
# ─────────────────────────────────────────────────────────────────────────────


def _field_data(pcd: Any, name: str) -> np.ndarray:
    value = pcd.scalar_fields[name]
    return np.asarray(getattr(value, "data", value))


def lift_labels(
    pcd: Any,
    boxes: list[BoxAnnotation],
    instance_img: np.ndarray,
    class_names: list[str],
    regions: list[tuple[float, float, float, float]],
) -> None:
    """Write ``instances`` and ``classes`` scalar fields on ``pcd`` from the 2D labels.

    ``pcd`` must be in the scanner frame the images were projected from.
    """
    from tls2dseg.lifting.masks_to_pcd import lift_mask_to_pcd, lift_masks_to_pcd

    ids = class_id_map(class_names)
    inst = instance_img.astype(np.int32)
    lut = np.zeros(int(inst.max(initial=0)) + 1, dtype=np.int32)
    for b in boxes:
        if b.instance_id < lut.size:
            lut[b.instance_id] = ids[b.class_name]
    sem = lut[inst]
    inst[sem == 0] = 0  # drop pixels of instances without a box

    lift_masks_to_pcd(pcd, inst, sem)
    if regions:
        lift_mask_to_pcd(pcd, regions_mask(regions, inst.shape), "gt_region")
        region = _field_data(pcd, "gt_region")
        del pcd.scalar_fields["gt_region"]
        classes = _field_data(pcd, "classes").copy()
        instances = _field_data(pcd, "instances").copy()
        ignore = (region == 0) & (classes != -1)
        classes[ignore] = -2
        instances[ignore] = -2
        pcd.scalar_fields["classes"] = classes
        pcd.scalar_fields["instances"] = instances


def export_3d(
    gt_dir: Path,
    session: Any,
    boxes: list[BoxAnnotation],
    instance_img: np.ndarray,
    class_names: list[str],
    regions: list[tuple[float, float, float, float]],
) -> Path:
    """Lift the 2D labels onto the full-resolution scan and write ``<name>_gt.ply`` in PRCS."""
    from pchandler.data_io import save_ply
    from pchandler.geometry.transforms import toggle_prcs2socs, toggle_socs2prcs

    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pcd = session.pcd
    lift_labels(pcd, boxes, instance_img, class_names, regions)

    out_path = gt_dir / f"{session.gt_name}_gt.ply"
    fields = ["intensity", "instances", "classes"]
    if pcd.tmat_socs2prcs is None:
        logger.warning("Scan has no pose; writing the ground truth in the scanner frame")
        save_ply(out_path, pcd, retain_colors=True, retain_normals=False, scalar_fields=fields)
    else:
        if not getattr(session, "tmat_compensated", False):
            object.__setattr__(pcd, "tmat_socs2prcs", pcd.tmat_socs2prcs @ np.linalg.inv(session.socs_rotation))
            session.tmat_compensated = True
        toggle_socs2prcs(pcd)
        try:
            save_ply(out_path, pcd, retain_colors=True, retain_normals=False, scalar_fields=fields)
        finally:
            toggle_prcs2socs(pcd)

    inverted = {str(i): name for name, i in class_id_map(class_names).items()}
    inverted["0"] = "background"
    (gt_dir / "class_names_id_map.txt").write_text(json.dumps(inverted), encoding="ascii")

    classes = _field_data(pcd, "classes")
    logger.info(
        "Exported 3D ground truth to %s (%d labeled points, %d instances)",
        out_path,
        int((classes > 0).sum()),
        len(np.unique(_field_data(pcd, "instances")[classes > 0])),
    )
    return out_path

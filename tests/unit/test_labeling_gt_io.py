"""2D ground-truth persistence and label lifting of the ``tls2dseg label`` tool."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tls2dseg.labeling.gt_io import (
    BoxAnnotation,
    build_coco,
    lift_labels,
    load_2d,
    save_2d,
    sparse_to_coco_rle,
)


@pytest.mark.tier_b_light
def test_sparse_rle_matches_pycocotools() -> None:
    mask_util = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(1)
    for density in (0.0, 0.05, 0.5, 1.0):
        for _ in range(20):
            h, w = (int(v) for v in rng.integers(1, 40, 2))
            mask = rng.random((h, w)) < density
            rle = sparse_to_coco_rle(*np.nonzero(mask), h, w)
            ref = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            assert rle["counts"] == ref["counts"].decode()
            decoded = mask_util.decode({"size": rle["size"], "counts": rle["counts"].encode()})
            assert (decoded == mask).all()


@pytest.mark.tier_b_light
def test_save_load_2d_roundtrip(tmp_path) -> None:
    pytest.importorskip("cv2")
    inst = np.zeros((30, 40), dtype=np.uint16)
    inst[2:8, 3:9] = 1
    inst[20:25, 30:38] = 300
    boxes = [
        BoxAnnotation(1, "plant", (3.0, 2.0, 9.0, 8.0), "sam2", "intensity"),
        BoxAnnotation(300, "cabinet", (30.0, 20.0, 38.0, 25.0), "manual"),
        BoxAnnotation(301, "plant", (0.0, 0.0, 5.0, 5.0)),
    ]
    regions = [(0.0, 0.0, 20.0, 30.0)]
    save_2d(tmp_path, {"scan_path": "x.e57"}, ["plant", "cabinet"], boxes, inst, regions)

    meta, boxes2, inst2, regions2 = load_2d(tmp_path)
    assert boxes2 == boxes
    assert np.array_equal(inst2, inst)
    assert regions2 == regions
    assert meta["class_ids"] == {"plant": 1, "cabinet": 2}

    coco = json.loads((tmp_path / "gt_coco.json").read_text())
    anns = {a["attributes"]["instance_id"]: a for a in coco["annotations"]}
    assert anns[1]["category_id"] == 1 and anns[1]["area"] == 36
    assert anns[300]["bbox"] == [30.0, 20.0, 8.0, 5.0]
    assert "segmentation" not in anns[301]


@pytest.mark.tier_b_light
def test_build_coco_skips_orphan_mask_pixels() -> None:
    inst = np.zeros((5, 5), dtype=np.uint16)
    inst[0, 0] = 9
    coco = build_coco([], inst, ["plant"], "img.png")
    assert coco["annotations"] == []


@pytest.mark.tier_b_light
def test_lift_labels_classes_instances_and_ignore(synthetic_pcd) -> None:
    pcd = synthetic_pcd
    h, w = 40, 60
    a0, e0, a1, e1 = pcd.fov.as_numpy(unit="rad")
    u = np.round((pcd.spherical_coordinates[:, 2] - a0) / (a1 - a0) * (w - 1)).astype(int)
    v = np.round((pcd.spherical_coordinates[:, 1] - e0) / (e1 - e0) * (h - 1)).astype(int)

    inst = np.zeros((h, w), dtype=np.uint16)
    inst[v[:3], u[:3]] = 5  # three points belong to instance 5
    inst[v[3], u[3]] = 6  # instance without a box → background
    boxes = [BoxAnnotation(5, "cabinet", (0.0, 0.0, 1.0, 1.0))]
    left = u < w // 2
    regions = [(0.0, 0.0, w / 2 - 0.5, float(h))]

    lift_labels(pcd, boxes, inst, ["plant", "cabinet"], regions)
    classes = np.asarray(pcd.scalar_fields["classes"])
    instances = np.asarray(pcd.scalar_fields["instances"])

    assert "gt_region" not in pcd.scalar_fields
    expected_inst = np.where(inst[v, u] == 5, 5, 0)
    expected_cls = np.where(expected_inst == 5, 2, 0)
    np.testing.assert_array_equal(instances, np.where(left, expected_inst, -2))
    np.testing.assert_array_equal(classes, np.where(left, expected_cls, -2))
    assert (instances[:3][left[:3]] == 5).all()

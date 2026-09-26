"""``tls2dseg label`` on the committed fixture scans: SAM2 box refinement, 3D export, headless app round-trip."""

from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_OFFICE = _REPO_ROOT / "examples" / "data" / "office_small"
_MOUNTAIN_SCAN = _REPO_ROOT / "examples" / "data" / "mountains_small" / "Epoch_1.e57"

pytestmark = [
    pytest.mark.tier_b_heavy,
    pytest.mark.skipif(importlib.util.find_spec("pchandler") is None, reason="requires pchandler"),
    pytest.mark.skipif(importlib.util.find_spec("pc2img") is None, reason="requires pc2img"),
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch"),
]


def _sam2_paths() -> tuple[str, str]:
    from tls2dseg.config.loader import load_config

    cfg = load_config(_REPO_ROOT / "examples" / "configs" / "office_small.yaml")
    checkpoint = Path(cfg.inference.sam2_checkpoint)
    if not checkpoint.is_file():
        pytest.skip(f"SAM2 checkpoint not found: {checkpoint}")
    return str(checkpoint), cfg.inference.sam2_model_config


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _reference_boxes(session, csv_path: Path, size_m: float) -> np.ndarray:
    """xyxy pixel boxes of ``size_m`` around the PRCS reference points of ``csv_path``."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    prcs = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"]), 1.0] for r in rows]).T
    socs = (session.socs_rotation @ np.linalg.inv(session.pcd.tmat_socs2prcs) @ prcs)[:3].T
    rng = np.linalg.norm(socs, axis=1)
    azim = -np.arctan2(socs[:, 1], socs[:, 0])
    elev = np.arctan2(np.hypot(socs[:, 0], socs[:, 1]), socs[:, 2])
    a0, e0, a1, e1 = session.pcd.fov.as_numpy(unit="rad")
    h, w = session.image_hw
    u = (azim - a0) / (a1 - a0) * (w - 1)
    v = (elev - e0) / (e1 - e0) * (h - 1)
    half = size_m / rng / ((a1 - a0) / w) / 2
    return np.unique(np.stack([u - half, v - half, u + half, v + half], axis=1), axis=0)


def test_refine_boxes_segments_reference_plants() -> None:
    from tls2dseg.labeling.sam2_refine import load_sam2_predictor, refine_boxes
    from tls2dseg.labeling.session import load_scan

    checkpoint, model_config = _sam2_paths()
    session = load_scan(_OFFICE / "Scan_1.e57", 0)
    boxes = _reference_boxes(session, _OFFICE / "reference_data_office.csv", size_m=0.35)
    predictor = load_sam2_predictor(checkpoint, model_config, _device())
    masks = refine_boxes(predictor, session.images["intensity"], boxes, _device())

    assert len(masks) == len(boxes)
    for (x0, y0, x1, y1), mask in zip(boxes, masks, strict=True):
        assert len(mask) > 100
        inside = (mask[:, 1] >= x0 - 2) & (mask[:, 1] <= x1 + 2) & (mask[:, 0] >= y0 - 2) & (mask[:, 0] <= y1 + 2)
        assert inside.mean() > 0.95
        assert len(mask) < 0.8 * (x1 - x0) * (y1 - y0)  # an object, not the whole box


def test_export_3d_compensates_projection_rotation(tmp_path) -> None:
    plyfile = pytest.importorskip("plyfile")
    from pchandler.data_io import load_e57
    from pchandler.geometry.transforms import toggle_socs2prcs

    from tls2dseg.labeling.gt_io import BoxAnnotation, export_3d
    from tls2dseg.labeling.session import load_scan

    session = load_scan(_MOUNTAIN_SCAN, 0, {"rotate_pcd": 37.0})
    assert session.projection_params["rotate_pcd"] == pytest.approx(37.0)
    h, w = session.image_hw
    inst = np.zeros((h, w), dtype=np.uint16)
    inst[h // 3 : h // 2, w // 3 : w // 2] = 1
    boxes = [BoxAnnotation(1, "tree", (w / 3, h / 3, w / 2, h / 2), "manual")]

    out = export_3d(tmp_path, session, boxes, inst, ["tree"], [])
    export_3d(tmp_path, session, boxes, inst, ["tree"], [])  # re-export must not drift

    vertex = plyfile.PlyData.read(str(out))["vertex"]
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    ref = load_e57(_MOUNTAIN_SCAN, stay_prcs=False, save_prcs_info=True)
    toggle_socs2prcs(ref)
    ref_xyz = ref.xyz.astype(np.float64)
    if ref.global_coordinate_shift is not None:
        ref_xyz += ref.global_coordinate_shift
    assert np.abs(xyz - ref_xyz).max() < 0.01
    assert (vertex["scalar_classes"] == 1).sum() > 100
    assert json.loads((tmp_path / "class_names_id_map.txt").read_text()) == {"1": "tree", "0": "background"}


def test_labeling_app_headless_roundtrip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("napari")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import napari
    from qtpy.QtWidgets import QApplication

    from tls2dseg.labeling.app import LabelingApp

    def wait(cond, timeout: float = 600.0) -> None:
        start = time.time()
        while not cond():
            QApplication.processEvents()
            time.sleep(0.02)
            assert time.time() - start < timeout, "timed out waiting for the labeling app"

    checkpoint, model_config = _sam2_paths()
    kwargs = {
        "class_names": ["tree", "rock"],
        "out_dir": tmp_path,
        "projection_params": {},
        "sam2_checkpoint": checkpoint,
        "sam2_model_config": model_config,
        "device": _device(),
        "scan_path": _MOUNTAIN_SCAN,
    }
    viewer = napari.Viewer(show=False)
    app = LabelingApp(viewer, **kwargs)
    wait(lambda: app.session is not None and not app.busy)

    app.boxes.add_rectangles([np.array([[100, 200], [150, 260]]), np.array([[160, 300], [200, 360]])])
    app.choose_class("rock")
    app.boxes.add_rectangles([np.array([[50, 400], [90, 450]])])
    assert list(app.boxes.features["instance_id"]) == [1, 2, 3]
    assert list(app.boxes.features["class_name"]) == ["tree", "tree", "rock"]

    app.refine(selected_only=False)
    wait(lambda: not app.busy)
    assert set(app.mask_state) == {1, 2, 3}
    assert (app.inst == 2).any()

    app.boxes.selected_data = {1}
    app.boxes.remove_selected()
    assert not (app.inst == 2).any()
    app.masks.data[0:3, 0:3] = 1  # brush edit of instance 1
    app.save()
    viewer.close()

    meta = json.loads((tmp_path / "Epoch_1" / "gt_meta.json").read_text())
    sources = {b["instance_id"]: b["mask_source"] for b in meta["boxes"]}
    assert sources == {1: "sam2_edited", 3: "sam2"}

    viewer2 = napari.Viewer(show=False)
    app2 = LabelingApp(viewer2, **kwargs)
    wait(lambda: app2.session is not None and not app2.busy)
    assert np.array_equal(app2.inst, app.inst)
    assert list(app2.boxes.features["instance_id"]) == [1, 3]
    assert app2.next_id == 4
    viewer2.close()

"""Stage 1 on a scan rotated during projection must write its results in the scan's own PRCS pose."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent

pytestmark = [
    pytest.mark.tier_b_heavy,
    pytest.mark.skipif(importlib.util.find_spec("pchandler") is None, reason="requires pchandler"),
    pytest.mark.skipif(importlib.util.find_spec("pc2img") is None, reason="requires pc2img"),
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch"),
]


def test_stage1_output_overlays_input_scan_after_forced_rotation(tmp_path) -> None:
    from pchandler.data_io import load_e57
    from pchandler.geometry.transforms import toggle_socs2prcs
    from scipy.spatial import cKDTree

    from tls2dseg.pipeline.pipeline import Pipeline

    cfg = yaml.safe_load((_REPO_ROOT / "examples" / "configs" / "mountain_small.yaml").read_text())
    cfg["io"]["input_path"] = str(_REPO_ROOT / "examples" / "data" / "mountains_small")
    cfg["io"]["output_dir"] = str(tmp_path)
    cfg["projection"]["rotate_pcd"] = 90.0
    config_path = tmp_path / "rotated.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    pipeline = Pipeline.from_yaml(config_path)
    result = pipeline.stage1()

    for scan_path, seg in zip(result.pcd_collection.raw_pcd_paths, result.pcd_collection.seg_pcds, strict=True):
        assert seg is not None and seg.nbPoints > 0
        seg_xyz = seg.xyz.astype(np.float64)
        if seg.global_coordinate_shift is not None:
            seg_xyz += seg.global_coordinate_shift
        ref = load_e57(Path(scan_path), stay_prcs=False, save_prcs_info=True)
        toggle_socs2prcs(ref)
        ref_xyz = ref.xyz.astype(np.float64)
        if ref.global_coordinate_shift is not None:
            ref_xyz += ref.global_coordinate_shift
        # Voxel subsampling moves points by up to ~half a voxel; a missed un-rotation moves them by metres.
        dist, _ = cKDTree(ref_xyz).query(seg_xyz)
        voxel = cfg["preprocessing"]["output_resolution_m"]
        assert np.quantile(dist, 0.95) < voxel, f"{Path(scan_path).name}: segmented points are off the input scan"

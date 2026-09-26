"""Pure-numpy helpers of the ``tls2dseg label`` tool."""

from __future__ import annotations

import numpy as np
import pytest

from tls2dseg.labeling.gt_io import (
    class_id_map,
    instance_pixels,
    regions_mask,
    sparse_to_coco_rle,
)
from tls2dseg.labeling.sam2_refine import crop_window


@pytest.mark.tier_a
def test_crop_window_pads_and_clips() -> None:
    assert crop_window(np.array([100, 200, 180, 240]), (1000, 1000)) == (188, 252, 80, 200)
    # small box grows to min_px, clipped at the image border
    r0, r1, c0, c1 = crop_window(np.array([0, 0, 4, 4]), (50, 1000), min_px=64)
    assert (r0, c0) == (0, 0)
    assert r1 == 34 and c1 == 34
    assert crop_window(np.array([0, 0, 4, 4]), (20, 1000))[1] == 20


@pytest.mark.tier_a
def test_crop_window_degenerate_box_is_non_empty() -> None:
    r0, r1, c0, c1 = crop_window(np.array([999, 499, 999, 499]), (500, 1000))
    assert r1 > r0 and c1 > c0
    assert r1 <= 500 and c1 <= 1000


@pytest.mark.tier_a
def test_sparse_to_coco_rle_known_values() -> None:
    # 3x2 image, column-major pixels: (0,0) (1,0) (2,0) | (0,1) (1,1) (2,1); mask = (1,0), (2,0), (0,1)
    rle = sparse_to_coco_rle(np.array([1, 2, 0]), np.array([0, 0, 1]), 3, 2)
    assert rle["size"] == [3, 2]
    assert rle["counts"] == "13" + "2"  # counts [1, 3, 2]: 0-run 1, 1-run 3, 0-run 2
    assert sparse_to_coco_rle(np.array([], dtype=int), np.array([], dtype=int), 2, 2)["counts"] == "4"


@pytest.mark.tier_a
def test_instance_pixels_groups_by_id() -> None:
    img = np.zeros((4, 5), dtype=np.uint16)
    img[0, 1] = 3
    img[2, 2:4] = 7
    pixels = instance_pixels(img)
    assert set(pixels) == {3, 7}
    assert pixels[3][0].tolist() == [0] and pixels[3][1].tolist() == [1]
    assert sorted(pixels[7][1].tolist()) == [2, 3]


@pytest.mark.tier_a
def test_regions_mask_union_and_clipping() -> None:
    mask = regions_mask([(1, 1, 3, 2), (8, 3, 20, 20)], (5, 10))
    assert mask.dtype == np.int32
    assert mask[1, 1:3].tolist() == [1, 1]
    assert mask[3:, 8:].all()
    assert mask.sum() == 2 + 4


@pytest.mark.tier_a
def test_class_id_map_is_one_based_in_order() -> None:
    assert class_id_map(["plant", "cabinet"]) == {"plant": 1, "cabinet": 2}


@pytest.mark.tier_a
def test_label_cmd_help_and_missing_napari(monkeypatch) -> None:
    import importlib.util
    import re

    from typer.testing import CliRunner

    from tls2dseg.cli import app

    result = CliRunner().invoke(app, ["label", "--help"])
    assert result.exit_code == 0
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    for flag in ("--scan", "--classes", "--config", "--out", "--sam2-checkpoint"):
        assert flag in out

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "napari" else real_find_spec(name, *a)
    )
    result = CliRunner().invoke(app, ["label", "--classes", "plant"])
    assert result.exit_code == 1
    assert "tls2dseg[label]" in result.output

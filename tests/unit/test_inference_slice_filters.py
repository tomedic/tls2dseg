"""Unit tests for the drop_incomplete_slices toggle in shared.check_if_img_slice_complete.

Tier: tier_a — stdlib + numpy only; no torch/sam2/supervision/pchandler at import time.
"""

from __future__ import annotations

import numpy as np
import pytest

from tls2dseg.engines.inference.shared import check_if_img_slice_complete


@pytest.mark.tier_a
def test_complete_slice_accepted_regardless_of_toggle() -> None:
    """A slice whose shape matches expected is accepted whether toggle is True or False."""
    img = np.zeros((644, 644, 3), dtype=np.uint8)
    params_true = {"slice_width_height": (644, 644), "drop_incomplete_slices": True}
    params_false = {"slice_width_height": (644, 644), "drop_incomplete_slices": False}
    assert check_if_img_slice_complete(img, params_true) is True
    assert check_if_img_slice_complete(img, params_false) is True


@pytest.mark.tier_a
def test_clamped_slice_dropped_when_toggle_true() -> None:
    """Test B (drop_incomplete_slices=True): clamped slice (301x586) with expected 644 -> False."""
    img = np.zeros((301, 586, 3), dtype=np.uint8)
    params = {"slice_width_height": (644, 644), "drop_incomplete_slices": True}
    assert check_if_img_slice_complete(img, params) is False


@pytest.mark.tier_a
def test_clamped_slice_accepted_when_toggle_false() -> None:
    """Test B (drop_incomplete_slices=False): clamped slice is accepted regardless of size mismatch."""
    img = np.zeros((301, 586, 3), dtype=np.uint8)
    params = {"slice_width_height": (644, 644), "drop_incomplete_slices": False}
    assert check_if_img_slice_complete(img, params) is True


@pytest.mark.tier_a
def test_missing_key_defaults_to_drop_true_behaviour() -> None:
    """drop_incomplete_slices key absent -> defaults True (current behaviour preserved)."""
    img = np.zeros((301, 586, 3), dtype=np.uint8)
    params = {"slice_width_height": (644, 644)}
    assert check_if_img_slice_complete(img, params) is False

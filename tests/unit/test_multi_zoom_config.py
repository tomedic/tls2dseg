"""Wave-0 tier_a tests for ClassSpec + MultiZoomConfig (MZ-01 / MZ-06 / MZ-10).

Covers:
- ClassSpec and MultiZoomConfig are frozen + extra=forbid.
- len(sizes_m) cross-validation raises ValidationError on mismatch;
  accepts a correctly-sized pair.
- sizes_m rejects non-positive values.
- footprint_band_frac rejects p_min >= p_max and out-of-(0,1) values.
- cross_class_iou_threshold rejects values outside [0, 1].
- YAML round-trip with a multi_zoom block preserves mode, classes, and
  tuning fields.
- Typo'd key under multi_zoom raises ValidationError (extra=forbid).

No torch / pchandler / pc2img imports.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config.models import ClassSpec, MultiZoomConfig

# ─────────────────────────────────────────────────────────────────────────────
# Frozen + extra=forbid invariants
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_class_spec_frozen_and_extra_forbid() -> None:
    """ClassSpec advertises frozen=True + extra=forbid."""
    assert ClassSpec.model_config.get("frozen") is True
    assert ClassSpec.model_config.get("extra") == "forbid"


@pytest.mark.tier_a
def test_multi_zoom_config_frozen_and_extra_forbid() -> None:
    """MultiZoomConfig advertises frozen=True + extra=forbid."""
    assert MultiZoomConfig.model_config.get("frozen") is True
    assert MultiZoomConfig.model_config.get("extra") == "forbid"


# ─────────────────────────────────────────────────────────────────────────────
# ClassSpec — len(sizes_m) cross-validation (D-CFG-01)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_class_spec_len_cross_validation() -> None:
    """len(sizes_m) must equal number of class tokens in text_prompt."""
    with pytest.raises(ValidationError, match="sizes_m has 1 entries but text_prompt has 2"):
        ClassSpec(text_prompt="chair . table .", sizes_m=[1.0])


@pytest.mark.tier_a
def test_class_spec_len_cross_validation_accepts_matching_pair() -> None:
    """A correctly-matched text_prompt + sizes_m constructs without error."""
    cs = ClassSpec(text_prompt="chair . table .", sizes_m=[0.9, 1.2])
    assert cs.text_prompt == "chair . table ."
    assert cs.sizes_m == [0.9, 1.2]


@pytest.mark.tier_a
def test_class_spec_len_cross_validation_single_class() -> None:
    """Single class: one token, one size — valid."""
    cs = ClassSpec(text_prompt="tree .", sizes_m=[3.0])
    assert cs.sizes_m == [3.0]


# ─────────────────────────────────────────────────────────────────────────────
# ClassSpec — sizes_m non-positive rejection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_class_spec_sizes_m_rejects_zero() -> None:
    """sizes_m rejects zero values (gt=0 contract)."""
    with pytest.raises(ValidationError):
        ClassSpec(text_prompt="chair .", sizes_m=[0.0])


@pytest.mark.tier_a
def test_class_spec_sizes_m_rejects_negative() -> None:
    """sizes_m rejects negative values."""
    with pytest.raises(ValidationError):
        ClassSpec(text_prompt="chair . table .", sizes_m=[-1.0, 1.2])


# ─────────────────────────────────────────────────────────────────────────────
# MultiZoomConfig — footprint_band_frac validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_multi_zoom_footprint_band_rejects_equal_bounds() -> None:
    """footprint_band_frac rejects p_min == p_max."""
    with pytest.raises(ValidationError, match="0 < p_min < p_max < 1"):
        MultiZoomConfig(footprint_band_frac=(0.2, 0.2))


@pytest.mark.tier_a
def test_multi_zoom_footprint_band_rejects_inverted() -> None:
    """footprint_band_frac rejects p_min > p_max."""
    with pytest.raises(ValidationError, match="0 < p_min < p_max < 1"):
        MultiZoomConfig(footprint_band_frac=(0.5, 0.1))


@pytest.mark.tier_a
def test_multi_zoom_footprint_band_rejects_out_of_unit_interval() -> None:
    """footprint_band_frac rejects values outside (0, 1)."""
    with pytest.raises(ValidationError):
        MultiZoomConfig(footprint_band_frac=(0.0, 0.5))
    with pytest.raises(ValidationError):
        MultiZoomConfig(footprint_band_frac=(0.1, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# MultiZoomConfig — cross_class_iou_threshold bounds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_multi_zoom_cross_class_iou_rejects_above_one() -> None:
    """cross_class_iou_threshold rejects values > 1."""
    with pytest.raises(ValidationError):
        MultiZoomConfig(cross_class_iou_threshold=1.5)


@pytest.mark.tier_a
def test_multi_zoom_cross_class_iou_rejects_below_zero() -> None:
    """cross_class_iou_threshold rejects values < 0."""
    with pytest.raises(ValidationError):
        MultiZoomConfig(cross_class_iou_threshold=-0.1)


@pytest.mark.tier_a
def test_multi_zoom_cross_class_iou_accepts_boundary_values() -> None:
    """cross_class_iou_threshold accepts 0.0 and 1.0 (ge/le bounds)."""
    mz = MultiZoomConfig(cross_class_iou_threshold=0.0)
    assert mz.cross_class_iou_threshold == 0.0
    mz2 = MultiZoomConfig(cross_class_iou_threshold=1.0)
    assert mz2.cross_class_iou_threshold == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# YAML round-trip with multi_zoom block
# ─────────────────────────────────────────────────────────────────────────────

_MULTI_ZOOM_YAML = textwrap.dedent("""\
    mode: single-view
    io:
      input_path: /tmp/tls2dseg_test_input
      output_dir: /tmp/tls2dseg_test_output
    prompt:
      text: chair . table
    preprocessing:
      output_resolution_m: 0.05
    projection:
      features: [intensity]
    inference:
      type: grounded_sam2_hf
      sam2_hf_model_id: facebook/sam2.1-hiera-large
      multi_zoom:
        mode: single-zoom
        classes:
          text_prompt: "chair . table ."
          sizes_m: [0.5, 1.2]
        cross_class_iou_threshold: 0.65
        footprint_band_frac: [0.08, 0.20]
        range_percentiles: [5.0, 95.0]
        ios_enabled: true
    d3d_extraction: {}
    fusion: {}
""")


@pytest.mark.tier_a
def test_yaml_round_trip_multi_zoom_block(tmp_path: Path) -> None:
    """YAML with a multi_zoom block round-trips: mode, classes, tuning fields preserved."""
    from tls2dseg.config.loader import load_config

    yaml_path = tmp_path / "test_multi_zoom.yaml"
    yaml_path.write_text(_MULTI_ZOOM_YAML)

    cfg = load_config(yaml_path)

    mz = cfg.inference.multi_zoom
    assert mz.mode == "single-zoom"
    assert mz.classes is not None
    assert mz.classes.text_prompt == "chair . table ."
    assert mz.classes.sizes_m == [0.5, 1.2]
    assert mz.cross_class_iou_threshold == pytest.approx(0.65)
    assert mz.footprint_band_frac == pytest.approx((0.08, 0.20))
    assert mz.range_percentiles == pytest.approx((5.0, 95.0))
    assert mz.ios_enabled is True


# ─────────────────────────────────────────────────────────────────────────────
# extra=forbid: typo'd key under multi_zoom
# ─────────────────────────────────────────────────────────────────────────────

_TYPO_KEY_YAML = textwrap.dedent("""\
    mode: single-view
    io:
      input_path: /tmp/tls2dseg_test_input
      output_dir: /tmp/tls2dseg_test_output
    prompt:
      text: chair
    preprocessing:
      output_resolution_m: 0.05
    projection:
      features: [intensity]
    inference:
      type: grounded_sam2_hf
      sam2_hf_model_id: facebook/sam2.1-hiera-large
      multi_zoom:
        mod: multi-zoom
    d3d_extraction: {}
    fusion: {}
""")


@pytest.mark.tier_a
def test_typo_key_under_multi_zoom_raises_validation_error(tmp_path: Path) -> None:
    """A typo'd key under multi_zoom raises ValidationError (extra=forbid)."""
    from tls2dseg.config.loader import load_config

    yaml_path = tmp_path / "typo_key.yaml"
    yaml_path.write_text(_TYPO_KEY_YAML)

    with pytest.raises(ValidationError):
        load_config(yaml_path)

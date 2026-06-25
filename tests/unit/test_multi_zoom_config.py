"""Tier-A tests for PromptConfig.sizes_m + MultiZoomConfig + RunConfig cross-validation.

Covers:
- MultiZoomConfig is frozen + extra=forbid.
- MultiZoomConfig.active defaults False; toggles to True.
- PromptConfig.sizes_m length/positivity validation.
- RunConfig cross-validation: active=true + no sizes -> ValueError;
  active=true + length mismatch -> ValueError; active=true + valid -> ok;
  active=false + no sizes -> ok.
- footprint_band_frac rejects p_min >= p_max and out-of-(0,1) values.
- cross_class_iou_threshold rejects values outside [0, 1].
- YAML round-trip with a multi_zoom block preserves active and tuning fields.
- Typo'd key under multi_zoom raises ValidationError (extra=forbid).

No torch / pchandler / pc2img imports.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config.models import MultiZoomConfig, PromptConfig

# ─────────────────────────────────────────────────────────────────────────────
# Frozen + extra=forbid invariants
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_multi_zoom_config_frozen_and_extra_forbid() -> None:
    """MultiZoomConfig advertises frozen=True + extra=forbid."""
    assert MultiZoomConfig.model_config.get("frozen") is True
    assert MultiZoomConfig.model_config.get("extra") == "forbid"


# ─────────────────────────────────────────────────────────────────────────────
# MultiZoomConfig.active — default + toggle
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_multi_zoom_active_default_false() -> None:
    """MultiZoomConfig.active defaults to False (single-zoom by default)."""
    mz = MultiZoomConfig()
    assert mz.active is False


@pytest.mark.tier_a
def test_multi_zoom_active_toggle_true() -> None:
    """MultiZoomConfig.active can be set to True."""
    mz = MultiZoomConfig(active=True)
    assert mz.active is True


# ─────────────────────────────────────────────────────────────────────────────
# PromptConfig.sizes_m — length / positivity validation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_prompt_sizes_m_accepts_none() -> None:
    """sizes_m=None (default) constructs PromptConfig without error."""
    p = PromptConfig(text="chair . table")
    assert p.sizes_m is None


@pytest.mark.tier_a
def test_prompt_sizes_m_accepts_matching_pair() -> None:
    """sizes_m with correct length constructs PromptConfig without error."""
    p = PromptConfig(text="chair . table", sizes_m=[0.9, 1.2])
    assert p.sizes_m == [0.9, 1.2]


@pytest.mark.tier_a
def test_prompt_sizes_m_accepts_single_class() -> None:
    """Single class token: one size — valid."""
    p = PromptConfig(text="tree", sizes_m=[3.0])
    assert p.sizes_m == [3.0]


@pytest.mark.tier_a
def test_prompt_sizes_m_rejects_length_mismatch() -> None:
    """sizes_m with wrong length raises ValidationError."""
    with pytest.raises(ValidationError, match="sizes_m"):
        PromptConfig(text="chair . table", sizes_m=[1.0])


@pytest.mark.tier_a
def test_prompt_sizes_m_rejects_zero() -> None:
    """sizes_m rejects zero values (all values must be > 0)."""
    with pytest.raises(ValidationError):
        PromptConfig(text="chair", sizes_m=[0.0])


@pytest.mark.tier_a
def test_prompt_sizes_m_rejects_negative() -> None:
    """sizes_m rejects negative values."""
    with pytest.raises(ValidationError):
        PromptConfig(text="chair . table", sizes_m=[-1.0, 1.2])


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig cross-validation (active=true requires sizes_m)
# ─────────────────────────────────────────────────────────────────────────────

_BASE_CFG: dict = {
    "mode": "single-view",
    "io": {
        "input_path": "/tmp/tls2dseg_test_input",
        "output_dir": "/tmp/tls2dseg_test_output",
    },
    "preprocessing": {"output_resolution_m": 0.05},
    "projection": {"features": ["intensity"]},
    "d3d_extraction": {},
    "fusion": {},
}


def _make_cfg_dict(**prompt_overrides: object) -> dict:
    import copy

    cfg = copy.deepcopy(_BASE_CFG)
    cfg["inference"] = {
        "type": "grounded_sam2_hf",
        "sam2_hf_model_id": "facebook/sam2.1-hiera-large",
    }
    cfg["prompt"] = {"text": "tree. pole.", **prompt_overrides}
    return cfg


@pytest.mark.tier_a
def test_run_config_active_false_no_sizes_ok() -> None:
    """active=false + no sizes_m -> valid config."""
    from tls2dseg.config.models import RunConfig

    cfg_dict = _make_cfg_dict()
    cfg_dict["inference"]["multi_zoom"] = {"active": False}
    cfg = RunConfig(**cfg_dict)
    assert cfg.inference.multi_zoom.active is False
    assert cfg.prompt.sizes_m is None


@pytest.mark.tier_a
def test_run_config_active_true_no_sizes_raises() -> None:
    """active=true + no sizes_m -> ValueError (fail-fast)."""
    from tls2dseg.config.models import RunConfig

    cfg_dict = _make_cfg_dict()
    cfg_dict["inference"]["multi_zoom"] = {"active": True}
    with pytest.raises(ValidationError, match="sizes_m"):
        RunConfig(**cfg_dict)


@pytest.mark.tier_a
def test_run_config_active_true_length_mismatch_raises() -> None:
    """active=true + sizes_m length != class tokens -> ValueError."""
    from tls2dseg.config.models import RunConfig

    cfg_dict = _make_cfg_dict(sizes_m=[5.0])  # "tree. pole." has 2 tokens; 1 size given
    cfg_dict["inference"]["multi_zoom"] = {"active": True}
    with pytest.raises(ValidationError, match="sizes_m"):
        RunConfig(**cfg_dict)


@pytest.mark.tier_a
def test_run_config_active_true_valid_sizes_ok() -> None:
    """active=true + sizes_m matching class tokens -> valid config."""
    from tls2dseg.config.models import RunConfig

    cfg_dict = _make_cfg_dict(sizes_m=[5.0, 0.1])  # "tree. pole." has 2 tokens
    cfg_dict["inference"]["multi_zoom"] = {"active": True}
    cfg = RunConfig(**cfg_dict)
    assert cfg.inference.multi_zoom.active is True
    assert cfg.prompt.sizes_m == [5.0, 0.1]


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
      sizes_m: [0.5, 1.2]
    preprocessing:
      output_resolution_m: 0.05
    projection:
      features: [intensity]
    inference:
      type: grounded_sam2_hf
      sam2_hf_model_id: facebook/sam2.1-hiera-large
      multi_zoom:
        active: true
        cross_class_iou_threshold: 0.65
        footprint_band_frac: [0.08, 0.20]
        range_percentiles: [5.0, 95.0]
        ios_enabled: true
    d3d_extraction: {}
    fusion: {}
""")


@pytest.mark.tier_a
def test_yaml_round_trip_multi_zoom_block(tmp_path: Path) -> None:
    """YAML with a multi_zoom block round-trips: active and tuning fields preserved."""
    from tls2dseg.config.loader import load_config

    yaml_path = tmp_path / "test_multi_zoom.yaml"
    yaml_path.write_text(_MULTI_ZOOM_YAML)

    cfg = load_config(yaml_path)

    mz = cfg.inference.multi_zoom
    assert mz.active is True
    assert mz.cross_class_iou_threshold == pytest.approx(0.65)
    assert mz.footprint_band_frac == pytest.approx((0.08, 0.20))
    assert mz.range_percentiles == pytest.approx((5.0, 95.0))
    assert mz.ios_enabled is True
    assert cfg.prompt.sizes_m == [0.5, 1.2]


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
        actve: true
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


@pytest.mark.tier_a
def test_multi_zoom_overview_pass_default_true() -> None:
    """MultiZoomConfig.overview_pass defaults to True (always-on overview)."""
    mz = MultiZoomConfig()
    assert mz.overview_pass is True


@pytest.mark.tier_a
def test_multi_zoom_overview_pass_toggle_false() -> None:
    """MultiZoomConfig.overview_pass can be disabled."""
    mz = MultiZoomConfig(overview_pass=False)
    assert mz.overview_pass is False

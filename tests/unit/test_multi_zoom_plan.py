"""Unit tests for multi_zoom_plan.py — ZoomPass + compute_zoom_passes.

Encodes the locked 1-D interval-cover-in-log-footprint-space algorithm:
  - Each class spans [fp_far, fp_near] in native px.
  - F_lo / F_hi = global min far / max near across all classes (unclamped).
  - n_s = ceil(log(F_hi/F_lo) / log(K)), capped at max_zoom_passes.
  - Band b covers [F_lo*K^b, F_lo*K^(b+1)]; center fc_b = F_lo*K^(b+0.5).
  - Class c is in band b iff fp_far[c] < band_hi AND fp_near[c] > band_lo.
  - Z_b = target_effective / fc_b:
      Z_b <= 1 -> downscale path (resize=Z_b, tile=model_short_side)
      Z_b > 1  -> small-object path (resize=1.0, tile=round(model_short_side/Z_b))
  - Always-on overview pass (needs_tiling=False) first in returned list.

Tier: tier_a — stdlib + numpy only; no torch/sam2/supervision/pchandler at import time.
"""

from __future__ import annotations

import logging
import math

import pytest

from tls2dseg.engines.inference.multi_zoom_plan import (
    ZoomPass,
    compute_zoom_passes,
    engine_short_side,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants matching defaults
# ─────────────────────────────────────────────────────────────────────────────

_P_MIN_FRAC = 0.075
_P_MAX_FRAC = 0.225
_MODEL_SHORT_SIDE = 800
_P_MIN = _P_MIN_FRAC * _MODEL_SHORT_SIDE  # 60
_P_MAX = _P_MAX_FRAC * _MODEL_SHORT_SIDE  # 180
_K = _P_MAX / _P_MIN  # 3.0
_TARGET_EFF = math.sqrt(_P_MIN * _P_MAX)  # ~103.9


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _passes(
    class_sizes: dict[str, float],
    d_azim_rad: float,
    range_near_m: float,
    range_far_m: float | None = None,
    *,
    p_min_frac: float = _P_MIN_FRAC,
    p_max_frac: float = _P_MAX_FRAC,
    model_short_side: int = _MODEL_SHORT_SIDE,
    max_zoom_passes: int = 6,
) -> list[ZoomPass]:
    return compute_zoom_passes(
        class_sizes=class_sizes,
        d_azim_rad=d_azim_rad,
        range_near_m=range_near_m,
        range_far_m=range_far_m if range_far_m is not None else range_near_m,
        p_min_frac=p_min_frac,
        p_max_frac=p_max_frac,
        model_short_side=model_short_side,
        max_zoom_passes=max_zoom_passes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Oracle / worked example (the canonical spec)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_oracle_wheat_leaf_footprints() -> None:
    """Verify fp_near/fp_far values for oracle case match locked spec."""
    d_azim = 1.442283e-4
    range_near = 1.51
    range_far = 11.87
    # fp_near = size_m / (range_near * d_azim_rad)
    fp_near_wheat = 0.1 / (range_near * d_azim)
    fp_near_leaf = 0.2 / (range_near * d_azim)
    fp_far_wheat = 0.1 / (range_far * d_azim)
    fp_far_leaf = 0.2 / (range_far * d_azim)
    assert 450 < fp_near_wheat < 470, f"fp_near wheat expected ~459, got {fp_near_wheat:.1f}"
    assert 900 < fp_near_leaf < 930, f"fp_near leaf expected ~918, got {fp_near_leaf:.1f}"
    assert 55 < fp_far_wheat < 62, f"fp_far wheat expected ~58, got {fp_far_wheat:.1f}"
    assert 110 < fp_far_leaf < 124, f"fp_far leaf expected ~117, got {fp_far_leaf:.1f}"


@pytest.mark.tier_a
def test_oracle_n_bands() -> None:
    """Oracle case: F_lo~58.4, F_hi~918 → n_s == 3."""
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        max_zoom_passes=6,
    )
    tiled = [p for p in passes if p.needs_tiling]
    assert len(tiled) == 3, f"Oracle case: expected 3 tiled passes, got {len(tiled)}"
    assert len(passes) == 4, f"Oracle case: expected 4 total passes (1 overview + 3), got {len(passes)}"


@pytest.mark.tier_a
def test_oracle_overview_pass_first_and_both_classes() -> None:
    """Oracle: overview pass is index 0, has both classes, needs_tiling=False."""
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    ov = passes[0]
    assert not ov.needs_tiling
    assert ov.tile_size_px is None
    assert ov.overlap_px is None
    assert "wheat" in ov.class_names
    assert "leaf" in ov.class_names


@pytest.mark.tier_a
def test_oracle_band0_small_object_path() -> None:
    """Oracle band 0: center ~101 < target_eff=103.9 → Z~1.03 > 1 → small-object path.
    resize_factor >= 0.95, tile_size_px ~779 (in [700, 800]).
    Both wheat and leaf are members (both fp intervals overlap band 0).
    """
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    # tiled passes ordered fine->coarse (band 0 first)
    tiled = [p for p in passes if p.needs_tiling]
    b0 = tiled[0]
    assert b0.resize_factor >= 0.95, f"Band 0 small-object path: resize_factor must be ~1.0, got {b0.resize_factor}"
    assert b0.tile_size_px is not None
    assert 700 <= b0.tile_size_px <= 800, f"Band 0 tile ~779 expected, got {b0.tile_size_px}"
    assert "wheat" in b0.class_names, "wheat must be in band 0"
    assert "leaf" in b0.class_names, "leaf must be in band 0"


@pytest.mark.tier_a
def test_oracle_band1_downscale_path() -> None:
    """Oracle band 1: center ~304 > target_eff → Z~0.34 <= 1 → downscale path.
    resize_factor in [0.30, 0.38], tile_size_px == model_short_side == 800.
    Both wheat and leaf are members.
    """
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    tiled = [p for p in passes if p.needs_tiling]
    b1 = tiled[1]
    assert 0.30 <= b1.resize_factor <= 0.38, f"Band 1: resize_factor expected ~0.34, got {b1.resize_factor:.4f}"
    assert b1.tile_size_px == _MODEL_SHORT_SIDE, f"Band 1: tile=800 expected, got {b1.tile_size_px}"
    assert "wheat" in b1.class_names, "wheat must be in band 1"
    assert "leaf" in b1.class_names, "leaf must be in band 1"


@pytest.mark.tier_a
def test_oracle_band2_downscale_leaf_only() -> None:
    """Oracle band 2: center ~910 → Z~0.114 → downscale.
    resize_factor in [0.10, 0.20], tile=800.
    ONLY leaf is a member (wheat fp_near=459 < band_lo=525.7).
    """
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    tiled = [p for p in passes if p.needs_tiling]
    b2 = tiled[2]
    assert 0.10 <= b2.resize_factor <= 0.20, f"Band 2: resize_factor expected ~0.11, got {b2.resize_factor:.4f}"
    assert b2.tile_size_px == _MODEL_SHORT_SIDE, f"Band 2: tile=800 expected, got {b2.tile_size_px}"
    assert "leaf" in b2.class_names, "leaf must be in band 2"
    assert "wheat" not in b2.class_names, "wheat must NOT be in band 2 (fp_near=459 < band_lo~526)"


# ─────────────────────────────────────────────────────────────────────────────
# Class-in-multiple-bands
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_class_appears_in_multiple_bands() -> None:
    """A class whose [fp_far, fp_near] span > K appears in more than one band."""
    # wheat from oracle: fp_far~58, fp_near~459, span=459/58=7.9 >> K=3
    # So wheat should appear in at least 2 bands.
    passes = _passes(
        {"wheat": 0.1},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        max_zoom_passes=6,
    )
    tiled = [p for p in passes if p.needs_tiling]
    bands_with_wheat = [p for p in tiled if "wheat" in p.class_names]
    assert len(bands_with_wheat) >= 2, f"wheat with span>K must appear in >=2 bands, got {len(bands_with_wheat)}"


# ─────────────────────────────────────────────────────────────────────────────
# Single class needing >1 band (span > K)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_single_class_span_greater_than_K_yields_multiple_bands() -> None:
    """Single class with near/far span >> K → multiple tiled bands.

    Setup: size=1.0m, d_azim=0.001, range_near=2m, range_far=20m
    fp_near = 1.0/(2*0.001) = 500, fp_far = 1.0/(20*0.001) = 50
    F_lo=50, F_hi=500, span=10 → n_s = ceil(log(10)/log(3)) = 3
    """
    passes = _passes(
        {"pole": 1.0},
        d_azim_rad=0.001,
        range_near_m=2.0,
        range_far_m=20.0,
        max_zoom_passes=6,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # n_s = ceil(log(500/50)/log(3)) = ceil(log(10)/log(3)) = 3
    expected_n_s = math.ceil(math.log(10) / math.log(_K))
    assert len(tiled) == expected_n_s, f"Single class, span 10x: expected {expected_n_s} tiled passes, got {len(tiled)}"
    # The class must appear in all 3 bands (its interval covers everything)
    assert all("pole" in p.class_names for p in tiled), "pole must be in all bands"


# ─────────────────────────────────────────────────────────────────────────────
# max_zoom_passes cap + warning
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_max_zoom_passes_cap_enforced(caplog: pytest.LogCaptureFixture) -> None:
    """When computed n_s > max_zoom_passes, result is capped and warning is emitted.

    Setup: span >> K^6 so uncapped n_s would be large; max_zoom_passes=2 triggers cap.
    """
    # fp_near=2000, fp_far=1 → span=2000 → n_s=ceil(log(2000)/log(3))=7 > cap of 2
    with caplog.at_level(logging.WARNING, logger="tls2dseg.engines.inference.multi_zoom_plan"):
        passes = _passes(
            {"giant": 2.0},
            d_azim_rad=0.001,
            range_near_m=1.0,
            range_far_m=2000.0,
            max_zoom_passes=2,
        )
    tiled = [p for p in passes if p.needs_tiling]
    assert len(tiled) <= 2, f"max_zoom_passes=2 must cap to <=2 tiled passes, got {len(tiled)}"
    # Warning must have been emitted
    warning_texts = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("cap" in w.lower() or "max" in w.lower() or "exceed" in w.lower() for w in warning_texts), (
        f"Expected a cap warning, got: {warning_texts}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Degenerate inputs
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_degenerate_empty_classes_returns_overview_only() -> None:
    """Empty class_sizes → only the always-on overview pass (no tiled passes)."""
    passes = _passes({}, d_azim_rad=0.001, range_near_m=10.0)
    assert len(passes) == 1
    assert not passes[0].needs_tiling


@pytest.mark.tier_a
def test_degenerate_zero_d_azim_returns_overview_only() -> None:
    """d_azim_rad <= 0 → degenerate guard fires, overview only."""
    passes = _passes({"door": 1.0}, d_azim_rad=0.0, range_near_m=10.0)
    assert len(passes) == 1
    assert not passes[0].needs_tiling


@pytest.mark.tier_a
def test_degenerate_zero_range_near_returns_overview_only() -> None:
    """range_near_m <= 0 → degenerate guard fires, overview only."""
    passes = _passes({"door": 1.0}, d_azim_rad=0.001, range_near_m=0.0)
    assert len(passes) == 1
    assert not passes[0].needs_tiling


@pytest.mark.tier_a
def test_degenerate_zero_range_does_not_crash() -> None:
    """T-06-06: zero range_m must not raise ZeroDivisionError."""
    try:
        passes = _passes({"door": 1.0}, d_azim_rad=0.001, range_near_m=0.0)
        assert isinstance(passes, list)
    except ZeroDivisionError:
        pytest.fail("compute_zoom_passes raised ZeroDivisionError for range_m=0")


# ─────────────────────────────────────────────────────────────────────────────
# Overview pass invariants
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_overview_pass_always_first() -> None:
    """D-B-05: overview pass is always at index 0."""
    for class_sizes, d_azim, r_near, r_far in [
        ({"a": 1.0}, 0.001, 10.0, 10.0),
        ({"a": 1.0, "b": 0.1}, 0.001, 5.0, 50.0),
        ({}, 0.001, 10.0, 10.0),
    ]:
        passes = _passes(class_sizes, d_azim_rad=d_azim, range_near_m=r_near, range_far_m=r_far)
        assert not passes[0].needs_tiling, "First pass must always be the overview (needs_tiling=False)"
        assert passes[0].tile_size_px is None
        assert passes[0].overlap_px is None


@pytest.mark.tier_a
def test_overview_pass_exactly_one() -> None:
    """D-B-05: exactly one needs_tiling=False pass in every result."""
    for class_sizes, d_azim, r_near, r_far in [
        ({"door": 1.0}, 0.001, 10.0, 10.0),
        ({"a": 1.0, "b": 1.2}, 0.001, 10.0, 10.0),
        ({}, 0.001, 10.0, 10.0),
    ]:
        passes = _passes(class_sizes, d_azim_rad=d_azim, range_near_m=r_near, range_far_m=r_far)
        overview = [p for p in passes if not p.needs_tiling]
        assert len(overview) == 1, f"Exactly one overview pass expected, got {len(overview)}"


@pytest.mark.tier_a
def test_overview_pass_covers_all_classes() -> None:
    """D-B-05: the overview pass must include all configured classes."""
    passes = _passes(
        {"door": 1.0, "window": 0.5, "column": 3.0},
        d_azim_rad=0.001,
        range_near_m=10.0,
        range_far_m=30.0,
    )
    ov = passes[0]
    for cls in ("door", "window", "column"):
        assert cls in ov.class_names, f"'{cls}' missing from overview pass"


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm invariants
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_resize_factor_never_exceeds_one() -> None:
    """resize_factor <= 1.0 for every pass (D-A-07: never invent detail)."""
    test_cases = [
        ({"chair": 0.5}, 0.001, 2.0, 50.0),
        ({"building": 20.0}, 0.001, 5.0, 50.0),
        ({"bolt": 0.01}, 0.0005, 5.0, 100.0),
        ({"tree": 5.0}, 0.002, 5.0, 20.0),
    ]
    for class_sizes, d_azim_rad, range_near, range_far in test_cases:
        passes = _passes(class_sizes, d_azim_rad=d_azim_rad, range_near_m=range_near, range_far_m=range_far)
        for p in passes:
            assert p.resize_factor <= 1.0, f"resize_factor={p.resize_factor} > 1.0 for {class_sizes}"


@pytest.mark.tier_a
def test_overlap_is_p_max_frac_times_tile() -> None:
    """D-B-02: overlap_px == round(p_max_frac * tile_size_px) for every tiled pass."""
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    for p in passes:
        if p.needs_tiling:
            assert p.tile_size_px is not None
            assert p.overlap_px is not None
            expected = round(_P_MAX_FRAC * p.tile_size_px)
            assert p.overlap_px == expected, f"overlap_px={p.overlap_px} != round(p_max_frac*tile)={expected}"


@pytest.mark.tier_a
def test_small_object_path_resize_is_1() -> None:
    """Small-object path (Z>1): resize_factor==1.0, tile < model_short_side.

    Setup: size=0.1m, range_near=range_far=1.51m, d_azim=1.442e-4
    fp_near = fp_far = 0.1/(1.51*1.442e-4) ~459 px > p_max -> wait, that's big.
    Use size=0.005, range=1m, d_azim=1.442e-4 → fp = 0.005/1.442e-4 ~34.7 < p_min=60.
    fc_b of band 0 = F_lo*sqrt(K) = 34.7*sqrt(3)=60.1 → Z~103.9/60.1=1.73 > 1 → small-obj.
    """
    passes = _passes(
        {"tiny": 0.005},
        d_azim_rad=1.442283e-4,
        range_near_m=1.0,
        range_far_m=1.0,
        max_zoom_passes=6,
    )
    tiled = [p for p in passes if p.needs_tiling]
    assert len(tiled) >= 1
    # The fine-end pass should be the small-object path
    b0 = tiled[0]
    assert b0.resize_factor == 1.0, f"Small-object pass: resize_factor must be 1.0, got {b0.resize_factor}"
    assert b0.tile_size_px is not None and b0.tile_size_px < _MODEL_SHORT_SIDE


@pytest.mark.tier_a
def test_downscale_path_tile_equals_model_short_side() -> None:
    """Downscale path (Z<=1): tile_size_px == model_short_side == 800."""
    # Band center > target_effective → Z < 1. Use oracle band 1 config (leaf@band2).
    passes = _passes(
        {"leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        max_zoom_passes=6,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # Some bands should use the downscale path (resize < 1)
    downscale_passes = [p for p in tiled if p.resize_factor < 1.0]
    assert len(downscale_passes) >= 1
    for p in downscale_passes:
        assert p.tile_size_px == _MODEL_SHORT_SIDE, (
            f"Downscale-path pass: tile must be {_MODEL_SHORT_SIDE}, got {p.tile_size_px}"
        )


@pytest.mark.tier_a
def test_tiled_passes_ordered_fine_to_coarse() -> None:
    """Tiled passes are ordered fine->coarse (band 0 first = smallest fc, largest Z)."""
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # Fine pass has the smallest resize_factor or largest tile (small-obj path).
    # Check monotonically: resize_factor should be non-decreasing going coarse->fine,
    # i.e., non-increasing going fine->coarse? Not necessarily if small-obj is band 0.
    # More robust: band 0 (fine) has the largest tile or Z~1.
    # Band 0 tile should be < 800 (small-obj path) and bands 1,2 should have tile=800.
    assert tiled[0].tile_size_px is not None and tiled[0].tile_size_px < _MODEL_SHORT_SIDE, (
        "Band 0 (fine/native) should have tile < 800 (small-obj path)"
    )
    for b in tiled[1:]:
        assert b.tile_size_px == _MODEL_SHORT_SIDE, "Coarser bands should have tile=800 (downscale path)"


# ─────────────────────────────────────────────────────────────────────────────
# Combined prompt
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_combined_prompt_contains_all_pass_classes() -> None:
    """text_prompt for each pass contains all its class names."""
    passes = _passes(
        {"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
    )
    for p in passes:
        for cls in p.class_names:
            assert cls in p.text_prompt, f"'{cls}' not in text_prompt {p.text_prompt!r}"


@pytest.mark.tier_a
def test_prompt_longest_name_first() -> None:
    """T-06-07: in a multi-class pass, longer names come first in text_prompt."""
    passes = _passes(
        {"street tree": 1.0, "tree": 1.0},
        d_azim_rad=0.001,
        range_near_m=10.0,
    )
    # Both classes have the same footprint → share one or more bands
    for p in passes:
        if "street tree" in p.class_names and "tree" in p.class_names:
            assert p.text_prompt.index("street tree") < p.text_prompt.index("tree"), (
                f"Longer name must come first: {p.text_prompt!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Engine short-side registry + overview self-downscale + overview_pass flag
# (mz-overview-collapse fix)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_engine_short_side_registry() -> None:
    """Known engine types map to their short side; unknown falls back to 800."""
    assert engine_short_side("grounded_sam2") == 800
    assert engine_short_side("grounded_sam2_hf") == 1024
    assert engine_short_side("nonexistent_engine") == 800


@pytest.mark.tier_a
def test_overview_self_downscale_when_native_larger() -> None:
    """native_short_side > model_short_side → overview resize_factor < 1.0 (self-downscale)."""
    passes = compute_zoom_passes(
        class_sizes={"wheat": 0.1},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        model_short_side=800,
        native_short_side=4000,
    )
    ov = passes[0]
    assert not ov.needs_tiling
    assert ov.resize_factor == 800 / 4000
    assert ov.resize_factor < 1.0


@pytest.mark.tier_a
def test_overview_native_when_smaller_or_unset() -> None:
    """native_short_side None or <= model → overview stays native (resize_factor == 1.0)."""
    passes_unset = compute_zoom_passes(
        class_sizes={"wheat": 0.1},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        model_short_side=800,
    )
    assert passes_unset[0].resize_factor == 1.0

    passes_small = compute_zoom_passes(
        class_sizes={"wheat": 0.1},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        model_short_side=800,
        native_short_side=600,
    )
    assert passes_small[0].resize_factor == 1.0


@pytest.mark.tier_a
def test_overview_pass_disabled_omits_overview() -> None:
    """overview_pass=False → no needs_tiling=False overview when tiled bands exist."""
    passes = compute_zoom_passes(
        class_sizes={"wheat": 0.1, "leaf": 0.2},
        d_azim_rad=1.442283e-4,
        range_near_m=1.51,
        range_far_m=11.87,
        overview_pass=False,
    )
    assert len(passes) >= 1
    assert all(p.needs_tiling for p in passes), "overview_pass=False must drop the full-image pass"


@pytest.mark.tier_a
def test_overview_pass_disabled_degenerate_still_returns_one_pass() -> None:
    """overview_pass=False + degenerate inputs → never returns an empty plan."""
    passes = compute_zoom_passes(
        class_sizes={"door": 1.0},
        d_azim_rad=0.001,
        range_near_m=0.0,
        range_far_m=10.0,
        overview_pass=False,
    )
    assert len(passes) == 1


@pytest.mark.tier_a
def test_degenerate_guard_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Degenerate near-range fires a WARNING (operator visibility, not silent DEBUG)."""
    with caplog.at_level(logging.WARNING, logger="tls2dseg.engines.inference.multi_zoom_plan"):
        compute_zoom_passes(
            class_sizes={"door": 1.0},
            d_azim_rad=0.001,
            range_near_m=0.0,
            range_far_m=10.0,
        )
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "degenerate guard must emit a WARNING"
    assert any("SINGLE inference pass" in r.message for r in warnings)

"""Unit tests for multi_zoom_plan.py — ZoomPass + compute_zoom_passes.

Covers: MZ-02 (footprint math + resize_factor + needs_tiling),
        MZ-11 (grouping = scales, combined-prompt routing, always-on full-image pass).
Tier: tier_a — stdlib + numpy only; no torch/sam2/supervision/pchandler at import time.
"""

from __future__ import annotations

import math

import pytest

from tls2dseg.engines.inference.multi_zoom_plan import ZoomPass, compute_zoom_passes

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODEL_SHORT_SIDE = 800
_DEFAULT_P_MIN_FRAC = 0.075  # → 60 px at 800
_DEFAULT_P_MAX_FRAC = 0.225  # → 180 px at 800


def _passes(
    class_sizes: dict[str, float],
    d_azim_rad: float,
    range_near_m: float,
    range_far_m: float | None = None,
    *,
    p_min_frac: float = _DEFAULT_P_MIN_FRAC,
    p_max_frac: float = _DEFAULT_P_MAX_FRAC,
    model_short_side: int = _DEFAULT_MODEL_SHORT_SIDE,
) -> list[ZoomPass]:
    """Thin wrapper so tests share one call site."""
    return compute_zoom_passes(
        class_sizes=class_sizes,
        d_azim_rad=d_azim_rad,
        range_near_m=range_near_m,
        range_far_m=range_far_m if range_far_m is not None else range_near_m,
        p_min_frac=p_min_frac,
        p_max_frac=p_max_frac,
        model_short_side=model_short_side,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 tests — footprint math, greedy cover, resize invariant, overlap
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_footprint_math_big_object_gets_downscale() -> None:
    """Big object (footprint > p_max) → resize_factor < 1.0 and needs_tiling=True.

    Setup: size_m=3.0, range_m=10.0, d_azim_rad=0.001 rad/px
    → p0 = 3.0 / (10.0 * 0.001) = 300 px  >  p_max=180 px
    → the scaled pass must have resize_factor = p_max/p0 = 180/300 = 0.6
      and needs_tiling=True.
    """
    passes = _passes({"door": 3.0}, d_azim_rad=0.001, range_near_m=10.0)

    scaled = [p for p in passes if p.needs_tiling]
    assert len(scaled) >= 1, "At least one tiled pass expected for big object"
    zf = scaled[0].resize_factor
    assert zf <= 1.0, f"resize_factor must be <= 1.0, got {zf}"
    # resize_factor should put the footprint in-band: p0 * zf ≈ p_max
    p0 = 3.0 / (10.0 * 0.001)
    p_max = _DEFAULT_P_MAX_FRAC * _DEFAULT_MODEL_SHORT_SIDE
    assert abs(p0 * zf - p_max) < 5, f"resize_factor {zf:.4f} should target p_max={p_max}, got p0*zf={p0 * zf:.1f}"


@pytest.mark.tier_a
def test_footprint_math_small_object_keeps_native_and_shrinks_tile() -> None:
    """Small object (footprint < p_min) → resize_factor=1.0, tiled pass with small tile.

    Setup: size_m=0.05, range_m=50.0, d_azim_rad=0.001 rad/px
    → p0 = 0.05 / (50.0 * 0.001) = 1 px  <  p_min=60 px
    → small-object strategy: resize_factor=1.0, shrink tile so model upscales.
    """
    passes = _passes({"screw": 0.05}, d_azim_rad=0.001, range_near_m=50.0)

    scaled = [p for p in passes if p.needs_tiling]
    assert len(scaled) >= 1, "Tiled pass expected for small object"
    for p in scaled:
        assert p.resize_factor == 1.0, f"Small object: resize_factor must be 1.0 (no downscale), got {p.resize_factor}"
        assert p.tile_size_px is not None and p.tile_size_px > 0
        assert p.overlap_px is not None and p.overlap_px >= 0


@pytest.mark.tier_a
def test_resize_factor_never_exceeds_one() -> None:
    """D-A-07: resize_factor <= 1.0 for every pass on every class."""
    # Vary size and range to stress-test the invariant
    test_cases = [
        ({"chair": 0.5}, 0.001, 2.0),
        ({"building": 20.0}, 0.001, 5.0),
        ({"bolt": 0.01}, 0.0005, 100.0),
        ({"tree": 5.0}, 0.002, 20.0),
    ]
    for class_sizes, d_azim_rad, range_near in test_cases:
        passes = _passes(class_sizes, d_azim_rad=d_azim_rad, range_near_m=range_near)
        for p in passes:
            assert p.resize_factor <= 1.0, (
                f"resize_factor={p.resize_factor} > 1.0 for {class_sizes}, d_azim={d_azim_rad}, range={range_near}"
            )


@pytest.mark.tier_a
def test_greedy_cover_band_count_10x_span() -> None:
    """Greedy cover with ~10x footprint span yields ceil(log(10)/log(K)) scaled passes.

    Default K = p_max/p_min = 0.225/0.075 = 3.0
    span = 10, n_s = ceil(log(10)/log(3)) = ceil(2.096) = 3
    The always-on full-image pass is NOT counted here — only tiled passes.
    """
    # Two classes: one near (big footprint) and one far (small footprint), 10x span
    # near: size=2.0, range=5m  → p0 = 2.0/(5*0.001) = 400 px
    # far:  size=0.2, range=50m → p0 = 0.2/(50*0.001) = 4 px
    # span ≈ 400/4 = 100 in absolute px — well above 10x; use a tighter example
    # near: size=1.0, range=5m  → p0 = 200 px
    # far:  size=0.3, range=15m → p0 = 20 px
    # span = 200/20 = 10 → n_s = ceil(log(10)/log(3)) = 3
    passes = _passes(
        {"big": 1.0, "small": 0.3},
        d_azim_rad=0.001,
        range_near_m=5.0,
        range_far_m=15.0,
    )
    tiled = [p for p in passes if p.needs_tiling]
    K = _DEFAULT_P_MAX_FRAC / _DEFAULT_P_MIN_FRAC
    p0_near = 1.0 / (5.0 * 0.001)
    p0_far = 0.3 / (15.0 * 0.001)
    p_min = _DEFAULT_P_MIN_FRAC * _DEFAULT_MODEL_SHORT_SIDE
    p_max = _DEFAULT_P_MAX_FRAC * _DEFAULT_MODEL_SHORT_SIDE
    log_near = math.log(min(p0_near, p_max))
    log_far = math.log(max(p0_far, p_min))
    log_K = math.log(K)
    expected_n_s = math.ceil((log_near - log_far) / log_K)
    assert len(tiled) == expected_n_s, f"Expected {expected_n_s} tiled passes for 10x span, got {len(tiled)}"


@pytest.mark.tier_a
def test_greedy_cover_single_band_yields_one_scaled_pass() -> None:
    """Footprint within one band → exactly 1 tiled scaled pass (plus 1 full-image)."""
    # size=1.0, range=10m, d_azim=0.001 → p0 = 100 px in [60,180] → in-band
    passes = _passes({"tree": 1.0}, d_azim_rad=0.001, range_near_m=10.0)
    tiled = [p for p in passes if p.needs_tiling]
    assert len(tiled) == 1, f"Single-band footprint should yield exactly 1 tiled pass, got {len(tiled)}"


@pytest.mark.tier_a
def test_overlap_is_pixel_derived_not_ratio() -> None:
    """D-B-02: overlap_px == int(p_max_frac * tile_size_px) for every tiled pass."""
    passes = _passes({"door": 1.0}, d_azim_rad=0.001, range_near_m=10.0)
    for p in passes:
        if p.needs_tiling:
            assert p.tile_size_px is not None
            assert p.overlap_px is not None
            expected_overlap = int(_DEFAULT_P_MAX_FRAC * p.tile_size_px)
            assert p.overlap_px == expected_overlap, (
                f"overlap_px={p.overlap_px} != int(p_max_frac * tile_size)={expected_overlap}"
            )


@pytest.mark.tier_a
def test_empty_classes_returns_only_full_image_pass() -> None:
    """Degenerate: empty class dict → only the always-on full-image pass returned."""
    passes = _passes({}, d_azim_rad=0.001, range_near_m=10.0)
    assert len(passes) >= 1, "At least one pass (full-image) always returned"
    full_image = [p for p in passes if not p.needs_tiling]
    assert len(full_image) == 1, "Exactly one full-image pass expected"


@pytest.mark.tier_a
def test_degenerate_zero_range_does_not_crash() -> None:
    """T-06-06: zero range_m must not raise ZeroDivisionError."""
    try:
        passes = _passes({"door": 1.0}, d_azim_rad=0.001, range_near_m=0.0)
        assert isinstance(passes, list)
    except ZeroDivisionError:
        pytest.fail("compute_zoom_passes raised ZeroDivisionError for range_m=0")


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 tests — grouping, combined prompt, always-on full-image pass (MZ-11)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_grouping_same_band_shares_one_pass() -> None:
    """MZ-11: two classes in same footprint band → one shared ZoomPass."""
    # Both at 100 px footprint (in-band): same size, same range
    passes = _passes(
        {"chair": 1.0, "table": 1.0},
        d_azim_rad=0.001,
        range_near_m=10.0,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # Both classes have identical footprints → same band → share one pass
    assert len(tiled) == 1, f"Same-band classes must share one pass, got {len(tiled)}"
    assert "chair" in tiled[0].class_names
    assert "table" in tiled[0].class_names


@pytest.mark.tier_a
def test_grouping_distinct_bands_produce_distinct_passes() -> None:
    """MZ-11 fallback: classes in distinct bands produce separate single-class passes."""
    # near: size=1.0, range=5m  → p0 = 200 px (band 2 or 3)
    # far:  size=0.3, range=15m → p0 = 20 px  (band 0)
    passes = _passes(
        {"big": 1.0, "small": 0.3},
        d_azim_rad=0.001,
        range_near_m=5.0,
        range_far_m=15.0,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # The two classes are far apart in log space → different bands → different passes
    # Each pass's class_names must be a non-empty subset
    assert all(len(p.class_names) >= 1 for p in tiled)
    # No pass should contain BOTH classes (they are in distinct bands)
    both_in_one = any("big" in p.class_names and "small" in p.class_names for p in tiled)
    assert not both_in_one, "Distinct-band classes must not share a pass"


@pytest.mark.tier_a
def test_substring_order_longest_name_first_in_prompt() -> None:
    """T-06-07: combined prompt must order longer names first.

    'street tree' must come before 'tree' in the combined text_prompt so that
    downstream resolve_class_names substring matching routes 'street tree' correctly.
    """
    # Same band: both at 100 px footprint
    passes = _passes(
        {"street tree": 1.0, "tree": 1.0},
        d_azim_rad=0.001,
        range_near_m=10.0,
    )
    tiled = [p for p in passes if p.needs_tiling]
    # Both in same band → one shared pass
    assert len(tiled) == 1, "Same-band classes must share one pass"
    prompt = tiled[0].text_prompt
    assert prompt.index("street tree") < prompt.index("tree"), (
        f"Longer name 'street tree' must appear before 'tree' in prompt: {prompt!r}"
    )


@pytest.mark.tier_a
def test_always_on_full_image_pass_exactly_one() -> None:
    """D-B-05: exactly one needs_tiling=False pass in every result."""
    test_cases = [
        ({"door": 1.0}, 0.001, 10.0),
        ({"chair": 1.0, "table": 1.2}, 0.001, 10.0),
        ({}, 0.001, 10.0),
    ]
    for class_sizes, d_azim_rad, range_near in test_cases:
        passes = _passes(class_sizes, d_azim_rad=d_azim_rad, range_near_m=range_near)
        full_image = [p for p in passes if not p.needs_tiling]
        assert len(full_image) == 1, f"Exactly 1 full-image pass expected for {class_sizes}, got {len(full_image)}"
        fi = full_image[0]
        assert fi.tile_size_px is None
        assert fi.overlap_px is None


@pytest.mark.tier_a
def test_full_image_pass_covers_all_classes() -> None:
    """D-B-05: the full-image pass must include all configured classes."""
    passes = _passes(
        {"door": 1.0, "window": 0.5, "column": 3.0},
        d_azim_rad=0.001,
        range_near_m=10.0,
        range_far_m=30.0,
    )
    full_image = [p for p in passes if not p.needs_tiling]
    assert len(full_image) == 1
    fi = full_image[0]
    for cls in ("door", "window", "column"):
        assert cls in fi.class_names, f"Class '{cls}' missing from full-image pass"


@pytest.mark.tier_a
def test_combined_prompt_contains_all_pass_classes() -> None:
    """text_prompt for a multi-class pass contains all class names."""
    passes = _passes(
        {"chair": 1.0, "table": 1.1},
        d_azim_rad=0.001,
        range_near_m=10.0,
    )
    for p in passes:
        for cls in p.class_names:
            assert cls in p.text_prompt, f"Class '{cls}' not found in text_prompt {p.text_prompt!r}"

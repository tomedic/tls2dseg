"""Per-class adaptive multi-zoom plan engine.

Pure-function strategy (D-B-01): given per-class physical sizes, scan angular
resolution, and representative range, computes a minimal set of ZoomPasses that
together cover the full footprint range at the target model-input resolution.

Design references
-----------------
- SNIP/SNIPER (Singh et al., NeurIPS 2018 / CVPR 2018): per-scale valid-size bands
  in a log-footprint pyramid.
- SAHI (Akyon et al., CVPR 2022): tiled inference with overlap to capture
  objects near tile boundaries.
- Greedy interval cover in log-footprint space (D-B-03): n_s = ceil(log(span)/log(K)).

Tier-A import contract: only stdlib + numpy. No torch/pyvips/supervision/sam2.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence

logger = logging.getLogger("tls2dseg.engines.inference.multi_zoom_plan")

# SAM2 internal resize cap (RESEARCH §DINO/SAM2 Internal Resize).
_SAM2_TILE_CAP_PX = 1024

# Per-engine GroundingDINO short-side input size (px). The overview pass
# downscales the native panorama to this short side ourselves (Lanczos3) so the
# inference engine never has to shrink a full-resolution image internally.
# Keep this tier_a-importable: no torch / engine-class imports at module level.
_ENGINE_SHORT_SIDE: dict[str, int] = {
    "grounded_sam2": 800,
    "grounded_sam2_hf": 1024,
}
_DEFAULT_SHORT_SIDE = 800


def engine_short_side(engine_type: str) -> int:
    """Return the GroundingDINO short-side input size for an inference engine.

    Falls back to the default short side for unknown engine types.
    """
    return _ENGINE_SHORT_SIDE.get(engine_type, _DEFAULT_SHORT_SIDE)


@dataclasses.dataclass(frozen=True)
class ZoomPass:
    """One inference pass in the multi-zoom plan.

    resize_factor : <=1.0; 1.0 = native (no downscale).
    tile_size_px  : None for full-image (no-tiling) pass.
    overlap_px    : None for full-image pass; in pixels (NOT ratio).
    needs_tiling  : True if SAHI slicing should be used.
    class_names   : classes active in this pass.
    text_prompt   : combined GroundingDINO prompt, names ordered longest-first.
    """

    resize_factor: float
    tile_size_px: int | None
    overlap_px: int | None
    needs_tiling: bool
    class_names: tuple[str, ...]
    text_prompt: str


def compute_zoom_passes(
    class_sizes: dict[str, float],
    d_azim_rad: float,
    range_near_m: float,
    range_far_m: float,
    p_min_frac: float = 0.075,
    p_max_frac: float = 0.225,
    model_short_side: int = 800,
    max_zoom_passes: int = 6,
    native_short_side: int | None = None,
    overview_pass: bool = True,
) -> list[ZoomPass]:
    """Compute the minimal ZoomPass list covering all class footprints.

    Parameters
    ----------
    class_sizes :
        Mapping class_name → representative physical size in metres.
    d_azim_rad :
        Native scanner angular resolution (radians per pixel).
    range_near_m :
        Nearest representative range (metres); drives largest footprint.
    range_far_m :
        Farthest representative range (metres); drives smallest footprint.
    p_min_frac, p_max_frac :
        Target footprint band as fractions of model_short_side (D-A-05).
    model_short_side :
        DINO short-side input size in pixels (default 800).
    max_zoom_passes :
        Maximum number of tiled bands. Caps n_s when the footprint span would
        require more; a warning is emitted when the cap is applied.

    Returns
    -------
    list[ZoomPass]
        Always starts with exactly one full-image pass (D-B-05), followed by
        the greedy-cover tiled passes ordered fine->coarse (band 0 first).
    """
    all_names = list(class_sizes.keys())

    # Overview self-downscale: downscale the native panorama
    # ourselves to the model short side rather than letting the inference engine
    # shrink a full-resolution image internally. resize_factor < 1.0 makes the
    # dispatcher run _lanczos3_resize AND _remap_to_native for this pass.
    if native_short_side is not None and native_short_side > model_short_side:
        overview_resize_factor = model_short_side / native_short_side
    else:
        overview_resize_factor = 1.0

    # Always-on full-image overview pass (D-B-05).
    overview = ZoomPass(
        resize_factor=overview_resize_factor,
        tile_size_px=None,
        overlap_px=None,
        needs_tiling=False,
        class_names=tuple(all_names),
        text_prompt=_build_prompt(all_names),
    )
    base_passes = [overview] if overview_pass else []

    if not class_sizes or d_azim_rad <= 0.0 or range_near_m <= 0.0:
        logger.warning(
            "Multi-zoom degenerate inputs (class_sizes=%s, d_azim_rad=%.6g, range_near_m=%.6g): "
            "running a SINGLE inference pass over the full-FoV image only (no per-band tiling). "
            "A non-positive near range or missing class sizes disables band planning. "
            "Set preprocessing.range_limits_m[0] >= 0.5 (or unset it to use range_percentiles).",
            list(class_sizes.keys()),
            d_azim_rad,
            range_near_m,
        )
        return base_passes if base_passes else [overview]

    p_min = p_min_frac * model_short_side
    p_max = p_max_frac * model_short_side
    K = p_max / p_min  # band ratio
    ov_frac = p_max_frac  # overlap fraction (D-B-02)
    target_effective = math.sqrt(p_min * p_max)  # sweet-spot center

    # Per-class native footprints in image pixels.
    # fp_near = biggest footprint (object at near range).
    # fp_far  = smallest footprint (object at far range, clamped to near if far < near).
    fp_near: dict[str, float] = {}
    fp_far: dict[str, float] = {}
    for name, size_m in class_sizes.items():
        fp_near[name] = size_m / (range_near_m * d_azim_rad)
        fp_far[name] = size_m / (max(range_far_m, range_near_m) * d_azim_rad)

    # Total span across ALL classes — unclamped (this is the core fix).
    F_lo = min(fp_far.values())  # smallest-object-at-far
    F_hi = max(fp_near.values())  # largest-object-at-near

    n_s = 1 if F_hi <= F_lo else math.ceil(math.log(F_hi / F_lo) / math.log(K))

    if n_s > max_zoom_passes:
        logger.warning(
            "Footprint span %.1f (F_lo=%.1f, F_hi=%.1f) requires %d bands but "
            "max_zoom_passes=%d caps it. Coarsest instances rely on the overview pass.",
            F_hi / F_lo,
            F_lo,
            F_hi,
            n_s,
            max_zoom_passes,
        )
        n_s = max_zoom_passes

    # Build tiled passes, one per non-empty band.
    tiled_passes: list[ZoomPass] = []
    for b in range(n_s):
        band_lo = F_lo * (K**b)
        band_hi = F_lo * (K ** (b + 1))
        fc_b = F_lo * (K ** (b + 0.5))  # geometric center

        # Class membership: interval overlap (D-B-06, DECISION 1).
        # fp_far < band_hi: object has instances smaller than band top.
        # fp_near >= band_lo: object has instances at least as large as band bottom.
        members = [name for name in all_names if fp_far[name] < band_hi and fp_near[name] >= band_lo]
        if not members:
            continue

        resize_factor, tile_size_px = _compute_pass_geometry(
            fc_b=fc_b,
            target_effective=target_effective,
            model_short_side=model_short_side,
        )

        if native_short_side is not None and tile_size_px >= native_short_side:
            logger.warning(
                "Band %d requested tile %d px >= native image short side %d px: "
                "image is too small to tile for this band. "
                "Running a full-image pass over the whole panorama instead.",
                b,
                tile_size_px,
                native_short_side,
            )
            tiled_passes.append(
                ZoomPass(
                    resize_factor=1.0,
                    tile_size_px=None,
                    overlap_px=None,
                    needs_tiling=False,
                    class_names=tuple(members),
                    text_prompt=_build_prompt(members),
                )
            )
            continue

        overlap_px = round(ov_frac * tile_size_px)

        tiled_passes.append(
            ZoomPass(
                resize_factor=resize_factor,
                tile_size_px=tile_size_px,
                overlap_px=overlap_px,
                needs_tiling=True,
                class_names=tuple(members),
                text_prompt=_build_prompt(members),
            )
        )

    # Emit structured debug dump so the operator is never blind.
    logger.info(
        "multi-zoom plan: %s",
        {
            "p_min": round(p_min, 1),
            "p_max": round(p_max, 1),
            "K": round(K, 2),
            "target_effective": round(target_effective, 1),
            "F_lo": round(F_lo, 2),
            "F_hi": round(F_hi, 2),
            "n_s": n_s,
            "per_class": {
                name: {
                    "fp_near": round(fp_near[name], 1),
                    "fp_far": round(fp_far[name], 1),
                }
                for name in all_names
            },
            "bands": [
                {
                    "idx": b,
                    "native_lo": round(F_lo * (K**b), 2),
                    "native_hi": round(F_lo * (K ** (b + 1)), 2),
                    "center": round(F_lo * (K ** (b + 0.5)), 2),
                    "resize_factor": round(p.resize_factor, 3),
                    "tile_size_px": p.tile_size_px,
                    "overlap_px": p.overlap_px,
                    "classes": list(p.class_names),
                }
                for b, p in enumerate(tiled_passes)
            ],
        },
    )

    return [*base_passes, *tiled_passes]


def _compute_pass_geometry(
    fc_b: float,
    target_effective: float,
    model_short_side: int,
) -> tuple[float, int]:
    """Compute (resize_factor, tile_size_px) for one band.

    Z_b = target_effective / fc_b is the effective scale needed to bring
    the band center to the sweet-spot footprint (D-A-06).

    Two strategies:
    - Z_b <= 1.0 (band center bigger than target): DOWNSCALE image.
      resize_factor = Z_b  (< 1); tile_size_px = model_short_side.
    - Z_b > 1.0 (band center smaller than target): keep native, SHRINK tile
      so the model upscales the crop.
      resize_factor = 1.0; tile_size_px = round(model_short_side / Z_b).
      tile_size_px clamped to [model_short_side // 4, model_short_side].
    """
    Z_b = target_effective / fc_b

    if Z_b <= 1.0:
        resize_factor = Z_b
        tile_size_px = model_short_side
    else:
        resize_factor = 1.0
        tile_size_px = round(model_short_side / Z_b)
        tile_size_px = max(model_short_side // 4, min(tile_size_px, model_short_side))

    return resize_factor, tile_size_px


def _build_prompt(names: Sequence[str]) -> str:
    """Build a GroundingDINO combined prompt, longest names first (T-06-07).

    Longest-first ordering ensures substring routing in resolve_class_names
    does not misroute (e.g. 'street tree' must precede 'tree').
    """
    ordered = sorted(names, key=len, reverse=True)
    if not ordered:
        return ""
    return ". ".join(ordered) + "."

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

    Returns
    -------
    list[ZoomPass]
        Always starts with exactly one full-image pass (D-B-05), followed by
        the greedy-cover tiled passes.
    """
    all_names = list(class_sizes.keys())

    # Always-on full-image pass (D-B-05): Lanczos3-downscale to model short-side.
    # resize_factor = min(model_short_side / max(native_h, native_w), 1.0)
    # We don't know the image dimensions here, so use 1.0 as the safe default;
    # the dispatcher will apply the actual resize. The important invariant is
    # needs_tiling=False and covering all classes.
    full_image_pass = ZoomPass(
        resize_factor=1.0,
        tile_size_px=None,
        overlap_px=None,
        needs_tiling=False,
        class_names=tuple(all_names),
        text_prompt=_build_prompt(all_names),
    )

    if not class_sizes or d_azim_rad <= 0.0 or range_near_m <= 0.0:
        logger.debug("Degenerate inputs: returning full-image pass only")
        return [full_image_pass]

    p_min = p_min_frac * model_short_side
    p_max = p_max_frac * model_short_side
    ov_frac = p_max_frac  # overlap fraction per D-B-02

    # Compute native footprint p0 = size_m / (range_m * d_azim_rad) for each class.
    # Use range_near for near-side (biggest footprint) and range_far for far-side
    # (smallest footprint) to bracket the full span seen in a scan.
    footprints_near: dict[str, float] = {}
    footprints_far: dict[str, float] = {}
    for name, size_m in class_sizes.items():
        footprints_near[name] = size_m / (range_near_m * d_azim_rad)
        footprints_far[name] = size_m / (max(range_far_m, range_near_m) * d_azim_rad)

    # Clamp footprints for the log-space cover computation.
    # Footprints outside [p_min, p_max] are mapped to the band edge (they'll still
    # be bucketed to the nearest band; clamping only affects n_s calculation).
    all_far_fps = [max(fp, p_min) for fp in footprints_far.values()]
    all_near_fps = [min(fp, p_max) for fp in footprints_near.values()]

    log_far = math.log(min(all_far_fps))
    log_near = math.log(max(all_near_fps))
    log_K = math.log(p_max / p_min)

    n_s = 1 if log_near <= log_far else math.ceil((log_near - log_far) / log_K)

    # Band boundaries: greedy sweep from far end in log space.
    # Band s covers [log_far + s*log_K, log_far + (s+1)*log_K].
    def _band_index(log_fp: float) -> int:
        idx = int((log_fp - log_far) / log_K)
        return max(0, min(idx, n_s - 1))

    # Bucket each class by its representative footprint (use near = near_m → biggest footprint,
    # which determines what band it needs most aggressive scaling for).
    band_to_classes: dict[int, list[str]] = {i: [] for i in range(n_s)}
    for name in all_names:
        # Use near-side footprint for band assignment (determines downscale needed).
        fp_near = footprints_near[name]
        log_fp = math.log(max(fp_near, p_min))
        log_fp = min(log_fp, log_near)  # clamp to cover range
        band_idx = _band_index(log_fp)
        band_to_classes[band_idx].append(name)

    tiled_passes: list[ZoomPass] = []
    for band_idx in range(n_s):
        names_in_band = band_to_classes[band_idx]
        if not names_in_band:
            continue

        # Band center in log space; derive target footprint (geometric mean of band edges).
        log_lo = log_far + band_idx * log_K
        log_hi = log_lo + log_K
        log_center = (log_lo + log_hi) / 2.0
        target_fp = math.exp(log_center)

        # Representative class footprint for resize math: use near-side (largest).
        # For the band's resize factor we pick the largest footprint in the band.
        max_fp_near = max(footprints_near[n] for n in names_in_band)

        resize_factor, tile_size_px = _compute_pass_geometry(
            p0=max_fp_near,
            target_fp=target_fp,
            p_min=p_min,
            p_max=p_max,
            model_short_side=model_short_side,
        )
        overlap_px = int(ov_frac * tile_size_px)

        tiled_passes.append(
            ZoomPass(
                resize_factor=resize_factor,
                tile_size_px=tile_size_px,
                overlap_px=overlap_px,
                needs_tiling=True,
                class_names=tuple(names_in_band),
                text_prompt=_build_prompt(names_in_band),
            )
        )

    return [full_image_pass, *tiled_passes]


def _compute_pass_geometry(
    p0: float,
    target_fp: float,
    p_min: float,
    p_max: float,
    model_short_side: int,
) -> tuple[float, int]:
    """Compute (resize_factor, tile_size_px) for one band.

    Two strategies (D-A-06):
    - Big object (p0 >= p_max): Lanczos3 downscale to bring footprint in-band.
      resize_factor = clamp(p_max / p0, 0 < z <= 1.0).
      tile_size_px = model_short_side (one tile = model input, no tiling overhead).
    - Small object (p0 < p_min): keep native (resize_factor=1.0), shrink tile so
      model upscales the crop to see sub-pixel objects.
      tile_size_px derived so model_short_side / tile_size_px brings p0 in-band.
    """
    if p0 >= p_max:
        # Big-object path: downscale image.
        resize_factor = min(p_max / p0, 1.0)
        tile_size_px = model_short_side
    elif p0 < p_min:
        # Small-object path: shrink tile so DINO upscales the crop.
        # effective_fp = p0 * (model_short_side / tile_size_px) = p_min
        # → tile_size_px = p0 * model_short_side / p_min
        tile_size_px = max(1, int(p0 * model_short_side / p_min))
        tile_size_px = min(tile_size_px, _SAM2_TILE_CAP_PX)
        resize_factor = 1.0
    else:
        # In-band: use native resolution, tile at model_short_side.
        resize_factor = 1.0
        tile_size_px = model_short_side

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

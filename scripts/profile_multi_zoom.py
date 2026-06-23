"""Standalone profiling harness for the multi-zoom dispatcher.

Measures per-pass peak VRAM, total runtime (multi-zoom vs optional single-zoom
baseline), confirms DINO/SAM2 internal resize assumptions (A1/A2), and saves
per-pass input images + detection overlays for visual inspection.

Run (from the tls2dseg/ repo root, GPU stack active):

    python scripts/profile_multi_zoom.py \\
        --config examples/configs/mountain.yaml \\
        --scan /path/to/scan.e57 \\
        [--classes "tree:5,pole:0.1"] \\
        [--feature intensity] \\
        [--passes N] \\
        [--overlay-dir ./profile_out/myscan] \\
        [--baseline]

Outputs:
  - Summary table to stdout (DINO/SAM2 sizes, per-pass VRAM, runtime).
  - Per-pass PNG images under --overlay-dir:
      pass{i}_input_resize{r}_tile{t}.png  — normalized input feature image
      pass{i}_overlay_resize{r}_tile{t}.png — same with detection boxes/labels
  - final_combined_overlay.png  — combined detections on a downscaled native image
  - plan_dump.txt               — zoom-pass plan parameters

No results are written automatically to .md — copy the printed table into
06-PROFILE.md after inspecting the values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("profile_multi_zoom")


def _parse_classes(classes_str: str) -> dict[str, float]:
    """Parse 'name1:size1,name2:size2' into {name: float} dict."""
    result: dict[str, float] = {}
    for item in classes_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected 'name:size_m' pairs, got: {item!r}")
        name, size_str = item.split(":", 1)
        result[name.strip()] = float(size_str.strip())
    return result


def _confirm_dino_sizes(inference_engine: object) -> dict[str, object]:
    """Read back DINO processor resize params to confirm A1.

    Returns dict with keys: 'size', 'max_size', 'confirmed_a1'.
    """
    info: dict[str, object] = {"size": None, "max_size": None, "confirmed_a1": False}
    try:
        processor = getattr(inference_engine, "_processor", None)
        if processor is None:
            processor = getattr(inference_engine, "processor", None)
        if processor is None:
            logger.warning("Could not locate DINO processor on inference_engine — skipping A1 check")
            return info

        img_proc = getattr(processor, "image_processor", None)
        if img_proc is None:
            logger.warning("processor.image_processor not found — skipping A1 check")
            return info

        info["size"] = getattr(img_proc, "size", None)
        info["max_size"] = getattr(img_proc, "max_size", None)

        size_val = info["size"]
        max_size_val = info["max_size"]
        expected_shortest = 800
        expected_max = 1333
        if isinstance(size_val, dict):
            shortest = size_val.get("shortest_edge")
        else:
            shortest = size_val
        info["confirmed_a1"] = (shortest == expected_shortest and max_size_val == expected_max)
        if not info["confirmed_a1"]:
            logger.warning(
                "A1 MISMATCH: expected shortest_edge=%d max_size=%d, got size=%r max_size=%r",
                expected_shortest,
                expected_max,
                size_val,
                max_size_val,
            )
    except Exception as exc:
        logger.warning("A1 check failed: %s", exc)
    return info


def _confirm_sam2_resize(inference_engine: object) -> dict[str, object]:
    """Probe SAM2 predictor image size to confirm A2 (1024 internal resize).

    Returns dict with keys: 'image_size', 'confirmed_a2'.
    """
    info: dict[str, object] = {"image_size": None, "confirmed_a2": False}
    try:
        predictor = getattr(inference_engine, "_sam2_predictor", None)
        if predictor is None:
            predictor = getattr(inference_engine, "sam2_predictor", None)
        if predictor is None:
            logger.warning("Could not locate SAM2 predictor — skipping A2 check")
            return info

        image_size = getattr(predictor, "image_size", None)
        info["image_size"] = image_size
        info["confirmed_a2"] = image_size == 1024
        if not info["confirmed_a2"]:
            logger.warning("A2 MISMATCH: expected SAM2 image_size=1024, got %r", image_size)
    except Exception as exc:
        logger.warning("A2 check failed: %s", exc)
    return info


def _run_single_zoom_baseline(
    image: "np.ndarray",
    inference_engine: object,
    base_request: "InferenceRequest",
) -> tuple[float, "Detections2D"]:
    """Run a single full-image detect() call and return (elapsed_s, detections)."""
    import time

    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    detections = inference_engine.detect(image, request=base_request)
    elapsed_s = time.perf_counter() - t0
    return elapsed_s, detections


def _run_multi_zoom_profiled(
    image: "np.ndarray",
    inference_engine: object,
    base_request: "InferenceRequest",
    zoom_passes: list,
    ctx: object,
    mz_cfg: object,
    on_pass_cb: object = None,
) -> tuple[float, list[float], "Detections2D"]:
    """Run run_multi_zoom with per-pass VRAM recording.

    Wraps run_multi_zoom's loop manually to record max_memory_allocated after
    each pass.  on_pass_cb is forwarded directly to run_multi_zoom's on_pass
    parameter so overlays are written from inside the official hook.

    Returns (total_elapsed_s, per_pass_vram_gb_list, combined_detections).
    """
    import dataclasses
    import gc
    import time

    import torch

    from tls2dseg.engines.inference import multi_zoom_dispatch as _mzd
    from tls2dseg.engines.inference.dedup import (
        _concat_detections,
        dedup_cross_class,
        dedup_same_class_ios,
        dedup_same_class_nms,
    )
    from tls2dseg.types import Detections2D

    class_id_map: dict[str, int] = getattr(ctx, "class_id_map", {})
    per_class_metadata: dict[str, object] = getattr(ctx, "per_class_metadata", {})

    all_detections: list[Detections2D] = []
    per_pass_vram: list[float] = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for pass_idx, zoom_pass in enumerate(zoom_passes):
        torch.cuda.reset_peak_memory_stats()

        if zoom_pass.resize_factor < 1.0:
            image_input = _mzd._lanczos3_resize(image, zoom_pass.resize_factor)
            logger.info(
                "Pass %d: Lanczos3 resize %.3f -> shape %s",
                pass_idx,
                zoom_pass.resize_factor,
                image_input.shape,
            )
        else:
            image_input = image

        if zoom_pass.needs_tiling:
            tile_wh = (zoom_pass.tile_size_px, zoom_pass.tile_size_px)
            overlap_wh = (zoom_pass.overlap_px, zoom_pass.overlap_px)
        else:
            tile_wh = (0, 0)
            overlap_wh = (0, 0)

        per_pass_request = dataclasses.replace(
            base_request,
            text_prompt=zoom_pass.text_prompt,
            slicing_enabled=zoom_pass.needs_tiling,
            slice_width_height=tile_wh,
            overlap_width_height=overlap_wh,
            resize_factor=zoom_pass.resize_factor,
            is_full_image_pass=not zoom_pass.needs_tiling,
            pass_class_names=zoom_pass.class_names,
        )

        det_i: Detections2D = inference_engine.detect(image_input, request=per_pass_request)

        # Fire on_pass before remap (same contract as run_multi_zoom's hook)
        if on_pass_cb is not None:
            try:
                on_pass_cb(pass_idx, zoom_pass, image_input, det_i)
            except Exception:
                logger.debug("on_pass_cb raised; ignoring", exc_info=True)

        pass_peak_gb = torch.cuda.max_memory_allocated() / 1e9
        per_pass_vram.append(pass_peak_gb)
        logger.info(
            "Pass %d (classes=%s, tiled=%s, resize=%.3f): %d detections, peak VRAM=%.3f GB",
            pass_idx,
            zoom_pass.class_names,
            zoom_pass.needs_tiling,
            zoom_pass.resize_factor,
            len(det_i.input_boxes),
            pass_peak_gb,
        )

        if zoom_pass.resize_factor < 1.0:
            det_i = _mzd._remap_to_native(det_i, zoom_pass.resize_factor)
        det_i = _mzd._remap_class_ids(det_i, class_id_map)
        all_detections.append(det_i)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_s = time.perf_counter() - t0

    combined = _concat_detections(all_detections)
    combined = dedup_same_class_nms(combined, iou_threshold=base_request.iou_threshold)
    combined = dedup_same_class_ios(
        combined,
        ios_threshold=getattr(mz_cfg, "ios_threshold", 0.8),
        enabled=getattr(mz_cfg, "ios_enabled", False),
    )
    combined = dedup_cross_class(
        combined,
        iou_threshold=getattr(mz_cfg, "cross_class_iou_threshold", 0.7),
    )

    return elapsed_s, per_pass_vram, combined


def _check_monotonic(vram_list: list[float]) -> bool:
    """Return True if VRAM grew monotonically (every pass >= prior)."""
    if len(vram_list) <= 1:
        return False
    return all(vram_list[i] >= vram_list[i - 1] for i in range(1, len(vram_list)))


def _print_summary(
    *,
    scan_path: str,
    feature: str,
    image_shape: tuple,
    dino_info: dict,
    sam2_info: dict,
    zoom_passes: list,
    per_pass_vram: list[float],
    mz_elapsed_s: float,
    sz_elapsed_s: float | None,
    sz_n_detections: int,
    mz_n_detections: int,
) -> None:
    """Print the profiling summary table to stdout."""
    sep = "-" * 70
    print(sep)
    print("MULTI-ZOOM PROFILING RESULTS")
    print(sep)
    print(f"Scan:    {scan_path}")
    print(f"Feature: {feature}")
    print(f"Image:   {image_shape[1]} x {image_shape[0]} px (W x H)")
    print()

    print("--- Model Resize Confirmation ---")
    print(f"  DINO image_processor.size:     {dino_info.get('size')!r}")
    print(f"  DINO image_processor.max_size: {dino_info.get('max_size')!r}")
    a1_verdict = "OK (shortest_edge=800, max_size=1333)" if dino_info.get("confirmed_a1") else "MISMATCH — recalibrate defaults"
    print(f"  A1 verdict:                    {a1_verdict}")
    print(f"  SAM2 predictor.image_size:     {sam2_info.get('image_size')!r}")
    a2_verdict = "OK (1024)" if sam2_info.get("confirmed_a2") else "MISMATCH — update tile_size guidance"
    print(f"  A2 verdict:                    {a2_verdict}")
    print()

    print("--- Per-Pass VRAM (peak, GB) ---")
    print(f"  {'Pass':>4}  {'Type':<20}  {'Classes':<40}  {'VRAM (GB)':>10}")
    for i, (zoom_pass, vram_gb) in enumerate(zip(zoom_passes, per_pass_vram, strict=True)):
        pass_type = "full-image" if not zoom_pass.needs_tiling else f"tiled {zoom_pass.tile_size_px}px"
        classes_str = ", ".join(zoom_pass.class_names)
        print(f"  {i:>4}  {pass_type:<20}  {classes_str:<40}  {vram_gb:>10.3f}")
    print()

    monotonic = _check_monotonic(per_pass_vram)
    vram_verdict = "FLAT (OK — no monotonic growth)" if not monotonic else "MONOTONIC GROWTH — review gc/empty_cache hygiene"
    print(f"  VRAM-flat verdict: {vram_verdict}")
    print()

    print("--- Runtime Comparison ---")
    if sz_elapsed_s is not None:
        print(f"  Single-zoom baseline:  {sz_elapsed_s:.2f} s  ({sz_n_detections} detections)")
    else:
        print("  Single-zoom baseline:  skipped (run with --baseline to enable)")
    print(f"  Multi-zoom total:      {mz_elapsed_s:.2f} s  ({mz_n_detections} detections)")
    if sz_elapsed_s is not None and sz_elapsed_s > 0:
        ratio = mz_elapsed_s / sz_elapsed_s
        print(f"  Overhead factor:       {ratio:.2f}x")
    print()
    print(sep)


# ---------------------------------------------------------------------------
# Overlay helpers (heavy deps; only called from main() after imports)
# ---------------------------------------------------------------------------


def _feature_image_to_bgr_uint8(image: "np.ndarray") -> "np.ndarray":
    """Convert a 2-D float feature image to a 3-channel BGR uint8 for cv2.

    Handles float images that exceed [0, 1] via min-max normalisation.
    NaN values are zeroed before normalisation.
    """
    import numpy as np

    img = image.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    img_u8 = img.astype("uint8")
    # Convert single-channel to BGR
    import cv2
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def _build_on_pass_callback(
    overlay_dir: "Path",
) -> object:
    """Return an on_pass callback that writes per-pass input + overlay PNGs.

    Filenames: pass{idx}_input_resize{r}_tile{t}.png
               pass{idx}_overlay_resize{r}_tile{t}.png

    The callback captures pass-count per resize band for unique naming.
    Heavy imports (cv2, supervision) happen inside the returned closure.
    """

    per_resize_tile_count: dict[float, int] = {}

    def _on_pass(
        idx: int,
        zoom_pass: object,
        image_input: "np.ndarray",
        det: "Detections2D",
    ) -> None:
        import cv2
        import numpy as np
        import supervision as sv
        from supervision.draw.color import ColorPalette

        from tls2dseg.supervision_utils import CUSTOM_COLOR_MAP

        resize_factor = getattr(zoom_pass, "resize_factor", 1.0)
        tile_size = getattr(zoom_pass, "tile_size_px", None)

        r_str = f"{resize_factor:.3f}".replace(".", "p")
        t_str = str(tile_size) if tile_size is not None else "none"
        stem = f"pass{idx}_resize{r_str}_tile{t_str}"

        # --- input image ---
        bgr = _feature_image_to_bgr_uint8(image_input)
        input_path = overlay_dir / f"{stem}_input.png"
        cv2.imwrite(str(input_path), bgr)
        logger.info("Saved input image: %s", input_path)

        # --- overlay image ---
        n = len(det.input_boxes) if hasattr(det, "input_boxes") else 0
        if n > 0:
            xyxy = np.asarray(det.input_boxes, dtype=np.float32)
            class_ids = np.asarray(det.class_ids, dtype=np.int32)
            confs = np.asarray(det.confidences, dtype=np.float32)
            sv_det = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confs)
            labels = [
                f"{name} {conf:.2f}"
                for name, conf in zip(det.class_names, confs, strict=False)
            ]
            palette = ColorPalette.from_hex(CUSTOM_COLOR_MAP)
            annotated = bgr.copy()
            annotated = sv.BoxAnnotator(color=palette).annotate(annotated, sv_det)
            annotated = sv.LabelAnnotator(color=palette).annotate(annotated, sv_det, labels=labels)
        else:
            annotated = bgr.copy()

        overlay_path = overlay_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(overlay_path), annotated)
        logger.info(
            "Saved overlay (pass %d, %d detections): %s",
            idx,
            n,
            overlay_path,
        )

    return _on_pass


def _save_final_combined_overlay(
    image_native: "np.ndarray",
    combined: "Detections2D",
    overlay_dir: "Path",
    max_dim: int = 4000,
) -> None:
    """Save final_combined_overlay.png on a downscaled native image.

    The native panorama can be ~8k x 43k pixels; cap the longest dimension at
    max_dim and scale the combined bounding boxes by the same factor.
    """
    import cv2
    import numpy as np
    import supervision as sv
    from supervision.draw.color import ColorPalette

    from tls2dseg.supervision_utils import CUSTOM_COLOR_MAP

    h, w = image_native.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        logger.info(
            "Downscaling native image %.3f -> (%d x %d) for final overlay",
            scale,
            new_w,
            new_h,
        )
        bgr_full = _feature_image_to_bgr_uint8(image_native)
        bgr = cv2.resize(bgr_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        bgr = _feature_image_to_bgr_uint8(image_native)

    n = len(combined.input_boxes) if hasattr(combined, "input_boxes") else 0
    if n > 0:
        xyxy = np.asarray(combined.input_boxes, dtype=np.float32) * scale
        class_ids = np.asarray(combined.class_ids, dtype=np.int32)
        confs = np.asarray(combined.confidences, dtype=np.float32)
        sv_det = sv.Detections(xyxy=xyxy, class_id=class_ids, confidence=confs)
        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(combined.class_names, confs, strict=False)
        ]
        palette = ColorPalette.from_hex(CUSTOM_COLOR_MAP)
        annotated = bgr.copy()
        annotated = sv.BoxAnnotator(color=palette).annotate(annotated, sv_det)
        annotated = sv.LabelAnnotator(color=palette).annotate(annotated, sv_det, labels=labels)
    else:
        annotated = bgr.copy()

    out_path = overlay_dir / "final_combined_overlay.png"
    cv2.imwrite(str(out_path), annotated)
    logger.info(
        "Saved final combined overlay (%d detections, scale=%.3f): %s",
        n,
        scale,
        out_path,
    )


def _write_plan_dump(zoom_passes: list, overlay_dir: "Path") -> None:
    """Write zoom-pass plan parameters to plan_dump.txt in overlay_dir."""
    lines = ["# Multi-zoom plan dump\n"]
    for i, zp in enumerate(zoom_passes):
        lines.append(f"pass {i}:")
        lines.append(f"  class_names: {list(zp.class_names)}")
        lines.append(f"  resize_factor: {zp.resize_factor}")
        lines.append(f"  needs_tiling: {zp.needs_tiling}")
        lines.append(f"  tile_size_px: {zp.tile_size_px}")
        lines.append(f"  overlap_px: {zp.overlap_px}")
        lines.append(f"  text_prompt: {zp.text_prompt!r}")
        lines.append("")
    dump_path = overlay_dir / "plan_dump.txt"
    dump_path.write_text("\n".join(lines))
    logger.info("Saved plan dump: %s", dump_path)


def main() -> None:
    """Entry point — all heavy imports happen here."""
    import time

    parser = argparse.ArgumentParser(
        description="Profile multi-zoom dispatcher on a real scan."
    )
    parser.add_argument("--config", required=True, help="Path to tls2dseg YAML config")
    parser.add_argument("--scan", required=True, help="Path to one .e57 scan file")
    parser.add_argument(
        "--classes",
        default="tree:5,pole:0.1",
        help="Comma-separated 'name:size_m' pairs (default: tree:5,pole:0.1)",
    )
    parser.add_argument(
        "--feature",
        default="intensity",
        help="Which projection feature to use (default: intensity)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=0,
        help="Override number of tiled passes (0 = compute from footprint math)",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        help=(
            "Directory for per-pass input/overlay PNGs and plan_dump.txt. "
            "Defaults to ./profile_out/<scan_stem>/"
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run single-zoom baseline (expensive — skipped by default for fast re-runs).",
    )
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        help="Skip single-zoom baseline (default).",
    )
    args = parser.parse_args()

    # --- heavy imports inside main() ---
    import gc

    import numpy as np
    import torch
    from pchandler.data_io import load_e57

    from tls2dseg.config.loader import load_config
    from tls2dseg.engines import build_inference_engine, build_projection_engine
    from tls2dseg.engines.inference.multi_zoom_plan import compute_zoom_passes
    from tls2dseg.types import InferenceRequest

    # --- parse inputs ---
    config_path = Path(args.config).resolve()
    scan_path = Path(args.scan).resolve()
    class_sizes = _parse_classes(args.classes)

    # --- resolve overlay-dir ---
    if args.overlay_dir is not None:
        overlay_dir = Path(args.overlay_dir).resolve()
    else:
        overlay_dir = Path("profile_out") / scan_path.stem
    overlay_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Overlay directory: %s", overlay_dir)

    logger.info("Loading config from %s", config_path)
    cfg = load_config(config_path)

    logger.info("Loading scan %s", scan_path)
    pcd = load_e57(scan_path, stay_prcs=False, save_prcs_info=True)

    # --- build inference engine ---
    logger.info("Building inference engine (%s)", cfg.inference.type)
    engine_kwargs: dict = {
        "object_detection_model_id": cfg.inference.object_detection_model_id,
        "sam_box_prompt_batch_size": cfg.inference.sam_box_prompt_batch_size,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if cfg.inference.type == "grounded_sam2":
        engine_kwargs["sam2_checkpoint"] = str(cfg.inference.sam2_checkpoint)
        engine_kwargs["sam2_model_config"] = cfg.inference.sam2_model_config
    elif cfg.inference.type == "grounded_sam2_hf":
        engine_kwargs["sam2_hf_model_id"] = cfg.inference.sam2_hf_model_id

    inference_engine = build_inference_engine(cfg.inference.type, **engine_kwargs)

    # --- confirm model resize assumptions (A1/A2) ---
    dino_info = _confirm_dino_sizes(inference_engine)
    sam2_info = _confirm_sam2_resize(inference_engine)

    # --- project one scan to get a real image ---
    logger.info("Projecting scan (feature=%s, skip_image_reduction=True)", args.feature)
    image_generation_parameters: dict = {
        "image_width": cfg.projection.image_width,
        "scan_resolution": cfg.projection.scan_resolution,
        "rotate_pcd": cfg.projection.rotate_pcd,
        "rasterization_method": cfg.projection.rasterization_method,
        "features": [args.feature],
    }
    projection_engine = build_projection_engine(
        "spherical",
        image_generation_parameters=image_generation_parameters,
        pcd_path=scan_path,
    )
    projection_results = projection_engine.project(
        pcd,
        features=[args.feature],
        resolution=(0, 0),
        skip_image_reduction=True,
    )
    pr = projection_results[0]
    image_native: np.ndarray = pr.image
    d_azim_rad: float = pr.d_azim_rad

    logger.info("Native image shape: %s (H x W)", image_native.shape)
    logger.info("d_azim_rad: %.6e rad/px", d_azim_rad)

    # --- compute per-class range quantiles ---
    ranges: np.ndarray = pcd.spherical_coordinates[:, 0]
    range_near_m = float(np.nanpercentile(ranges, 10))
    range_far_m = float(np.nanpercentile(ranges, 90))
    logger.info("Range 10th pctile (near): %.2f m, 90th pctile (far): %.2f m", range_near_m, range_far_m)

    # --- compute zoom passes ---
    zoom_passes = compute_zoom_passes(
        class_sizes=class_sizes,
        d_azim_rad=d_azim_rad,
        range_near_m=range_near_m,
        range_far_m=range_far_m,
    )
    logger.info("Computed %d zoom pass(es) (including full-image pass)", len(zoom_passes))
    for i, zp in enumerate(zoom_passes):
        logger.info(
            "  Pass %d: classes=%s, resize=%.3f, tiled=%s, tile=%s px",
            i,
            zp.class_names,
            zp.resize_factor,
            zp.needs_tiling,
            zp.tile_size_px,
        )

    # --- write plan dump ---
    _write_plan_dump(zoom_passes, overlay_dir)

    # --- build base InferenceRequest ---
    text_prompt = ". ".join(sorted(class_sizes.keys(), key=len, reverse=True)) + "."
    base_request = InferenceRequest(
        text_prompt=text_prompt,
        box_threshold=cfg.inference.box_threshold,
        text_threshold=cfg.inference.text_threshold,
        slicing_enabled=cfg.inference.slicing.enabled,
        slice_width_height=cfg.inference.slicing.slice_width_height,
        overlap_width_height=cfg.inference.slicing.overlap_width_height,
        iou_threshold=cfg.inference.slicing.iou_threshold,
        overlap_filter_strategy=cfg.inference.slicing.overlap_filter_strategy,
        large_object_removal_threshold=cfg.inference.large_object_removal_threshold,
        partial_detection_edge_touching_threshold=cfg.inference.partial_detection_edge_touching_threshold,
        thread_workers=cfg.runtime.n_workers,
        empty_slice_removal_threshold=cfg.inference.slicing.empty_slice_removal_threshold,
    )

    # --- build minimal ctx for class_id_map ---
    class DataClass:
        pass

    ctx = DataClass()
    ctx.class_id_map = {name: i + 1 for i, name in enumerate(class_sizes)}
    ctx.per_class_metadata = {}

    # --- build minimal mz_cfg ---
    mz_cfg = DataClass()
    mz_cfg.ios_enabled = False
    mz_cfg.ios_threshold = 0.8
    mz_cfg.cross_class_iou_threshold = 0.7

    # Warm up the engine with a small blank image to ensure CUDA is initialised
    # before timing measurements.
    logger.info("Warming up inference engine...")
    warm_h = min(image_native.shape[0], 64)
    warm_w = min(image_native.shape[1], 64)
    warm_img = np.zeros((warm_h, warm_w), dtype=np.float32)
    try:
        inference_engine.detect(warm_img, request=base_request)
    except Exception as exc:
        logger.warning("Warmup detect() raised (expected for degenerate input): %s", exc)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- optional single-zoom baseline ---
    sz_elapsed_s: float | None = None
    sz_n_detections = 0
    if args.baseline:
        logger.info("Running single-zoom baseline...")
        sz_elapsed_s, sz_detections = _run_single_zoom_baseline(
            image_native, inference_engine, base_request
        )
        sz_n_detections = len(sz_detections.input_boxes)
        logger.info(
            "Single-zoom baseline: %.2f s, %d detections", sz_elapsed_s, sz_n_detections
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("Skipping single-zoom baseline (use --baseline to enable).")

    # --- build the per-pass overlay callback ---
    on_pass_cb = _build_on_pass_callback(overlay_dir)

    # --- multi-zoom profiled run ---
    logger.info("Running multi-zoom (profiled)...")
    mz_elapsed_s, per_pass_vram, mz_detections = _run_multi_zoom_profiled(
        image_native,
        inference_engine,
        base_request,
        zoom_passes,
        ctx,
        mz_cfg,
        on_pass_cb=on_pass_cb,
    )

    mz_n_detections = len(mz_detections.input_boxes)
    logger.info("Multi-zoom: %.2f s, %d detections", mz_elapsed_s, mz_n_detections)

    # Per-pass count summary
    logger.info(
        "Count summary: total combined=%d (after dedup); check per-pass overlays in %s",
        mz_n_detections,
        overlay_dir,
    )

    # --- save final combined overlay ---
    _save_final_combined_overlay(image_native, mz_detections, overlay_dir)

    # --- print summary ---
    _print_summary(
        scan_path=str(scan_path),
        feature=args.feature,
        image_shape=image_native.shape,
        dino_info=dino_info,
        sam2_info=sam2_info,
        zoom_passes=zoom_passes,
        per_pass_vram=per_pass_vram,
        mz_elapsed_s=mz_elapsed_s,
        sz_elapsed_s=sz_elapsed_s,
        sz_n_detections=sz_n_detections,
        mz_n_detections=mz_n_detections,
    )


if __name__ == "__main__":
    main()

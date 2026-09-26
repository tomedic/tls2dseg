"""Load one scan and project it to full-resolution spherical images for labeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("tls2dseg.labeling.session")

DEFAULT_FEATURES: tuple[str, ...] = ("intensity", "range")


@dataclass
class LabelingSession:
    """A loaded scan plus its spherical images.

    ``pcd`` stays in the (possibly rotated) scanner frame used for projection;
    ``socs_rotation`` is the 4x4 rotation ``project()`` applied to it, needed to
    recover the correct SOCS→PRCS pose on export.
    """

    pcd: Any
    scan_path: Path
    scan_index: int
    scan_count: int
    images: dict[str, np.ndarray]  # float16 in [0, 1], (H, W)
    d_azim_rad: float
    projection_params: dict
    socs_rotation: np.ndarray = field(default_factory=lambda: np.eye(4))
    tmat_compensated: bool = False

    @property
    def image_hw(self) -> tuple[int, int]:
        h, w = next(iter(self.images.values())).shape[:2]
        return int(h), int(w)

    @property
    def gt_name(self) -> str:
        stem = self.scan_path.stem
        return stem if self.scan_count == 1 else f"{stem}_s{self.scan_index}"


def default_projection_params() -> dict:
    """Projection defaults taken from ``ProjectionConfig`` so they stay in sync."""
    from tls2dseg.config.models import ProjectionConfig

    fields = ProjectionConfig.model_fields
    return {
        "image_width": fields["image_width"].default,
        "scan_resolution": fields["scan_resolution"].default,
        "rotate_pcd": fields["rotate_pcd"].default,
        "rasterization_method": fields["rasterization_method"].default,
        "flip_upsidedown_scans_deg": None,
    }


def e57_scan_count(scan_path: Path) -> int:
    import pye57

    e57 = pye57.E57(str(scan_path), mode="r")
    try:
        return int(e57.scan_count)
    finally:
        e57.close()


def load_scan(
    scan_path: Path,
    scan_index: int = 0,
    projection_params: dict | None = None,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> LabelingSession:
    """Load ``scan_path`` (one scan of an .e57) and project it at full resolution.

    No range/ROI filtering or subsampling is applied, so labels are lifted onto
    the full-resolution cloud. The returned ``projection_params`` are pinned
    (numeric width, rotation and resolution) so a re-projection with them yields
    the same image even where the originals were randomized ``"auto"`` estimates.
    """
    from pchandler.data_io import load_e57

    from tls2dseg.engines import build_projection_engine

    scan_path = Path(scan_path)
    scan_count = e57_scan_count(scan_path)
    if not 0 <= scan_index < scan_count:
        raise ValueError(f"{scan_path.name} has {scan_count} scan(s); scan index {scan_index} is out of range")

    params = {**default_projection_params(), **(projection_params or {})}
    params["features"] = list(features)
    params["output_resolution"] = None

    logger.info("Loading scan %d of %s", scan_index, scan_path)
    pcd = load_e57(scan_path, point_cloud_index=scan_index, stay_prcs=False, save_prcs_info=True)

    engine = build_projection_engine("spherical", image_generation_parameters=dict(params), pcd_path=scan_path)
    results = engine.project(pcd, features=list(features), resolution=(0, 0), skip_image_reduction=True)

    rotation = results[0].socs_rotation

    images = {r.feature_name: np.asarray(r.image, dtype=np.float16) for r in results}
    h, w = next(iter(images.values())).shape
    logger.info("Projected %s: %d x %d px, features=%s", scan_path.name, w, h, list(images))

    # rotation = Rz(theta) @ Rx(flip); Rx keeps the x-axis, so column 0 is (cos theta, sin theta, 0).
    d_azim_rad = float(results[0].d_azim_rad)
    pinned = {
        **params,
        "image_width": int(w),
        "rotate_pcd": round(float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))), 6),
        "scan_resolution": float(np.rad2deg(d_azim_rad)),
    }

    return LabelingSession(
        pcd=pcd,
        scan_path=scan_path,
        scan_index=scan_index,
        scan_count=scan_count,
        images=images,
        d_azim_rad=d_azim_rad,
        projection_params=pinned,
        socs_rotation=rotation,
    )


def to_uint8(image: np.ndarray) -> np.ndarray:
    """[0, 1] float image → uint8 (NaN → 255, matching the pipeline's NaN→max)."""
    img = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=1.0)
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_multiscale(image_u8: np.ndarray, min_side: int = 2048) -> list[np.ndarray]:
    """Factor-2 image pyramid (full resolution first) for napari multiscale display."""
    import cv2

    levels = [image_u8]
    while max(levels[-1].shape[:2]) > min_side:
        h, w = levels[-1].shape[:2]
        levels.append(cv2.resize(levels[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
    return levels

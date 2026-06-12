"""SphericalProjectionEngine — concrete ProjectionEngine for TLS spherical projection.

Phase 4 plan 02 Task 2 (ENG-03, D-B-03). Implements the ``ProjectionEngine``
Protocol by wrapping the existing ``pc2img_utils`` spherical projection logic:

* ``pc2img_run`` (image generation via PCDImageLink + SphericalImageGeneratorFromPCD)
* Rotation helpers (``rotate_pcd_around_z``, ``rotate_pcd_around_x``,
  ``check_was_scanner_upsidedown``, ``rotate_pcd_to_azimuth_gap``,
  ``resolve_rotate_pcd_parameter``)
* Resolution math (``resolve_scanning_resolution_parameter``,
  ``compute_image_dimensions``, ``resolve_necessary_image_resolution``,
  ``reduce_image_resolution``)

Design decisions:
- D-A-05 (ABSOLUTE heavy-import contract): all of ``pyvips``, ``pc2img``,
  ``pchandler``, and ``sklearn`` are imported INSIDE method bodies only — NEVER
  at module level.  Importing this module MUST NOT trigger those packages.
- D-B-03 (thickness lean): ``project()`` bundles the D-B-03 INSIDE steps
  (rotation, resolution math, rasterisation, image-resolution reduction).
  RoI/range filtering stays outside (orchestrator responsibility).
- The helper functions remain in ``pc2img_utils`` (not physically moved here)
  to keep Task 2 self-contained; run.py import re-pointing happens in Task 3.

Closest analog: ``tls2dseg.pc2img_utils`` (wraps lines 18-68, 71-112, 678-735).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tls2dseg.types import ProjectionResult

logger = logging.getLogger("tls2dseg.engines.projection.spherical")


class SphericalProjectionEngine:
    """Concrete ``ProjectionEngine`` for TLS spherical projection via pc2img.

    Wraps the existing ``pc2img_utils`` projection logic behind the
    ``ProjectionEngine`` Protocol.  All heavy imports (pyvips, pc2img,
    pchandler, sklearn) are confined to method bodies so that importing
    this class never triggers them (D-A-05).

    Parameters
    ----------
    image_generation_parameters :
        Dictionary with projection settings (``rasterization_method``,
        ``features``, ``scan_resolution``, ``image_width``, ``rotate_pcd``,
        ``output_resolution``, etc.).  Matches the current
        ``image_generation_parameters`` dict used throughout ``run.py``.
    pcd_path :
        Path to the source point-cloud file (used for output image naming).
    """

    def __init__(
        self,
        image_generation_parameters: dict,
        pcd_path: object | None = None,
    ) -> None:
        # Store config only — no heavy imports here (D-A-05).
        self._params = image_generation_parameters
        self._pcd_path = pcd_path

    def project(
        self,
        pcd: object,
        *,
        features: list[str],
        resolution: tuple[int, int],
    ) -> list[ProjectionResult]:
        """Project a point cloud scan to a list of per-feature spherical images.

        Bundles the full D-B-03 projection pipeline:
        1. Resolve scan resolution (``resolve_scanning_resolution_parameter``)
        2. Compute image dimensions (``compute_image_dimensions``)
        3. Optional up-side-down flip (``check_was_scanner_upsidedown`` +
           ``rotate_pcd_around_x``)
        4. Rotate to azimuth gap (``resolve_rotate_pcd_parameter`` +
           ``rotate_pcd_to_azimuth_gap`` internally)
        5. Rasterise with pc2img (``pc2img_run``)
        6. Optional resolution reduction (``resolve_necessary_image_resolution``
           + ``reduce_image_resolution``)

        Heavy imports happen inside this method (D-A-05).

        Parameters
        ----------
        pcd :
            ``PointCloudData`` instance (typed as ``object`` here to avoid
            pchandler at Protocol definition time; callers pass the real type).
        features :
            List of scalar-field names to render (overrides params default if
            non-empty; falls back to ``self._params["features"]`` otherwise).
        resolution :
            ``(height, width)`` target image resolution.  Stored in params as
            ``image_width`` (height is derived from the FoV aspect ratio by
            pc2img_run when ``image_height=0``); the ``resolution`` argument
            here is accepted for Protocol conformance but the per-scan height is
            always derived from the scan's FoV ratio.

        Returns
        -------
        list[ProjectionResult]
            One ``ProjectionResult`` per entry in ``features``.
        """
        # Heavy imports live here — never at module level (D-A-05 absolute contract).
        from pathlib import Path

        from tls2dseg.pc2img_utils import (
            check_was_scanner_upsidedown,
            compute_image_dimensions,
            pc2img_run,
            reduce_image_resolution,
            resolve_necessary_image_resolution,
            resolve_rotate_pcd_parameter,
            resolve_scanning_resolution_parameter,
            rotate_pcd_around_x,
        )
        from tls2dseg.types import ProjectionResult

        params = dict(self._params)  # shallow copy — avoid mutating the shared dict

        # Override features in params if caller passed a non-empty list
        if features:
            params["features"] = features

        # Resolve image width from resolution tuple (protocol contract uses (H, W))
        _, width = resolution
        if width > 0:
            params["image_width"] = width

        pcd_path = self._pcd_path if self._pcd_path is not None else Path("unknown.e57")

        # --- Step 1: scan-resolution + image-dimension math ---
        d_azim_rad, d_elev_rad = resolve_scanning_resolution_parameter(pcd, params)
        image_width, image_height = compute_image_dimensions(pcd, params, d_azim_rad, d_elev_rad)

        # --- Step 2: optional upside-down flip ---
        if check_was_scanner_upsidedown(pcd):
            rotate_pcd_around_x(pcd, alpha_deg=180.0)
            logger.debug("Scanner was upside-down — applied 180° X-axis rotation.")

        # --- Step 3: rotate to azimuth gap ---
        resolve_rotate_pcd_parameter(pcd, params)

        # --- Step 4: rasterise ---
        images_raw = pc2img_run(pcd, pcd_path, params, image_width=image_width, image_height=image_height)

        # --- Step 5: optional resolution reduction ---
        if "output_resolution" in params and params.get("output_resolution") is not None:
            reduction_coeff = resolve_necessary_image_resolution(pcd, params, d_azim_rad)
            if reduction_coeff < 1.0:
                images_raw = reduce_image_resolution(images_raw, reduction_coeff, params, pcd_path)

        # --- Step 6: wrap as ProjectionResult ---
        results: list[ProjectionResult] = []
        for feature_name, image, path in images_raw:
            path_obj = Path(path) if isinstance(path, str) else path
            results.append(ProjectionResult(feature_name=feature_name, image=image, path=path_obj))

        return results

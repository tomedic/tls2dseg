"""Typed pydantic v2 + pydantic-settings YAML config models for tls2dseg.

Replaces the 6 free-floating parameter dicts at ``pipeline/run.py:99-217``
with 11 strictly-typed sub-models. Phase 3 ground floor — every Phase 4+
plan layers on top of this.

Conventions (locked by CONTEXT.md decisions D-A1-00 through D-A1-17):

- Every model (including ``RunConfig``) uses ``extra="forbid"`` and
  ``frozen=True`` (CFG-02 + D-A1-17 + D-A1-05). Unknown YAML keys raise
  ``ValidationError(type="extra_forbidden")``; attribute assignment on a
  constructed instance raises ``ValidationError(type="frozen_instance")``.
- Every ``Field`` carries ``json_schema_extra={"tag": "<tag>"}`` where
  ``<tag>`` is one of ``{"primary","tuning","pipings","other"}`` (D-A1-04
  4-tag vocab). Phase 8 DOC-02 groups generated schema docs by tag.
- Fields ordered within each sub-model by tag: primary → tuning → pipings
  → other (D-A1-02). Inline ``# tag: <tag>`` comments mirror the field.
- ``snake_case`` everywhere (D-A1-03). Legacy hyphenated keys (e.g.
  ``sam2-checkpoint``) are hard-rejected by ``extra="forbid"``. No
  deprecated-alias clutter — v1-internal scope, no legacy YAMLs.
- Sub-model defaults use ``SubModel()`` shape — NEVER
  ``Field(default_factory=SubModel)``. Hydra-zen compatibility (D-A1-00).
- Phase 4 engine-abstraction discriminator slots ship as
  ``projection.type``, ``inference.type``, ``fusion.type`` Literal fields
  with single-value defaults (D-A1-12/13/16).

Required-no-default fields (raise ``ValidationError`` when omitted):
- ``RunConfig.mode``                  (MODE-01, D-A1-06)
- ``IOConfig.input_path``             (D-A1-07)
- ``IOConfig.output_dir``             (D-A1-07)
- ``PromptConfig.text``               (D-A1-09)
- ``PreprocessingConfig.output_resolution_m``  (D-A1-11)
- ``ProjectionConfig.features``       (MODE-06, D-A1-12; min_length=1)
- ``GroundedSAM2Config.sam2_checkpoint`` (D-A1-13; supports ``${ENV_VAR}``; required for ``type: grounded_sam2``)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tls2dseg.config.text import split_class_keys

logger = logging.getLogger("tls2dseg.config.models")


# ─────────────────────────────────────────────────────────────────────────────
# IOConfig — input/output paths + run-id strategy
# ─────────────────────────────────────────────────────────────────────────────
class IOConfig(BaseModel):
    """Input/output paths + per-run-dir strategy (D-A1-07)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    input_path: Path = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Path to input point clouds (file or directory).",
    )
    # tag: primary
    output_dir: Path = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Root directory under which per-run output dirs are created.",
    )
    # tag: pipings
    resume_from_checkpoint: bool = Field(
        True,
        json_schema_extra={"tag": "pipings"},
        description="Check current run's intermediate/ for partial state on startup (D-A2-08).",
    )
    # tag: pipings
    save_intermediate: bool = Field(
        True,
        json_schema_extra={"tag": "pipings"},
        description="Persist stage-1 partials to intermediate/stage_1_partial/.",
    )
    # tag: other
    file_format: Literal["e57"] = Field(
        "e57",
        json_schema_extra={"tag": "other"},
        description="Input point cloud file format. Only e57 supported in v1.",
    )
    # tag: other
    save_d2d: bool = Field(
        False,
        json_schema_extra={"tag": "other"},
        description="Save 2D detections (debug aid).",
    )
    # tag: other
    run_id_strategy: Literal["timestamp", "timestamp_scanset"] = Field(
        "timestamp_scanset",
        json_schema_extra={"tag": "other"},
        description="How to construct the per-run dir name (D-A2-06).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeConfig — device + workers + CPU-fallback policy
# ─────────────────────────────────────────────────────────────────────────────
class RuntimeConfig(BaseModel):
    """Device + worker + CPU-fallback policy (D-A1-08)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: pipings
    device: Literal["auto", "cpu", "cuda"] = Field(
        "auto",
        json_schema_extra={"tag": "pipings"},
        description="Compute device selection. 'auto' picks CUDA if available, else CPU.",
    )
    # tag: pipings
    n_workers: int = Field(
        12,
        json_schema_extra={"tag": "pipings"},
        description="Parallel workers for stage-1 per-scan loop.",
    )
    # tag: pipings
    accept_cpu_fallback: bool = Field(
        True,
        json_schema_extra={"tag": "pipings"},
        description="Allow loud CPU fallback when CUDA absent (CPU-03 contract).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PromptConfig — text prompt for Grounded-DINO
# ─────────────────────────────────────────────────────────────────────────────
class PromptConfig(BaseModel):
    """Text prompt for Grounded-DINO (D-A1-09)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    text: str = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description=(
            "Space-or-dot-separated open-vocab class prompt for Grounded-DINO. "
            "Auto-lowercased + trailing period appended (closes pipeline/run.py:170 TODO)."
        ),
    )
    # tag: primary
    sizes_m: list[float] | None = Field(
        None,
        json_schema_extra={"tag": "primary"},
        description=(
            "Representative physical size (m) per class token in `text`, in order. "
            "Required when multi-zoom is active; otherwise optional."
        ),
    )

    @field_validator("text", mode="after")
    @classmethod
    def _normalize_text(cls, v: str) -> str:
        """Lowercase, drop empty class segments, end in exactly one period."""
        keys = split_class_keys(v.lower())
        if not keys:
            raise ValueError("prompt.text must be non-empty")
        return ". ".join(keys) + "."

    @model_validator(mode="after")
    def _validate_sizes_m(self) -> PromptConfig:
        if self.sizes_m is None:
            return self
        if any(s <= 0 for s in self.sizes_m):
            raise ValueError("All sizes_m values must be > 0")
        keys = split_class_keys(self.text)
        if len(self.sizes_m) != len(keys):
            raise ValueError(
                f"sizes_m has {len(self.sizes_m)} entries but prompt.text has {len(keys)} class token(s): {keys}"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# PreprocessingConfig — point cloud preprocessing
# ─────────────────────────────────────────────────────────────────────────────
class PreprocessingConfig(BaseModel):
    """Point cloud preprocessing (D-A1-11).

    ``output_resolution_m`` has NO default — must declare per dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    output_resolution_m: float = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Voxel downsample resolution in meters. Must be set per dataset.",
    )
    # tag: primary
    range_limits_m: tuple[float, float] | None = Field(
        None,
        json_schema_extra={"tag": "primary"},
        description="(min, max) range from scanner origin in meters. None = no limits.",
    )
    # tag: primary
    roi_polygon_m: list[tuple[float, float]] | None = Field(
        None,
        json_schema_extra={"tag": "primary"},
        description=(
            "List of (x, y) vertices defining 2D ROI polygon in scanner-local meters. "
            "None = no ROI filter. Renamed from misleading 'roi_limits'."
        ),
    )
    # tag: tuning
    flip_upsidedown_scans_deg: float | None = Field(
        None,
        json_schema_extra={"tag": "tuning"},
        description="Rotation about x-axis (degrees) to flip upside-down scans. None = no flip.",
    )
    # tag: other
    keep_confidences: bool = Field(
        False,
        json_schema_extra={"tag": "other"},
        description="Retain per-point detection confidences in output.",
    )
    # tag: other
    assign_random_color_per_instance: bool = Field(
        False,
        json_schema_extra={"tag": "other"},
        description="Color each detected instance with a random RGB (visualization aid).",
    )

    @field_validator("range_limits_m", mode="after")
    @classmethod
    def _validate_range_limits(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is None:
            return v
        near, far = v
        if near < 0.5:
            raise ValueError(
                f"range_limits_m[0] (near bound) must be >= 0.5 m, got {near}. "
                "It also feeds the multi-zoom near range and must be strictly positive "
                "and meaningful; a near bound of 0 collapses multi-zoom to a single "
                "full-image pass. Use range_percentiles (leave range_limits_m unset) "
                "for an automatic near bound."
            )
        if far <= near:
            raise ValueError(f"range_limits_m must satisfy near < far, got near={near}, far={far}.")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# ProjectionConfig — spherical 2D projection from 3D
# ─────────────────────────────────────────────────────────────────────────────
# Pre-compiled regex for the scan_resolution string arm — anchored at full
# string, allows decimal point counts (incl. trailing-dot/leading-dot floats),
# requires the "<NUM><UNIT>@<NUM>m" shape. Examples: "1.6mm@10m", ".5cm@2m".
_SCAN_RES_STRING_RE = re.compile(r"^[0-9.]+(mm|cm|m)@[0-9.]+m$")

# Whitelist for the projection.features list — validated per element. Phase 3
# ships these 3; Phase 5 MODE-02..05 may extend.
_KNOWN_FEATURES: tuple[str, ...] = ("intensity", "range", "rgb")


class ProjectionConfig(BaseModel):
    """Spherical 2D projection from 3D point cloud (D-A1-12)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    features: list[Literal["intensity", "range", "rgb"]] = Field(
        ...,
        min_length=1,
        json_schema_extra={"tag": "primary"},
        description=(
            "List of point cloud features projected to 2D image channels. "
            "Validated against the projection engine's known features (MODE-06)."
        ),
    )
    # tag: other
    type: Literal["spherical"] = Field(
        "spherical",
        json_schema_extra={"tag": "other"},
        description="Projection engine discriminator (Phase 4 ENG-* expansion slot).",
    )
    # tag: other
    image_width: int | str = Field(
        "scan_resolution",
        json_schema_extra={"tag": "other"},
        description=(
            "Image width in pixels (int), a fractional string of the form "
            "'<frac>-scan_resolution' (e.g. '0.5-scan_resolution'), "
            "or 'scan_resolution' (match scanner native angular increment). "
            "Bare floats are rejected — use the string form for fractions."
        ),
    )
    # tag: other
    scan_resolution: float | Literal["auto"] | str = Field(
        "auto",
        json_schema_extra={"tag": "other"},
        description=(
            "Angular point spacing. 'auto' = compute from data; "
            "float = explicit angular increment (degrees); "
            "str of shape '<num><unit>@<dist>m' (e.g. '1.6mm@10m') = derived from physical distance."
        ),
    )
    # tag: other
    rotate_pcd: float | Literal["auto", False] = Field(
        "auto",
        json_schema_extra={"tag": "other"},
        description=(
            "Rotate point cloud around z-axis (degrees) before projection to shift "
            "spherical image horizontal seam. 'auto' = compute from data; "
            "false = no rotation (YAML-natural sentinel per feedback_yaml_natural_over_pythonic)."
        ),
    )
    # tag: other
    rasterization_method: Literal["raw", "nanconv", "bary_delaunay", "bary_knn"] = Field(
        "nanconv",
        json_schema_extra={"tag": "other"},
        description="Rasterization method for projecting points onto image grid.",
    )

    @field_validator("image_width", mode="before")
    @classmethod
    def _coerce_image_width(cls, v: object) -> object:
        """Dispatch value-type union for image_width (RESEARCH.md Pattern 6).

        Accepted: int pixel width, 'scan_resolution', '<frac>-scan_resolution'
        (e.g. '0.5-scan_resolution'), or a numeric string that converts to int.
        Bare floats are rejected — use '<frac>-scan_resolution' for fractions.
        """
        import re

        _ERR = (
            "image_width must be int (pixels), 'scan_resolution', or "
            "'<frac>-scan_resolution' (e.g. '0.5-scan_resolution'). Got: {v!r}"
        )
        if isinstance(v, bool):
            raise ValueError(_ERR.format(v=v))
        if isinstance(v, float):
            raise ValueError(
                f"image_width does not accept bare floats. "
                f"Use '<frac>-scan_resolution' (e.g. '{v}-scan_resolution') instead. Got: {v!r}"
            )
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "scan_resolution":
                return v
            if re.match(r"^[\d.]+[\s-]+scan_resolution$", s):
                return v
            # Numeric string — coerce to int only (no bare-float coercion).
            try:
                f = float(v)
                if f != int(f):
                    raise ValueError(
                        f"image_width string '{v}' converts to a non-integer float. "
                        f"Use '{f}-scan_resolution' for a fractional width."
                    )
                return int(f)
            except ValueError as exc:
                raise ValueError(_ERR.format(v=v)) from exc
        return v

    @field_validator("scan_resolution", mode="after")
    @classmethod
    def _validate_scan_resolution_str(cls, v: float | str) -> float | str:
        """Enforce '<num><unit>@<dist>m' regex on the string arm (D-A1-12)."""
        if isinstance(v, str) and v != "auto" and not _SCAN_RES_STRING_RE.match(v):
            raise ValueError(f"scan_resolution string must match '<num><unit>@<dist>m' (e.g. '1.6mm@10m'). Got: {v!r}")
        return v

    @field_validator("rotate_pcd", mode="before")
    @classmethod
    def _coerce_rotate_pcd(cls, v: object) -> object:
        """Preserve ``False`` as bool (don't coerce to 0.0) per RESEARCH.md Pattern 6."""
        # pydantic v2 smart mode handles this correctly for bool False, but YAML may
        # deliver it as a string "false" depending on quoting. Be explicit.
        if isinstance(v, str):
            lowered = v.lower()
            if lowered == "auto":
                return "auto"
            if lowered in ("false",):
                return False
            # Otherwise try numeric.
            try:
                return float(v)
            except ValueError as exc:
                raise ValueError(f"rotate_pcd must be float (degrees), 'auto', or false. Got: {v!r}") from exc
        return v

    @field_validator("features", mode="after")
    @classmethod
    def _validate_known_features(cls, v: list[str]) -> list[str]:
        """Reject unknown feature names (MODE-06).

        Pydantic Literal validation also catches this at parse time; the
        explicit validator surfaces a clearer error message listing all
        known features.
        """
        unknown = [f for f in v if f not in _KNOWN_FEATURES]
        if unknown:
            raise ValueError(f"Unknown projection feature(s): {unknown}. Allowed: {list(_KNOWN_FEATURES)}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# SlicingConfig — SAHI-style slicing sub-block of InferenceConfig
# ─────────────────────────────────────────────────────────────────────────────
class SlicingConfig(BaseModel):
    """SAHI-style slicing sub-block (D-A1-14).

    When ``enabled=False`` other slicing fields are ignored at runtime but
    still validated at config-load time (no conditional schema).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: tuning
    enabled: bool = Field(
        True,
        json_schema_extra={"tag": "tuning"},
        description="Run SAHI-style sliced inference (replaces legacy 'with_slice_inference').",
    )
    # tag: tuning
    slice_width_height: tuple[int, int] = Field(
        (200, 200),
        json_schema_extra={"tag": "tuning"},
        description="(width, height) of each image slice in pixels.",
    )
    # tag: tuning
    overlap_width_height: tuple[int, int] = Field(
        (150, 150),
        json_schema_extra={"tag": "tuning"},
        description="(width, height) of overlap between adjacent slices in pixels.",
    )
    # tag: tuning
    iou_threshold: float = Field(
        0.80,
        json_schema_extra={"tag": "tuning"},
        description="IoU threshold for cross-slice detection merging.",
    )
    # tag: tuning
    nms_combine_class_agnostic: bool = Field(
        False,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Class semantics for the single-view multi-feature NMS combine. "
            "false (default) = class-aware: overlapping detections of *different* classes "
            "are kept; only same-class duplicates are suppressed. "
            "true = class-agnostic: overlapping detections are suppressed regardless of class."
        ),
    )
    # tag: other
    overlap_filter_strategy: Literal["nms", "nmm"] = Field(
        "nms",
        json_schema_extra={"tag": "other"},
        description="Cross-slice overlap filter: NMS (Non-Max Suppression) or NMM (Non-Max Merging).",
    )
    # tag: other
    empty_slice_removal_threshold: float = Field(
        0.95,
        json_schema_extra={"tag": "other"},
        description="Skip slices whose empty-pixel fraction exceeds this threshold.",
    )
    # tag: tuning
    drop_incomplete_slices: bool = Field(
        True,
        json_schema_extra={"tag": "tuning"},
        description=(
            "When true, SAHI border/clamped slices smaller than the requested tile are dropped "
            "(current behaviour). When false, clamped edge and whole-image slices are still inferred."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MultiZoomConfig — per-class adaptive multi-zoom sub-block (MZ-01/06/10)
# ─────────────────────────────────────────────────────────────────────────────
class MultiZoomConfig(BaseModel):
    """Per-class adaptive multi-zoom inference sub-block (MZ-01 / D-CFG-01..04)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    active: bool = Field(
        False,
        json_schema_extra={"tag": "primary"},
        description=("Enable per-class adaptive multi-zoom. False (default) = single-zoom."),
    )
    # tag: tuning
    overview_pass: bool = Field(
        True,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Always run an extra full-FoV overview pass alongside the per-band tiled passes "
            "(N+1 passes). False = tiled bands only. The overview pass catches objects that "
            "fall outside every band and provides a safety net for the coarsest instances."
        ),
    )
    # tag: tuning
    footprint_band_frac: tuple[float, float] = Field(
        (0.075, 0.225),
        json_schema_extra={"tag": "tuning"},
        description=(
            "Target footprint band as fraction of DINO short-side (800 px). "
            "p_min = detection-quality floor; p_max = containment ceiling (D-A-05). "
            "Must satisfy 0 < p_min < p_max < 1."
        ),
    )
    # tag: tuning
    range_percentiles: tuple[float, float] = Field(
        (10.0, 90.0),
        json_schema_extra={"tag": "tuning"},
        description=(
            "Robust per-scan range bounds as percentiles (D-A-02). Used when preprocessing.range_limits_m is not set."
        ),
    )
    # tag: tuning
    cross_class_iou_threshold: float = Field(
        0.7,
        json_schema_extra={"tag": "tuning"},
        ge=0.0,
        le=1.0,
        description=(
            "IoU threshold for cross-class deduplication (MZ-06 / D-D-04). "
            "Overlapping detections of different classes above this threshold "
            "→ keep higher-confidence one."
        ),
    )
    # tag: tuning
    area_filter_px: tuple[int, int] | None = Field(
        None,
        json_schema_extra={"tag": "tuning"},
        description=("(min_px, max_px) area filter applied to detections. None = no area filter (D-CFG-03)."),
    )
    # tag: tuning
    ios_enabled: bool = Field(
        False,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Enable intersection-over-smaller (IoS) for same-class cross-scale dedup "
            "(D-D-03). IoS is restricted to same class to avoid killing nested objects."
        ),
    )
    # tag: tuning
    max_zoom_passes: int = Field(
        6,
        gt=0,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Maximum number of tiled zoom bands. Caps the greedy interval-cover when the "
            "footprint span would require more bands; a warning is emitted and coarsest "
            "instances rely on the always-on overview pass."
        ),
    )

    @field_validator("footprint_band_frac", mode="after")
    @classmethod
    def _validate_footprint_band(cls, v: tuple[float, float]) -> tuple[float, float]:
        p_min, p_max = v
        if not (0 < p_min < p_max < 1):
            raise ValueError(f"footprint_band_frac must satisfy 0 < p_min < p_max < 1. Got: {v}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# InferenceConfig — Grounded-DINO + SAM2 inference (discriminated union, D-C-02)
# ─────────────────────────────────────────────────────────────────────────────


class InferenceSharedConfig(BaseModel):
    """Shared inference settings — common to all engine types (D-C-02).

    Fields ordered: tuning → pipings → other.
    Every Field carries json_schema_extra={"tag": ...}.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: tuning
    box_threshold: float = Field(
        0.10,
        json_schema_extra={"tag": "tuning"},
        description="Grounded-DINO box confidence threshold.",
    )
    # tag: tuning
    text_threshold: float = Field(
        0.10,
        json_schema_extra={"tag": "tuning"},
        description="Grounded-DINO text-similarity threshold.",
    )
    # tag: tuning
    large_object_removal_threshold: float = Field(
        0.9,
        json_schema_extra={"tag": "tuning"},
        description="Drop detections whose bbox area exceeds this fraction of image area.",
    )
    # tag: tuning
    partial_detection_edge_touching_threshold: int = Field(
        5,
        json_schema_extra={"tag": "tuning"},
        description="Pixel margin within which a bbox is treated as edge-touching (partial detection).",
    )
    # tag: pipings
    sam_box_prompt_batch_size: int = Field(
        32,
        json_schema_extra={"tag": "pipings"},
        description="SAM2 box-prompt batch size; GPU memory/throughput trade-off.",
    )
    # tag: other (renamed from bbox_model_id per D-A1-13)
    object_detection_model_id: str = Field(
        "IDEA-Research/grounding-dino-base",
        json_schema_extra={"tag": "other"},
        description="HuggingFace model id for the object detection model (renamed from bbox_model_id).",
    )
    # tag: — (nested sub-block — has its own field tags)
    slicing: SlicingConfig = Field(
        default=SlicingConfig(),
        json_schema_extra={"tag": "other"},
        description="SAHI-style sliced inference sub-block.",
    )
    # tag: — (nested sub-block — has its own field tags)
    multi_zoom: MultiZoomConfig = Field(
        default=MultiZoomConfig(),
        json_schema_extra={"tag": "other"},
        description="Per-class adaptive multi-zoom inference sub-block.",
    )


class GroundedSAM2Config(InferenceSharedConfig):
    """Per-engine config for GroundedSAM2Engine (direct sam2 package, D-C-01).

    Required: sam2_checkpoint (no default — CFG-03).
    """

    # tag: other — discriminator (first, so pydantic finds it)
    type: Literal["grounded_sam2"] = Field(
        "grounded_sam2",
        json_schema_extra={"tag": "other"},
        description="Inference engine discriminator — selects GroundedSAM2Engine.",
    )
    # tag: pipings (required — no default; CFG-03 removes the legacy hardcoded path)
    sam2_checkpoint: Path = Field(
        ...,
        json_schema_extra={"tag": "pipings"},
        description=(
            "Path to the SAM2 checkpoint (.pt). Supports ${ENV_VAR} interpolation in YAML "
            "(e.g. '${SAM2_CHECKPOINT_PATH}'). Existence is checked lazily at runtime "
            "(NOT at validate-config time) per D-CD-05."
        ),
    )
    # tag: other (renamed from hyphenated sam2-model-config per D-A1-03)
    sam2_model_config: str = Field(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        json_schema_extra={"tag": "other"},
        description="SAM2 model config file path (relative to SAM2 package install).",
    )


class GroundedSAM2HFConfig(InferenceSharedConfig):
    """Per-engine config for GroundedSAM2HFEngine (HF transformers, D-C-01).

    ENG-05: uses Sam2Processor/Sam2Model.from_pretrained (transformers 5.11.0).
    Default model id confirmed by Task 1 probe (facebook/sam2.1-hiera-large).
    """

    # tag: other — discriminator
    type: Literal["grounded_sam2_hf"] = Field(
        "grounded_sam2_hf",
        json_schema_extra={"tag": "other"},
        description="Inference engine discriminator — selects GroundedSAM2HFEngine.",
    )
    # tag: other
    sam2_hf_model_id: str = Field(
        "facebook/sam2.1-hiera-large",
        json_schema_extra={"tag": "other"},
        description=(
            "HuggingFace model id for the SAM2 model. "
            "Confirmed: Sam2Processor/Sam2Model under transformers 5.11.0 (ENG-05 probe)."
        ),
    )


# Discriminated union — the public type alias used everywhere (RunConfig.inference).
# Pydantic resolves the concrete sub-model by matching ``inference.type``.
InferenceConfig = Annotated[
    GroundedSAM2Config | GroundedSAM2HFConfig,
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────────────────────────────────────
# D3DExtractionConfig — stage-1 per-scan 3D detection extraction
# ─────────────────────────────────────────────────────────────────────────────
class D3DExtractionConfig(BaseModel):
    """Stage-1 per-scan 3D detection extraction (D-A1-15).

    Split out from the legacy ``d3d_parameters`` dict — the stage-1 half.
    Active in both single-view and multi-view modes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: tuning
    bounding_box_type: Literal["obb", "aabb"] = Field(
        "obb",
        json_schema_extra={"tag": "tuning"},
        description="3D bounding box type: OBB (oriented) or AABB (axis-aligned).",
    )
    # tag: tuning (renamed from min_d3d_pcd_point_count per D-A1-15)
    min_point_count: int = Field(
        50,
        json_schema_extra={"tag": "tuning"},
        description="Drop instances with fewer points (renamed from min_d3d_pcd_point_count).",
    )
    # tag: other
    centroid_type: Literal["mean", "median", "bbox_c"] = Field(
        "bbox_c",
        json_schema_extra={"tag": "other"},
        description=(
            "Instance centroid method: points mean ('mean'), points median ('median'), or bbox centre ('bbox_c')."
        ),
    )
    # tag: other
    preprocess: bool = Field(
        True,
        json_schema_extra={"tag": "other"},
        description="Run per-instance preprocessing (SOR filter etc.) before bbox extraction.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FusionConfig — stage-2 cross-scan graph fusion (multi-view only)
# ─────────────────────────────────────────────────────────────────────────────
class FusionConfig(BaseModel):
    """Stage-2 cross-scan graph fusion (D-A1-16).

    Split out from the legacy ``d3d_parameters`` dict — the stage-2 half.
    Only consumed when ``mode == 'multi-view'``; otherwise parsed normally
    and ignored at runtime (soft handling).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: tuning
    merge_inst_of_same_class_only: bool = Field(
        False,
        json_schema_extra={"tag": "tuning"},
        description="Merge instances across scans only when they share the same class.",
    )
    # tag: tuning
    sparse_connectivity_threshold: int | float = Field(
        1,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Per-method threshold: 'knn' → k (int), 'radius' → distance in m (float). "
            "Semantics depends on sparse_connectivity_method."
        ),
    )
    # tag: tuning
    supporters_iou_threshold: float = Field(
        0.15,
        json_schema_extra={"tag": "tuning"},
        description="3D bbox IoU threshold above which two detections count as mutual supporters.",
    )
    # tag: tuning
    remove_outliers_by_support: bool = Field(
        True,
        json_schema_extra={"tag": "tuning"},
        description="Drop detections with anomalously large support (likely under-segmented).",
    )
    # tag: tuning
    outlier_detection_threshold: float = Field(
        0.01,
        json_schema_extra={"tag": "tuning"},
        description="Threshold for the outlier detection method (semantics depends on method).",
    )
    # tag: tuning
    graph_clustering_method: Literal["leiden", "hcs", "pcc"] = Field(
        "hcs",
        json_schema_extra={"tag": "tuning"},
        description="Graph clustering algorithm for cross-scan instance fusion.",
    )
    # tag: tuning
    min_supporters: int = Field(
        1,
        json_schema_extra={"tag": "tuning"},
        description="Minimum supporters required for a valid cross-scan cluster.",
    )
    # tag: tuning
    leiden_resolution: float = Field(
        250.0,
        json_schema_extra={"tag": "tuning"},
        description=(
            "Resolution hyperparameter for Leiden clustering (<1 = fewer/larger, >1 = more/smaller). "
            "Ignored when graph_clustering_method != 'leiden'."
        ),
    )
    # tag: tuning
    small_cluster_removal_threshold: int = Field(
        1,
        json_schema_extra={"tag": "tuning"},
        description="Drop clusters appearing fewer than this many times across scans.",
    )
    # tag: other (Phase 4 ENG-* discriminator slot)
    type: Literal["graph_cluster"] = Field(
        "graph_cluster",
        json_schema_extra={"tag": "other"},
        description="Fusion engine discriminator (Phase 4 ENG-* expansion slot).",
    )
    # tag: other
    sparse_connectivity_method: Literal["knn", "radius"] = Field(
        "knn",
        json_schema_extra={"tag": "other"},
        description="Cross-scan sparse connectivity construction method (KD-tree based).",
    )
    # tag: other
    outlier_detection_method: Literal["iqr", "mad", "percentile", "negative_binomial"] = Field(
        "negative_binomial",
        json_schema_extra={"tag": "other"},
        description="Statistical method for over-support outlier detection.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# LoggingConfig — log levels (default + per-package overrides)
# ─────────────────────────────────────────────────────────────────────────────
class LoggingConfig(BaseModel):
    """Logging config (D-A1-10).

    Phase 3 plan 03-05 wires this into ``logging.config.dictConfig`` at CLI
    entry (LOG-02/03). Empty ``per_package`` = nothing silenced (LOG-03
    conservative default).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # tag: primary
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        json_schema_extra={"tag": "primary"},
        description="Default log level for the tls2dseg.* logger hierarchy.",
    )
    # tag: other
    per_package: dict[str, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default_factory=dict,
        json_schema_extra={"tag": "other"},
        description=("Per-package logger level overrides (e.g. {'pchandler': 'WARNING'}). Empty = no overrides."),
    )
    # tag: other
    log_to_file: bool = Field(
        True,
        json_schema_extra={"tag": "other"},
        description="Write {run_dir}/logs/run.log in addition to console.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig — top-level BaseSettings; the public API surface
# ─────────────────────────────────────────────────────────────────────────────
class RunConfig(BaseSettings):
    """Top-level run configuration (D-A1-01..D-A1-06 + D-A3-04).

    Multi-source loader: YAML file + ``TLS2DSEG_*`` env vars + CLI overrides.
    Source precedence (handled by ``config.loader.load_config``):

        cli_overrides  >  env vars  >  YAML file  >  schema defaults

    Frozen post-construction; attribute assignment raises
    ``ValidationError(type='frozen_instance')`` (NOT AttributeError; pydantic v2
    differs from frozen dataclass per RESEARCH.md Pitfall 4).

    Use ``config.loader.load_config(yaml_path, cli_overrides)`` — do NOT
    instantiate ``RunConfig`` directly (no YAML source wired).
    """

    model_config = SettingsConfigDict(
        env_prefix="TLS2DSEG_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    # tag: primary (no default — must declare explicitly, D-A1-06 + MODE-01)
    mode: Literal["single-view", "multi-view"] = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description=(
            "Pipeline operating mode. 'single-view' = stage-1 only (per-scan); "
            "'multi-view' = stage-1 + stage-2 cross-scan fusion. "
            "Phase 3 runs full pipeline for both; Phase 5 ORC-02 adds dispatch."
        ),
    )

    # Required sub-blocks (must appear in YAML — their own required fields
    # propagate the no-default behavior).
    io: IOConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Input/output paths + per-run-dir strategy.",
    )
    prompt: PromptConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Grounded-DINO text prompt.",
    )
    preprocessing: PreprocessingConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Point cloud preprocessing.",
    )
    projection: ProjectionConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Spherical 2D projection from 3D.",
    )
    inference: InferenceConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Grounded-DINO + SAM2 inference.",
    )
    d3d_extraction: D3DExtractionConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Stage-1 per-scan 3D detection extraction.",
    )
    fusion: FusionConfig = Field(
        ...,
        json_schema_extra={"tag": "primary"},
        description="Stage-2 cross-scan graph fusion (multi-view only).",
    )

    # Sub-blocks with defaults — instantiated directly (NOT via default_factory)
    # per D-A1-00 hydra-zen compat. No mypy call-arg ignores: the pydantic.mypy
    # plugin is not configured (it's the plugin that flags no-arg construction per
    # pydantic GH #6300), so plain mypy never raises call-arg here. The ignores
    # were dead under `warn_unused_ignores` and broke CI (which runs mypy without
    # project deps); removed 2026-06-11.
    runtime: RuntimeConfig = Field(
        default=RuntimeConfig(),
        json_schema_extra={"tag": "pipings"},
        description="Compute device + workers + CPU-fallback policy.",
    )
    logging: LoggingConfig = Field(
        default=LoggingConfig(),
        json_schema_extra={"tag": "other"},
        description="Log levels (default + per-package overrides).",
    )

    @model_validator(mode="after")
    def _validate_multi_zoom_requires_sizes_m(self) -> RunConfig:
        if not self.inference.multi_zoom.active:
            return self
        if self.prompt.sizes_m is None:
            raise ValueError(
                "inference.multi_zoom.active=true requires prompt.sizes_m with one size per class token in prompt.text"
            )
        keys = split_class_keys(self.prompt.text)
        if len(self.prompt.sizes_m) != len(keys):
            raise ValueError(
                f"inference.multi_zoom.active=true requires prompt.sizes_m with "
                f"one size per class token in prompt.text — "
                f"got {len(self.prompt.sizes_m)} sizes_m but {len(keys)} tokens: {keys}"
            )
        return self

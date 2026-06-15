"""Pipeline orchestrator — builds engines and dispatches stage1 / stage2.

Importing this module is cheap: all heavy dependencies (torch, pchandler,
pc2img, sam2) live inside __init__ and stage method bodies, never at module level.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tls2dseg.config.models import RunConfig
    from tls2dseg.engines.protocols import FusionEngine, InferenceEngine, ProjectionEngine
    from tls2dseg.pipeline.stage1 import Stage1Result
    from tls2dseg.runtime.context import RunContext
    from tls2dseg.types import InferenceRequest

logger = logging.getLogger("tls2dseg.pipeline.pipeline")


class Pipeline:
    """Run the segmentation pipeline.

    Parameters
    ----------
    cfg : RunConfig
        Frozen resolved config.
    ctx : RunContext
        Per-run runtime context.

    Notes
    -----
    Heavy imports (torch, pchandler, pc2img, sam2) stay inside __init__ and
    stage method bodies — importing this module must be cheap.
    """

    def __init__(self, cfg: RunConfig, ctx: RunContext) -> None:
        # Heavy imports confined here:
        import torch

        from tls2dseg.engines import (
            build_fusion_engine,
            build_inference_engine,
            build_projection_engine,
        )
        from tls2dseg.types import InferenceRequest

        self._cfg = cfg
        self._ctx = ctx

        device = ctx.device

        # Torch autocast + TF32 setup (same block as run.py:156-160):
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
        if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Frozen per-call request spec — constructed once and reused for every
        # detect() call in the loop (text prompt and tuning knobs are invariant
        # within a run).
        self._inference_request: InferenceRequest = InferenceRequest(
            text_prompt=cfg.prompt.text,
            box_threshold=cfg.inference.box_threshold,
            text_threshold=cfg.inference.text_threshold,
            slicing_enabled=cfg.inference.slicing.enabled,
            slice_width_height=cfg.inference.slicing.slice_width_height,
            overlap_width_height=cfg.inference.slicing.overlap_width_height,
            iou_threshold=cfg.inference.slicing.iou_threshold,
            overlap_filter_strategy=cfg.inference.slicing.overlap_filter_strategy,
            large_object_removal_threshold=cfg.inference.large_object_removal_threshold,
            partial_detection_edge_touching_threshold=cfg.inference.partial_detection_edge_touching_threshold,
        )

        image_generation_parameters: dict = {
            "image_width": cfg.projection.image_width,
            "scan_resolution": cfg.projection.scan_resolution,
            "rotate_pcd": cfg.projection.rotate_pcd,
            "rasterization_method": cfg.projection.rasterization_method,
            "features": list(cfg.projection.features),
        }
        self._projection_engine: ProjectionEngine = build_projection_engine(
            "spherical",
            image_generation_parameters=image_generation_parameters,
            pcd_path=None,
        )

        engine_kwargs: dict = {
            "object_detection_model_id": cfg.inference.object_detection_model_id,
            "sam_box_prompt_batch_size": cfg.inference.sam_box_prompt_batch_size,
            "device": device,
        }
        if cfg.inference.type == "grounded_sam2":
            engine_kwargs["sam2_checkpoint"] = str(cfg.inference.sam2_checkpoint)
            engine_kwargs["sam2_model_config"] = cfg.inference.sam2_model_config
        elif cfg.inference.type == "grounded_sam2_hf":
            engine_kwargs["sam2_hf_model_id"] = cfg.inference.sam2_hf_model_id
        self._inference_engine: InferenceEngine = build_inference_engine(cfg.inference.type, **engine_kwargs)

        self._fusion_engine: FusionEngine = build_fusion_engine(
            cfg.fusion.type,
            sparse_connectivity_method=cfg.fusion.sparse_connectivity_method,
            sparse_connectivity_threshold=cfg.fusion.sparse_connectivity_threshold,
            supporters_iou_threshold=cfg.fusion.supporters_iou_threshold,
            remove_outliers_by_support=cfg.fusion.remove_outliers_by_support,
            outlier_detection_method=cfg.fusion.outlier_detection_method,
            outlier_detection_threshold=cfg.fusion.outlier_detection_threshold,
            graph_clustering_method=cfg.fusion.graph_clustering_method,
            min_supporters=cfg.fusion.min_supporters,
            leiden_resolution=cfg.fusion.leiden_resolution,
            small_cluster_removal_threshold=cfg.fusion.small_cluster_removal_threshold,
            merge_inst_of_same_class_only=cfg.fusion.merge_inst_of_same_class_only,
        )

    @classmethod
    def from_yaml(cls, path: Path | str, **overrides: object) -> Pipeline:
        """Convenience constructor: load config from a YAML path."""
        from tls2dseg.config.loader import load_config
        from tls2dseg.runtime import build_context, probe_all

        cfg = load_config(path, cli_overrides=dict(overrides))
        ctx = build_context(cfg, probe_all())
        return cls(cfg, ctx)

    def run(self) -> None:
        """Run end-to-end; dispatches stage1 always, stage2 only in multi-view."""
        result = self.stage1()
        if self._cfg.mode == "multi-view":
            self.stage2(result)

    def stage1(self) -> Stage1Result:
        """Per-scan projection, inference, 2D-NMS combine (single-view), lift, checkpoint."""
        from tls2dseg.pipeline.stage1 import run_stage1

        return run_stage1(
            self._cfg,
            self._ctx,
            self._projection_engine,
            self._inference_engine,
            self._inference_request,
        )

    def stage2(self, stage1_result: Stage1Result) -> None:
        """Cross-scan graph fusion and final merged PLY write (multi-view only)."""
        from tls2dseg.pipeline.stage2 import run_stage2

        run_stage2(self._cfg, self._ctx, self._fusion_engine, stage1_result)

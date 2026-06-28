from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pchandler.geometry import PointCloudData

    from tls2dseg.config.models import RunConfig
    from tls2dseg.engines.protocols import FusionEngine
    from tls2dseg.pipeline.stage1 import Stage1Result
    from tls2dseg.runtime.context import RunContext

logger = logging.getLogger("tls2dseg.pipeline.stage2")


def run_stage2(
    cfg: RunConfig,
    ctx: RunContext,
    fusion_engine: FusionEngine,
    stage1_result: Stage1Result,
) -> PointCloudData:
    """Cross-scan graph fusion, label application, and merged PLY write (multi-view).

    Heavy imports (pchandler, detections_3d) live inside this function body
    so that importing tls2dseg.pipeline.stage2 is cheap (tier_a import-cost contract).

    Returns the fused, labelled merged point cloud (also written to results/).
    """
    from pathlib import Path

    from tls2dseg.detections_3d import d3d_outlier_removal, merge_detections3d
    from tls2dseg.pc_preprocessing import save_segmented_pcd
    from tls2dseg.preprocessing.cleanup import (
        color_pcd_instances_by_random,
        remove_small_instances,
        remove_unclassified_points,
        subsample_pcd_to_output_resolution,
    )
    from tls2dseg.types import FusionInput
    from tls2dseg.utils_main import get_segmented_and_merged_point_cloud

    pcp_parameters: dict = {
        "output_resolution": cfg.preprocessing.output_resolution_m,
        "range_limits": list(cfg.preprocessing.range_limits_m) if cfg.preprocessing.range_limits_m else None,
        "roi_limits": cfg.preprocessing.roi_polygon_m,
        "keep_confidences": cfg.preprocessing.keep_confidences,
        "assign_random_color_per_instance": cfg.preprocessing.assign_random_color_per_instance,
        "flip_upsidedown_scans": cfg.preprocessing.flip_upsidedown_scans_deg or False,
    }
    task_parameters: dict = {
        "task": "object_detection",
        "n_workers": ctx.n_workers,
    }
    class_id_map = dict(ctx.class_id_map)
    data_folder_path = Path(str(cfg.io.input_path)).resolve()

    # 1. merge all per-feature Detections3D into one flat collection
    d3d_collection = merge_detections3d(stage1_result.d3d_collection)

    # 2. outlier removal BEFORE fuse() (fuse() does not run it internally)
    d3d_collection, pcd_or, inst_or = d3d_outlier_removal(
        d3d_collection, per_class_separation=False, confidence_interval=0.99
    )
    stage1_result.pcd_collection.filter_out_instances(pcd_or, inst_or)

    # 3. fuse: single call replaces the entire inline connectivity/clustering block
    fusion_input = FusionInput(
        detections_list=[d3d_collection],
        scan_ids=np.arange(stage1_result.n_scans, dtype=np.float64),
    )
    logger.info("Running fusion engine on %d scans", stage1_result.n_scans)
    fusion_result = fusion_engine.fuse(fusion_input)

    # 4. apply cluster labels and merge point clouds
    pcd_merged = get_segmented_and_merged_point_cloud(
        stage1_result.pcd_collection, d3d_collection, fusion_result.cluster_ids, pcp_parameters
    )

    # 5. post-merge cleanup
    pcd_merged = subsample_pcd_to_output_resolution(pcd_merged, pcp_parameters)
    pcd_merged = remove_unclassified_points(pcd_merged, task_parameters)
    pcd_merged = remove_small_instances(pcd_merged, min_pts=150)

    if pcp_parameters["assign_random_color_per_instance"]:
        color_pcd_instances_by_random(pcd_merged)

    # 6. write one merged PLY to results/
    logger.info("Writing merged PLY to results/")
    save_segmented_pcd(data_folder_path, ctx.results_dir, pcd_merged, class_id_map)

    # Return the fused, labelled point cloud so callers (e.g. the tier_b_heavy
    # gate) can score the fused output, not just the on-disk PLY.
    return pcd_merged

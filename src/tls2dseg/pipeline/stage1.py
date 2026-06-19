from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

np.seterr(invalid="ignore")
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Normals, and colors are not retained during `voxel_downsample`!",
)
warnings.filterwarnings("ignore", category=FutureWarning, message="The key `labels` is will")
warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writable")

if TYPE_CHECKING:
    from tls2dseg.config.models import RunConfig
    from tls2dseg.engines.protocols import InferenceEngine, ProjectionEngine
    from tls2dseg.runtime.context import RunContext
    from tls2dseg.types import InferenceRequest

logger = logging.getLogger("tls2dseg.pipeline.stage1")


@dataclasses.dataclass
class Stage1Result:
    """Container for stage-1 outputs passed to stage-2 or returned to the caller."""

    pcd_collection: Any
    d3d_collection: list
    n_scans: int


def run_stage1(
    cfg: RunConfig,
    ctx: RunContext,
    projection_engine: ProjectionEngine,
    inference_engine: InferenceEngine,
    inference_request: InferenceRequest,
) -> Stage1Result:
    """Per-scan projection, inference, optional 2D-NMS combine, lift, and checkpoint.

    Heavy imports (torch, pchandler, pc2img, sam2) live inside this function body
    so that importing tls2dseg.pipeline.stage1 is cheap (tier_a import-cost contract).
    """
    import gc
    import pickle
    from pathlib import Path

    import torch  # noqa: F401
    from pchandler.data_io import load_e57
    from pchandler.geometry import PointCloudData  # noqa: F401
    from pchandler.geometry.transforms import toggle_socs2prcs

    from tls2dseg.detections_3d import clean_pcd_instances_and_get_detections3d
    from tls2dseg.engines.inference.shared import (
        get_instance_and_semantic_mask_with_confidence,
        get_per_mask_depth_parallel,
    )
    from tls2dseg.lifting.masks_to_pcd import lift_mask_to_pcd, lift_masks_to_pcd
    from tls2dseg.pc_preprocessing import save_segmented_pcd, save_segmented_pcd_ij
    from tls2dseg.pcd_collection import SegPCDCollection
    from tls2dseg.preprocessing.cleanup import (
        apply_robust_sor_filter,
        filter_pcd_roi_range,
        remove_small_instances,
        remove_unclassified_points,
        subsample_pcd_to_output_resolution,
    )
    from tls2dseg.preprocessing.nms_combine import nms_combine_detections
    from tls2dseg.utils_main import (
        assure_common_global_shift,
        id_from_path,
        load_previously_saved_inference_results_if_any,
    )

    # --- checkpoint discovery ---
    stage1_odir_partial = ctx.stage1_dir
    stage1_odir_partial.mkdir(parents=True, exist_ok=True)
    d3d_partial_paths = list(stage1_odir_partial.glob("d3d_ij_*.pkl"))
    pcd_partial_paths = list(stage1_odir_partial.glob("pcd_ij_*.pkl"))
    d3d_map = {id_from_path(p): p for p in d3d_partial_paths}
    pcd_map = {id_from_path(p): p for p in pcd_partial_paths}

    # --- scan file discovery ---
    data_folder_path = Path(str(cfg.io.input_path)).resolve()
    file_format = cfg.io.file_format
    pcd_file_paths = list(data_folder_path.glob(f"*.{file_format}"))

    if len(pcd_file_paths) == 0:
        raise ValueError(f"No .{file_format} files found in {data_folder_path}")

    n_scans = len(pcd_file_paths)
    features: list[str] = list(cfg.projection.features)
    n_features = len(features)
    class_id_map = dict(ctx.class_id_map)
    text_prompt = inference_request.text_prompt
    save_intermediate_results = ctx.dump_json_results

    pcp_parameters: dict = {
        "output_resolution": cfg.preprocessing.output_resolution_m,
        "range_limits": list(cfg.preprocessing.range_limits_m) if cfg.preprocessing.range_limits_m else None,
        "roi_limits": cfg.preprocessing.roi_polygon_m,
        "keep_confidences": cfg.preprocessing.keep_confidences,
        "assign_random_color_per_instance": cfg.preprocessing.assign_random_color_per_instance,
        "flip_upsidedown_scans": cfg.preprocessing.flip_upsidedown_scans_deg or False,
    }
    d3d_parameters: dict = {
        "bounding_box_type": cfg.d3d_extraction.bounding_box_type,
        "centroid_type": cfg.d3d_extraction.centroid_type,
        "preprocess": cfg.d3d_extraction.preprocess,
        "min_d3d_pcd_point_count": cfg.d3d_extraction.min_point_count,
    }
    task_parameters: dict = {
        "task": "object_detection",
        "n_workers": ctx.n_workers,
        "save_d2d": cfg.io.save_d2d,
    }

    if save_intermediate_results:
        from tls2dseg.utils_main import make_output_folders

        inference_models_parameters: dict = {"dump_json_results": ctx.dump_json_results}
        _img_params: dict = {"features": features}
        make_output_folders(ctx.run_dir, _img_params, inference_models_parameters)
        projection_engine.set_output_dir_images(_img_params["output_dir_images"])
        inference_models_parameters["stage1_output_dir"] = ctx.stage1_dir
    else:
        inference_models_parameters = {"dump_json_results": False}

    # --- full checkpoint: mode-correct count ---
    if cfg.mode == "single-view":
        have_processed_all = len(pcd_map) == n_scans and len(d3d_map) == n_scans
    else:
        have_processed_all = len(pcd_map) == n_scans * n_features and len(d3d_map) == n_scans * n_features
    checkpoint_enabled = cfg.io.resume_from_checkpoint

    sv_slots = 1 if cfg.mode == "single-view" else None
    pcd_collection = SegPCDCollection(
        raw_pcd_paths=pcd_file_paths, features=features, class_id_map=class_id_map, slots_per_scan=sv_slots
    )
    d3d_collection: list = []

    if checkpoint_enabled and have_processed_all:
        all_loaded = True
        for pcd_i_id in range(1, n_scans + 1):
            if cfg.mode == "single-view":
                scan_expected_ids = [(pcd_i_id - 1) * n_features + 1]
                slot_start = pcd_i_id - 1
            else:
                scan_expected_ids = list(range((pcd_i_id - 1) * n_features + 1, pcd_i_id * n_features + 1))
                slot_start = (pcd_i_id - 1) * n_features
            pcd_collection, d3d_collection, load_flag = load_previously_saved_inference_results_if_any(
                pcd_i_id, pcd_collection, pcd_map, d3d_collection, d3d_map, scan_expected_ids, slot_start
            )
            if not load_flag:
                all_loaded = False
        if all_loaded:
            return Stage1Result(
                pcd_collection=pcd_collection,
                d3d_collection=d3d_collection,
                n_scans=n_scans,
            )
        # One or more scans failed to load — reset and fall through to per-scan loop
        pcd_collection = SegPCDCollection(
            raw_pcd_paths=pcd_file_paths, features=features, class_id_map=class_id_map, slots_per_scan=sv_slots
        )
        d3d_collection = []

    common_global_shift = np.zeros((3,), dtype=np.float64)

    for pcd_i_id, pcd_path_i in enumerate(pcd_file_paths, start=1):
        # --- per-scan checkpoint resume ---
        if checkpoint_enabled:
            if cfg.mode == "single-view":
                scan_expected_ids = [(pcd_i_id - 1) * n_features + 1]
                slot_start = pcd_i_id - 1
            else:
                scan_expected_ids = list(range((pcd_i_id - 1) * n_features + 1, pcd_i_id * n_features + 1))
                slot_start = (pcd_i_id - 1) * n_features
            pcd_collection, d3d_collection, load_flag = load_previously_saved_inference_results_if_any(
                pcd_i_id, pcd_collection, pcd_map, d3d_collection, d3d_map, scan_expected_ids, slot_start
            )
            if load_flag:
                continue

        # --- load scan ---
        pcd_i = load_e57(pcd_path_i, stay_prcs=False, save_prcs_info=True)

        # RoI/range filter (flip + rotate are now inside project())
        filter_pcd_roi_range(pcd_i, pcp_parameters)

        # --- projection via engine (replaces inline pc2img_run block) ---
        logger.info("Projecting scan %d/%d: %s", pcd_i_id, n_scans, pcd_path_i.name)
        projection_engine.set_pcd_path(pcd_path_i)
        projection_results = projection_engine.project(
            pcd_i,
            features=features,
            resolution=(0, 0),
        )

        # subsample + global-shift after projection (project() may mutate pcd_i in-place).
        # Safe ordering: lift_masks_to_pcd re-derives pixel↔point correspondence from
        # pcd.fov (azimuth/elevation spherical coordinates), not from projection-time
        # point ordering, so the subsampled cloud's FoV still maps correctly.
        pcd_i = subsample_pcd_to_output_resolution(pcd_i, pcp_parameters)
        pcd_i, common_global_shift = assure_common_global_shift(pcd_i, common_global_shift, pcd_i_id)
        pcd_collection.global_shift = common_global_shift

        # generalized (scan, feature) index: pcd_ij_id is 1-based across all pairs
        pcd_ij_id_base = (pcd_i_id - 1) * n_features

        # build images_of_pcd_i list for depth helper (list of (name, array) tuples)
        scan_images: list = [(pr.feature_name, pr.image) for pr in projection_results]

        # accumulate Detections2D across features (single-view only)
        scan_d2d_list = []

        for j, projection_result in enumerate(projection_results):
            pcd_ij_id = pcd_ij_id_base + j + 1
            image_j_numpy = projection_result.image
            image_j_tuple = (projection_result.feature_name, image_j_numpy, projection_result.path)

            logger.info(
                "Inference — scan %d/%d feature %s",
                pcd_i_id,
                n_scans,
                projection_result.feature_name,
            )
            detections_2d = inference_engine.detect(image_j_numpy, request=inference_request)

            # dict adapter for dict-consuming helpers
            results: dict = {
                "masks": detections_2d.masks,
                "input_boxes": detections_2d.input_boxes,
                "confidences": detections_2d.confidences.tolist(),
                "class_names": detections_2d.class_names,
                "class_ids": detections_2d.class_ids,
                "mask_labels": detections_2d.mask_labels,
            }

            get_per_mask_depth_parallel(results, scan_images, n_jobs=task_parameters["n_workers"])

            if save_intermediate_results:
                logger.info("Saving intermediate results")
                from tls2dseg.grounded_sam2 import save_gsam2_results

                save_gsam2_results(
                    image=image_j_tuple,
                    results=results,
                    inference_models_parameters=inference_models_parameters,
                )

            if cfg.mode == "single-view":
                # collect for NMS-combine after the inner loop
                from tls2dseg.types import Detections2D

                scan_d2d_list.append(
                    Detections2D(
                        masks=list(detections_2d.masks),
                        input_boxes=detections_2d.input_boxes,
                        confidences=detections_2d.confidences,
                        class_names=list(detections_2d.class_names),
                        class_ids=detections_2d.class_ids,
                        mask_labels=list(detections_2d.mask_labels),
                    )
                )
                if save_intermediate_results:
                    # per-feature intermediate dump — does not disturb NMS-combine flow
                    feat_instance_mask, feat_semantic_mask, feat_confidence_mask, feat_class_id_map = (
                        get_instance_and_semantic_mask_with_confidence(
                            results, text_prompt, image_hw=image_j_numpy.shape[:2]
                        )
                    )
                    pcd_feat = pcd_i.copy()
                    lift_masks_to_pcd(pcd_feat, feat_instance_mask, feat_semantic_mask)
                    if pcp_parameters["keep_confidences"]:
                        lift_mask_to_pcd(pcd_feat, mask=feat_confidence_mask, mask_name="confidence")
                    del feat_instance_mask, feat_semantic_mask, feat_confidence_mask, feat_class_id_map
                    gc.collect()
                    pcd_feat = remove_unclassified_points(pcd_feat, task_parameters)
                    pcd_feat = remove_small_instances(pcd_feat, min_pts=50)
                    pcd_feat = toggle_socs2prcs(pcd_feat)
                    apply_robust_sor_filter(pcd_feat, k_neighbors=50, std_ratio=2)
                    save_segmented_pcd_ij(
                        pcd_path_i, pcd_feat, inference_models_parameters, class_id_map, image_j_tuple
                    )
                    del pcd_feat
                    gc.collect()
                # single-view: lift happens once after combine, not per-feature
            else:
                # multi-view: lift each feature set independently
                instance_mask, semantic_mask, confidence_mask, image_class_id_map = (
                    get_instance_and_semantic_mask_with_confidence(
                        results, text_prompt, image_hw=image_j_numpy.shape[:2]
                    )
                )

                pcd_ij = pcd_i.copy()
                lift_masks_to_pcd(pcd_ij, instance_mask, semantic_mask)
                if pcp_parameters["keep_confidences"]:
                    lift_mask_to_pcd(pcd_ij, mask=confidence_mask, mask_name="confidence")

                del instance_mask, semantic_mask, confidence_mask, image_class_id_map
                gc.collect()

                pcd_ij = remove_unclassified_points(pcd_ij, task_parameters)
                pcd_ij = remove_small_instances(pcd_ij, min_pts=50)

                pcd_ij = toggle_socs2prcs(pcd_ij)
                apply_robust_sor_filter(pcd_ij, k_neighbors=50, std_ratio=2)

                d3d_i, pcd_ij = clean_pcd_instances_and_get_detections3d(
                    pcd_ij, pcd_ij_id - 1, d3d_parameters, pcp_parameters
                )

                if save_intermediate_results:
                    save_segmented_pcd_ij(pcd_path_i, pcd_ij, inference_models_parameters, class_id_map, image_j_tuple)

                odir_pcd_ij = stage1_odir_partial / Path(f"pcd_ij_{pcd_ij_id}.pkl")
                odir_d3d_ij = stage1_odir_partial / Path(f"d3d_ij_{pcd_ij_id}.pkl")

                pcd_collection.seg_pcds[pcd_ij_id - 1] = pcd_ij
                inst_count = np.unique(pcd_ij.scalar_fields["instances"]).size
                pcd_collection.pcd_n_instances[pcd_ij_id - 1] = inst_count

                with open(odir_pcd_ij, "wb") as f:
                    pickle.dump(pcd_ij, f)

                d3d_collection.append(d3d_i)
                with open(odir_d3d_ij, "wb") as f:
                    pickle.dump(d3d_i, f)

                del pcd_ij, d3d_i
                gc.collect()

        # --- single-view: NMS-combine → lift once → write final PLY per scan ---
        if cfg.mode == "single-view":
            logger.info("Single-view: NMS-combining %d feature detection sets", len(scan_d2d_list))
            combined_d2d = nms_combine_detections(
                scan_d2d_list,
                iou_threshold=inference_request.iou_threshold,
                overlap_filter_strategy=inference_request.overlap_filter_strategy,
                class_agnostic=cfg.inference.slicing.nms_combine_class_agnostic,
            )

            combined_results: dict = {
                "masks": combined_d2d.masks,
                "input_boxes": combined_d2d.input_boxes,
                "confidences": combined_d2d.confidences.tolist(),
                "class_names": combined_d2d.class_names,
                "class_ids": combined_d2d.class_ids,
                "mask_labels": combined_d2d.mask_labels,
            }

            get_per_mask_depth_parallel(combined_results, scan_images, n_jobs=task_parameters["n_workers"])

            instance_mask, semantic_mask, confidence_mask, image_class_id_map = (
                get_instance_and_semantic_mask_with_confidence(
                    combined_results,
                    text_prompt,
                    image_hw=projection_results[0].image.shape[:2],
                )
            )
            del image_class_id_map

            pcd_i_sv = pcd_i.copy()
            lift_masks_to_pcd(pcd_i_sv, instance_mask, semantic_mask)
            if pcp_parameters["keep_confidences"]:
                lift_mask_to_pcd(pcd_i_sv, mask=confidence_mask, mask_name="confidence")

            del instance_mask, semantic_mask, confidence_mask
            gc.collect()

            pcd_i_sv = remove_unclassified_points(pcd_i_sv, task_parameters)
            pcd_i_sv = remove_small_instances(pcd_i_sv, min_pts=50)

            pcd_i_sv = toggle_socs2prcs(pcd_i_sv)
            apply_robust_sor_filter(pcd_i_sv, k_neighbors=50, std_ratio=2)

            sv_pcd_ij_id = pcd_ij_id_base + 1
            d3d_i, pcd_i_sv = clean_pcd_instances_and_get_detections3d(
                pcd_i_sv, sv_pcd_ij_id - 1, d3d_parameters, pcp_parameters
            )

            odir_pcd_sv = stage1_odir_partial / Path(f"pcd_ij_{sv_pcd_ij_id}.pkl")
            odir_d3d_sv = stage1_odir_partial / Path(f"d3d_ij_{sv_pcd_ij_id}.pkl")

            # single-view collection is sized one slot per scan (pcd_i_id-1)
            pcd_collection.seg_pcds[pcd_i_id - 1] = pcd_i_sv
            inst_count = np.unique(pcd_i_sv.scalar_fields["instances"]).size
            pcd_collection.pcd_n_instances[pcd_i_id - 1] = inst_count

            with open(odir_pcd_sv, "wb") as f:
                pickle.dump(pcd_i_sv, f)

            d3d_collection.append(d3d_i)
            with open(odir_d3d_sv, "wb") as f:
                pickle.dump(d3d_i, f)

            # write one final PLY per scan to results/
            logger.info("Writing single-view final PLY for scan %d/%d", pcd_i_id, n_scans)
            save_segmented_pcd(data_folder_path, ctx.results_dir, pcd_i_sv, class_id_map, output_stem=pcd_path_i.stem)

            del pcd_i_sv, d3d_i
            gc.collect()

        del pcd_i, projection_results, scan_images
        gc.collect()

    return Stage1Result(
        pcd_collection=pcd_collection,
        d3d_collection=d3d_collection,
        n_scans=n_scans,
    )

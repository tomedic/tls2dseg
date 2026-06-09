# Point Cloud Segmentation Using 2D point cloud representation and DL foundational models (e.g. GroundedSAM)
# Author: Tomislav Medic & ChatGPT, 22.04.2025
#
# Relocated from tls2dseg/main.py to tls2dseg.pipeline.run in Phase 2 (02-01-PLAN, D-01..D-04).
# Phase 3 plan 06 — main() now takes (cfg: RunConfig, ctx: RunContext); module-level
# parameter dicts deleted; all print() converted to logger; pchandler hardcoded
# ERROR setLevel removed; mode=single-view once-per-process warning added.
# Invocation: via `tls2dseg run --config <yaml>` (cli.py:run_cmd).
from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

# np.seterr / warnings filters must run before any inference call that emits
# the suppressed messages; safe at module level since they don't import torch.
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
    from tls2dseg.runtime.context import RunContext

logger = logging.getLogger("tls2dseg.pipeline.run")


@cache
def _warn_single_view_dispatch_pending() -> None:
    """Fire exactly one WARNING per process when mode=single-view is requested.

    Phase 3 still runs the full pipeline regardless of mode. Phase 5 ORC-02 adds
    per-mode dispatch. Until then, this warning signals the user that the
    single-view code path is captured but not yet specialized.

    Wrapped in ``functools.cache`` per D-A1-06 + RESEARCH §Pattern 5 — the
    cache key is the empty arg-tuple, so the body runs at most once per process.
    """
    logger.warning(
        "mode=single-view captured but Phase 3 still runs full pipeline; "
        "per-mode dispatch lands in Phase 5 ORC-02. (Once per process.)"
    )


def main(cfg: RunConfig, ctx: RunContext) -> None:
    """Run the segmentation pipeline.

    Parameters
    ----------
    cfg
        Resolved :class:`tls2dseg.config.RunConfig` (plan 03-01).
    ctx
        Per-run :class:`tls2dseg.runtime.RunContext` (plan 03-04). Provides
        ``run_dir``, ``stage1_dir``, ``device``, ``class_id_map``, etc.

    Notes
    -----
    Heavy imports (torch, pchandler, pc2img, sam2) happen inside this function
    so that importing :mod:`tls2dseg.pipeline.run` is cheap (the CLI dry-run
    + the test_warn_once tier_a tests rely on this property).
    """
    import gc
    import json
    import pickle
    from pathlib import Path

    import torch
    from pchandler.data_io import load_e57
    from pchandler.geometry import PointCloudData
    from pchandler.geometry.transforms import toggle_socs2prcs

    from tls2dseg.detections_3d import (
        clean_pcd_instances_and_get_detections3d,
        d3d_outlier_removal,
        merge_detections3d,
    )
    from tls2dseg.graph_clustering import (
        count_significant_overlaps,
        detect_upper_tail_outliers,
        filter_outlier_detections3d_edges_and_nodes,
        get_edge_weights,
        get_initial_sparse_connectivity,
        graph_clustering,
    )
    from tls2dseg.grounded_sam2 import (
        initialize_gdino,
        initialize_sam2,
        run_grounded_sam2,
        run_grounded_sam2_with_sahi,
        save_gsam2_results,
    )
    from tls2dseg.pc2img_utils import (
        check_was_scanner_upsidedown,
        compute_image_dimensions,
        get_instance_and_semantic_mask_with_confidence,
        get_per_mask_depth_parallel,
        pc2img_run,
        project_a_mask_2_pcd_as_scalarfield,
        project_masks2pcd_as_scalarfields,
        reduce_image_resolution,
        resolve_necessary_image_resolution,
        resolve_rotate_pcd_parameter,
        resolve_scanning_resolution_parameter,
        rotate_pcd_around_x,
        rotate_pcd_around_z,
    )
    from tls2dseg.pc_preprocessing import (
        apply_robust_sor_filter,
        color_pcd_instances_by_random,
        filter_pcd_roi_range,
        remove_small_instances,
        remove_unclassified_points,
        save_segmented_pcd,
        save_segmented_pcd_ij,
        subsample_pcd_to_output_resolution,
    )
    from tls2dseg.pcd_collection import SegPCDCollection
    from tls2dseg.utils_main import (
        assure_common_global_shift,
        get_segmented_and_merged_point_cloud,
        id_from_path,
        load_previously_saved_inference_results_if_any,
        small_cluster_removal,
    )

    # 0. Initial Set-up
    # _______________________________________________________________________

    if cfg.mode == "single-view":
        _warn_single_view_dispatch_pending()

    # Output paths: write to ctx subdirs (per-run isolation per D-A2-05),
    # NOT directly to cfg.io.output_dir.
    output_dir_pathlib = ctx.run_dir
    save_intermediate_results = ctx.dump_json_results

    # Resume-from-checkpoint partial map.
    stage1_odir_partial = ctx.stage1_dir
    stage1_odir_partial.mkdir(parents=True, exist_ok=True)
    d3d_partial_paths = list(stage1_odir_partial.glob("d3d_ij_*.pkl"))
    pcd_partial_paths = list(stage1_odir_partial.glob("pcd_ij_*.pkl"))
    d3d_map = {id_from_path(p): p for p in d3d_partial_paths}
    pcd_map = {id_from_path(p): p for p in pcd_partial_paths}

    # Device resolution already done by build_context (CPU-04). ctx.device is
    # the canonical "cpu" | "cuda" answer.
    device = ctx.device

    # Set the environment settings (this is necessary for SAM2 with bf16):
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Adapter dict for the legacy inference functions (still consume a dict).
    # Phase 4 ENG-* replaces these with engine objects receiving (cfg, ctx).
    inference_models_parameters = {
        "with_slice_inference": cfg.inference.slicing.enabled,
        "bbox_model_id": cfg.inference.object_detection_model_id,
        "box_threshold": cfg.inference.box_threshold,
        "text_threshold": cfg.inference.text_threshold,
        "sam2-model-config": cfg.inference.sam2_model_config,
        "sam2-checkpoint": str(cfg.inference.sam2_checkpoint),
        "large_object_removal_threshold": cfg.inference.large_object_removal_threshold,
        "partial_detection_edge_touching_threshold": cfg.inference.partial_detection_edge_touching_threshold,
        "sam_box_prompt_batch_size": cfg.inference.sam_box_prompt_batch_size,
        "device": device,
        "dump_json_results": ctx.dump_json_results,
    }
    slice_inference_parameters = {
        "slice_width_height": cfg.inference.slicing.slice_width_height,
        "overlap_width_height": cfg.inference.slicing.overlap_width_height,
        "iou_threshold": cfg.inference.slicing.iou_threshold,
        "overlap_filter_strategy": cfg.inference.slicing.overlap_filter_strategy,
        "empty_slice_removal_threshold": cfg.inference.slicing.empty_slice_removal_threshold,
        "thread_workers": ctx.n_workers,
    }
    image_generation_parameters = {
        "image_width": cfg.projection.image_width,
        "scan_resolution": cfg.projection.scan_resolution,
        "rotate_pcd": cfg.projection.rotate_pcd,
        "rasterization_method": cfg.projection.rasterization_method,
        "features": list(cfg.projection.features),
    }
    pcp_parameters = {
        "output_resolution": cfg.preprocessing.output_resolution_m,
        "range_limits": list(cfg.preprocessing.range_limits_m) if cfg.preprocessing.range_limits_m else None,
        "roi_limits": cfg.preprocessing.roi_polygon_m,
        "keep_confidences": cfg.preprocessing.keep_confidences,
        "assign_random_color_per_instance": cfg.preprocessing.assign_random_color_per_instance,
        "flip_upsidedown_scans": cfg.preprocessing.flip_upsidedown_scans_deg or False,
    }
    task_parameters = {
        "input_path": str(cfg.io.input_path),
        "checkpoint": cfg.io.resume_from_checkpoint,
        "file_format": cfg.io.file_format,
        "output_path": str(ctx.run_dir),
        "save_intermediate_results": ctx.dump_json_results,
        "task": "object_detection",
        "results_aggregation_strategy": "object_memory_bank",
        "n_workers": ctx.n_workers,
        "save_d2d": cfg.io.save_d2d,
    }
    d3d_parameters = {
        "bounding_box_type": cfg.d3d_extraction.bounding_box_type,
        "centroid_type": cfg.d3d_extraction.centroid_type,
        "preprocess": cfg.d3d_extraction.preprocess,
        "min_d3d_pcd_point_count": cfg.d3d_extraction.min_point_count,
        "merge_inst_of_same_class_only": cfg.fusion.merge_inst_of_same_class_only,
        "sparse_connectivity_method": cfg.fusion.sparse_connectivity_method,
        "sparse_connectivity_threshold": cfg.fusion.sparse_connectivity_threshold,
        "supporters_iou_threshold": cfg.fusion.supporters_iou_threshold,
        "remove_outliers_by_support": cfg.fusion.remove_outliers_by_support,
        "outlier_detection_method": cfg.fusion.outlier_detection_method,
        "outlier_detection_threshold": cfg.fusion.outlier_detection_threshold,
        "graph_clustering_method": cfg.fusion.graph_clustering_method,
        "min_supporters": cfg.fusion.min_supporters,
        "leiden_resolution": cfg.fusion.leiden_resolution,
        "small_cluster_removal_threshold": cfg.fusion.small_cluster_removal_threshold,
    }
    text_prompt = cfg.prompt.text

    if save_intermediate_results:
        # make_output_folders was a Phase-1-2 helper; the stage1_dir is already
        # created by build_context, so the call below would have been a no-op
        # in modern flow. Keep the marker; Phase 4 ENG-* replaces.
        inference_models_parameters["stage1_output_dir"] = ctx.stage1_dir

    # Use ctx.class_id_map (derived once by build_context per D-A2-03), then
    # mirror it onto disk for downstream consumers.
    class_id_map = dict(ctx.class_id_map)
    inverted_id_map = {v: k for k, v in class_id_map.items()}
    inverted_map_path = ctx.run_info_dir / "class_names_id_map.txt"
    with open(str(inverted_map_path), "w", encoding="ascii") as f:
        json.dump(inverted_id_map, f, ensure_ascii=True)

    # 1. Point Cloud Processing
    # _______________________________________________________________________

    # Initialize Grounded SAM2 (Grounded DINO + SAM2)
    gdino_model, gdino_processor = initialize_gdino(inference_models_parameters)
    sam2_predictor = initialize_sam2(inference_models_parameters)

    # Find all point cloud files of defined "file_format" within data folder
    data_folder_path = Path(task_parameters["input_path"]).resolve()
    file_format = task_parameters["file_format"]
    pcd_file_paths = list(data_folder_path.glob(f"*.{file_format}"))
    n_scans = len(pcd_file_paths)

    # Generate SegPCDCollection object:
    features = image_generation_parameters["features"]
    n_features = len(features)
    # Jump over segmentation if already fully solved
    checkpoint = task_parameters["checkpoint"]

    have_processed_all = False
    if len(pcd_map) == n_scans * n_features and len(d3d_map) == n_scans * n_features:
        have_processed_all = True

    # Start the inference loop if no wish to start from checkpoint or not all pcds already processed:
    if not checkpoint or not have_processed_all:
        # Set-up results structure
        how_aggregate_results = task_parameters["results_aggregation_strategy"]
        pcd_collection = SegPCDCollection(raw_pcd_paths=pcd_file_paths, features=features, class_id_map=class_id_map)
        if how_aggregate_results == "object_memory_bank":
            d3d_collection = []
            if task_parameters["save_d2d"]:
                d2d_collection = []
        else:
            raise ValueError("Chosen results_aggregation_strategy is currently not supported!")

        # Set common global shift for all point clouds (precaution, should not be necessary for small projects)
        common_global_shift = np.zeros((3,), dtype=np.float64)

        for pcd_i_id, pcd_path_i in enumerate(pcd_file_paths, start=1):
            # Check if some point clouds already processed (step 2/2) - pcd_i
            pcd_collection, d3d_collection, load_flag = load_previously_saved_inference_results_if_any(
                pcd_i_id, pcd_collection, pcd_map, d3d_collection, d3d_map
            )

            # Skip the rest of the inference loop if load_flag is True:
            if load_flag:
                continue

            # Load data
            pcd: PointCloudData = load_e57(pcd_path_i, stay_prcs=False, save_prcs_info=True)  # Load point cloud

            # Get scan resolution along azimuth and elevation in radians (before any manipulations)
            imw = image_generation_parameters["image_width"]
            if isinstance(imw, str) and "scan_resolution" in imw:
                d_azim_rad, d_elev_rad = resolve_scanning_resolution_parameter(pcd, image_generation_parameters)
            else:
                # BUGS-02 (02-01-PLAN Task 1, D-16): chained assignment
                d_azim_rad = d_elev_rad = np.nan

            # Filter point cloud for ranges and RoI (region of interest)
            filter_pcd_roi_range(pcd, pcp_parameters)

            # Optional: Detect if point cloud upside-down, if yes - flip for theta degrees:
            test_upsidedown, alpha_deg = False, 0.0
            if pcp_parameters["flip_upsidedown_scans"]:
                alpha_deg = pcp_parameters["flip_upsidedown_scans"]
                test_upsidedown = check_was_scanner_upsidedown(pcd)
                if test_upsidedown:
                    rotate_pcd_around_x(pcd, alpha_deg=alpha_deg)

            # Rotate point cloud around z (if necessary), return rotation angle theta in degrees
            theta_deg = resolve_rotate_pcd_parameter(pcd, image_generation_parameters)

            # Compute image dimensions (w and h) in pixels and scan resolution (azimuth and elevation) in radians
            image_width, image_height = compute_image_dimensions(
                pcd, image_generation_parameters, d_azim_rad, d_elev_rad
            )

            logger.info("Image height x width: %d x %d", image_height, image_width)

            # Resolve necessary image resolution
            reduction_coefficient = resolve_necessary_image_resolution(pcd, pcp_parameters, d_azim_rad)

            # Generate images of point cloud i
            logger.info("Generating desired image(s)")
            images_of_pcd_i = pc2img_run(pcd, pcd_path_i, image_generation_parameters, image_width, image_height)

            # Reducing image resolution (if necessary)
            if reduction_coefficient < 1:
                logger.info("Reducing image resolution")
                images_of_pcd_i = reduce_image_resolution(
                    images_of_pcd_i, reduction_coefficient, image_generation_parameters, pcd_path_i
                )

            # Subsample point cloud to desired output resolution (once images generated):
            pcd = subsample_pcd_to_output_resolution(pcd, pcp_parameters)
            # Assure common global shift for further operations
            pcd, common_global_shift = assure_common_global_shift(pcd, common_global_shift, pcd_i_id)
            pcd_collection.global_shift = common_global_shift

            # 2. Inference: Instance + semantic segmentation -------------------
            # Set segmented point cloud (pcd_ij) counter:
            pcd_ij_id = pcd_i_id * 2 - 2

            # Run Grounded SAM2 inference (for all images of a point cloud pcd_i)
            for j, image_j in enumerate(images_of_pcd_i):
                pcd_ij_id += 1
                image_j_numpy = image_j[1]

                if inference_models_parameters["with_slice_inference"] is True:
                    logger.info("Grounded SAM2 - Inference on image slices")
                    results = run_grounded_sam2_with_sahi(
                        image=image_j_numpy,
                        text_prompt=text_prompt,
                        gdino_model=gdino_model,
                        gdino_processor=gdino_processor,
                        sam2_predictor=sam2_predictor,
                        inference_models_parameters=inference_models_parameters,
                        slice_inference_parameters=slice_inference_parameters,
                    )
                else:
                    logger.info("Grounded SAM2 - Inference on a whole image")
                    results = run_grounded_sam2(
                        image=image_j_numpy,
                        text_prompt=text_prompt,
                        gdino_model=gdino_model,
                        gdino_processor=gdino_processor,
                        sam2_predictor=sam2_predictor,
                        inference_models_parameters=inference_models_parameters,
                    )

                # Get per-mask depths:
                get_per_mask_depth_parallel(results, images_of_pcd_i, n_jobs=task_parameters["n_workers"])

                # Save object detection (gdino) and segmentation (SAM2) results as .jpeg images and corresponding data in .json:
                if save_intermediate_results:
                    logger.info("Saving intermediate results")
                    save_gsam2_results(
                        image=images_of_pcd_i[j],
                        results=results,
                        inference_models_parameters=inference_models_parameters,
                    )

                # From individual per-object bool masks get:
                #   - 1 instance mask (each instance having one int ID),
                #   - 1 semantic mask (each class having one int ID),
                #   - class_id_map which maps semantic classes provided in text_prompt to semantic class IDs
                logger.info("Getting unified instance and semantic mask from individual masks")
                instance_mask, semantic_mask, confidence_mask, class_id_map = (
                    get_instance_and_semantic_mask_with_confidence(
                        results, text_prompt, image_hw=image_j_numpy.shape[:2]
                    )
                )

                # Add the generated masks to ImageStack related to the point cloud pcd
                logger.info("Lifting 2d masks to 3d")
                # Create a point cloud copy for further data processing:
                pcd_ij = pcd.copy()
                project_masks2pcd_as_scalarfields(pcd_ij, instance_mask, semantic_mask)
                if pcp_parameters["keep_confidences"]:
                    project_a_mask_2_pcd_as_scalarfield(pcd_ij, mask=confidence_mask, mask_name="confidence")

                del instance_mask, semantic_mask, confidence_mask
                gc.collect()

                # Remove background class (if task = object detection)
                pcd_ij = remove_unclassified_points(pcd_ij, task_parameters)

                # Remove too small object detections
                pcd_ij = remove_small_instances(pcd_ij, min_pts=50)

                # Transform point cloud to global (project-related) coordinate system
                if theta_deg != 0.0:
                    rotate_pcd_around_z(pcd_ij, theta_deg=-theta_deg)

                if test_upsidedown:
                    rotate_pcd_around_x(pcd_ij, alpha_deg=-alpha_deg)

                pcd_ij = toggle_socs2prcs(pcd_ij)

                # Apply global robust SOR filter to kick-out spurious points / floaters
                apply_robust_sor_filter(pcd_ij, k_neighbors=50, std_ratio=2)

                # Preprocess (clean) per-instance point clouds and extract related Detections3D data
                d3d_i, pcd_ij = clean_pcd_instances_and_get_detections3d(
                    pcd_ij, pcd_ij_id - 1, d3d_parameters, pcp_parameters
                )

                # Save individual station point clouds (currently aligned in PRCS, if toggle_socs2prcs works)
                if save_intermediate_results:
                    save_segmented_pcd_ij(pcd_path_i, pcd_ij, inference_models_parameters, class_id_map, image_j)

                # Create path to pickled point cloud, d2d and d3d objects
                odir_pcd_ij = stage1_odir_partial / Path(f"pcd_ij_{pcd_ij_id}.pkl")
                odir_d3d_ij = stage1_odir_partial / Path(f"d3d_ij_{pcd_ij_id}.pkl")
                odir_d2d_ij = stage1_odir_partial / Path(f"d2d_ij_{pcd_ij_id}.pkl")

                # Store segmented point cloud
                pcd_collection.seg_pcds[pcd_ij_id - 1] = pcd_ij
                inst_count = np.unique(pcd_ij.scalar_fields["instances"]).size
                pcd_collection.pcd_n_instances[pcd_ij_id - 1] = inst_count

                with open(odir_pcd_ij, "wb") as f:
                    pickle.dump(pcd_ij, f)

                # Store 2d detections (bounding boxes, masks, confidences, class_ids,...)
                if task_parameters["save_d2d"]:
                    d2d_collection.append(results)

                    with open(odir_d2d_ij, "wb") as f:
                        pickle.dump(results, f)

                # Store 3d detections
                d3d_collection.append(d3d_i)
                with open(odir_d3d_ij, "wb") as f:
                    pickle.dump(d3d_i, f)

                del pcd_ij, d3d_i, image_j, image_j_numpy, results
                gc.collect()

            del pcd, images_of_pcd_i
            gc.collect()

        # All point clouds looped through
        # ----------------------------------------------------------------

        # Pickle and save Stage 1 results
        if save_intermediate_results is True:
            stage1_output_dir = inference_models_parameters["stage1_output_dir"]
            odir_pcd_collection = stage1_output_dir / Path("pcd_collection.pkl")
            if task_parameters["save_d2d"]:
                odir_d2d_collection = stage1_output_dir / Path("d2d_collection.pkl")
            odir_d3d_collection = stage1_output_dir / Path("d3d_collection.pkl")
            with open(odir_pcd_collection, "wb") as f:
                pickle.dump(pcd_collection, f)
            if task_parameters["save_d2d"]:
                with open(odir_d2d_collection, "wb") as f:
                    pickle.dump(d2d_collection, f)
            with open(odir_d3d_collection, "wb") as f:
                pickle.dump(d3d_collection, f)

    else:
        # Load Stage 1 results (if Grounded SAM2 was already applied on the data)
        stage1_output_dir = inference_models_parameters["stage1_output_dir"]
        odir_pcd_collection = stage1_output_dir / Path("pcd_collection.pkl")
        if task_parameters["save_d2d"]:
            odir_d2d_collection = stage1_output_dir / Path("d2d_collection.pkl")
        odir_d3d_collection = stage1_output_dir / Path("d3d_collection.pkl")

        with open(odir_pcd_collection, "rb") as f:
            pcd_collection = pickle.load(f)
        if task_parameters["save_d2d"]:
            with open(odir_d2d_collection, "rb") as f:
                d2d_collection = pickle.load(f)
        with open(odir_d3d_collection, "rb") as f:
            d3d_collection = pickle.load(f)

    # SECOND PART OF THE CODE: ---------------------------------------------

    # Merge all Detections3D objects into 1 large object
    d3d_collection = merge_detections3d(d3d_collection)

    # Apply statistical outlier removal using multivariate-normal distribution and mahalanobis distance
    d3d_collection, pcd_or, inst_or = d3d_outlier_removal(
        d3d_collection, per_class_separation=False, confidence_interval=0.99
    )
    pcd_collection.filter_out_instances(pcd_or, inst_or)

    # Get initial sparse connectivity (relevant/sparse nodes for the graph)
    semantic_gate = d3d_parameters["merge_inst_of_same_class_only"]
    spcon_method = d3d_parameters["sparse_connectivity_method"]
    spcon_threshold = d3d_parameters["sparse_connectivity_threshold"]
    pairs = get_initial_sparse_connectivity(
        centroids=d3d_collection.centroids,
        class_ids=d3d_collection.classes,
        n_scans=n_scans,
        method=spcon_method,
        knn_ps=spcon_threshold,
        radius=spcon_threshold,
        semantic_gate=semantic_gate,
    )

    # get edges for graph
    iou_threshold = d3d_parameters["supporters_iou_threshold"]
    edges_iou, edges_supp = get_edge_weights(
        detections3d=d3d_collection, pairs=pairs, iou_threshold=iou_threshold, mode="both"
    )

    # Remove Detections3D that overlap with too many other Detections3D (likely under-segmented)
    remove_outliers_by_support = d3d_parameters["remove_outliers_by_support"]
    or_method = d3d_parameters["outlier_detection_method"]
    or_threshold = d3d_parameters["outlier_detection_threshold"]
    if remove_outliers_by_support:
        n_d3d = d3d_collection.pcd_ids.shape[0]
        counts = count_significant_overlaps(pairs=pairs, bbox_overlap=edges_iou, iou_threshold=iou_threshold, N=n_d3d)
        outliers, cutoff = detect_upper_tail_outliers(data=counts, method=or_method, alpha=or_threshold)
        d3d_collection, pairs, edges_supp = filter_outlier_detections3d_edges_and_nodes(
            d3d_collection=d3d_collection, pairs=pairs, edge_weights=edges_supp, outliers=outliers
        )
        n_d3d = d3d_collection.pcd_ids.shape[0]

    # Find corresponding Detections3D instances using graph clustering
    clustering_method = d3d_parameters["graph_clustering_method"]
    min_supporters = d3d_parameters["min_supporters"]
    leiden_resolution = d3d_parameters["leiden_resolution"]
    clusters_ids = graph_clustering(
        num_nodes=n_d3d,
        pairs=pairs,
        edge_weights=edges_supp,
        method=clustering_method,
        min_supporters=min_supporters,
        leiden_resolution=leiden_resolution,
    )

    # Kicking-out clusters with too small support
    clusters_ids = small_cluster_removal(clusters_ids, d3d_parameters)

    # Assign new instance labels to point clouds and merge them together
    pcd_merged = get_segmented_and_merged_point_cloud(pcd_collection, d3d_collection, clusters_ids, pcp_parameters)

    # Subsample point cloud to desired output resolution:
    pcd_merged = subsample_pcd_to_output_resolution(pcd_merged, pcp_parameters)

    # Remove background class (if task = object detection)
    pcd_merged = remove_unclassified_points(pcd_merged, task_parameters)

    # Remove too small merged instances:
    pcd_merged = remove_small_instances(pcd_merged, min_pts=150)

    # Replace pcd RGB colour by random colors for each instance
    if pcp_parameters["assign_random_color_per_instance"] is True:
        color_pcd_instances_by_random(pcd_merged)

    # Save point cloud with final results
    save_segmented_pcd(data_folder_path, output_dir_pathlib, pcd_merged, class_id_map)

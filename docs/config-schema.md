# tls2dseg Config Schema Reference

This document gives an overview of all modifiable parameters and tuning knobs that steer the implemented pipeline and are exposed to end users through the config YAML files. Fields are grouped by **relevance tag** and organised by **parameter block**.

The example configs in `examples/configs/` (`office_small.yaml`, `mountain_small.yaml`) carry the same tags and descriptions as inline comments — start from one of them.

## Relevance tags

- **primary** — should be set/adjusted for each new dataset for the pipeline to work as intended.
- **tuning** — adjusting directly affects result quality; use these to tweak performance and improve the segmentation.
- **pipings** — adjusting affects compute/infra; can make the process more efficient (RAM, speed) without changing what is detected.
- **other** — parameters with strong defaults (not expected to be modified), but which can have an indirect effect on quality/efficiency; also documentation/logging knobs and rarely-touched switches.

## Parameter blocks

A config is a tree of blocks. The ten top-level entries below are always present; each one *branches* into the individual fields documented in the tables further down (which of those fields are required depends on the block). Set the block, then reach into its fields as needed.

| Block | What it steers / packs |
|-------|------------------------|
| `mode` | Pipeline operating mode. `single-view` = stage-1 only (per-scan segmentation); `multi-view` = stage-1 + stage-2 (cross-scan label fusion). |
| `io` | Input/output: where scans are read from, where per-run output dirs are written, run-id naming, and what intermediate artifacts are kept. |
| `prompt` | The open-vocabulary Grounded-DINO text prompt (the classes to detect) and the per-class physical sizes multi-zoom needs. |
| `preprocessing` | Point-cloud conditioning before projection: output resolution (voxel size), optional range/ROI cropping, upside-down flip, and output colouring. |
| `projection` | How the 3D cloud becomes a 2D spherical image the detector sees: which point features become image channels, image width, rasterization, rotation. |
| `inference` | The Grounded-DINO + SAM2 detection/segmentation step: engine choice, DINO/SAM thresholds, model ids, checkpoint, and the two sub-blocks below (`multi_zoom`, `slicing`). |
| `d3d_extraction` | Stage-1 lift of 2D masks back to 3D per-scan instances: 3D bounding-box type, point-count filter, centroid method, per-instance cleanup. |
| `fusion` | Stage-2 cross-scan graph fusion (multi-view only): how per-scan detections are linked, clustered, and outlier-filtered into merged instances. Parsed but ignored for `single-view`. |
| `runtime` | Compute policy: device selection (GPU/CPU), worker count, CPU-fallback behaviour. |
| `logging` | Log verbosity: default level, per-package overrides, and whether a run log file is written. |

### Sub-blocks: `inference.multi_zoom` and `inference.slicing`

The `inference` block carries two mutually-influencing sub-blocks that decide **how the image is tiled for detection**:

- **`multi_zoom`** — per-class adaptive multi-zoom. When `active: true` (default), the pipeline auto-derives a set of zoom "bands" from each object's physical `sizes_m` and the scan geometry, tiling every band so each class is detected at a favourable footprint, plus an optional full-image overview pass. This is the recommended path.
- **`slicing`** — fixed SAHI-style single-zoom tiling with manually-set tile/overlap sizes. It is the fallback used when `multi_zoom.active: false`: `slicing.enabled`, `slice_width_height`, and `overlap_width_height` only take effect in that single-zoom mode (multi-zoom computes its own tile geometry). The remaining `slicing` fields (`iou_threshold`, `overlap_filter_strategy`, `empty_slice_removal_threshold`, `drop_incomplete_slices`, `nms_combine_class_agnostic`) apply in **both** modes.

## Engine types

Several blocks expose a `type` discriminator, a forward-compatibility slot so alternative engines can be added later without changing the YAML shape. In v1:

- **`inference.type`** selects the detection/segmentation engine and is a discriminated union — the chosen value determines which engine-specific fields are valid:
  - **`grounded_sam2`** (default) — Grounded-DINO + the direct SAM2 package. Best masks. Requires a local `sam2_checkpoint` (`.pt`) and uses `sam2_model_config`.
  - **`grounded_sam2_hf`** — Grounded-DINO + SAM2 via HuggingFace transformers (`sam2_hf_model_id`, auto-downloaded). Softer masks, no manual checkpoint. Kept behind the flag as an alternative.
- **`projection.type`** — only `spherical` in v1.
- **`fusion.type`** — only `graph_cluster` in v1.

## Notes

- **`Default = (required)`** means the user must always set the value explicitly; there is no fallback.
- Some parameters are only relevant in combination with another, which is not obvious from the flat hierarchy (a better separation is pending). For example `fusion.leiden_resolution` matters only when `fusion.graph_clustering_method: leiden`, and the single-zoom `slicing` sizing fields matter only when `inference.multi_zoom.active: false`.
- This document is **hand-maintained**. `tls2dseg schema` regenerates the flat per-tag tables from the pydantic models as a starting point; the overview sections above are authored by hand and should be preserved.

---

## primary

| Field | Type | Default | Description | Block |
|-------|------|---------|-------------|-------|
| `input_path` | `Path` | *(required)* | Path to input point clouds (file or directory). | io |
| `output_dir` | `Path` | *(required)* | Root directory under which per-run output dirs are created. | io |
| `text` | `str` | *(required)* | Open-vocab class prompt for Grounded-DINO; each class string separated by a period. Auto-lowercased + trailing period appended. | prompt |
| `sizes_m` | `list[float] \| None` | null | One representative physical size (m) per class token in `text`, in order. Mandatory when `multi_zoom.active: true`; otherwise optional. | prompt |
| `output_resolution_m` | `float` | *(required)* | Desired output point-cloud resolution (voxel downsample size, m). Set per dataset. | preprocessing |
| `active` | `bool` | false | Enable per-class adaptive multi-zoom. true (default in the examples) = multi-zoom; false = manual single-zoom slicing. | inference.multi_zoom |

## tuning

| Field | Type | Default | Description | Block |
|-------|------|---------|-------------|-------|
| `box_threshold` | `float` | 0.10 | Grounded-DINO box confidence threshold. | inference |
| `text_threshold` | `float` | 0.10 | Grounded-DINO box–text alignment threshold. | inference |
| `large_object_removal_threshold` | `float` | 0.9 | Remove detections whose bbox area exceeds this fraction of the image area. | inference |
| `partial_detection_edge_touching_threshold` | `int` | 5 | Buffer in pixels within which an edge-touching bbox is treated as a partial detection. | inference |
| `footprint_band_frac` | `tuple` | (0.075, 0.225) | Allowed object footprint as a fraction of the DINO input side (800 px); [0.075, 0.225] = 7.5%–22.5% of the side, i.e. 60–180 px. Must satisfy 0 < p_min < p_max < 1. | inference.multi_zoom |
| `range_percentiles` | `tuple` | (10.0, 90.0) | Robust per-scan range bounds (percentiles) used when `preprocessing.range_limits_m` is null. | inference.multi_zoom |
| `cross_class_iou_threshold` | `float` | 0.7 | Cross-zoom cross-class dedup IoU threshold; different-class detections overlapping above this keep the higher-confidence one. | inference.multi_zoom |
| `ios_enabled` | `bool` | false | Enable same-class intersection-over-smaller (IoS) dedup across scales (restricted to same class to avoid killing nested objects). | inference.multi_zoom |
| `overview_pass` | `bool` | true | Run one extra inference pass on a coarse overview image of the whole RoI (catches large objects missed in the sliced zoom bands). | inference.multi_zoom |
| `slice_width_height` | `tuple` | (200, 200) | (width, height) of each image slice in px. Single-zoom only (`multi_zoom.active: false`). | inference.slicing |
| `overlap_width_height` | `tuple` | (150, 150) | (width, height) of overlap between adjacent slices in px. Single-zoom only. | inference.slicing |
| `iou_threshold` | `float` | 0.80 | IoU threshold for merging overlapping slice detections. | inference.slicing |
| `nms_combine_class_agnostic` | `bool` | false | Single-view multi-feature NMS-combine class semantics. false = class-aware (keep different-class overlaps); true = suppress all overlaps regardless of class. | inference.slicing |
| `min_point_count` | `int` | 50 | Minimum points for a 3D detection to be kept. | d3d_extraction |
| `graph_clustering_method` | `Literal` | 'hcs' | Cross-scan clustering method. hcs = hierarchical clustering with support (default). | fusion |
| `min_supporters` | `int` | 1 | Minimum supporting detections for a valid cluster (hcs). | fusion |
| `leiden_resolution` | `float` | 250.0 | Leiden resolution (only used when `graph_clustering_method: leiden`). | fusion |
| `supporters_iou_threshold` | `float` | 0.15 | Minimum 3D-bbox IoU for two detections to count as mutual supporters (valid graph edge). | fusion |
| `remove_outliers_by_support` | `bool` | true | Remove detections with anomalously large support relative to their cluster (likely under-segmented). | fusion |
| `outlier_detection_method` | `Literal` | 'negative_binomial' | Statistical method for over-support outlier detection. | fusion |
| `outlier_detection_threshold` | `float` | 0.01 | Threshold for the outlier method (0.01 ≈ 1% of detections removed as outliers). | fusion |
| `small_cluster_removal_threshold` | `int` | 1 | Drop clusters appearing fewer than this many times across scans. | fusion |
| `merge_inst_of_same_class_only` | `bool` | false | Merge instances across scans only when they share the same class (else allow cross-class merges, tolerating faulty classification). | fusion |

## pipings

| Field | Type | Default | Description | Block |
|-------|------|---------|-------------|-------|
| `resume_from_checkpoint` | `bool` | true | On startup, check the run's `intermediate/stage_1_partial/` for partial state and resume from the last completed (scan, feature). | io |
| `save_intermediate` | `bool` | true | Also save stage-1 visual intermediates under `intermediate/` (projection images, object-detection + SAM2 overlays, mask JSON, per-feature segmented clouds). Resume checkpoints are written regardless. | io |
| `file_format` | `Literal` | 'e57' | Input point-cloud file format; only e57 is fully tested in v1. | io |
| `run_id_strategy` | `Literal` | 'timestamp_scanset' | Strategy for constructing the per-run dir name. | io |
| `sam_box_prompt_batch_size` | `int` | 32 | SAM2 box-prompt batch size; tune for speed vs. GPU memory. | inference |
| `object_detection_model_id` | `str` | 'IDEA-Research/grounding-dino-base' | HuggingFace model id for the object-detection model. | inference |
| `sam2_checkpoint` | `Path` | *(required)* | Path to the SAM2 checkpoint (`.pt`), provided by the SAM2 repo. Supports `${ENV_VAR}` interpolation; existence checked lazily at runtime. (engine `grounded_sam2`) | inference |
| `sam2_model_config` | `str` | 'configs/sam2.1/sam2.1_hiera_l.yaml' | SAM2 model config path (relative to the SAM2 package install). (engine `grounded_sam2`) | inference |
| `device` | `Literal` | 'auto' | Compute device. 'auto' picks CUDA if available, else CPU. | runtime |
| `n_workers` | `int` | 12 | Parallel workers for the stage-1 per-scan loop; tune for speed vs. memory. | runtime |
| `accept_cpu_fallback` | `bool` | true | If true, fall back to CPU when CUDA is absent (loud); if false, raise instead. | runtime |

## other

| Field | Type | Default | Description | Block |
|-------|------|---------|-------------|-------|
| `range_limits_m` | `tuple[float, float] \| None` | null | (min, max) range from the scanner origin (m). null = no limits; setting it can drastically speed up inference. | preprocessing |
| `roi_polygon_m` | `list[tuple[float, float]] \| None` | null | 2D (x, y) polygon in scanner-local metres to crop to a region of interest. null = no ROI; can drastically speed up inference. | preprocessing |
| `flip_upsidedown_scans_deg` | `float \| None` | null | Rotation about the x-axis (deg) to flip upside-down scans (RoI directly under the scanner). null = no flip. | preprocessing |
| `keep_confidences` | `bool` | false | Keep per-point detection confidences in the output cloud (e.g. for custom post-processing). | preprocessing |
| `assign_random_color_per_instance` | `bool` | false | Colour each detected instance with a random RGB (visualization aid). | preprocessing |
| `type` | `Literal` | 'spherical' | Projection engine discriminator (only `spherical` in v1). | projection |
| `features` | `list` | *(required)* | Point features projected to 2D image channels (intensity, range, rgb). Both intensity and range currently required; extensible to arbitrary ScalarFields / RGB. | projection |
| `image_width` | `int \| str` | 'scan_resolution' | Image width: 'scan_resolution' (match scanner angular increment, best results), a fixed int (e.g. 1024), or a fractional string '<frac>-scan_resolution'. Bare floats rejected. | projection |
| `scan_resolution` | `float \| 'auto' \| str` | 'auto' | Angular point spacing. 'auto' = estimate from data; float = explicit increment (deg); '<num><unit>@<dist>m' (e.g. '1.6mm@10m') = derived from physical distance. | projection |
| `rotate_pcd` | `float \| 'auto' \| false` | 'auto' | Rotate the cloud about z (deg) before projection to move the spherical image seam off detail-dense regions. 'auto' = compute; false = no rotation. | projection |
| `rasterization_method` | `Literal` | 'nanconv' | Rasterization method for projecting points onto the image grid (efficient default). | projection |
| `type` | `Literal` | 'grounded_sam2' | Inference engine discriminator — selects the direct SAM2 engine (default). | inference |
| `enabled` | `bool` | true | Run SAHI-style single-zoom slicing. Only takes effect when `multi_zoom.active: false`. | inference.slicing |
| `overlap_filter_strategy` | `Literal` | 'nms' | Cross-slice overlap filter: nms (Non-Max Suppression) or nmm (Non-Max Merging). | inference.slicing |
| `empty_slice_removal_threshold` | `float` | 0.95 | Skip slices whose empty-pixel fraction exceeds this threshold (speeds up inference). | inference.slicing |
| `drop_incomplete_slices` | `bool` | true | Drop SAHI border/edge tiles smaller than the requested slice size (image-border clamped tiles); speeds up and drops questionable edge detections. | inference.slicing |
| `max_zoom_passes` | `int` | 6 | Cap on tiled zoom bands per scan (helps memory/runtime); coarsest instances then rely on the overview pass. | inference.multi_zoom |
| `bounding_box_type` | `Literal` | 'obb' | 3D bounding box type: obb (oriented) or aabb (axis-aligned). | d3d_extraction |
| `centroid_type` | `Literal` | 'bbox_c' | Instance centroid method: 'mean', 'median', or 'bbox_c' (bbox centre). | d3d_extraction |
| `preprocess` | `bool` | true | Per-instance cleanup (statistical outlier removal + dbscan) before saving the 3D instance. | d3d_extraction |
| `type` | `Literal` | 'graph_cluster' | Fusion engine discriminator (only `graph_cluster` in v1). | fusion |
| `sparse_connectivity_method` | `Literal` | 'knn' | Cross-scan sparse-connectivity construction (KD-tree based): knn or radius. | fusion |
| `sparse_connectivity_threshold` | `int \| float` | 1 | Per-method threshold: knn → k (int), radius → distance in m (float). | fusion |
| `type` | `Literal` | 'grounded_sam2_hf' | Inference engine discriminator — selects the HuggingFace SAM2 engine. | inference (grounded_sam2_hf) |
| `sam2_hf_model_id` | `str` | 'facebook/sam2.1-hiera-large' | HuggingFace model id for SAM2 (auto-downloaded). (engine `grounded_sam2_hf`) | inference |
| `level` | `Literal` | 'INFO' | Default log level for the `tls2dseg.*` logger hierarchy (DEBUG/INFO/WARNING/ERROR/CRITICAL). | logging |
| `per_package` | `dict` | {} | Per-package logger level overrides (e.g. {'pchandler': 'WARNING'}). Empty = no overrides. | logging |
| `log_to_file` | `bool` | true | Write `{run_dir}/run_info/run.log` in addition to the console. | logging |

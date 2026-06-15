# tls2dseg Output Contract

This document defines the per-run output layout and the per-mode artifact contract.

## Per-run directory layout

Every run creates a directory under `output_dir/<run_id>/` with this structure:

```
<output_dir>/
└── <run_id>/
    ├── run_info/
    │   ├── run.log              # structured run log
    │   ├── provenance.json      # git sha, config snapshot, timing
    │   └── class_names_id_map.txt  # class name → integer id map
    ├── intermediate/
    │   └── stage_1_partial/
    │       ├── pcd_ij_<N>.pkl   # per-(scan, feature) point cloud checkpoint
    │       └── d3d_ij_<N>.pkl   # per-(scan, feature) Detections3D checkpoint
    └── results/
        └── <final output files>  # mode-specific; see below
```

There is no `stage_1_results/` directory and no `logs/` directory.

## Single-view mode

Stage 1 processes each scan independently. For each input scan, one final
segmented point cloud is written to `results/`:

```
results/
└── <scan_folder_name>_segmented.ply   # one file per input scan
```

Per-feature segmented point clouds (`*_<feature>_prcs_seg.ply` or
`*_<feature>_socs_seg.ply`) are intermediate artifacts. They are written to
the stage-1 area only when `save_intermediate_results` is enabled; they are
never placed in `results/`.

Stage 2 (cross-scan fusion) does not run in single-view mode.

## Multi-view mode

Stage 1 runs per-scan inference and accumulates detections. Stage 2 fuses
detections across all scans via graph clustering and writes one merged final
point cloud to `results/`:

```
results/
└── <scan_folder_name>_segmented.ply   # one merged file for all input scans
```

Per-scan intermediate files are kept under `intermediate/stage_1_partial/`
as checkpoints (same as single-view).

## Artifact shape invariant

In both modes, the files under `results/` are per-instance, per-class
segmented point clouds stored as `.ply`. Each point carries instance id and
class id scalar fields. Downstream tools (e.g. CloudCompare) can load either
mode's output with the same workflow — the only difference is cardinality:
N per-scan files (single-view) versus one merged file (multi-view).

The `class_names_id_map.txt` written alongside the final output maps integer
class ids back to the user-supplied class names from the config prompt.

## Checkpoint resume

If a run is interrupted, the `pcd_ij_<N>.pkl` / `d3d_ij_<N>.pkl` files in
`intermediate/stage_1_partial/` allow stage 1 to resume from the last
completed (scan, feature) pair. Set `resume_from_checkpoint: true` in the
config to activate.

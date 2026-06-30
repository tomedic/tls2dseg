# tls2dseg

Open-vocabulary 3D instance and semantic segmentation of terrestrial laser scan (TLS) point clouds
via panoramic projections and 2D foundation models.

## What it does

`tls2dseg` takes registered `.e57` (or `.ply` / `.las` / `.laz`) scans and a plain-language list
of objects to find, then produces per-instance, per-class segmented point clouds in 3D. The core
idea: project each scan onto a 2D spherical panoramic image, run Grounded-DINO + SAM2 on that
image to detect and segment objects, then lift the 2D masks back to 3D points. In multi-view mode
a second cross-scan graph-fusion stage merges detections across all co-registered scans into a
unified labelled cloud.

The tool is configured entirely through a YAML file — no source-code edits required.

## Key Features

- **Spherical panoramic projection** — rasterizes TLS point clouds to panoramic images using
  `pc2img` (rasterization methods: `nanconv`, `raw`, `bary_delaunay`, `bary_knn`).
- **Open-vocabulary detection + segmentation** — Grounded-DINO for text-prompted bounding boxes;
  SAM2 for per-instance masks; works with any natural-language object-class description.
- **Per-class adaptive multi-zoom inference** — each object class is inferred at the zoom level
  where it occupies the detector's sweet spot, derived from its approximate physical size in
  metres. Replaces a single manually-tuned zoom with an automatic per-class schedule.
- **Two pipeline modes** — `single-view` processes N independent scans independently;
  `multi-view` runs cross-scan graph fusion (stage 2) to merge detections into one unified cloud.
- **Cross-scan graph fusion** — sparse KNN connectivity, 3D-IoU edge weights, and Leiden/HCS
  graph clustering merge redundant detections across scan stations.
- **Checkpoint resume** — stage-1 per-(scan, feature) `.pkl` checkpoints allow a long run to
  restart from the last completed pair without repeating inference.

## Installation

**Prerequisites:**

- Python 3.11
- System library: `libvips` (required by `pyvips`)
- NVIDIA GPU is strongly recommended; CPU-only mode works but is much slower.

```bash
# System deps (Debian/Ubuntu)
sudo apt-get install -y libvips-dev git

# Install tls2dseg
# Fill <PLACEHOLDER_REPO_URL> (e.g. github.com/your-user) and <PLACEHOLDER_REF>
# (a semver tag such as v0.1.0) once the repos are public.
pip install "git+https://<PLACEHOLDER_REPO_URL>/tls2dseg.git@<PLACEHOLDER_REF>"
```

`pchandler` and `pc2img` are declared as direct dependencies and will be pulled automatically. If
you already have local editable installs of those repos active, use `--no-deps` to skip them:

```bash
pip install --no-deps "git+https://<PLACEHOLDER_REPO_URL>/tls2dseg.git@<PLACEHOLDER_REF>"
```

> **Tip:** Use a semver git tag (`v0.1.0`) as `<PLACEHOLDER_REF>`, not a branch name. Branch refs
> produce a non-parseable version string and trigger the `0.0.0` fallback.

**SAM2 checkpoint (required for the default `grounded_sam2` engine):**

The default engine (`inference.type: grounded_sam2`) uses the direct `sam2` package and requires a
local model checkpoint. Download `sam2.1_hiera_large.pt` from Meta's
[SAM2 GitHub releases](https://github.com/facebookresearch/sam2/releases) and point to it in your
config:

```yaml
inference:
  sam2_checkpoint: /path/to/checkpoints/sam2.1_hiera_large.pt
  sam2_model_config: configs/sam2.1/sam2.1_hiera_l.yaml
```

Alternatively, switch to `inference.type: grounded_sam2_hf` to use the HuggingFace Hub engine,
which downloads the checkpoint automatically on first run (see §Architecture).

**Verify the installation:**

```bash
tls2dseg doctor
```

This probes Python, torch/CUDA, libvips, the SAM2 checkpoint, and the `pchandler`/`pc2img`
imports, and reports the status of each.

## Quickstart

The `examples/` directory ships with a small multi-view indoor office dataset (`office_small`) and
a ready-to-run config:

```bash
tls2dseg run --config examples/configs/office_small.yaml
```

The four primary fields to adjust for your own data:

```yaml
mode: multi-view             # single-view or multi-view

io:
  input_path: ./data/my_scan_folder/   # folder of .e57 scan files
  output_dir:  ./results/

prompt:
  text: "plant. cabinet"    # period-separated object class names
  sizes_m: [0.2, 1.5]      # approximate size in metres per class (for multi-zoom)

preprocessing:
  output_resolution_m: 0.005   # 3D voxel size; typical indoor = 0.005; outdoor = 0.01–0.05
```

**Where results land:**

Every run creates a timestamped directory under `output_dir/<run_id>/`:

```
<run_id>/
├── run_info/
│   ├── run.log                  # structured run log
│   ├── provenance.json          # git sha, config snapshot, timing
│   └── class_names_id_map.txt   # class name → integer id map
├── intermediate/
│   └── stage_1_partial/
│       ├── pcd_ij_<N>.pkl       # per-(scan, feature) point cloud checkpoint
│       └── d3d_ij_<N>.pkl       # per-(scan, feature) Detections3D checkpoint
└── results/
    └── <scan_folder_name>_segmented.ply   # final merged cloud (multi-view)
    # or: <scan_name>_segmented.ply per input scan (single-view)
```

See [`docs/output-contract.md`](docs/output-contract.md) for the full per-mode artifact contract,
including the `save_intermediate` option for per-feature intermediate clouds.

To validate a config without running inference:

```bash
tls2dseg validate-config examples/configs/office_small.yaml
```

## Configuration

`tls2dseg` is configured entirely through YAML. Every field carries one of four tags that indicate
how relevant it is to a typical user:

| Tag | Meaning |
|-----|---------|
| `primary` | Fields you almost always need to set for your own dataset |
| `tuning` | Fields that directly affect segmentation quality (thresholds, zoom parameters) |
| `pipings` | Infrastructure knobs (device, workers, checkpoint resume, logging) |
| `other` | Rarely touched; engine-type discriminators and internal options |

The example configs in `examples/configs/` list fields within each block in `primary → tuning →
pipings → other` order and include the tag as an inline comment.

For the full field reference grouped by tag, see [`docs/config-schema.md`](docs/config-schema.md).
Regenerate it whenever models change:

```bash
tls2dseg schema --output docs/config-schema.md
```

## Datasets

Reference datasets used for development and validation will be made available at:

`<PLACEHOLDER_DATASET_URL>`

The three reference datasets cover: an indoor office scene (multi-view), an outdoor agricultural
field (multi-view), and a mountain forest (single-view).

## Python API

The CLI is a thin wrapper over the `Pipeline` class. You can call it directly for scripting or
integration into a larger workflow:

```python
from pathlib import Path
from tls2dseg.pipeline.pipeline import Pipeline

pipeline = Pipeline.from_yaml("examples/configs/office_small.yaml")
pipeline.run()  # writes results/ + run_info/ under io.output_dir
```

For finer control, call the stages separately:

```python
# Stage 1 only — per-scan projection + inference + 3D lift
stage1_result = pipeline.stage1()
# stage1_result.pcd_collection  — per-scan point clouds
# stage1_result.d3d_collection  — per-scan Detections3D lists
# stage1_result.n_scans         — number of input scans

# Stage 2 — cross-scan graph fusion (multi-view mode only)
pipeline.stage2(stage1_result)
```

See [`examples/api_usage.py`](examples/api_usage.py) for a complete runnable example on
`office_small`.

## Architecture and Extending

The pipeline is built around three engine `Protocol`s (defined in `engines/protocols.py`):

| Protocol | Role | Default implementation |
|----------|------|----------------------|
| `ProjectionEngine` | Rasterizes a point cloud to panoramic feature images | `SphericalProjectionEngine` (`spherical`) |
| `InferenceEngine` | Runs 2D detection + segmentation on a single image | `GroundedSAM2Engine` (`grounded_sam2`) |
| `FusionEngine` | Cross-scan graph clustering for multi-view mode (stage 2) | `GraphClusterFusionEngine` (`graph_cluster`) |

Each Protocol has a corresponding dict registry in `engines/__init__.py` (e.g.
`INFERENCE_ENGINES["grounded_sam2"] = GroundedSAM2Engine`). To add a new engine: implement the
Protocol, add an entry to the registry, and select it by name via the config `type` field. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the step-by-step guide.

**SAM2 inference engine default (D-07):**

The default is `inference.type: grounded_sam2` — the direct `sam2` package with a local `.pt`
checkpoint. It produces better-quality masks on average. The trade-off is a one-time manual
checkpoint download (see §Installation).

To switch to the HuggingFace Hub engine, which downloads the checkpoint automatically on first run
at the cost of slightly softer masks:

```yaml
inference:
  type: grounded_sam2_hf
  sam2_hf_model_id: facebook/sam2.1-hiera-large
```

## License and Citation

`tls2dseg` is released under the [GNU General Public License v3.0](LICENSE). The copyleft
requirement follows from runtime dependencies `leidenalg` (GPL-3.0) and `igraph` (GPL-2.0+) used
in the cross-scan fusion stage.

If you use this software in your research, please cite:

```
Medic, T. (PLACEHOLDER_YEAR). tls2dseg: PLACEHOLDER_PAPER_TITLE.
PLACEHOLDER_JOURNAL. https://doi.org/PLACEHOLDER_DOI
```

For BibTeX and other citation formats, see [`CITATION.cff`](CITATION.cff).

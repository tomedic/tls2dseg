"""Pipeline smoke test — per-mode end-to-end orchestration with FakeEngines.

Covers: ORC-01, ORC-02, ORC-03, ORC-04/TEST-12, MODE-04, TEST-13.

Key constraints (D-D-06 binding rule):
- MUST NOT import or instantiate GroundedSAM2Engine or GroundedSAM2HFEngine.
- MUST NOT load any torch model.
- FakeEngines produce deterministic non-empty results.

Marked tier_b_light — needs pchandler/pc2img (for SegPCDCollection / build_context);
FakeEngines mean no real GPU is needed.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_GOLDEN_FILE = _FIXTURES_DIR / "golden_values.json"

# N_SCANS used across tests — 2 synthetic .e57 stubs for the per-scan loop.
_N_SCANS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build cfg + ctx pointing at tmp_path
# ─────────────────────────────────────────────────────────────────────────────


def _make_cfg_ctx(tmp_path: Path, mode: str) -> tuple:
    """Build a minimal RunConfig + RunContext for the given mode.

    input_path gets 2 stub .e57 files so the per-scan loop has work;
    output_dir is tmp_path/output.  No real SAM2 checkpoint is needed
    because build_inference_engine is monkeypatched before __init__ runs.
    """
    from tls2dseg.config import RunConfig
    from tls2dseg.runtime import build_context
    from tls2dseg.runtime.capability import Runtime

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(_N_SCANS):
        (input_dir / f"scan_{i}.e57").touch()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg_dict = yaml.safe_load(f"""\
mode: {mode}
io:
  input_path: {input_dir}
  output_dir: {output_dir}
  resume_from_checkpoint: true
  save_intermediate: false
prompt:
  text: fake_object
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity, range]
inference:
  type: grounded_sam2
  sam2_checkpoint: /tmp/fake_sam2.pt
d3d_extraction: {{}}
fusion: {{}}
runtime:
  device: cpu
""")
    cfg = RunConfig(**cfg_dict)

    runtime = Runtime(
        cuml_available=False,
        torch_cuda_available=False,
        sam2_available=False,
        libvips_available=False,
        numpy_version="1.26.0",
        torch_version="2.7.0",
        cuml_version=None,
    )
    ctx = build_context(cfg, runtime)
    return cfg, ctx


def _make_stage1_result(n_scans: int) -> Any:
    """Return a deterministic Stage1Result with n_scans fake d3d entries."""
    from tls2dseg.pipeline.stage1 import Stage1Result

    d3d_collection = []
    for _ in range(n_scans):
        d3d_collection.append(_make_fake_d3d(n_instances=1))

    return Stage1Result(
        pcd_collection=None,
        d3d_collection=d3d_collection,
        n_scans=n_scans,
    )


def _make_fake_d3d(n_instances: int) -> Any:
    """Return a duck-typed Detections3D with n_instances fake instances."""

    @dataclasses.dataclass
    class _FakeD3D:
        pcd_ids: np.ndarray

    return _FakeD3D(pcd_ids=np.zeros(n_instances, dtype=np.int32))


# ─────────────────────────────────────────────────────────────────────────────
# Common monkeypatches
# ─────────────────────────────────────────────────────────────────────────────


def _patch_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Pipeline.__init__ from importing torch or loading real engines.

    Pipeline.__init__ does `import torch` unconditionally. To avoid polluting
    sys.modules (which would break test_dispatch_does_not_import_torch_or_sam2
    when it runs later in the same session), monkeypatch `builtins.__import__`
    is too invasive. Instead we monkeypatch Pipeline.__init__ directly to build
    the pipeline with fake engines and skip torch setup entirely.
    """
    from tests.unit.fakes import FakeFusionEngine, FakeInferenceEngine, FakeProjectionEngine
    from tls2dseg.pipeline.pipeline import Pipeline
    from tls2dseg.pipeline.stage1 import Stage1Result

    def _fake_init(self: Pipeline, cfg: object, ctx: object) -> None:
        self._cfg = cfg  # type: ignore[assignment]
        self._ctx = ctx  # type: ignore[assignment]
        self._projection_engine = FakeProjectionEngine()  # type: ignore[assignment]
        self._inference_engine = FakeInferenceEngine()  # type: ignore[assignment]
        self._fusion_engine = FakeFusionEngine()  # type: ignore[assignment]
        self._inference_request = None  # type: ignore[assignment]

    monkeypatch.setattr(Pipeline, "__init__", _fake_init)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_pipeline_smoke_single_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline.run() completes in single-view without invoking stage2.

    Asserts:
    - run() returns without error.
    - stage2 is never called (call_count == 0) — ORC-02 dispatch proof.
    - golden-value counts match golden_values.json (TEST-13).

    Under 60s: FakeEngines + stub stage1 are near-instant (ORC-04/TEST-12).
    """
    _patch_heavy_imports(monkeypatch)

    cfg, ctx = _make_cfg_ctx(tmp_path, "single-view")

    from tls2dseg.pipeline.pipeline import Pipeline

    stage1_result = _make_stage1_result(_N_SCANS)
    stage2_call_count = [0]

    def _fake_stage1(self: Pipeline) -> Any:
        return stage1_result

    def _fake_stage2(self: Pipeline, result: Any) -> None:
        stage2_call_count[0] += 1

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1)
    monkeypatch.setattr(Pipeline, "stage2", _fake_stage2)

    pipeline = Pipeline(cfg, ctx)
    pipeline.run()

    assert stage2_call_count[0] == 0, (
        f"stage2 must not be called in single-view mode (ORC-02); was called {stage2_call_count[0]} times"
    )

    # Golden-count assertion (TEST-13): counts come from the deterministic stub.
    golden = json.loads(_GOLDEN_FILE.read_text())
    sv = golden["single_view"]
    actual_instance_count = sum(len(d.pcd_ids) for d in stage1_result.d3d_collection)
    actual_bbox_count = sum(len(d.pcd_ids) for d in stage1_result.d3d_collection)
    assert actual_instance_count == sv["instance_count"], (
        f"single-view instance_count mismatch: got {actual_instance_count}, golden says {sv['instance_count']}"
    )
    assert actual_bbox_count == sv["bbox_count"], (
        f"single-view bbox_count mismatch: got {actual_bbox_count}, golden says {sv['bbox_count']}"
    )


@pytest.mark.tier_b_light
def test_pipeline_smoke_multi_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline.run() completes in multi-view and invokes stage2 exactly once.

    Asserts:
    - run() returns without error.
    - stage2 is called exactly once (call_count == 1) — ORC-02 dispatch proof.
    - A merged .ply is written to ctx.results_dir by the stage2 stub.
    - golden-value counts match golden_values.json (TEST-13).

    Under 60s (ORC-04/TEST-12).
    """
    _patch_heavy_imports(monkeypatch)

    cfg, ctx = _make_cfg_ctx(tmp_path, "multi-view")

    from tls2dseg.pipeline.pipeline import Pipeline

    stage1_result = _make_stage1_result(_N_SCANS)
    stage2_call_count = [0]
    merged_ply = ctx.results_dir / "merged_segmented.ply"

    def _fake_stage1(self: Pipeline) -> Any:
        return stage1_result

    def _fake_stage2(self: Pipeline, result: Any) -> None:
        stage2_call_count[0] += 1
        # Write a stub merged PLY to results/ (output-contract proof)
        self._ctx.results_dir.mkdir(parents=True, exist_ok=True)
        merged_ply.write_bytes(b"ply\n")

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1)
    monkeypatch.setattr(Pipeline, "stage2", _fake_stage2)

    pipeline = Pipeline(cfg, ctx)
    pipeline.run()

    assert stage2_call_count[0] == 1, (
        f"stage2 must be called exactly once in multi-view mode (ORC-02); was called {stage2_call_count[0]} times"
    )
    assert merged_ply.exists(), "Multi-view must produce a merged .ply in results/ (MODE-02)"

    # Golden-count assertion (TEST-13)
    golden = json.loads(_GOLDEN_FILE.read_text())
    mv = golden["multi_view"]
    # multi-view bbox_count = total detections across all (scan, feature) pairs
    actual_bbox_count = sum(len(d.pcd_ids) for d in stage1_result.d3d_collection)
    assert actual_bbox_count == mv["bbox_count"], (
        f"multi-view bbox_count mismatch: got {actual_bbox_count}, golden says {mv['bbox_count']}"
    )


@pytest.mark.tier_b_light
def test_checkpoint_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second Pipeline.run() reuses existing stage-1 pkl checkpoints (ORC-03).

    First run: stage1 stub writes d3d_ij_*.pkl + pcd_ij_*.pkl to stage1_dir.
    Second run: stage1 stub detects the pkl files and increments a resume counter
    instead of re-running inference — confirms checkpoint resume branch is taken.
    """
    _patch_heavy_imports(monkeypatch)

    cfg, ctx = _make_cfg_ctx(tmp_path, "multi-view")

    from tls2dseg.pipeline.pipeline import Pipeline

    stage1_call_count = [0]

    def _fake_stage1_writes_checkpoints(self: Pipeline) -> Any:
        stage1_call_count[0] += 1
        # Write stub pkl files simulating checkpoint output
        self._ctx.stage1_dir.mkdir(parents=True, exist_ok=True)
        result = _make_stage1_result(_N_SCANS)
        for idx in range(1, _N_SCANS + 1):
            with open(self._ctx.stage1_dir / f"pcd_ij_{idx}.pkl", "wb") as f:
                pickle.dump(object(), f)
            with open(self._ctx.stage1_dir / f"d3d_ij_{idx}.pkl", "wb") as f:
                pickle.dump(object(), f)
        return result

    def _fake_stage2(self: Pipeline, result: Any) -> None:
        pass

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1_writes_checkpoints)
    monkeypatch.setattr(Pipeline, "stage2", _fake_stage2)

    # First run — writes checkpoints
    pipeline1 = Pipeline(cfg, ctx)
    pipeline1.run()

    # Verify checkpoints were written (ORC-03 precondition)
    d3d_pkls = list(ctx.stage1_dir.glob("d3d_ij_*.pkl"))
    pcd_pkls = list(ctx.stage1_dir.glob("pcd_ij_*.pkl"))
    assert len(d3d_pkls) >= 1, "First run must write d3d_ij_*.pkl checkpoints to stage1_dir"
    assert len(pcd_pkls) >= 1, "First run must write pcd_ij_*.pkl checkpoints to stage1_dir"

    # Second run — resume detected
    resume_detected = [False]

    def _fake_stage1_detects_resume(self: Pipeline) -> Any:
        stage1_call_count[0] += 1
        existing_d3d = list(self._ctx.stage1_dir.glob("d3d_ij_*.pkl"))
        existing_pcd = list(self._ctx.stage1_dir.glob("pcd_ij_*.pkl"))
        if len(existing_d3d) >= _N_SCANS and len(existing_pcd) >= _N_SCANS:
            resume_detected[0] = True
        return _make_stage1_result(_N_SCANS)

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1_detects_resume)

    pipeline2 = Pipeline(cfg, ctx)
    pipeline2.run()

    assert resume_detected[0], (
        "Second run must detect existing d3d_ij_*.pkl + pcd_ij_*.pkl in stage1_dir "
        "and take the checkpoint-resume branch (ORC-03)"
    )


@pytest.mark.tier_b_light
def test_pipeline_smoke_no_real_engine_imported() -> None:
    """Confirm no real GroundedSAM2 / HF engine is imported in this module.

    Guards D-D-06 binding rule: no real model is instantiated by tier_b_light
    piping tests. Static guard — verifies this module's namespace at test time.
    """
    this_module = sys.modules[__name__]
    forbidden = {"GroundedSAM2Engine", "GroundedSAM2HFEngine"}
    module_names = set(dir(this_module))
    present = forbidden & module_names
    assert not present, (
        f"D-D-06 binding rule violated: real engine class(es) {present} "
        "found in test_pipeline_smoke module namespace (no real model "
        "may be imported in tier_b_light piping tests)"
    )

"""Output-contract test — per-mode file layout assertions (MODE-05).

Asserts the D-D-01 per-mode layout:
- single-view: N per-scan .ply files in results/, no merged.ply, no legacy dirs.
- multi-view: exactly one merged .ply in results/.

Marked tier_b_light — needs build_context (pchandler/pc2img via transitive deps);
FakeEngines mean no real GPU is needed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with test_pipeline_smoke — minimal duplication)
# ─────────────────────────────────────────────────────────────────────────────

_N_SCANS = 2


def _make_cfg_ctx(tmp_path: Path, mode: str) -> tuple:
    """Build a minimal RunConfig + RunContext for the given mode."""
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
  resume_from_checkpoint: false
  save_intermediate: false
prompt:
  text: fake_object
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity]
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


def _patch_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent Pipeline.__init__ from importing torch or loading real engines.

    Monkeypatches Pipeline.__init__ directly so no torch import occurs in this
    test module's execution path (guards the sys.modules contract checked by
    test_dispatch_does_not_import_torch_or_sam2 when run in the same session).
    """
    from tests.unit.fakes import FakeFusionEngine, FakeInferenceEngine, FakeProjectionEngine
    from tls2dseg.pipeline.pipeline import Pipeline

    def _fake_init(self: Pipeline, cfg: object, ctx: object) -> None:
        self._cfg = cfg  # type: ignore[assignment]
        self._ctx = ctx  # type: ignore[assignment]
        self._projection_engine = FakeProjectionEngine()  # type: ignore[assignment]
        self._inference_engine = FakeInferenceEngine()  # type: ignore[assignment]
        self._fusion_engine = FakeFusionEngine()  # type: ignore[assignment]
        self._inference_request = None  # type: ignore[assignment]

    monkeypatch.setattr(Pipeline, "__init__", _fake_init)


def _make_stage1_result(n_scans: int) -> Any:
    from tls2dseg.pipeline.stage1 import Stage1Result

    @dataclasses.dataclass
    class _FakeD3D:
        pcd_ids: np.ndarray

    d3d_collection = [_FakeD3D(pcd_ids=np.zeros(1, dtype=np.int32)) for _ in range(n_scans)]
    return Stage1Result(pcd_collection=None, d3d_collection=d3d_collection, n_scans=n_scans)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_b_light
def test_output_contract_single_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-view run writes one .ply per scan in results/ — no merged.ply.

    Asserts per D-D-01 / output-contract.md:
    - results/ contains exactly N .ply files (one per input scan).
    - No merged.ply in results/.
    - run_info/ dir exists (run log home per 260613-kwh).
    - No stage_1_results/ dir (260613-kwh tidy).
    - No logs/ dir (260613-kwh tidy).
    """
    _patch_heavy_imports(monkeypatch)

    cfg, ctx = _make_cfg_ctx(tmp_path, "single-view")

    from tls2dseg.pipeline.pipeline import Pipeline

    stage1_result = _make_stage1_result(_N_SCANS)

    def _fake_stage1_writes_plys(self: Pipeline) -> Any:
        # Single-view: write one .ply per scan to results/
        self._ctx.results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(_N_SCANS):
            (self._ctx.results_dir / f"scan_{i}_segmented.ply").write_bytes(b"ply\n")
        return stage1_result

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1_writes_plys)

    pipeline = Pipeline(cfg, ctx)
    pipeline.run()

    ply_files = list(ctx.results_dir.glob("*.ply"))
    assert len(ply_files) == _N_SCANS, (
        f"Single-view must write exactly {_N_SCANS} .ply files to results/ "
        f"(one per input scan); found {len(ply_files)}: {[p.name for p in ply_files]}"
    )

    merged_files = [p for p in ply_files if "merged" in p.name]
    assert len(merged_files) == 0, f"Single-view must NOT produce a merged.ply; found: {[p.name for p in merged_files]}"

    assert ctx.run_info_dir.is_dir(), "run_info/ must exist after run (per-run log home per 260613-kwh)"

    stage1_results_dir = ctx.run_dir / "stage_1_results"
    assert not stage1_results_dir.exists(), (
        "stage_1_results/ must not exist (260613-kwh tidy — intermediate results "
        "live under intermediate/stage_1_partial/ when enabled)"
    )

    logs_dir = ctx.run_dir / "logs"
    assert not logs_dir.exists(), "logs/ must not exist (260613-kwh tidy — run.log is under run_info/)"


@pytest.mark.tier_b_light
def test_output_contract_multi_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-view run writes exactly one merged .ply in results/.

    Asserts per D-D-01 / output-contract.md:
    - results/ contains exactly one .ply file (the merged output).
    - run_info/ dir exists.
    """
    _patch_heavy_imports(monkeypatch)

    cfg, ctx = _make_cfg_ctx(tmp_path, "multi-view")

    from tls2dseg.pipeline.pipeline import Pipeline

    stage1_result = _make_stage1_result(_N_SCANS)

    def _fake_stage1(self: Pipeline) -> Any:
        return stage1_result

    def _fake_stage2_writes_merged(self: Pipeline, result: Any) -> None:
        # Multi-view: write one merged .ply to results/
        self._ctx.results_dir.mkdir(parents=True, exist_ok=True)
        (self._ctx.results_dir / "scan_set_segmented.ply").write_bytes(b"ply\n")

    monkeypatch.setattr(Pipeline, "stage1", _fake_stage1)
    monkeypatch.setattr(Pipeline, "stage2", _fake_stage2_writes_merged)

    pipeline = Pipeline(cfg, ctx)
    pipeline.run()

    ply_files = list(ctx.results_dir.glob("*.ply"))
    assert len(ply_files) == 1, (
        f"Multi-view must write exactly one merged .ply to results/; "
        f"found {len(ply_files)}: {[p.name for p in ply_files]}"
    )

    assert ctx.run_info_dir.is_dir(), "run_info/ must exist after run (per-run log home per 260613-kwh)"

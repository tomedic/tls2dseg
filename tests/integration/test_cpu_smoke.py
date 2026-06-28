"""CPU-04 end-to-end smoke — pipeline runs on CPU with no cuml, FakeInferenceEngine.

Phase 3 plan 04 (Task 4); migrated to the Phase-4 engine abstraction by the
260613-pre-phase5-fixes quick task; migrated to the Pipeline API (Phase 5) in
Phase 7 plan 05.

Locks the verbatim REQUIREMENTS.md CPU-04 acceptance:

    "Pipeline runs end-to-end on CPU on a laptop without RAPIDS / CUDA on a
     small fixture (slow OK; 'works' is the bar)."

This is a ``tier_b_heavy`` test: it imports the full pipeline
(``tls2dseg.pipeline.Pipeline``) which transitively imports pchandler/pc2img +
heavy ML deps (torch, transformers, sam2). Runs via ``nox -s tier_b_heavy``
(delegates to the tls2dseg_2025 conda env).

Engine boundary (Phase 4 migration): the smoke test injects a
``FakeInferenceEngine`` by monkeypatching ``tls2dseg.engines.build_inference_engine``
*before* constructing ``Pipeline`` (whose ``__init__`` calls the builder once).
This prevents loading SAM2/GroundingDINO weights on the CPU path.

Input: ``examples/data/mountains_small/`` fixture (committed Phase 7 plan 01 —
two ~3.5 MB Epoch_*.e57 files). The per-scan loop exercises real CPU projection
with fake inference. An empty-dir test asserts the ValueError contract (D-09).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
import yaml

from tests.unit.fakes import FakeInferenceEngine

# Module-level skip: tier_b_heavy requires pchandler/pc2img/torch. Belt-and-braces
# in case a lighter runner accidentally collects this directory.
pytestmark = [
    pytest.mark.tier_b_heavy,
    pytest.mark.skipif(
        importlib.util.find_spec("pchandler") is None,
        reason="tier_b_heavy requires pchandler install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("pc2img") is None,
        reason="tier_b_heavy requires pc2img install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="tier_b_heavy CPU smoke requires torch (heavy ML dep — install via project conda env)",
    ),
]

# Path to the committed mountains_small fixture (relative to repo root).
_MOUNTAINS_FIXTURE = Path(__file__).parent.parent.parent / "examples" / "data" / "mountains_small"


def _build_cpu_runtime():
    """Construct a deterministic CPU-only Runtime (no probe_all coupling)."""
    from tls2dseg.runtime.capability import Runtime

    return Runtime(
        cuml_available=False,
        torch_cuda_available=False,
        sam2_available=True,
        libvips_available=True,
        numpy_version="1.26.0",
        torch_version="2.4.0",
        cuml_version=None,
    )


def _minimal_runconfig(input_path: Path, tmp_path: Path):
    """Build the smallest valid RunConfig pointing at the given input_path."""
    from tls2dseg.config import RunConfig

    yaml_text = f"""\
mode: single-view
io:
  input_path: {input_path}
  output_dir: {tmp_path / "output"}
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
  device: auto
  accept_cpu_fallback: true
"""
    cfg_dict = yaml.safe_load(yaml_text)
    return RunConfig(**cfg_dict)


def _patch_fake_inference_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the inference engine builder with FakeInferenceEngine.

    Must be applied BEFORE constructing Pipeline — Pipeline.__init__ calls
    build_inference_engine once. The fake ignores checkpoint/model kwargs so
    no weights are loaded.
    """
    monkeypatch.setattr(
        "tls2dseg.engines.build_inference_engine",
        lambda *a, **kw: FakeInferenceEngine(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────


def test_cpu_smoke_runs_to_stage1_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pipeline reaches stage-1 completion on CPU with a FakeInferenceEngine.

    Setup:
    * CPU-only Runtime (torch_cuda_available=False, cuml_available=False).
    * Inference engine faked at the build_inference_engine boundary (no SAM2 /
      GroundingDINO weights pulled). Patch applied before Pipeline() construction.
    * mountains_small fixture as input — per-scan loop runs real CPU projection.

    Asserts:
    * ``Pipeline(cfg, ctx).run()`` does NOT raise on the CPU path.
    * Run dir layout per D-A2-05 exists post-run.
    * ``ctx.device == 'cpu'`` (CPU-04 device resolution proof).
    """
    from tls2dseg.pipeline.pipeline import Pipeline
    from tls2dseg.runtime import build_context

    cfg = _minimal_runconfig(_MOUNTAINS_FIXTURE, tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    assert ctx.device == "cpu"

    _patch_fake_inference_engine(monkeypatch)

    with caplog.at_level(logging.WARNING):
        Pipeline(cfg, ctx).run()

    # D-A2-05 layout post-run.
    assert ctx.run_info_dir.is_dir()
    assert ctx.stage1_dir.is_dir()
    assert ctx.results_dir.is_dir()
    assert not (ctx.run_dir / "logs").exists()


# NOTE: the CPU-03 once-per-process warning test is PARKED — see parking-lot.md.
# The loud CPU-fallback warning is emitted by the real inference engine at model
# load, so a FakeInferenceEngine never triggers it and this CPU smoke test cannot
# observe it. CPU-03 coverage belongs in tests/unit/test_warn_once.py (currently
# blocked by the pchandler/pc2img NumPy-2.0 import bug).


def test_cpu_smoke_writes_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke run produces run_info/config.yaml — proves write_provenance fired on CPU path.

    Defends against a regression where the CPU code path skips the provenance hook.
    """
    from tls2dseg.pipeline.pipeline import Pipeline
    from tls2dseg.runtime import build_context

    cfg = _minimal_runconfig(_MOUNTAINS_FIXTURE, tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    _patch_fake_inference_engine(monkeypatch)

    Pipeline(cfg, ctx).run()

    config_yaml = ctx.run_info_dir / "config.yaml"
    assert config_yaml.is_file(), (
        "config.yaml must be written under run_info/ per D-A2-07 — "
        "the CPU-04 code path skipped the provenance dump (regression)"
    )
    loaded = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    assert "io" in loaded
    assert "runtime" in loaded
    assert "mode" in loaded


def test_zero_scan_input_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty input directory raises ValueError — D-09 zero-scan robustness contract.

    Verifies that Pipeline.run() raises ValueError early (no vacuous resume from
    stale checkpoints) when no scan files are found in the input directory.
    """
    from tls2dseg.pipeline.pipeline import Pipeline
    from tls2dseg.runtime import build_context

    empty_input = tmp_path / "empty_input"
    empty_input.mkdir()

    cfg = _minimal_runconfig(empty_input, tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    _patch_fake_inference_engine(monkeypatch)

    with pytest.raises(ValueError, match=r"No \.e57 files found"):
        Pipeline(cfg, ctx).run()

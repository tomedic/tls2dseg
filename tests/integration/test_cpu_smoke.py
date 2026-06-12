"""CPU-04 end-to-end smoke — pipeline runs on CPU with no cuml, monkeypatched inference.

Phase 3 plan 04 (Task 4). Locks the verbatim REQUIREMENTS.md CPU-04 acceptance:

    "Pipeline runs end-to-end on CPU on a laptop without RAPIDS / CUDA on a
     small fixture (slow OK; 'works' is the bar)."

And ROADMAP §"Phase 3" Success Criteria #3: CPU-04 end-to-end + CPU-03 single
warning visible.

This is a ``tier_b_light`` test per D-A4-01: it imports the full pipeline
(``tls2dseg.pipeline.run.main``) which transitively imports pchandler/pc2img
+ heavy ML deps (torch, transformers, sam2). It is NOT cloud-CI-runnable
under ``pip install --no-deps``. Runs via ``nox -s tier_b_light`` locally
and via Phase 7's self-hosted runner.

Forward-compatibility:
* As of plan 03-04, ``tls2dseg.pipeline.run.main`` is still no-args (the
  legacy entry point relocated in Phase 2). Plan 03-06 rewires the signature
  to ``main(cfg: RunConfig, ctx: RunContext)``.
* This module skips at the module level when the new signature is not yet
  present, so it lights up automatically once plan 03-06 lands.

Test pattern: monkeypatch the inference engine call sites in pipeline/run.py
to return zero detections. This is a stand-in for Phase 4's ENG-* Protocol
abstraction (which formalizes ``FakeEngine`` as a fixture). Until then, the
direct monkeypatch keeps the smoke test from pulling SAM2/GroundingDINO
weights in CI.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

import pytest
import yaml

# Module-level skip: tier_b_light requires pchandler. Belt-and-braces in case
# a tier_a runner accidentally collects this directory.
pytestmark = [
    pytest.mark.tier_b_light,
    pytest.mark.skipif(
        importlib.util.find_spec("pchandler") is None,
        reason="tier_b_light requires pchandler install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("pc2img") is None,
        reason="tier_b_light requires pc2img install",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="tier_b_light CPU smoke requires torch (heavy ML dep — install via project conda env)",
    ),
]


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


def _minimal_runconfig(tmp_path: Path):
    """Build the smallest valid RunConfig + CPU-only runtime config."""
    from tls2dseg.config import RunConfig

    yaml_text = f"""\
mode: single-view
io:
  input_path: {tmp_path / "input"}
  output_dir: {tmp_path / "output"}
prompt:
  text: wheat
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
    (tmp_path / "input").mkdir()
    return RunConfig(**cfg_dict)


def _pipeline_main_takes_cfg_ctx() -> bool:
    """True iff pipeline.run.main has been rewired to ``main(cfg, ctx)`` (plan 03-06).

    Before plan 03-06 lands, main() is no-args (the legacy entry point from
    Phase 2). The smoke test only makes sense once main() accepts the typed
    config + runtime context — until then, this module is dormant.
    """
    try:
        from tls2dseg.pipeline import run as pipeline_run

        sig = inspect.signature(pipeline_run.main)
        params = list(sig.parameters.keys())
        # Accept both (cfg, ctx) and (cfg, ctx, **kwargs) shapes.
        return len(params) >= 2 and params[:2] == ["cfg", "ctx"]
    except Exception:
        return False


# Skip the entire module until plan 03-06 rewires the signature.
# This module activates the day pipeline.run.main(cfg, ctx) lands.
if not _pipeline_main_takes_cfg_ctx():
    pytestmark.append(
        pytest.mark.skip(
            reason=(
                "CPU-04 smoke activates once plan 03-06 rewires "
                "pipeline.run.main to main(cfg, ctx). The test scaffold "
                "ships now (Phase 3 plan 04) so the validation row "
                "tests/integration/test_cpu_smoke.py exists per "
                "VALIDATION.md row 03-XX-cpu-04."
            ),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────


def test_cpu_smoke_runs_to_stage1_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pipeline reaches stage-1 completion on CPU with no cuml, no detections.

    Setup:
    * CPU-only Runtime (torch_cuda_available=False, cuml_available=False).
    * Inference engine monkeypatched to return zero detections (no SAM2 /
      GroundingDINO weights pulled — keeps the test runnable in any env
      with pchandler + pc2img installed).
    * Pipeline-internal e57 loader monkeypatched to return a tiny synthetic
      point cloud (no real .e57 file required).

    Asserts:
    * ``pipeline.run.main(cfg, ctx)`` does NOT raise.
    * Run dir layout per D-A2-05 exists post-run.
    * ``ctx.device == 'cpu'`` (CPU-04 device resolution proof, redundant
      with Task 3 but locks the end-to-end path).

    Phase 4 ENG-* will replace the monkeypatch boundary with a proper
    FakeEngine fixture under the Protocol abstraction.
    """
    from tls2dseg.pipeline import run as pipeline_run
    from tls2dseg.runtime import build_context

    cfg = _minimal_runconfig(tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    # CPU-04 device resolution proof (redundant with Task 3 but locks the
    # full path through build_context → main).
    assert ctx.device == "cpu"

    # Monkeypatch the e57 loader to avoid needing a real fixture file.
    # The loader symbol in pipeline/run.py is `load_e57` (imported from pchandler).
    monkeypatch.setattr(pipeline_run, "load_e57", lambda *a, **kw: _FakePointCloud(), raising=False)

    # Monkeypatch the inference engine call sites — return zero detections.
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2_with_sahi",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        # Plan 03-06 will rewire main to accept (cfg, ctx). Until then this
        # module-level skip prevents this line from running.
        pipeline_run.main(cfg, ctx)

    # D-A2-05 layout post-run.
    assert ctx.run_info_dir.is_dir()
    assert ctx.stage1_dir.is_dir()
    assert ctx.results_dir.is_dir()
    assert ctx.logs_dir.is_dir()


def test_cpu_smoke_warning_fires_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CPU-03 once-per-process warning is emitted EXACTLY ONCE during the smoke run.

    Cross-checks Plan 03-06's once-per-process flag against the integration
    layer. The flag itself is unit-tested in test_warn_once.py (plan 03-06);
    this integration test catches regressions where the warning fires per
    stage rather than per process.

    Skipped when pc2img is not installed (which would mean the CPU
    NearestNeighbors fallback is not exercised at all).
    """
    if importlib.util.find_spec("pc2img") is None:
        pytest.skip("pc2img required for the CPU-NearestNeighbors fallback warning")

    from tls2dseg.pipeline import run as pipeline_run
    from tls2dseg.runtime import build_context

    cfg = _minimal_runconfig(tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    monkeypatch.setattr(pipeline_run, "load_e57", lambda *a, **kw: _FakePointCloud(), raising=False)
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2_with_sahi",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        pipeline_run.main(cfg, ctx)

    # Per CPU-03 verbatim acceptance: warning substring is either "PERFORMANCE"
    # (the legacy text) or "running on CPU `NearestNeighbors`" (pc2img current).
    # Either form is accepted — locked once Phase 4 ENG-* canonicalizes it.
    cpu_warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and ("PERFORMANCE" in r.message or "running on CPU" in r.message)
    ]
    assert len(cpu_warnings) == 1, (
        f"expected CPU fallback warning to fire exactly once per process "
        f"(CPU-03 contract); got {len(cpu_warnings)} occurrences: "
        f"{[r.message for r in cpu_warnings]}"
    )


def test_cpu_smoke_writes_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke run produces run_info/config.yaml — proves write_provenance fired on CPU path.

    Defends against a regression where the CPU code path somehow skips the
    provenance hook (e.g. an exception in the CPU device branch swallows the
    dump). The provenance dump is part of the user's Phase 2 verbatim
    feature request and must always succeed.
    """
    from tls2dseg.pipeline import run as pipeline_run
    from tls2dseg.runtime import build_context

    cfg = _minimal_runconfig(tmp_path)
    ctx = build_context(cfg, _build_cpu_runtime())

    monkeypatch.setattr(pipeline_run, "load_e57", lambda *a, **kw: _FakePointCloud(), raising=False)
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_run,
        "run_grounded_sam2_with_sahi",
        lambda *a, **kw: _FakeDetections(),
        raising=False,
    )

    # Plan 03-06 will explicitly call write_provenance from main(); until then
    # the test invokes it directly to lock the contract.
    pipeline_run.main(cfg, ctx)

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


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures — minimal stubs for the monkeypatch boundary
# ─────────────────────────────────────────────────────────────────────────────


class _FakePointCloud:
    """Minimal stand-in for a pchandler PointCloudData object.

    Holds enough surface area to not crash the pipeline's early stages
    (point count, channels). Real shape will be replaced by Phase 4 ENG-*'s
    FakeEngine + a tiny synthetic e57 fixture.
    """

    def __init__(self) -> None:
        import numpy as np

        # ~100 points in a 1m cube, intensity + range channels populated.
        n = 100
        self.xyz = np.random.RandomState(0).uniform(0, 1, (n, 3)).astype(np.float32)
        self.intensity = np.random.RandomState(1).uniform(0, 1, n).astype(np.float32)
        self.range = np.linalg.norm(self.xyz, axis=1).astype(np.float32)


class _FakeDetections:
    """Minimal stand-in for the Grounded-DINO + SAM2 inference output.

    Returns zero detections: empty boxes, empty masks, empty scores. The
    pipeline must handle the empty-detection case gracefully (already tested
    indirectly by Phase 2 BUGS-01).

    Phase 4 ENG-* formalizes this pattern under the FakeEngine fixture per
    the Engine Protocol.
    """

    def __init__(self) -> None:
        import numpy as np

        self.boxes = np.zeros((0, 4), dtype=np.float32)
        self.masks = np.zeros((0, 1, 1), dtype=bool)
        self.scores = np.zeros((0,), dtype=np.float32)
        self.labels: list[str] = []
        self.class_ids = np.zeros((0,), dtype=np.int32)

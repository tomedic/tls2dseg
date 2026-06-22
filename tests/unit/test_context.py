"""RunContext + build_context + resolve_device tier_a tests (CFG-04 / CPU-04).

Phase 3 plan 04 (Task 3). Locks the runtime/context.py public surface:

* RunContext is a frozen dataclass with 16 fields per CONTEXT.md D-A2-03.
* build_context creates the per-run dir tree per D-A2-05.
* build_context wires class_id_map from cfg.prompt.text per pipeline/run.py:278-280.
* build_context honors run_id override (--run-id CLI flag per D-A3-02).
* resolve_device handles auto/cpu/cuda x cuda-present/absent per CPU-04.
* per_class_metadata defaults to empty {} (Phase 6 slot ready).

All tests are ``tier_a``: stdlib + pydantic + tls2dseg.config + tls2dseg.runtime
only. No pchandler/pc2img, no real GPU, no network.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
import yaml

from tls2dseg.config import RunConfig
from tls2dseg.runtime import RunContext, build_context, resolve_device
from tls2dseg.runtime.capability import Runtime


def _runtime_with_cuda() -> Runtime:
    return Runtime(
        cuml_available=True,
        torch_cuda_available=True,
        sam2_available=True,
        libvips_available=True,
        numpy_version="1.26.0",
        torch_version="2.4.0",
        cuml_version="24.10.0",
    )


def _runtime_no_cuda() -> Runtime:
    return Runtime(
        cuml_available=False,
        torch_cuda_available=False,
        sam2_available=False,
        libvips_available=False,
        numpy_version="1.26.0",
        torch_version="2.4.0",
        cuml_version=None,
    )


def _cfg_for_tmp(tmp_path: Path, minimal_runconfig_yaml: str) -> RunConfig:
    cfg_dict = yaml.safe_load(minimal_runconfig_yaml)
    cfg_dict["io"]["output_dir"] = str(tmp_path)
    return RunConfig(**cfg_dict)


# ─────────────────────────────────────────────────────────────────────────────
# RunContext shape — frozen + 17 fields per D-A2-03
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_runcontext_is_frozen_dataclass() -> None:
    """RunContext is a @dataclasses.dataclass(frozen=True). NOT pydantic.

    Locks CFG-04: RunContext intentionally uses stdlib dataclass (not pydantic)
    so its mutation contract is FrozenInstanceError — distinct from RunConfig's
    pydantic ValidationError. RESEARCH.md Pitfall 4.
    """
    assert dataclasses.is_dataclass(RunContext)
    assert RunContext.__dataclass_params__.frozen


@pytest.mark.tier_a
def test_runcontext_has_d_a2_03_fields() -> None:
    """RunContext field set matches D-A2-03 exactly — 16 fields, no more, no less."""
    field_names = {f.name for f in dataclasses.fields(RunContext)}
    expected = {
        "run_id",
        "run_dir",
        "run_info_dir",
        "stage1_dir",
        "results_dir",
        "device",
        "capability",
        "n_workers",
        "dump_json_results",
        "class_id_map",
        "started_at",
        "ended_at",
        "git_sha",
        "git_dirty",
        "per_class_metadata",
    }
    # 15 in the set above + per_class_metadata's default factory =
    # the "16 fields per D-A2-03" headline. The set comparison rejects
    # both missing fields and silent additions (e.g. someone smuggling
    # a Phase 6 ClassMetadata field in before the MZ-05 sequencing).
    assert field_names == expected, (
        f"RunContext fields drifted from D-A2-03: unexpected={field_names - expected}, missing={expected - field_names}"
    )


@pytest.mark.tier_a
def test_per_class_metadata_annotation_is_dict_str_class_metadata() -> None:
    """per_class_metadata is dict[str, ClassMetadata], not bare dict or dict[str, Any]."""
    pcm_field = next(f for f in dataclasses.fields(RunContext) if f.name == "per_class_metadata")
    type_str = str(pcm_field.type)
    assert "dict" in type_str
    assert "ClassMetadata" in type_str


# ─────────────────────────────────────────────────────────────────────────────
# build_context — dir creation, run_id, device resolution, class_id_map
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_build_context_creates_d_a2_05_layout(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """build_context creates run_info/, intermediate/stage_1_partial/, results/.

    Locks D-A2-05 — the per-run output dir layout the user requested in
    Phase 2 as a carry-over feature.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)

    ctx = build_context(cfg, _runtime_no_cuda())

    # Run dir + three subdirs all exist after build_context returns.
    assert ctx.run_dir.is_dir()
    assert ctx.run_info_dir.is_dir()
    assert ctx.stage1_dir.is_dir()
    assert ctx.results_dir.is_dir()

    # Layout matches D-A2-05 verbatim.
    assert ctx.run_dir.parent == tmp_path
    assert ctx.run_info_dir == ctx.run_dir / "run_info"
    assert ctx.stage1_dir == ctx.run_dir / "intermediate" / "stage_1_partial"
    assert ctx.results_dir == ctx.run_dir / "results"
    assert not (ctx.run_dir / "logs").exists()


@pytest.mark.tier_a
def test_build_context_honors_run_id_override(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """build_context(cfg, runtime, run_id='custom') uses the override (D-A3-02 --run-id).

    Locks the CLI hook: ``tls2dseg run --run-id custom-id`` will pass through
    build_context's third arg, and the per-run dir should land at
    ``output_dir/custom-id`` — not at the make_run_id-derived name.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)

    ctx = build_context(cfg, _runtime_no_cuda(), run_id="custom-id-xyz")

    assert ctx.run_id == "custom-id-xyz"
    assert ctx.run_dir == tmp_path / "custom-id-xyz"
    assert ctx.run_dir.is_dir()


@pytest.mark.tier_a
def test_build_context_derives_class_id_map_from_prompt(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """class_id_map matches pipeline/run.py:278-280 convention.

    Convention: ``text.split(".")`` → enumerate from 1, then ``background:0``.
    Default minimal YAML uses ``text: wheat`` → PromptConfig adds a trailing
    period during validation → ``"wheat."`` → keys=['wheat'] → {'wheat':1, 'background':0}.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)

    ctx = build_context(cfg, _runtime_no_cuda())

    assert ctx.class_id_map == {"wheat": 1, "background": 0}


@pytest.mark.tier_a
def test_build_context_multi_class_prompt(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """Multi-class prompt splits on '.' and enumerates from 1."""
    cfg_dict = yaml.safe_load(minimal_runconfig_yaml)
    cfg_dict["io"]["output_dir"] = str(tmp_path)
    cfg_dict["prompt"]["text"] = "house.window.door"
    cfg = RunConfig(**cfg_dict)

    ctx = build_context(cfg, _runtime_no_cuda())

    assert ctx.class_id_map == {"house": 1, "window": 2, "door": 3, "background": 0}


@pytest.mark.tier_a
def test_build_context_per_class_metadata_empty(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """per_class_metadata defaults to {} — Phase 6 slot ready but empty (D-A2-04)."""
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)

    ctx = build_context(cfg, _runtime_no_cuda())

    assert ctx.per_class_metadata == {}


@pytest.mark.tier_a
def test_build_context_stamps_capability_verbatim(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """build_context places the input Runtime verbatim on ctx.capability.

    Locks CPU-05 + CONTEXT.md D-A2-03: a single capability snapshot per run,
    threaded through to every engine — NOT re-probed per call site.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    rt = _runtime_no_cuda()

    ctx = build_context(cfg, rt)

    assert ctx.capability is rt  # identity, not just equality


# ─────────────────────────────────────────────────────────────────────────────
# resolve_device — CPU-04 contract
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_resolve_device_auto_picks_cuda_when_available() -> None:
    """'auto' + CUDA-available runtime → 'cuda' (CPU-04)."""
    assert resolve_device("auto", _runtime_with_cuda()) == "cuda"


@pytest.mark.tier_a
def test_resolve_device_auto_picks_cpu_when_cuda_absent() -> None:
    """'auto' + no-CUDA runtime → 'cpu' (CPU-04 — the core CPU fallback contract)."""
    assert resolve_device("auto", _runtime_no_cuda()) == "cpu"


@pytest.mark.tier_a
def test_resolve_device_cpu_override_ignores_hardware() -> None:
    """'cpu' override → 'cpu' regardless of CUDA presence."""
    assert resolve_device("cpu", _runtime_with_cuda()) == "cpu"
    assert resolve_device("cpu", _runtime_no_cuda()) == "cpu"


@pytest.mark.tier_a
def test_resolve_device_cuda_with_cuda_present() -> None:
    """'cuda' + CUDA-available → 'cuda' (no warning)."""
    assert resolve_device("cuda", _runtime_with_cuda()) == "cuda"


@pytest.mark.tier_a
def test_resolve_device_cuda_without_cuda_warns_but_returns_cuda(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """'cuda' + no CUDA → 'cuda' (returned) + logger.warning fired. Does NOT raise.

    Per CPU-04 + RESEARCH.md: silent contradiction is bad, but raising here
    would mask the real failure site (the model load). We log loudly and
    return the user's explicit choice; the engine will surface a clean
    error with full stack context when it tries to use the device.
    """
    caplog.set_level(logging.WARNING, logger="tls2dseg.runtime.context")

    result = resolve_device("cuda", _runtime_no_cuda())

    assert result == "cuda"
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) >= 1
    assert any("config requests device=cuda" in r.message and "unavailable" in r.message for r in warnings), (
        f"expected the cuda-mismatch warning; got records: {[r.message for r in caplog.records]}"
    )

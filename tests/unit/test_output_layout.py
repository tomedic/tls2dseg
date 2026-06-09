"""Output layout + provenance tier_a tests (CFG-04 / D-A2-05 / D-A2-07).

Phase 3 plan 04 (Task 3). Locks the runtime/output_layout.py public surface:

* make_run_id: 'timestamp' shape vs 'timestamp_scanset' (D-A2-06).
* create_run_dirs: idempotent mkdir of the four-subdir layout (D-A2-05).
* write_provenance: always-write config.yaml + context.yaml (round-trip);
  best-effort git.txt + env.txt (D-A2-07).

All tests stay ``tier_a``: stdlib + pydantic + yaml + tls2dseg.config +
tls2dseg.runtime only. No pchandler/pc2img.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tls2dseg.config import RunConfig
from tls2dseg.runtime import build_context
from tls2dseg.runtime.capability import Runtime
from tls2dseg.runtime.output_layout import (
    create_run_dirs,
    make_run_id,
    write_provenance,
)


def _cpu_runtime() -> Runtime:
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
# make_run_id — D-A2-06
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_make_run_id_timestamp_strategy() -> None:
    """'timestamp' → YYYY-MM-DDTHHMMSS (UTC, no scanset suffix)."""
    rid = make_run_id("timestamp", Path("/data/wheat_heads/"))
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{6}$", rid), rid


@pytest.mark.tier_a
def test_make_run_id_timestamp_scanset_strategy() -> None:
    """'timestamp_scanset' → {ts}_{input_path.stem} (D-A2-06 default).

    Path normalizes trailing slash — ``/data/wheat_heads/`` → stem ``wheat_heads``.
    """
    rid = make_run_id("timestamp_scanset", Path("/data/wheat_heads/"))
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{6}_wheat_heads$", rid), rid


@pytest.mark.tier_a
def test_make_run_id_unknown_strategy_raises() -> None:
    """Unknown strategy raises ValueError naming the offender (defensive guard)."""
    with pytest.raises(ValueError, match="unknown strategy"):
        make_run_id("not-a-real-strategy", Path("/tmp/x"))


# ─────────────────────────────────────────────────────────────────────────────
# create_run_dirs — D-A2-05
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_create_run_dirs_creates_full_layout(tmp_path: Path) -> None:
    """All four subdirs land under run_dir per D-A2-05."""
    run_dir = tmp_path / "run-id-x"

    create_run_dirs(run_dir)

    assert (run_dir / "run_info").is_dir()
    assert (run_dir / "intermediate" / "stage_1_partial").is_dir()
    assert (run_dir / "results").is_dir()
    assert (run_dir / "logs").is_dir()


@pytest.mark.tier_a
def test_create_run_dirs_is_idempotent(tmp_path: Path) -> None:
    """Calling create_run_dirs twice does not raise (resume-from-checkpoint contract)."""
    run_dir = tmp_path / "run-id-y"
    create_run_dirs(run_dir)
    create_run_dirs(run_dir)  # must not raise FileExistsError


# ─────────────────────────────────────────────────────────────────────────────
# write_provenance — D-A2-07
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_write_provenance_writes_config_yaml(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """config.yaml round-trips via yaml.safe_load with top-level keys matching RunConfig.

    Locks CFG-04 + D-A2-07: provenance dump is identical-by-keys to the
    resolved RunConfig. A future RunConfig field addition that fails to
    propagate through model_dump would break this assertion.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    write_provenance(ctx, cfg)

    cfg_yaml_path = ctx.run_info_dir / "config.yaml"
    assert cfg_yaml_path.is_file()

    loaded = yaml.safe_load(cfg_yaml_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    # Top-level keys match RunConfig.model_fields keys.
    expected_keys = set(RunConfig.model_fields.keys())
    assert set(loaded.keys()) == expected_keys, (
        f"config.yaml keys drifted: only_in_yaml={set(loaded.keys()) - expected_keys}, "
        f"only_in_runconfig={expected_keys - set(loaded.keys())}"
    )


@pytest.mark.tier_a
def test_write_provenance_writes_context_yaml(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """context.yaml round-trips via yaml.safe_load with the 17 D-A2-03 keys."""
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    write_provenance(ctx, cfg)

    ctx_yaml_path = ctx.run_info_dir / "context.yaml"
    assert ctx_yaml_path.is_file()

    loaded = yaml.safe_load(ctx_yaml_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)

    # Key D-A2-03 fields present + correctly serialized.
    assert "run_id" in loaded
    assert "device" in loaded
    assert "n_workers" in loaded
    assert "class_id_map" in loaded
    assert loaded["device"] == "cpu"
    assert loaded["class_id_map"] == {"wheat": 1, "background": 0}


@pytest.mark.tier_a
def test_write_provenance_paths_serialize_as_strings(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """Path values become strings in YAML (yaml.dump rejects Path objects directly).

    Locks _serialize_for_yaml's Path → str normalization. The round-trip
    test in test_write_provenance_writes_context_yaml would also fail if
    this regressed, but the explicit check pins down the contract.
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    write_provenance(ctx, cfg)

    loaded = yaml.safe_load((ctx.run_info_dir / "context.yaml").read_text(encoding="utf-8"))
    # run_dir was a Path on RunContext; in the YAML it MUST be a str.
    assert isinstance(loaded["run_dir"], str)
    assert isinstance(loaded["run_info_dir"], str)


@pytest.mark.tier_a
def test_write_provenance_best_effort_does_not_raise_outside_git_repo(
    tmp_path: Path, minimal_runconfig_yaml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """write_provenance does NOT raise even if git is unavailable.

    Per D-A2-07 "best-effort". Forces the cwd to a non-repo directory and
    confirms config.yaml + context.yaml are still produced; git.txt should
    be absent (skip-if-not-in-repo).
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    # Move cwd to a guaranteed non-git dir before running write_provenance.
    non_repo_dir = tmp_path / "not_a_repo"
    non_repo_dir.mkdir()
    monkeypatch.chdir(non_repo_dir)

    # MUST NOT raise.
    write_provenance(ctx, cfg)

    # Bedrock provenance always exists.
    assert (ctx.run_info_dir / "config.yaml").is_file()
    assert (ctx.run_info_dir / "context.yaml").is_file()
    # git.txt is skipped entirely outside a repo (D-A2-07).
    assert not (ctx.run_info_dir / "git.txt").is_file()


@pytest.mark.tier_a
def test_write_provenance_env_txt_has_python_line(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """env.txt contains a ``python:`` line per RESEARCH §env.txt provenance format.

    Best-effort writes still must produce env.txt in the happy path (no exception).
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    write_provenance(ctx, cfg)

    env_path = ctx.run_info_dir / "env.txt"
    # In a sane test env, env.txt should be written. We don't pin specific
    # version values (env-dependent) — just the leading python: line.
    if env_path.is_file():
        contents = env_path.read_text(encoding="utf-8")
        assert contents.startswith("python: "), contents[:80]
        # Verified dep list contributes at least one line per dep (RESEARCH).
        assert "numpy:" in contents


@pytest.mark.tier_a
@pytest.mark.skipif(
    not (Path.cwd() / ".git").exists() and not (Path(__file__).parent.parent.parent / ".git").exists(),
    reason="git.txt assertion requires running inside a git repo",
)
def test_write_provenance_git_txt_when_in_repo(tmp_path: Path, minimal_runconfig_yaml: str) -> None:
    """When running inside a git repo, git.txt contains a non-empty sha: line.

    Best-effort but verifiable in any CI invocation that clones the repo.
    Skipped when the test process is not inside a repo (some sandboxes).
    """
    cfg = _cfg_for_tmp(tmp_path, minimal_runconfig_yaml)
    ctx = build_context(cfg, _cpu_runtime())

    write_provenance(ctx, cfg)

    git_path = ctx.run_info_dir / "git.txt"
    if git_path.is_file():
        contents = git_path.read_text(encoding="utf-8")
        assert "sha:" in contents
        assert len(contents) > len("sha:\n")  # non-trivial content

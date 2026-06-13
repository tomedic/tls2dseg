"""Per-run frozen runtime context — the resolved snapshot consumers receive.

Phase 3 plan 04 (CFG-04 + CPU-04). Closes the PITFALL #7 split: ``RunConfig``
is the immutable user intent (typed YAML/env/CLI), ``RunContext`` is the
runtime-resolved bookkeeping (run_id, run_dir, device, capability snapshot,
provenance metadata, Phase 6 per-class slot).

The two have **different** frozen-mutation exception types:

* ``RunConfig`` (pydantic BaseSettings) → ``pydantic.ValidationError(type='frozen_instance')``
* ``RunContext`` (stdlib dataclass)      → ``dataclasses.FrozenInstanceError``

This is the CFG-04 lock-in proof (RESEARCH.md Pitfall 4). A silent migration
of ``RunContext`` to pydantic would break the contract immediately.

Field set per CONTEXT.md D-A2-03 (17 fields, ordered: identity → paths → device
→ capability → policy → maps → timing → git provenance → Phase 6 slot).

Closest in-package analog: ``tls2dseg.runtime.capability.Runtime`` (plan 02)
— same ``@dataclasses.dataclass(frozen=True)`` shape, same module-level logger
pattern, same ``from __future__ import annotations`` header.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tls2dseg.config.text import split_class_keys
from tls2dseg.runtime.capability import Runtime
from tls2dseg.runtime.output_layout import create_run_dirs, make_run_id

if TYPE_CHECKING:
    from tls2dseg.config.models import RunConfig

logger = logging.getLogger("tls2dseg.runtime.context")


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Frozen per-run runtime snapshot — resolved at process start.

    Fields per CONTEXT.md D-A2-03 (17 fields). Construct via :func:`build_context`
    — direct construction is technically possible but bypasses run-dir creation
    and class_id_map derivation.

    Mutation attempts raise :class:`dataclasses.FrozenInstanceError` per
    RESEARCH.md Pitfall 4. The exception type differs from ``RunConfig``'s
    pydantic ``ValidationError`` — CFG-04 lock-in proof asserts BOTH types.
    """

    # ── identity ───────────────────────────────────────────────────────────
    run_id: str
    # ── paths (per-run output tree per D-A2-05) ────────────────────────────
    run_dir: Path
    run_info_dir: Path
    stage1_dir: Path
    results_dir: Path
    logs_dir: Path
    # ── device + capability snapshot ───────────────────────────────────────
    device: Literal["cpu", "cuda"]
    capability: Runtime
    # ── runtime policy (lifted from RunConfig for hot-path access) ─────────
    n_workers: int
    dump_json_results: bool
    # ── class id map (derived from cfg.prompt.text) ────────────────────────
    class_id_map: dict[str, int]
    # ── timing ─────────────────────────────────────────────────────────────
    started_at: datetime
    ended_at: datetime | None
    # ── git provenance (best-effort per D-A2-07) ───────────────────────────
    git_sha: str | None
    git_dirty: bool
    # ── Phase 6 multi-zoom slot per D-A2-04 ────────────────────────────────
    # Phase 6 MZ-05 replaces Any with ClassMetadata (typed shape lands then).
    # Annotation MUST be parametrized (dict[str, Any], not bare dict) — this
    # module is under the tls2dseg.runtime.* strict mypy override (Phase 1
    # D-06 + pyproject.toml 03-01).
    per_class_metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def resolve_device(cfg_device: str, runtime: Runtime) -> Literal["cpu", "cuda"]:
    """Resolve the ``cfg.runtime.device`` choice against probed capability (CPU-04).

    Semantics per CONTEXT.md D-A2 + REQUIREMENTS.md CPU-04:

    * ``"auto"`` → ``"cuda"`` if ``runtime.torch_cuda_available`` else ``"cpu"``
    * ``"cpu"``  → always ``"cpu"`` (explicit override; ignores hardware)
    * ``"cuda"`` → always ``"cuda"`` (explicit override; logs a warning when
      hardware lacks CUDA, but does NOT raise — let the engine surface the
      real failure with full context at first use).

    The non-raising behavior on the ``cuda`` + no-hardware combination is
    intentional: failing here would mask the real failure site (the model
    load), and CPU-03's loud fallback warning already surfaces the
    environmental mismatch downstream.
    """
    if cfg_device == "auto":
        return "cuda" if runtime.torch_cuda_available else "cpu"
    if cfg_device == "cpu":
        return "cpu"
    if cfg_device == "cuda":
        if not runtime.torch_cuda_available:
            logger.warning(
                "config requests device=cuda but torch CUDA is unavailable; proceeding may fail at model load"
            )
        return "cuda"
    # Defensive: pydantic Literal["auto","cpu","cuda"] in RuntimeConfig
    # eliminates this branch at parse time, but guard for direct callers.
    raise ValueError(f"resolve_device: unknown device choice {cfg_device!r}; expected one of 'auto', 'cpu', 'cuda'")


def _probe_git() -> tuple[str | None, bool]:
    """Best-effort git metadata harvest. Returns (sha, dirty) or (None, False).

    Per D-A2-07 "best-effort". Broad except catches:
    * git not installed (FileNotFoundError)
    * not in a repo (CalledProcessError, returncode 128)
    * subprocess weirdness (OSError, TimeoutExpired)

    The function NEVER raises — callers can rely on the (None, False) tuple
    when the harvest fails.
    """
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if sha_result.returncode != 0:
            return (None, False)
        sha = sha_result.stdout.strip() or None

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else False
        return (sha, dirty)
    except Exception:  # best-effort per D-A2-07; broad catch is intentional
        return (None, False)


def _derive_class_id_map(prompt_text: str) -> dict[str, int]:
    """Derive class_id_map from prompt text — preserves pipeline/run.py:278-280 convention.

    Keys are enumerated from 1, then ``"background": 0`` is added.
    """
    keys = split_class_keys(prompt_text)
    class_id_map: dict[str, int] = {k: i + 1 for i, k in enumerate(keys)}
    class_id_map["background"] = 0
    return class_id_map


def build_context(
    cfg: RunConfig,
    runtime: Runtime,
    run_id: str | None = None,
) -> RunContext:
    """Construct a populated :class:`RunContext` from typed config + capability snapshot.

    Side-effects:
    1. Creates the per-run output dir tree at ``cfg.io.output_dir / run_id``
       (run_info/, intermediate/stage_1_partial/, results/, logs/) — atomic
       in the sense that all four mkdirs are issued before this function
       returns; idempotent (``exist_ok=True``).
    2. Probes git via subprocess for sha + dirty status (best-effort).

    Parameters
    ----------
    cfg
        Resolved :class:`RunConfig`. ``io.output_dir`` and
        ``io.run_id_strategy`` drive the run-dir layout; ``runtime.device``
        is the source of the resolve-device check; ``prompt.text`` drives
        ``class_id_map`` derivation.
    runtime
        Capability snapshot from :func:`tls2dseg.runtime.capability.probe_all`.
        Stored verbatim on ``RunContext.capability`` so consumers (engines,
        provenance writers) read a single snapshot.
    run_id
        Override for the per-run identifier (e.g. ``--run-id`` CLI flag per
        D-A3-02). When None, derive from ``cfg.io.run_id_strategy``.

    Returns
    -------
    RunContext
        Frozen — subsequent attribute assignment raises FrozenInstanceError.
    """
    resolved_run_id = run_id if run_id is not None else make_run_id(cfg.io.run_id_strategy, cfg.io.input_path)

    run_dir = cfg.io.output_dir / resolved_run_id
    run_info_dir = run_dir / "run_info"
    stage1_dir = run_dir / "intermediate" / "stage_1_partial"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"

    # Atomic-ish: all mkdirs happen before we return. mkdir is the trust
    # boundary for cfg.io.output_dir (T-03-path-validate); pydantic already
    # validated the Path shape.
    create_run_dirs(run_dir)

    device = resolve_device(cfg.runtime.device, runtime)

    class_id_map = _derive_class_id_map(cfg.prompt.text)

    git_sha, git_dirty = _probe_git()

    return RunContext(
        run_id=resolved_run_id,
        run_dir=run_dir,
        run_info_dir=run_info_dir,
        stage1_dir=stage1_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
        device=device,
        capability=runtime,
        n_workers=cfg.runtime.n_workers,
        dump_json_results=cfg.io.save_intermediate,
        class_id_map=class_id_map,
        started_at=datetime.now(UTC),
        ended_at=None,
        git_sha=git_sha,
        git_dirty=git_dirty,
        # per_class_metadata uses dataclasses.field(default_factory=dict)
        # default; Phase 6 MZ-05 populates after build_context returns.
    )

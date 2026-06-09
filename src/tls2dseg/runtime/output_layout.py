"""Per-run output directory layout + provenance dump.

Phase 3 plan 04 — implements the D-A2-05 per-run dir shape and the D-A2-07
provenance contract. Closes the user's Phase 2 carry-over feature request
("Per-run output directory with provenance") verbatim.

Layout per D-A2-05::

    {cfg.io.output_dir}/{run_id}/
        run_info/                 # config.yaml, context.yaml, git.txt, env.txt
        intermediate/
            stage_1_partial/      # stage-1 per-scan partials (resumability)
        results/                  # final segmented point clouds
        logs/                     # run.log (when cfg.logging.log_to_file)

Provenance under ``run_info/`` per D-A2-07:

* ``config.yaml`` — ALWAYS WRITTEN. ``RunConfig.model_dump(mode='json')`` →
  ``yaml.dump`` with a header comment naming Phase 3 CFG-04.
* ``context.yaml`` — ALWAYS WRITTEN. ``dataclasses.asdict(RunContext)`` with
  Path → str + datetime → ISO normalization (yaml.dump rejects rich types).
* ``git.txt`` — BEST-EFFORT. Skipped entirely if git is unavailable or the
  cwd is not a repo. Contents: ``rev-parse HEAD``, ``branch --show-current``,
  ``status --short``, ``log -1 --oneline``.
* ``env.txt`` — BEST-EFFORT. Python version + per-dep ``importlib.metadata``
  versions from the verified Phase 3 dep list. Never shells out.

The ``git.txt`` and ``env.txt`` files use broad ``except`` per
``diagnostics.py`` D-11 — provenance harvest must never block a pipeline run.

Closest analog: pipeline/run.py:244-248 (existing dir-creation pattern —
``mkdir(parents=True, exist_ok=True)``).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from tls2dseg.config.models import RunConfig
    from tls2dseg.runtime.context import RunContext

logger = logging.getLogger("tls2dseg.runtime.output_layout")


# Verified dep list per Phase 3 RESEARCH §"env.txt provenance format" — these
# are the runtime deps + workspace-internal deps + optional GPU deps whose
# versions matter for reproducibility. Ordered to match the rest of the
# project's stack discussion (ML/vision → image processing → scientific →
# workspace deps → optional GPU). Pure stdlib helpers (yaml, json) are out.
_PROVENANCE_DEPS: tuple[str, ...] = (
    "torch",
    "transformers",
    "supervision",
    "sam2",
    "opencv-python",
    "pyvips",
    "numpy",
    "scipy",
    "scikit-learn",
    "igraph",
    "leidenalg",
    "networkx",
    "pchandler",
    "pc2img",
    "cuml",
)


def make_run_id(strategy: str, input_path: Path) -> str:
    """Construct a per-run identifier per D-A2-06.

    Two strategies:

    * ``"timestamp"`` → ``YYYY-MM-DDTHHMMSS`` (UTC). Concise; race-free at
      whole-second resolution (we rely on shell users not spawning >1
      run/sec — D-A2-06 explicitly accepts this).
    * ``"timestamp_scanset"`` (default) → ``{ts}_{input_path.stem}``. Human-
      friendly: makes the run dir name self-describing in a glob listing.

    Notes
    -----
    * UTC is used unconditionally. Local time would re-introduce DST
      ambiguity (RESEARCH §Pattern 12).
    * ``input_path.stem`` handles trailing slashes gracefully (Path
      normalizes ``/data/wheat_heads/`` to ``wheat_heads``).
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    if strategy == "timestamp":
        return ts
    if strategy == "timestamp_scanset":
        return f"{ts}_{input_path.stem}"
    # Defensive: pydantic Literal in IOConfig.run_id_strategy eliminates this
    # branch at parse time, but guard for direct callers (tests, future
    # strategies before schema migration).
    raise ValueError(f"make_run_id: unknown strategy {strategy!r}; expected one of 'timestamp', 'timestamp_scanset'")


def create_run_dirs(run_dir: Path) -> None:
    """mkdir the four-subdir layout per D-A2-05. Idempotent.

    Issues four ``mkdir(parents=True, exist_ok=True)`` calls — one per
    subdir. Pattern matches pipeline/run.py:244-248 (existing stage-1
    partial dir creation), kept here for the full set so consumers don't
    each re-derive the layout.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info").mkdir(parents=True, exist_ok=True)
    (run_dir / "intermediate" / "stage_1_partial").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)


def _serialize_for_yaml(obj: Any) -> Any:
    """Recursively normalize Python objects to YAML-safe primitives.

    yaml.dump handles primitives, lists, dicts, and (with default_flow_style
    False) renders them readably. Rich types it rejects or mis-renders:

    * ``Path``     → ``str(obj)``
    * ``datetime`` → ``obj.isoformat()`` (preserves tzinfo when present)
    * ``dataclass`` → recurse over ``dataclasses.asdict`` (rare — RunContext
      is the only dataclass in the call path, and we call asdict() before
      this function)
    * ``dict``     → recurse on values
    * ``list``/``tuple`` → recurse on elements (tuples become lists for YAML)
    * else         → passthrough (bool/int/float/str/None — primitives)
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_yaml(x) for x in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _serialize_for_yaml(dataclasses.asdict(obj))
    return obj


def _write_yaml_provenance(path: Path, payload: dict[str, Any], header_comment: str) -> None:
    """Write a YAML provenance file with a leading ``# comment`` header line."""
    serialized = _serialize_for_yaml(payload)
    body = yaml.dump(serialized, default_flow_style=False, sort_keys=False)
    path.write_text(f"# {header_comment}\n{body}", encoding="utf-8")


def _try_write_git_txt(run_info_dir: Path) -> None:
    """Best-effort git.txt under run_info/. Skip the file if NOT in a git repo.

    Per D-A2-07: "skip block if not in a repo". We probe with ``git
    rev-parse HEAD`` first; if it fails (nonzero or OSError), we abort
    without writing anything. If it succeeds, we collect three more
    optional probes and write whatever we got.
    """
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:  # FileNotFoundError when git is missing; OSError; TimeoutExpired
        return

    if sha_result.returncode != 0:
        # Not in a repo → skip the file entirely.
        return

    lines: list[str] = [f"sha: {sha_result.stdout.strip()}"]

    for label, argv in (
        ("branch", ["git", "branch", "--show-current"]),
        ("status", ["git", "status", "--short"]),
        ("last_commit", ["git", "log", "-1", "--oneline"]),
    ):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=2.0, check=False)
            if r.returncode == 0:
                lines.append(f"{label}: {r.stdout.strip()}")
        except Exception:  # best-effort per D-A2-07
            pass

    (run_info_dir / "git.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_write_env_txt(run_info_dir: Path) -> None:
    """Best-effort env.txt under run_info/. Always attempted; never raises.

    Format per RESEARCH §"env.txt provenance format" — leading ``python:``
    line from ``sys.version_info``, then one line per dep in the verified
    list. Missing packages emit ``<pkg>: not installed`` (so the diff
    against a working env is informative).
    """
    try:
        vi = sys.version_info
        lines: list[str] = [f"python: {vi.major}.{vi.minor}.{vi.micro}"]
        for dist in _PROVENANCE_DEPS:
            try:
                lines.append(f"{dist}: {importlib.metadata.version(dist)}")
            except importlib.metadata.PackageNotFoundError:
                lines.append(f"{dist}: not installed")
        (run_info_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # best-effort per D-A2-07
        pass


def write_provenance(ctx: RunContext, cfg: RunConfig) -> None:
    """Dump per-run provenance to ``ctx.run_info_dir`` per D-A2-07.

    Always writes ``config.yaml`` and ``context.yaml``. Best-effort
    ``git.txt`` (skipped if not in a repo) and ``env.txt``.

    Security note (T-03-provenance-secrets): ``cfg.model_dump`` includes
    ``inference.sam2_checkpoint`` and other Path fields that may live under
    a user home dir or scratch path. Paths are written verbatim after env
    expansion (the loader already expanded ``${ENV_VAR}`` references). The
    ``env.txt`` file logs only ``importlib.metadata.version`` strings — no
    env var values, no paths, no keys. See plan threat model T-03-*.
    """
    # ALWAYS WRITE — these two are the provenance bedrock per D-A2-07.
    _write_yaml_provenance(
        ctx.run_info_dir / "config.yaml",
        cfg.model_dump(mode="json"),
        header_comment=(
            "Resolved RunConfig dumped at run start (Phase 3 CFG-04 provenance "
            "per D-A2-07). Paths are post env-var expansion."
        ),
    )
    _write_yaml_provenance(
        ctx.run_info_dir / "context.yaml",
        dataclasses.asdict(ctx),
        header_comment=("Resolved RunContext dumped at run start (Phase 3 CFG-04 provenance per D-A2-07)."),
    )

    # BEST-EFFORT — never raises; absence is informative.
    _try_write_git_txt(ctx.run_info_dir)
    _try_write_env_txt(ctx.run_info_dir)

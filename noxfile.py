"""nox session definitions for tls2dseg local automation.

Sessions:
  - tier_a:      lightweight, no-GPU, no-pchandler/pc2img. Mirrors ci.yml `tier_a` job EXACTLY.
  - tier_b_light: moderate, requires pchandler + pc2img via local editable installs.

Single-source-of-truth rule (Phase 3 D-A4-01):
  The pytest flags in each nox session below MUST match the corresponding job in
  ``tls2dseg/.github/workflows/ci.yml``. If the flags change in one place, update
  both atomically. The Phase 3 CICD-01 invariant is that
  ``pytest -m tier_a -v`` runs identically locally and in cloud CI.

Local usage:
  - ``nox -s tier_a``         # run the same suite as cloud CI
  - ``nox -s tier_b_light``    # run the broader local suite (requires PCHandler + pc2img siblings)

This file is also the binding point for the Claude Code PreToolUse hook
(``.claude/settings.json`` at the workspace root): the hook runs
``nox -s tier_b_light`` before any ``git push`` to gate the push on a green local
suite (D-A4-03 Layer 1b — best-effort soft gate; the binding floor is GitHub
branch protection on the cloud CI tier_a check, configured by the user as a
one-time manual step).

Reuse-venv: ``nox.options.reuse_venv = "yes"`` at module level is the verified
canonical form per Phase 3 RESEARCH §Pattern 10 (verified 2026-06-09 against
nox docs). Equivalent to ``--reuse-existing-venvs`` on the CLI. Critical for
pre-push hook latency — without it, every push rebuilds the venv from scratch.
"""

import os

import nox

nox.options.reuse_venv = "yes"


# D-A4-01: pytest flags MUST match the tier_a job in ci.yml exactly.
# Update both atomically if flags change.
@nox.session(python="3.11", name="tier_a")
def tier_a(session: nox.Session) -> None:
    """Run tier_a tests (no-GPU, no-pchandler/pc2img, < 5 min)."""
    session.install("-e", ".", "--no-deps")
    session.install(
        "pydantic-settings[yaml]>=2.14",
        "typer>=0.13",
        "pytest",
        "numpy",
    )
    session.run("pytest", "-m", "tier_a", "-v", *session.posargs)


# D-A4-01: pytest flags MUST match any future tier_b_light cloud-CI job in ci.yml
# exactly. Today tier_b_light is local-only (CICD-02 deferred to Phase 7 per
# D-A4-02), so this session's flags are the authoritative source.
@nox.session(python="3.11", name="tier_b_light")
def tier_b_light(session: nox.Session) -> None:
    """Run tier_b_light tests — pchandler+pc2img with deps; torch allowed; no real model runs.

    D-D-06 (Phase 4 plan 02): installs tls2dseg + PCHandler + pc2img WITH their
    transitive dependencies (no ``--no-deps``) so that imports of
    ``pc2img_utils``, ``pchandler``, etc. resolve correctly.  Previously
    ``--no-deps`` was used, but that caused ``imageio`` and other transitive
    deps to be missing, breaking tier_b_light test collection.
    """
    session.install("-e", ".")  # installs tls2dseg WITH all its deps
    session.install("-e", "../PCHandler")  # installs PCHandler WITH deps
    session.install("-e", "../pc2img")  # installs pc2img WITH deps
    session.install("pytest")  # explicit in case not in transitive deps
    # manifold3d: CPU-only trimesh boolean backend so the OBB-boolean IoU
    # tests (TEST-06, test_bboxes_iou.py) execute instead of skip-guarding on
    # an absent mesh-boolean engine. Pure pip dep, no GPU.
    session.install("manifold3d")
    session.run("pytest", "-m", "tier_b_light", "-v", *session.posargs)


# tier_b_heavy: real models / GPU / cudf / large fixtures. Unlike tier_a and
# tier_b_light (isolated CPU venvs), this session uses NO isolated venv and
# delegates to a project conda env that already has torch+CUDA+cudf+sam2 plus the
# editable pchandler/pc2img installs. `nox` need NOT be installed in that conda env
# — nox stays the orchestrator in the base env and shells into the conda env via
# `conda run`. Override the env name with TLS2DSEG_HEAVY_CONDA_ENV.
@nox.session(venv_backend="none", name="tier_b_heavy")
def tier_b_heavy(session: nox.Session) -> None:
    """Run tier_b_heavy tests inside the project conda env (default tls2dseg_2025)."""
    env_name = os.environ.get("TLS2DSEG_HEAVY_CONDA_ENV", "tls2dseg_2025")
    session.run(
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        "-m",
        "pytest",
        "-m",
        "tier_b_heavy",
        "-v",
        *session.posargs,
        external=True,
    )

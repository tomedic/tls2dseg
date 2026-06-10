"""Typer-based CLI for tls2dseg (Phase 3 CFG-05 + CFG-06).

Replaces the Phase-1 argparse stub. Surface (4 subcommands per D-A3-01):

* ``tls2dseg --version`` — prints semver and exits 0.
* ``tls2dseg run --config <yaml>`` — load config, build runtime context, dump
  provenance, dispatch the pipeline. 8 flags per D-A3-02.
* ``tls2dseg validate-config <file>`` — load + validate without running.
  Exit 0 if valid, exit 1 with a field-level error to stderr if not (CFG-06).
* ``tls2dseg doctor`` — Phase-1 diagnostics report (preserved verbatim).

Two-phase logging per RESEARCH §Pattern 8 + WARNING-5 fix:

1. :func:`bootstrap_logger(log_level)` BEFORE :func:`load_config` so
   config-loader DEBUG/INFO is captured.
2. :func:`configure_logging(...)` AFTER load_config with the resolved
   ``cfg.logging`` values — dictConfig cleanly replaces the basicConfig
   handlers.

Entry point: ``[project.scripts] tls2dseg = "tls2dseg.cli:app"`` (HYGN-09
atomic swap from ``:main``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="tls2dseg — 3D point cloud segmentation via 2D foundation models.",
)

logger = logging.getLogger("tls2dseg.cli")


def _version_callback(value: bool) -> None:
    """Print version and exit. Used as ``--version`` eager callback."""
    if value:
        from tls2dseg._version import __version__

        typer.echo(f"tls2dseg {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Applies to all subcommands.",
    ),
) -> None:
    """tls2dseg — point cloud segmentation pipeline."""
    # Phase-1 logging fires at the ROOT callback so DEBUG/INFO from config loaders,
    # capability probes etc. is captured regardless of which subcommand follows (LOG-04).
    # `run` may re-bootstrap with its own --log-level to scope the level differently.
    from tls2dseg.runtime.logging_setup import bootstrap, bootstrap_logger

    bootstrap()  # TOKENIZERS_PARALLELISM env-var
    bootstrap_logger(level=log_level)


@app.command(name="run")
def run_cmd(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=False,  # Lazy check inside load_config (clearer error).
        help="Path to YAML config file.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    ),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="Resume from a specific run directory (overrides cfg.io.resume_from_checkpoint).",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Override the auto-generated run id (per D-A2-06).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override cfg.io.output_dir.",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Override cfg.mode (single-view | multi-view).",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override cfg.runtime.device (auto | cpu | cuda).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve config + context, dump summary, exit without running pipeline.",
    ),
) -> None:
    """Run the full segmentation pipeline (Phase 3 CFG-05)."""
    # ── PHASE 1 logging already fired at the root callback. Re-bootstrap here if
    # the run subcommand's --log-level differs from the global default (allows
    # `tls2dseg run --log-level DEBUG` to scope DEBUG to the run path).
    from tls2dseg.runtime.logging_setup import bootstrap_logger, configure_logging

    if log_level != "INFO":
        bootstrap_logger(level=log_level)

    # Build CLI overrides dict from non-None flags (D-A3-04 precedence: CLI > env > YAML > defaults).
    cli_overrides: dict[str, object] = {}
    if mode is not None:
        cli_overrides["mode"] = mode
    io_overrides: dict[str, object] = {}
    if output_dir is not None:
        io_overrides["output_dir"] = output_dir
    if resume_from is not None:
        io_overrides["resume_from_checkpoint"] = True
    if io_overrides:
        cli_overrides["io"] = io_overrides
    if device is not None:
        cli_overrides["runtime"] = {"device": device}

    # Load config — safe to import now; loader's DEBUG output reaches the
    # bootstrap_logger basicConfig handler.
    from tls2dseg.config.loader import load_config

    try:
        cfg = load_config(config, cli_overrides=cli_overrides)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Config load error: {e}", err=True)
        raise typer.Exit(code=1) from e

    # ── PHASE 2 logging: dictConfig with per_package overrides + optional file
    # handler. dictConfig replaces basicConfig handlers cleanly (RESEARCH §Pattern 8).
    # log_file is finalized AFTER build_context once ctx.logs_dir exists.
    configure_logging(level=log_level, per_package=cfg.logging.per_package)

    # Build runtime + context.
    from tls2dseg.runtime import build_context, probe_all
    from tls2dseg.runtime.output_layout import write_provenance

    runtime = probe_all()
    ctx = build_context(cfg, runtime, run_id=run_id)

    # Re-configure logging once we know ctx.logs_dir — wires the file handler.
    if cfg.logging.log_to_file:
        log_file = str(ctx.logs_dir / "run.log")
        configure_logging(level=log_level, per_package=cfg.logging.per_package, log_file=log_file)

    write_provenance(ctx, cfg)

    if dry_run:
        typer.echo(f"[dry-run] run_id={ctx.run_id}")
        typer.echo(f"[dry-run] mode={cfg.mode}")
        typer.echo(f"[dry-run] device={ctx.device}")
        typer.echo(f"[dry-run] run_dir={ctx.run_dir}")
        typer.echo(f"[dry-run] n_classes={len(ctx.class_id_map)}")
        raise typer.Exit(code=0)

    # Dispatch the pipeline.
    from tls2dseg.pipeline.run import main as pipeline_main

    pipeline_main(cfg, ctx)


@app.command(name="validate-config")
def validate_config_cmd(
    file: Path = typer.Argument(
        ...,
        help="Path to YAML config file.",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Treat unknown keys as errors (extra='forbid'). Currently always strict in v1.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="On success, print the resolved RunConfig as YAML.",
    ),
) -> None:
    """Validate a YAML config file without running the pipeline (CFG-06).

    Exit 0 if valid; exit 1 with a field-level error to stderr if not.
    """
    from pydantic import ValidationError

    from tls2dseg.config.loader import load_config

    try:
        cfg = load_config(file)
    except (FileNotFoundError, ValueError, ValidationError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from e

    if verbose:
        import yaml

        typer.echo(yaml.dump(cfg.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
    else:
        typer.echo("OK")
    raise typer.Exit(code=0)


@app.command(name="doctor")
def doctor_cmd() -> None:
    """Report Python/GPU/RAPIDS/libvips/SAM2 status (Phase 1 stub preserved per ROADMAP §Phase 3)."""
    from tls2dseg.diagnostics import doctor, format_report

    report = doctor()
    typer.echo(format_report(report))


# NOTE: ``main()`` is preserved as a thin shim for backward-compat with any
# importers using ``from tls2dseg.cli import main`` (e.g. external scripts).
# The Phase 3 entry point swap goes to ``app`` (HYGN-09).
def main(argv: list[str] | None = None) -> int:
    """Backward-compat shim — delegates to :data:`app`."""
    try:
        if argv is not None:
            app(argv)
        else:
            app()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

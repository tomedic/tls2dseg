"""Phase 1 CLI stub. Replaced by Typer in Phase 3 (CFG-05)."""

from __future__ import annotations

import argparse
import logging
import sys

from tls2dseg.diagnostics import doctor, format_report

logger = logging.getLogger("tls2dseg.cli")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (always 0 — Phase 1 is best-effort)."""
    parser = argparse.ArgumentParser(prog="tls2dseg")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("doctor", help="Report Python/GPU/RAPIDS/libvips/SAM2 status.")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        report = doctor()
        print(format_report(report))
        return 0

    # No subcommand -> help text + return 0 (D-11 best-effort)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

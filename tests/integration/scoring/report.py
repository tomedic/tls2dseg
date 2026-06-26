"""JSON + markdown report writer for tier_b_heavy fixture results.

Writes to a fixed path, merging each dataset into any existing report so the
two heavy tests produce one combined report — not a committed golden snapshot
(D-08). The report is informational; threshold gating lives in the heavy test.

Exports:
    write_report(metrics, out_dir) -> (json_path, md_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("tls2dseg.tests.integration.scoring.report")

_JSON_NAME = "tier_b_heavy_report.json"
_MD_NAME = "tier_b_heavy_report.md"


def write_report(metrics: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write a JSON and a markdown summary of heavy-test metrics.

    Parameters
    ----------
    metrics:
        Dict with at minimum::

            {
              "iou_thr": float,          # per-pair match threshold (tuning knob)
              "datasets": {
                "<dataset_name>": {
                  "recall": float,
                  "correspondence_rate": float,
                  "fp_rate": float,
                  "consistency_only": {   # D-07 classes with no reference_data
                    "<class>": {"detected": int, "consistent": bool}
                  }
                }
              }
            }

    out_dir:
        Directory in which to write the two files (created if absent).

    Returns
    -------
    json_path, md_path : tuple[Path, Path]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / _JSON_NAME
    md_path = out_dir / _MD_NAME

    # Merge into any existing report so each heavy test contributes its own
    # dataset instead of clobbering the others (D-08 combined report).
    merged = metrics
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if isinstance(existing.get("datasets"), dict):
            existing_datasets = dict(existing["datasets"])
            existing_datasets.update(metrics.get("datasets", {}))
            merged = {**metrics, "datasets": existing_datasets}

    json_path.write_text(json.dumps(merged, indent=2))
    logger.info("Wrote heavy-test report: %s", json_path)

    md_path.write_text(_render_markdown(merged))
    logger.info("Wrote heavy-test report: %s", md_path)

    return json_path, md_path


def _render_markdown(metrics: dict) -> str:
    iou_thr = metrics.get("iou_thr", "n/a")
    lines = [
        "# tier_b_heavy fixture report",
        "",
        f"Per-pair IoU match threshold (`iou_thr`): **{iou_thr}**  ",
        "_Threshold gates (recall ≥0.70, correspondence ≥0.70, FP ≤0.30) are asserted "
        "in the test; this report is informational only._",
        "",
    ]

    datasets: dict = metrics.get("datasets", {})
    for ds_name, ds in datasets.items():
        recall = ds.get("recall", "n/a")
        corr = ds.get("correspondence_rate", "n/a")
        fp = ds.get("fp_rate", "n/a")
        recall_str = f"{recall:.3f}" if isinstance(recall, float) else str(recall)
        corr_str = f"{corr:.3f}" if isinstance(corr, float) else str(corr)
        fp_str = f"{fp:.3f}" if isinstance(fp, float) else str(fp)
        lines += [
            f"## {ds_name}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Reference recall | {recall_str} |",
            f"| Cross-scan correspondence rate | {corr_str} |",
            f"| False-positive rate | {fp_str} |",
            "",
        ]

        consistency = ds.get("consistency_only", {})
        if consistency:
            lines += ["### Consistency-only classes (D-07)", ""]
            lines += ["| Class | Detected | Cross-scan consistent |", "|-------|----------|----------------------|"]
            for cls_name, cls_info in consistency.items():
                detected = cls_info.get("detected", "n/a")
                consistent = cls_info.get("consistent", "n/a")
                lines.append(f"| {cls_name} | {detected} | {consistent} |")
            lines.append("")

    return "\n".join(lines)

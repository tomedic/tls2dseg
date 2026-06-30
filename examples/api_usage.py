"""Runnable library-use example — runs the segmentation pipeline on the office_small fixture.

Usage::

    python examples/api_usage.py

The script reads ``examples/configs/office_small.yaml`` and writes outputs
under the directory configured in ``io.output_dir``.
"""

from __future__ import annotations

from pathlib import Path

from tls2dseg.pipeline.pipeline import Pipeline

CONFIG = Path(__file__).parent / "configs" / "office_small.yaml"


def main() -> None:
    pipeline = Pipeline.from_yaml(CONFIG)
    pipeline.run()  # writes results/ + run_info/ under io.output_dir

    # Optional: inspect stage-1 detections without re-running stage 2
    # stage1_result = pipeline.stage1()
    # for i, (pcd, d3d) in enumerate(
    #     zip(stage1_result.pcd_collection, stage1_result.d3d_collection)
    # ):
    #     print(i, d3d)


if __name__ == "__main__":
    main()

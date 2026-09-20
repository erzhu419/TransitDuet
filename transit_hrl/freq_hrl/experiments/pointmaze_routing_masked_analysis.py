"""Root-paired analysis for equal-shape PointMaze routing attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pointmaze_routing_analysis import (
    analyze_pointmaze_routing_cells,
    load_pointmaze_routing_cells,
    render_report,
)
from .pointmaze_routing_masked_attribution import (
    POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
    POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analysis = analyze_pointmaze_routing_cells(
        load_pointmaze_routing_cells(args.inputs),
        confidence=args.confidence,
        protocol_version=POINTMAZE_MASKED_ROUTING_PROTOCOL_VERSION,
        algorithm_path=POINTMAZE_MASKED_ROUTING_ALGORITHM_PATH,
        analysis_version=(
            "pointmaze_frequency_routing_stage4_masked_analysis_v2"
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

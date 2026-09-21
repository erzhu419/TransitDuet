#!/usr/bin/env python3
"""CLI entry point for external-stream PointMaze routing attribution."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())


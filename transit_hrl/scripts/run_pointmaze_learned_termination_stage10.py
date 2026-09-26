#!/usr/bin/env python3
"""CLI entry point for the Stage-10 learned-termination development baseline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.pointmaze_learned_termination import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dispatch the fixed Stage90 lower-budget diagnosis dynamically on node001-006."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import main
from scripts import pointmaze_lower_budget_stage90_spec as spec


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

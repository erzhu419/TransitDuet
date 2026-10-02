#!/usr/bin/env python3
"""Evaluate fixed Stage88 call-weighted actor compositions."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_pointmaze_actor_swap_stage84 import main
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as spec


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

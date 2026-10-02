#!/usr/bin/env python3
"""Dispatch Stage89 evaluation dynamically across node001-006."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submit_pointmaze_actor_swap_stage84_scheduleurm import main
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as spec


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

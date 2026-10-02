#!/usr/bin/env python3
"""Run Stage88 with the unchanged Stage87 native training core."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_pointmaze_call_weighted_stage87 import main
from scripts import pointmaze_call_weighted_replication_stage88_spec as spec


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

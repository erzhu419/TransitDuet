#!/usr/bin/env python3
"""Dispatch frozen Stage88 replication dynamically across node001-006."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import main
from scripts import pointmaze_call_weighted_replication_stage88_spec as spec


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

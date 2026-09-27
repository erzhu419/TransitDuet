#!/usr/bin/env python3
"""Submit the 50-step same-budget timing-pair development screen."""

from scripts import pointmaze_timing_pair_stage12_spec as spec
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import main


if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

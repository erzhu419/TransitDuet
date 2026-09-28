#!/usr/bin/env python3
"""Submit path-cross-fitted plan response through scheduler."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_crossfit_response_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import main

if __name__ == "__main__":
    raise SystemExit(main(protocol_spec=spec))

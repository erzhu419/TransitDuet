#!/usr/bin/env python3
"""Submit the fresh-seed Stage-5 V2 confirmation to the Linux CPU pool."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_exogenous_stage5_v2_spec as spec  # noqa: E402
from scripts.submit_pointmaze_exogenous_stage5_scheduleurm import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main(spec))


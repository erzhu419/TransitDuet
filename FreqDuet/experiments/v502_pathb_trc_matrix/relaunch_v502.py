#!/usr/bin/env python3
"""Cancel broken V502 R1 tasks (t223597-t223669) and immediately stage & run V502 R2 with checkpoint resume."""

from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCHED = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"

cproc = subprocess.run(
    [sys.executable, SCHED, "cancel", "t223597-t223669", "--force", "--lock-timeout", "600", "--force-lock-wait"],
    text=True,
    capture_output=True,
)
print("CANCEL_RC", cproc.returncode, cproc.stdout[-1500:], cproc.stderr[-1000:], flush=True)
if cproc.returncode != 0:
    raise RuntimeError(f"Cancel failed: {cproc.stderr}")

subprocess.run([sys.executable, str(HERE / "launch.py"), "run_all"], check=True)

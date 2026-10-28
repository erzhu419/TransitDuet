"""Read only compact native development reports and publish a phase gate."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import (
    PHASES, attempt_directory, development_cells, load_plan, summarize_phase,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=PHASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-bundle", type=Path)
    args = parser.parse_args(argv)
    plan = load_plan(args.preregistration)
    recovery = json.loads(args.recovery_bundle.read_text()) if args.recovery_bundle else None
    if recovery and (recovery["run_name"] != args.run_name or recovery["stage"] != args.stage):
        raise ValueError("recovery bundle belongs to another campaign/phase")
    recovered = {(row["task"], row["method"], row["optimizer_seed"]) for row in recovery["cells"]} if recovery else set()
    paths = [ROOT / attempt_directory(args.run_name, cell,
        recovery["attempt"] if (cell["task"], cell["method"], cell["optimizer_seed"]) in recovered else "")
        / "public/result.json" for cell in development_cells(plan, args.stage)]
    report = summarize_phase(plan, args.run_name, args.stage, [json.loads(path.read_text()) for path in paths if path.exists()])
    report["source_files"] = [str(path) for path in paths if path.exists()]
    if recovery:
        report["recovery_bundle"] = str(args.recovery_bundle)
        report["interrupted_measurements"] = recovery["interrupted_measurements"]
        report["lifecycle_costs_complete"] = all(all(value is not None for value in cost.values())
            for cost in recovery["interrupted_measurements"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())

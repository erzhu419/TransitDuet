"""Select budget-feasible frozen configurations from compact development reports."""

import argparse
import json
from pathlib import Path

from freq_hrl.experiments.compute_budget import select_with_budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--plans", type=Path, nargs="+", required=True)
    parser.add_argument("--cpu-cap-seconds", type=float, required=True)
    parser.add_argument("--environment-step-cap", type=int, required=True)
    parser.add_argument("--deployment-episodes", type=int, required=True)
    parser.add_argument("--latency-p99-cap-seconds", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [record for path in args.inputs for record in json.loads(path.read_text())["records"]]
    report = select_with_budget(records, cpu_cap_seconds=args.cpu_cap_seconds,
        environment_step_cap=args.environment_step_cap, deployment_episodes=args.deployment_episodes,
        latency_p99_cap_seconds=args.latency_p99_cap_seconds,
        expected_runs=[run for path in args.plans for run in json.loads(path.read_text())["protocol"]["expected_runs"]])
    report["measurement_files"] = [str(path) for path in args.inputs]
    report["plan_files"] = [str(path) for path in args.plans]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

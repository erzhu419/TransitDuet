"""Aggregate compact frozen episode-reliability reports, without replay or fitting."""

import argparse
import json
from pathlib import Path

from freq_hrl.experiments.plan_validity_reliability import aggregate_reliability
from freq_hrl.experiments.plan_validity_transfer import TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-roots", nargs="+", type=int, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--comparator-method", required=True)
    parser.add_argument("--risk-rate-margin", type=float, required=True)
    parser.add_argument("--tail-ise-margin", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = [json.loads(path.read_text()) for path in args.inputs]
    if any(row["protocol"]["registered_comparator_method"] != args.comparator_method for row in sources):
        raise ValueError("input comparator differs from the registered analysis")
    report = aggregate_reliability(sources,
        expected_roots=args.expected_roots, tasks=(args.task,),
        risk_rate_margin=args.risk_rate_margin, tail_ise_margin=args.tail_ise_margin)
    report["source_files"] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

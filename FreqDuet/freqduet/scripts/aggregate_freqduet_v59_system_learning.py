#!/usr/bin/env python3
"""Aggregate the four development learning smokes, not a policy-comparison test."""

import argparse
import json
from pathlib import Path

from run_freqduet_v58_system_credit import CASES, case_name


def aggregate(root):
    cases = [json.loads((root / "exports" / case_name(*case) / "done.json").read_text()) for case in CASES]
    if any(c["status"] != "learning_smoke_pass" or not c["exact_warm_start_day"]
           or c["main_policy_changed"] for c in cases):
        raise ValueError("one or more development cases did not pass the learning smoke")
    days = [day for case in cases for day in case["episodes"]]
    result = dict(status="four_case_learning_smoke_pass", cases=len(cases), days=len(days),
                  updates=sum(c["updates"] for c in cases),
                  batch_sizes=sorted({c["batch_size"] for c in cases}),
                  state_dimensions=sorted({c["state_dim"] for c in cases}),
                  decisions=sum(d["replay_added"] for d in days),
                  day_terminals=sum(d["day_terminals"] for d in days),
                  cross_trip_bootstraps=sum(d["stream_audit"]["cross_trip_bootstraps"] for d in days),
                  same_clock_bootstraps=sum(d["stream_audit"]["same_clock_bootstraps"] for d in days),
                  conservation_max_error=max(d["stream_audit"]["conservation_max_error"] for d in days),
                  exact_reference_days=len(cases), main_policy_changed=False,
                  policy_effectiveness_established=False,
                  diagnostic_commits=sorted({c["diagnostic_commit"] for c in cases}))
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    (summary / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V59_AGGREGATE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    aggregate(parser.parse_args().out_root)

#!/usr/bin/env python3
"""Posthoc concentration and secondary journey audit of fixed V70 choices."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v70_independent_labels as audit


JOURNEY_METRICS = ("journey_min", "waiting_min", "onboard_min")


def secondary_effects(groups, choices, native_action):
    native = audit.joint.ACTIONS.index(native_action)
    arrays = {m: np.asarray([[r[m] for r in g["rows"]] for g in groups]) for m in JOURNEY_METRICS}
    values = []
    for choice, future in zip(choices, (np.arange(16, 32), np.arange(16))):
        selected = audit.joint.ACTIONS.index(choice["action_s"])
        values.append({m: float((a[future, selected] - a[future, native]).mean()) for m, a in arrays.items()})
    return {m: float(np.mean([v[m] for v in values])) for m in JOURNEY_METRICS}


def interval(contexts, getter):
    days = sorted({c["scenario_seed"] for c in contexts})
    values = [float(np.mean([getter(c) for c in contexts if c["scenario_seed"] == day])) for day in days]
    result = audit.labels.expanded.paired_interval(values, 70073)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def concentration(effects):
    effects = np.asarray(effects, dtype=float)
    improvement = np.maximum(-effects, 0.)
    strongest = int(np.argmin(effects))
    return dict(strongest_day_index=strongest, improved_days=int((effects < 0).sum()),
                gross_improvement_share=float(improvement[strongest] / improvement.sum()),
                net_improvement_share=float(effects[strongest] / effects.sum()),
                mean_without_strongest=float(np.delete(effects, strongest).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"),
                    "-p", "test_v70_concentration.py", "-v"], check=True)
    contexts = json.loads((args.out_root / "groups/context_audit.json").read_text())
    for c in contexts:
        groups = [json.loads(audit.group_path(args.out_root, c["context_index"] * len(audit.FUTURES) + f).read_text())
                  for f in audit.FUTURES]
        c["secondary"] = secondary_effects(groups, c["halves"], c["native_action_s"])
    daily = [float(np.mean([c["cross_future"]["passenger_person_s"] for c in contexts if c["scenario_seed"] == day]))
             for day in audit.DEMAND_SEEDS]
    concentrated = concentration(daily)
    strongest = audit.DEMAND_SEEDS[concentrated.pop("strongest_day_index")]
    remaining = [c for c in contexts if c["scenario_seed"] != strongest]
    changed = [(c, h) for c in contexts for h in c["halves"] if h["changed"]]
    result = dict(stage="posthoc_fixed_choice_journey_and_concentration", strongest_demand_day=strongest,
                  concentration=concentrated,
                  secondary_journey={m: interval(contexts, lambda c, m=m: c["secondary"][m]) for m in JOURNEY_METRICS},
                  without_strongest_day={m: interval(remaining, lambda c, m=m: c["cross_future"][m])
                                         for m in audit.repeated.RESPONSE_METRICS},
                  signal_without_strongest={m: interval(remaining, lambda c, m=m:
                                                       c["signal_noise"][m]["debiased_signal_squared"])
                                            for m in ("passenger_person_s", "cv_squared")},
                  direction_counts={direction: dict(changed=sum(test(h["action_s"], c["native_action_s"]) for c, h in changed),
                                                     passenger_improved=sum(test(h["action_s"], c["native_action_s"])
                                                                            and h["passenger_person_s"] < 0 for c, h in changed))
                                    for direction, test in (("increase_holding", lambda a, b: a > b),
                                                            ("decrease_holding", lambda a, b: a < b))},
                  leading_contexts=[{k: c[k] for k in ("context_index", "scenario_seed", "train_seed", "actor_scenario",
                                                      "target_time_s", "native_action_s", "halves", "cross_future", "secondary")}
                                    for c in sorted(contexts, key=lambda c: c["cross_future"]["passenger_person_s"])[:6]],
                  rollouts_added=0, model_updates=0, main_policy_changed=False, gate_changed=False)
    output = args.out_root / "focused_summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V70_CONCENTRATION_AUDIT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

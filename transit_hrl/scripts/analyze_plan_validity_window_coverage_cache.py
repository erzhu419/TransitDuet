"""Post-hoc frozen-action contrasts on existing held states; no fit or simulation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_target_credit import predict
from freq_hrl.experiments.plan_validity_window_coverage import values
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.analyze_plan_validity_target_credit_cache import action_readout


CONTRASTS = {"coverage_df2": ("wide_df2", "dense_df2"), "coverage_df8": ("wide_df8", "dense_df8"),
             "complexity_dense": ("dense_df2", "dense_df8"), "complexity_wide": ("wide_df2", "wide_df8")}


def panel(rows, rules):
    scores = {k: predict(rule, rows) for k, rule in rules.items()}
    actions = {k: (scores[k] >= rule.threshold).astype(float) for k, rule in rules.items()}
    labels = {block: {target: values(rows, target, block).mean(axis=1) for target in ("W", "G")}
              for block in ("A", "B")}
    contrasts = {}
    for name, (candidate, comparator) in CONTRASTS.items():
        delta = actions[candidate] - actions[comparator]
        changed = delta != 0
        contrasts[name] = {"changed_states": int(changed.sum()),
            "changed_paths": len(set(row["seed"] for row, keep in zip(rows, changed) if keep)),
            "stop_fraction_change": float(delta.mean()), "blocks": {block: {
                "window_gain": float(np.mean(delta * y["W"])),
                "full_gain": float(np.mean(delta * y["G"])),
                "changed_state_window_full_sign_disagreement": float(np.mean(y["W"][changed] * y["G"][changed] < 0))
                    if changed.any() else None} for block, y in labels.items()}}
    paired_rows = [{**row, "conditional_replicas": row["A_replicas"] + row["B_replicas"]} for row in rows]
    full = np.column_stack((values(rows, "G", "A"), values(rows, "G", "B")))
    squares = np.square(full).sum(axis=1)
    return {"paths": len(set(row["seed"] for row in rows)), "states": len(rows),
        "G_future_range": [float(full.min()), float(full.max())],
        "largest_state_fraction_of_G_future_squares": float(squares.max() / squares.sum()) if squares.sum() else 0,
        "factorial_action_contrasts": contrasts,
        "head_attribution": {name: action_readout(paired_rows, scores[name], target="W",
            threshold=rule.threshold, training=False) for name, rule in rules.items()}}


def read_cache(manifest, *, root=ROOT):
    cells = []
    for cell in manifest["cells"]:
        directory = root / cell["output_directory"] / "run/private"
        record = json.loads((directory / "result.json").read_text())
        if record["status"] != "complete":
            raise ValueError("action attribution needs a completed frozen cell")
        rules = {k: TimingRule(**v) for k, v in json.loads((directory / "frozen_rules.json").read_text()).items()}
        cells.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
            **panel(record["raw_banks"]["held_bank"], rules)})
    return {"run_name": manifest["run_name"], "analysis_role": "posthoc_descriptive_frozen_held_cache",
        "interpretation": "one_intervention_on_original_history_states_not_repeated_policy_episode_gains",
        "fit_calls": 0, "new_simulator_steps": 0, "confirmation_released": False, "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(json.dumps(read_cache(json.loads(args.manifest.read_text())), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

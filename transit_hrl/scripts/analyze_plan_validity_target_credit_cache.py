"""Post-hoc frozen-action readout on server-resident banks; no fit or simulation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_target_credit import labels, predict
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule


def action_readout(rows, scores, *, target, threshold, training):
    local = np.column_stack([labels(rows, target, b).mean(axis=1) for b in ("A", "B")])
    full = np.column_stack([labels(rows, "G", b).mean(axis=1) for b in ("A", "B")])
    frozen = np.asarray(scores) >= threshold
    factual = np.asarray([r["factual_stop"] for r in rows])
    output = {}
    # Fitted heads use A: only B is independent on training prefixes.
    directions = [(0, 1, "A_to_B")] if training else [(0, 1, "A_to_B"), (1, 0, "B_to_A")]
    for choose, evaluate, name in directions:
        truth = full[:, evaluate]
        oracle = full[:, choose] >= 0
        constant = float(full[:, choose].mean() >= 0)
        same_cut = local[:, choose] >= threshold
        target_zero = local[:, choose] >= 0
        output[name] = {
            "frozen_stop_fraction": float(frozen.mean()),
            "same_cut_information_stop_fraction": float(same_cut.mean()),
            "frozen_gain_over_factual": float(np.mean((frozen.astype(float) - factual) * truth)),
            "same_cut_information_gain_over_frozen": float(np.mean((same_cut.astype(float) - frozen) * truth)),
            "target_zero_information_gain_over_frozen": float(np.mean((target_zero.astype(float) - frozen) * truth)),
            "full_information_gain_over_frozen": float(np.mean((oracle.astype(float) - frozen) * truth)),
            "full_information_gain_over_selected_constant": float(np.mean((oracle.astype(float) - constant) * truth)),
            "score_zero_gain_over_frozen": float(np.mean(((np.asarray(scores) >= 0).astype(float) - frozen) * truth))}
    return output


def read_cache(manifest, *, root=ROOT):
    cells = []
    for cell in manifest["cells"]:
        directory = root / cell["output_directory"] / "run/private"
        record = json.loads((directory / "result.json").read_text())
        if record["status"] != "complete":
            raise ValueError("cache attribution requires the completed frozen cell")
        rules = {k: TimingRule(**v) for k, v in json.loads((directory / "frozen_rules.json").read_text()).items()}
        panels = {}
        for panel, key in (("training", "training_bank"), ("held", "held_bank")):
            rows = record["raw_banks"][key]
            full = labels(rows, "G", "all")
            squared = np.square(full).sum(axis=1)
            panels[panel] = {"paths": len(set(r["seed"] for r in rows)), "states": len(rows),
                "G_minimum_future": float(full.min()), "G_maximum_future": float(full.max()),
                "G_largest_state_fraction_of_future_squares": float(squared.max() / squared.sum()) if squared.sum() else 0,
                "actions": {target: action_readout(rows, predict(rule, rows), target=target,
                    threshold=rule.threshold, training=panel == "training") for target, rule in rules.items()}}
        cells.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "panels": panels})
    return {"run_name": manifest["run_name"], "analysis_role": "posthoc_descriptive_frozen_cache_only",
        "interpretation": "one_intervention_on_original_policy_states_not_repeated_policy_episode_effects",
        "information_benchmarks": "finite_future_information_not_deployable_rules_or_upper_bounds",
        "fit_calls": 0, "new_simulator_steps": 0, "confirmation_released": False, "cells": cells}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = read_cache(json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

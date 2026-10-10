#!/usr/bin/env python3
"""Compare reference-credit learning with its own controls and Stage157 raw SAC."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_reference_credit_stage159 as spec
from scripts.analyze_native_transit_residual_plan_stage157 import validate_credit
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_training(run, root, *, mode, short):
    count, decisions, clock, updates = (2, 4, 5400, 2) if short else (120, 44, 61380, 5500)
    seeds = [(600000000 if short else 700000000) + root * 1000 + ep for ep in range(count)]
    if not (run["credit_mode"] == mode and run["reference_episodes"] == count
            and run["updates"] == updates and run["transitions"] == count * decisions
            and run["actor_change_max_abs"] > 0 and run["training_scene_seeds"] == seeds
            and len(run["credit_pairs"]) == count and all(np.isfinite(v) for v in run["learning_mean"].values())):
        raise ValueError("Changed reference-credit training or reference-rollout budget")
    for ep, pair in enumerate(run["credit_pairs"]):
        expected = 100 * (pair["reference_final_cost"] - pair["controlled_final_cost"])
        if not (pair["episode"] == ep and pair["scene_seed"] == seeds[ep]
                and pair["scenario"] == spec.source.contract()["scenarios"][ep % 5]
                and pair["decisions"] == decisions and pair["duration_s"] == clock and pair["terminal_transitions"] == 1
                and np.isclose(pair["terminal_advantage"], expected, rtol=1e-10, atol=1e-8)
                and np.isclose(pair["paired_reward_sum"], expected, rtol=1e-10, atol=1e-8)
                and np.isclose(pair["replay_reward_sum"], expected, rtol=1e-6, atol=1e-4)):
            raise ValueError("Reference replay changed the terminal physical objective or scene pairing")
    for row in run["training_curve"]:
        validate_credit(row["credit"], row, clock)
        if not np.isclose(row["credit"]["final_cost"], run["credit_pairs"][row["episode"]]["controlled_final_cost"], rtol=0, atol=1e-8):
            raise ValueError("Paired credit refers to a different factual training episode")


def summarize(cells, raw_cells):
    if set(cells) != set(spec.ROOTS) or set(raw_cells) != set(spec.ROOTS):
        raise ValueError("Incomplete paired/raw roots")
    metrics = spec.source.source.source_spec.authority.routing.METRICS
    means, panels, diagnostics, raw_deltas = {}, [], [], []
    for root, cell in cells.items():
        raw = raw_cells[root]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["software_qualified"] and cell["native_training_updates"] == 0
                and cell["native_steps"] == 320 * 61380
                and cell["worker_preflight_native_steps"] == 61380 + 8 * 5400
                and cell["forecast_source_reproduced"] and cell["raw_short_reproduced"]):
            raise ValueError("Unqualified paired-reference native training")
        for key, mode, short in (("training", "paired", False), ("raw_short", "raw", True), ("paired_short", "paired", True)):
            validate_training(cell[key], root, mode=mode, short=short)
        spec.check_raw_reproduction(cell["raw_short"], raw["preflight_learning"])
        required = {(c, s, scene) for c in spec.CONDITIONS for s in spec.source.contract()["scenarios"]
            for scene in spec.source.source.source_spec.scene_seeds(spec.source.LOWER_ROOT[root], s, preflight=False)}
        rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
        if set(rows) != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Incomplete or duplicate reference-credit evaluation")
        constant = np.asarray(cell["training"]["constant_action"])
        if constant.shape != (2,) or not np.all(np.isfinite(constant)) or np.any(np.abs(constant) > 1):
            raise ValueError("Invalid training-state constant control")
        raw_rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in raw["evaluation"]}
        for (condition, scenario, scene), row in rows.items():
            spec.check_episode(row, rows["forecast", scenario, scene], baseline=False)
            if not (row["simulation_end_time_s"] == 61380 and row["N_fleet"] == 12
                    and np.all(np.isfinite([row[k] for k in metrics])) and row["execution"]["endpoint_error_max_s"] == 0):
                raise ValueError("Changed physical evaluation protocol")
            if condition in {"forecast", "nominal"}:
                if not row["source_control_reproduced"]:
                    raise ValueError("Source controls were not reproduced")
                spec.check_episode(row, raw_rows[condition, scenario, scene], baseline=True)
            else:
                validate_credit(row["credit"], row, 61380)
            if condition == "forecast":
                validate_credit(row["credit"], row, 61380)
            if condition == "constant_residual" and not np.allclose(row["residual_action_mean"], constant, rtol=0, atol=1e-6):
                raise ValueError("Constant action changed during evaluation")
        for condition in spec.CONDITIONS:
            means[root, condition] = {k: float(np.mean([r[k] for (c, _, _), r in rows.items() if c == condition])) for k in metrics}
            for scenario in spec.source.contract()["scenarios"]:
                panels.append({"root": root, "condition": condition, "scenario": scenario,
                    **{k: float(np.mean([r[k] for (c, s, _), r in rows.items() if c == condition and s == scenario])) for k in metrics}})
        raw_deltas.append({"root": root, **{k: means[root, "learned"][k] - float(np.mean([
            r[k] for r in raw["evaluation"] if r["condition"] == "learned"])) for k in metrics}})
        pairs = cell["training"]["credit_pairs"]
        diagnostics.append({"root": root, "paired_critic_loss": cell["training"]["learning_mean"]["critic_loss"],
            "raw_critic_loss": raw["training"]["learning_mean"]["critic_loss"],
            "mean_raw_reward_std": float(np.mean([p["raw_reward_std"] for p in pairs])),
            "mean_paired_reward_std": float(np.mean([p["paired_reward_std"] for p in pairs]))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "stage": "same_two_roots_scene_reuse_development_not_confirmation",
        "learned_minus_control": {c: [{"root": r, **{k: means[r, "learned"][k] - means[r, c][k] for k in metrics}}
            for r in spec.ROOTS] for c in spec.CONDITIONS if c != "learned"},
        "paired_minus_registered_raw": raw_deltas, "learning_diagnostics": diagnostics, "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {r: json.loads((directory / "cells" / spec.METHOD / f"seed_{r}/result.json").read_text()) for r in spec.ROOTS}
    raw = {r: spec.diagnosis.source_cell(r)[1] for r in spec.ROOTS}
    result = summarize(cells, raw)
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()

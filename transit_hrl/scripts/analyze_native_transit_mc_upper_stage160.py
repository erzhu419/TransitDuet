#!/usr/bin/env python3
"""Validate MC training and compare physical outcomes with own controls and SAC."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mc_upper_stage160 as spec
from scripts.analyze_native_transit_reference_credit_stage159 import control_diagnostics, validate_credit
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_training(run, root, preflight=False):
    settings = spec.contract(preflight)
    count, decisions = settings["training_episodes"], 4 if preflight else 44
    seeds = [(600000000 if preflight else 700000000) + root * 1000 + ep for ep in range(count)]
    if not (run["updates"] == settings["upper_actor_optimizer_steps"] and run["transitions"] == count * decisions
            and run["actor_change_max_abs"] > 0 and run["training_scene_seeds"] == seeds
            and run["reference_episodes"] == count and run["mc_targets_qualified"]
            and run["unused_lower_actor_unchanged"] and len(run["credit_pairs"]) == count
            and len(run["training_curve"]) == count
            and len(run["update_curve"]) == count // settings["batch_episodes"]
            and sum(r["upper_actor_optimizer_steps"] for r in run["update_curve"]) == run["updates"]):
        raise ValueError("Changed MC episode, on-policy batch or optimizer budget")
    for ep, (pair, row) in enumerate(zip(run["credit_pairs"], run["training_curve"])):
        expected = 100 * (pair["reference_final_cost"] - pair["controlled_final_cost"])
        if not (pair["episode"] == row["episode"] == ep and pair["scene_seed"] == seeds[ep]
                and pair["scenario"] == row["scenario"] == spec.source.contract()["scenarios"][ep % 5]
                and pair["duration_s"] == settings["clock_s"] and pair["decisions"] == decisions
                and pair["terminal_transitions"] == 1 and pair["mc_returns_qualified"]
                and np.isfinite(pair["mc_return_std"])
                and np.isclose(pair["paired_reward_sum"], expected, rtol=1e-10, atol=1e-8)
                and np.isclose(pair["terminal_advantage"], expected, rtol=1e-10, atol=1e-8)
                and np.isclose(pair["replay_reward_sum"], expected, rtol=1e-6, atol=1e-4)):
            raise ValueError("Changed MC terminal objective, scene pairing or complete-episode credit")
        validate_credit(row["credit"], row, settings["clock_s"])
        if not np.isclose(row["credit"]["final_cost"], pair["controlled_final_cost"], rtol=0, atol=1e-8):
            raise ValueError("MC credit refers to a different training episode")
    for i, row in enumerate(run["update_curve"]):
        if row["last_episode"] != (i + 1) * settings["batch_episodes"] - 1 or not all(np.isfinite(v) for v in row.values()):
            raise ValueError("MC PPO updated before its registered on-policy batch completed")


def summarize(cells, sac_cells):
    if set(cells) != set(spec.ROOTS) or set(sac_cells) != set(spec.ROOTS):
        raise ValueError("Incomplete MC/SAC roots")
    metrics = spec.source.source.source_spec.authority.routing.METRICS
    means, controls, panels, deltas = {}, [], [], []
    for root, cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["software_qualified"] and cell["native_training_updates"] == 0
                and cell["native_steps"] == 320 * 61380 and cell["worker_preflight_native_steps"] == 61380 + 4 * 5400
                and cell["forecast_source_reproduced"]):
            raise ValueError("Unqualified native MC upper experiment")
        validate_training(cell["training"], root)
        validate_training(cell["short_learning"], root, preflight=True)
        rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
        required = {(c, s, scene) for c in spec.CONDITIONS for s in spec.source.contract()["scenarios"]
            for scene in spec.source.source.source_spec.scene_seeds(spec.source.LOWER_ROOT[root], s, preflight=False)}
        if set(rows) != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Incomplete or duplicate MC evaluation")
        constant = np.asarray(cell["training"]["constant_action"])
        if constant.shape != (2,) or not np.all(np.isfinite(constant)) or np.any(np.abs(constant) > 1):
            raise ValueError("Invalid training-only constant control")
        sac = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in sac_cells[root]["evaluation"]}
        for (condition, scenario, scene), row in rows.items():
            spec.previous.check_episode(row, sac["forecast", scenario, scene], baseline=False)
            if not (row["simulation_end_time_s"] == 61380 and row["N_fleet"] == 12
                    and np.all(np.isfinite([row[k] for k in metrics])) and row["execution"]["endpoint_error_max_s"] == 0):
                raise ValueError("Changed MC physical protocol")
            if condition in {"forecast", "nominal"}:
                if not row["source_control_reproduced"]:
                    raise ValueError("Missing SAC source-control reproduction")
                spec.previous.check_episode(row, sac[condition, scenario, scene], baseline=True)
            if condition != "nominal":
                validate_credit(row["credit"], row, 61380)
            if condition == "constant_residual" and not np.allclose(row["residual_action_mean"], constant, rtol=0, atol=1e-6):
                raise ValueError("Constant changed during MC evaluation")
        for condition in spec.CONDITIONS:
            means[root, condition] = {k: float(np.mean([r[k] for (c, _, _), r in rows.items() if c == condition])) for k in metrics}
            for scenario in spec.source.contract()["scenarios"]:
                panels.append({"root": root, "condition": condition, "scenario": scenario,
                    **{k: float(np.mean([r[k] for (c, s, _), r in rows.items() if c == condition and s == scenario])) for k in metrics}})
        controls.extend(control_diagnostics(cell["evaluation"], root, c) for c in ("forecast", "constant_residual"))
        deltas.append({"root": root, **{k: means[root, "learned"][k] - float(np.mean([
            r[k] for r in sac_cells[root]["evaluation"] if r["condition"] == "learned"])) for k in metrics}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "native_steps": sum(c["native_steps"] for c in cells.values()), "stage": spec.contract()["boundary"],
        "learned_minus_control": {c: [{"root": r, **{k: means[r, "learned"][k] - means[r, c][k] for k in metrics}}
            for r in spec.ROOTS] for c in spec.CONDITIONS if c != "learned"},
        "mc_minus_registered_paired_sac": deltas, "control_diagnostics": controls, "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {r: json.loads((directory / "cells" / spec.METHOD / f"seed_{r}/result.json").read_text()) for r in spec.ROOTS}
    result = summarize(cells, {r: spec.source_cell(r) for r in spec.ROOTS})
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()

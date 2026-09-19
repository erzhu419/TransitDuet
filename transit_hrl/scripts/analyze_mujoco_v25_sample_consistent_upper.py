#!/usr/bin/env python3
"""Apply the frozen v25 gates to small exports; never select the diagnostic arm."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mujoco_v25_sample_consistent_upper_spec as spec
from scripts.submit_mujoco_v25_sample_consistent_upper_scheduleurm import cell_dir, cells

METRICS = {
    "reward": "episode_return",
    "component": "terminal_reserve_component_correction_rms_mean",
    "total": "terminal_reserve_correction_rms_mean",
}


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def truth(value):
    return str(value).lower() in ("true", "1", "1.0")


def row_validity_failures(row):
    failures = []
    if not truth(row["protocol_valid"]):
        failures.append("protocol_invalid")
    if not truth(row["terminal_reserve_projection_enabled"]):
        failures.append("projection_disabled")
    if not truth(row["terminal_reserve_context_enabled"]):
        failures.append("context_disabled")
    if float(row["terminal_reserve_certificate_violation_count"]) != 0:
        failures.append("certificate_violation")
    if float(row["terminal_reserve_recursive_fallback_rate"]) > 0.05:
        failures.append("fallback_rate")
    if float(row["terminal_reserve_upper_prefix_power_max"]) > 0.075 ** 2 + 1e-8:
        failures.append("upper_prefix_power")
    if float(row["terminal_reserve_lower_prefix_power_max"]) > 0.0475 ** 2 + 1e-8:
        failures.append("lower_prefix_power")
    return failures


def validate_cell(summary, rows, cell, *, preflight):
    env, arm, seed = cell
    options = spec.options(arm, preflight=preflight)
    aliases = {
        "train-seeds": "rollout_seed_roots", "selection-seeds": "checkpoint_selection_seed_roots",
        "checkpoint-minimum-iteration": "checkpoint_minimum_eligible_iteration",
        "terminal-reserve-context": "terminal_reserve_context_enabled",
        "terminal-reserve-projection": "terminal_reserve_projection_enabled",
        "control-protocol-version": "protocol_version",
    }
    # These are the shared training and target contracts exposed in the small summary.
    keys = ("train-seeds", "selection-seeds", "eval-seeds", "iterations", "steps", "upper-period",
            "training-disturbance-modes", "evaluation-disturbance-modes", "ppo-clip-ratio",
            "upper-projection-consistency-coef", "lower-projection-consistency-coef",
            "upper-projection-target-aggregation", "upper-projection-consistency-objective",
            "projection-consistency-update-mode", "projection-consistency-weighting",
            "projection-consistency-training-schedule", "projection-consistency-warmup-fraction",
            "projection-consistency-ramp-fraction", "terminal-reserve-context", "terminal-reserve-projection",
            "terminal-reserve-upper-window", "terminal-reserve-lower-window", "lower-lf-rms-budget",
            "upper-hf-rms-budget", "upper-action-scale", "lower-action-scale", "checkpoint-selection-mode",
            "checkpoint-score-mode", "checkpoint-minimum-iteration", "control-protocol-version", "code-revision")
    expected = {aliases.get(key, key.replace("-", "_")): options[key] for key in keys}
    expected.update(environment=env, optimizer_seed=seed, method="freq_hrl",
                    upper_projection_target_space="unit_box_action" if arm == spec.CANDIDATE else "gaussian_raw")
    expected = json.loads(json.dumps(expected))
    drift = {key: [summary.get(key), value] for key, value in expected.items() if summary.get(key) != value}
    if drift:
        raise ValueError(f"cell contract mismatch {cell}: {drift}")
    selected = summary["selected_checkpoint_iteration"]
    if not options["checkpoint-minimum-iteration"] <= selected < options["iterations"]:
        raise ValueError(f"checkpoint outside frozen window: {cell}")
    paths = {(row["disturbance_mode"], int(row["seed"])) for row in rows}
    expected_paths = {(mode, seed) for mode in spec.EVAL_MODES for seed in options["eval-seeds"]}
    if paths != expected_paths or len(rows) != len(expected_paths):
        raise ValueError(f"missing or duplicate evaluation paths: {cell}")
    training = summary["projection_consistency_weight_training"]
    for level in ("upper", "lower"):
        if not all(math.isfinite(float(value)) for value in training[level].values()):
            raise ValueError(f"nonfinite training consistency: {cell}/{level}")
        active = training[level]["active_iteration_count"] > 0
        if active != (arm != spec.ZERO):
            raise ValueError(f"consistency failed to activate as specified: {cell}/{level}")
    numeric_keys = (*METRICS.values(), "terminal_reserve_certificate_violation_count",
                    "terminal_reserve_recursive_fallback_rate", "terminal_reserve_upper_prefix_power_max",
                    "terminal_reserve_lower_prefix_power_max")
    if not all(math.isfinite(float(row[key])) for row in rows for key in numeric_keys):
        raise ValueError(f"nonfinite evaluation metrics: {cell}")
    failures = []
    for row in rows:
        reasons = row_validity_failures(row)
        if reasons:
            failures.append({
                "environment": cell[0],
                "arm": cell[1],
                "optimizer_seed": int(cell[2]),
                "disturbance_mode": row["disturbance_mode"],
                "evaluation_seed": int(row["seed"]),
                "reasons": reasons,
                "fallback_rate": float(
                    row["terminal_reserve_recursive_fallback_rate"]
                ),
                "upper_prefix_power": float(
                    row["terminal_reserve_upper_prefix_power_max"]
                ),
                "lower_prefix_power": float(
                    row["terminal_reserve_lower_prefix_power_max"]
                ),
            })
    return failures


def development_gates(values):
    reports, wins = {}, []
    for env in spec.ENVIRONMENTS:
        arms = {arm: [values[(env, arm, seed)] for seed in spec.OPTIMIZER_SEEDS] for arm in spec.ARMS}
        means = {arm: {key: mean(rows, key) for key in METRICS} for arm, rows in arms.items()}
        candidate = means[spec.CANDIDATE]
        reward_delta = {arm: (candidate["reward"] - means[arm]["reward"]) / max(abs(means[arm]["reward"]), 1.)
                        for arm in (spec.ZERO, spec.CONTROL)}
        reduction = {arm: {key: 1. - candidate[key] / max(means[arm][key], 1e-12) for key in ("component", "total")}
                     for arm in (spec.ZERO, spec.CONTROL)}
        paired_wins = sum(a["reward"] > b["reward"] for a, b in zip(arms[spec.CANDIDATE], arms[spec.CONTROL]))
        wins.append(paired_wins)
        reports[env] = dict(means=means, reward_relative_delta=reward_delta,
                            correction_relative_reduction=reduction, paired_reward_wins=paired_wins)
    gates = {
        "reward_wins": sum(wins) >= 8 and min(wins) >= 2,
        "reward_improved_environments": sum(r["reward_relative_delta"][spec.CONTROL] > 0 for r in reports.values()) >= 2,
        "reward_regression": all(delta >= -0.05 for r in reports.values() for delta in r["reward_relative_delta"].values()),
        "correction_regression": all(value >= -0.05 for r in reports.values() for arm in r["correction_relative_reduction"].values() for value in arm.values()),
        "hopper_total_correction": reports["Hopper-v5"]["means"][spec.CANDIDATE]["total"] <= 0.25,
    }
    for key in ("component", "total"):
        gates[key + "_improved_environments"] = sum(r["correction_relative_reduction"][spec.ZERO][key] >= 0.05 for r in reports.values()) >= 2
    return reports, gates


def analyze(directory):
    registration = json.loads((directory / "preregistration.json").read_text())
    preflight = registration["preflight"]
    expected = dict(protocol=spec.PROTOCOL, algorithm_revision=spec.ALGORITHM_REVISION,
                    cells=cells(preflight), contract=spec.CONTRACT,
                    options={arm: spec.options(arm, preflight=preflight) for arm in spec.ARMS})
    for key, value in json.loads(json.dumps(expected)).items():
        if registration.get(key) != value:
            raise ValueError(f"registration mismatch: {key}")
    values, capacities, rows_by_cell = {}, {}, {}
    validity_failures = []
    for cell in cells(preflight):
        path = directory / cell_dir(directory.name, *cell).relative_to(Path("results") / directory.name)
        summary = json.loads((path / "cell_summary.json").read_text())
        with (path / "evaluation_rows.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validity_failures.extend(
            validate_cell(summary, rows, cell, preflight=preflight)
        )
        capacities[cell] = summary["capacity_actual_parameter_count"]
        values[cell] = {key: mean(rows, metric) for key, metric in METRICS.items()}
        rows_by_cell[cell] = rows
    capacity_matched = all(len({v for (env, _, _), v in capacities.items() if env == environment}) == 1 for environment in spec.ENVIRONMENTS)
    all_rows = [row for rows in rows_by_cell.values() for row in rows]
    validity_maxima = {
        "certificate_violation_count": max(
            float(row["terminal_reserve_certificate_violation_count"])
            for row in all_rows
        ),
        "fallback_rate": max(
            float(row["terminal_reserve_recursive_fallback_rate"])
            for row in all_rows
        ),
        "upper_prefix_power": max(
            float(row["terminal_reserve_upper_prefix_power_max"])
            for row in all_rows
        ),
        "lower_prefix_power": max(
            float(row["terminal_reserve_lower_prefix_power_max"])
            for row in all_rows
        ),
    }
    reason_counts = Counter(
        reason for failure in validity_failures for reason in failure["reasons"]
    )
    gates = dict(
        validity=not validity_failures,
        capacity_matched=capacity_matched,
    )
    report = dict(protocol=spec.PROTOCOL, preflight=preflight, cell_count=len(values),
                  evaluation_row_count=sum(map(len, rows_by_cell.values())))
    report["validity_diagnostics"] = {
        "invalid_row_count": len(validity_failures),
        "invalid_cell_count": len({
            (row["environment"], row["arm"], row["optimizer_seed"])
            for row in validity_failures
        }),
        "reason_counts": dict(sorted(reason_counts.items())),
        "max_observed": validity_maxima,
        "failures": validity_failures,
    }
    if not preflight:
        report["environments"], performance = development_gates(values)
        gates.update(performance)
        report["by_condition"] = {env: {arm: {mode: {
            key: mean([row for (e, a, _), rows in rows_by_cell.items() if (e, a) == (env, arm)
                       for row in rows if row["disturbance_mode"] == mode], metric)
            for key, metric in METRICS.items()} for mode in spec.EVAL_MODES} for arm in spec.ARMS} for env in spec.ENVIRONMENTS}
    report["gates"] = gates
    report["status"] = ("preflight_valid" if preflight else "development_advances") if all(gates.values()) else ("preflight_invalid" if preflight else "development_stops")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    (args.directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("environments", "by_condition")}, indent=2))


if __name__ == "__main__":
    main()

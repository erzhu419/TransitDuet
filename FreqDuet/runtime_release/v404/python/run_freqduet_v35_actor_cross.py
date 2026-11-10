#!/usr/bin/env python3
"""Cross 40/200-episode execution actors under one frozen 200-episode context."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from complete_freqduet_v34_optimized_evaluation import (
    OPTIMIZATION_COMMIT, OPTIMIZED_ROOT, optimized_runtime,
)
from run_freqduet_protocol_v6_checkpoint_trajectory import (
    CONFIGS, CURRENT, EVAL_SEEDS, METRICS, MODEL_ROOT, SOURCE_COMMIT, TRAIN_SEEDS,
    V9_TRAIN_SEEDS, V33_TRAIN_SEEDS, V9_EVAL_SEEDS, V33_EVAL_SEEDS,
    assert_same_state, evaluation_rows, runtime_config, seed_runtime,
)


RUN_NAME = "protocol_v6_v35_actor_cross_b200"
TRAIN_ROOT = OPTIMIZED_ROOT.parents[2] / (
    "FreqDuet-v4-worktree/FreqDuet/freqduet/results_freqduet/"
    "protocol_v6_v34_checkpoint_trajectory")
BASE_EP = 199
ACTOR_EPS = (39, 199)
PRIMARY = ("headway_cv", "restricted_total_journey_horizon_min",
           "holding_vehicle_seconds_per_launched_trip", "fleet_readiness_delay_mean_s")
MECHANISM = (
    "lower_load_hold_penalty_mean", "lower_load_ratio_mean",
    "lower_normalized_person_delay_mean", "lower_departure_regularity_reward_mean",
    "upper_interval_wait_cost_sum", "upper_interval_dispatch_backlog_cost_sum",
)
KEYS = ["config", "train_seed", "eval_seed"]


def job_name(config, seed, upper_ep, lower_ep):
    return f"{config}_seed{seed}_U{upper_ep + 1}_L{lower_ep + 1}"


def non_actor_state(runner):
    lower, upper = runner.lower_trainer, runner.upper_trainer
    return copy.deepcopy(dict(
        deployment=runner._deployment_state_dict(BASE_EP),
        lower_q=lower.q_net.state_dict(), lower_target_q=lower.target_q_net.state_dict(),
        lower_cost_q=lower.cost_q_net.state_dict(),
        lower_target_cost_q=lower.target_cost_q_net.state_dict(),
        upper_q=upper.q_net.state_dict(), upper_target_q=upper.target_q_net.state_dict(),
        lower_alpha=lower.alpha, lower_log_alpha=lower.log_alpha,
        lower_log_lambda=lower.log_lambda, upper_alpha=upper.alpha,
        upper_log_alpha=upper.log_alpha,
        lower_sampling=lower.policy_net.sampling_state(),
        upper_sampling=upper.policy_net.sampling_state(),
    ))


def transplant_actors(runner, lower_checkpoint, upper_checkpoint):
    before = non_actor_state(runner)
    runner.lower_trainer.policy_net.load_state_dict(lower_checkpoint["policy"])
    runner.upper_trainer.policy_net.load_state_dict(upper_checkpoint["policy"])
    assert_same_state(before, non_actor_state(runner))


def run_cell(out_root, config, seed, upper_ep, lower_ep, eval_seeds=EVAL_SEEDS):
    _, Runner, load_config, torch = optimized_runtime(out_root)
    name = job_name(config, seed, upper_ep, lower_ep)
    cell = out_root / "cells" / name
    checkpoints = TRAIN_ROOT / "logs" / f"{config}_seed{seed}/checkpoints"
    contract = dict(config=config, train_seed=seed, eval_seeds=list(eval_seeds),
                    upper_actor_ep=upper_ep, lower_actor_ep=lower_ep,
                    deployment_ep=BASE_EP, policy_episode=BASE_EP,
                    training_source_commit=SOURCE_COMMIT,
                    runtime_optimization_commit=OPTIMIZATION_COMMIT,
                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
                    model_root=str(OPTIMIZED_ROOT), checkpoint_dir=str(checkpoints),
                    intervention="execution_actor_state_dict_only", stage="exploratory_attribution")
    cell.mkdir(parents=True, exist_ok=True)
    manifest = cell / "actor_cross_contract.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError(f"actor cross contract differs: {cell}")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")
    output = cell / "evaluation"
    if not (output / "evaluation.csv").is_file():
        seed_runtime(torch, seed)
        source_config = load_config(str(MODEL_ROOT / "configs_freqduet" / f"{config}.yaml"))
        runner = Runner(runtime_config(source_config, seed, cell / "runner_logs", 40), device="cpu")
        runner.load_checkpoint(checkpoints, ep=BASE_EP, require_deployment_state=True)
        lower = torch.load(checkpoints / f"lower_ep{lower_ep}.pt", map_location="cpu", weights_only=False)
        upper = torch.load(checkpoints / f"upper_ep{upper_ep}.pt", map_location="cpu", weights_only=False)
        transplant_actors(runner, lower, upper)
        print(f"ACTOR_CROSS_READY {name} deployment_ep={BASE_EP} scenarios={len(eval_seeds)}", flush=True)
        runner.evaluate(eval_seeds, output_dir=output, policy_ep=BASE_EP)
        del runner
    frame = evaluation_rows(output / "evaluation.csv", eval_seeds, BASE_EP)
    selected = frame[["eval_seed", "scenario_tape_id", *METRICS, *MECHANISM]].copy()
    for field, value in dict(config=config, train_seed=seed, upper_actor_ep=upper_ep,
                             lower_actor_ep=lower_ep, deployment_ep=BASE_EP,
                             policy_episode=BASE_EP).items():
        selected[field] = value
    records = out_root / "records"
    records.mkdir(parents=True, exist_ok=True)
    selected.to_csv(records / f"{name}.csv", index=False)
    ready = out_root / "ready" / name
    ready.mkdir(parents=True, exist_ok=True)
    (ready / "ready.json").write_text(json.dumps(dict(contract, rollouts=len(frame))) + "\n")
    print(f"ACTOR_CROSS_DONE {name}", flush=True)
    return frame


def smoke(out_root, config):
    seed, scenarios = TRAIN_SEEDS[0], (EVAL_SEEDS[0], V33_EVAL_SEEDS[0])
    candidate = run_cell(out_root, config, seed, BASE_EP, BASE_EP, scenarios)
    old_path = TRAIN_ROOT / "logs" / f"{config}_seed{seed}/frozen_trajectory/ep199/evaluation.csv"
    original = pd.read_csv(old_path).set_index("eval_seed").loc[list(scenarios)].reset_index()
    ignored = {"wall_env_s", "wall_train_s", "config_fingerprint_sha256", "policy_digest"}
    columns = [column for column in original if column not in ignored]
    pd.testing.assert_frame_equal(original[columns], candidate[columns], check_exact=True)
    run_cell(out_root, config, seed, BASE_EP, ACTOR_EPS[0], scenarios[:1])
    ready = out_root / "smoke_ready" / config
    ready.mkdir(parents=True, exist_ok=True)
    (ready / "smoke_pass.json").write_text(json.dumps(dict(
        config=config, noop_columns=len(columns), noop_scenarios=list(scenarios),
        mixed_actor_frozen_evaluation=True, non_actor_state_unchanged=True)) + "\n")
    print(f"ACTOR_CROSS_SMOKE_PASS {config} columns={len(columns)}", flush=True)


def factorial_effects(frame, original_early):
    panels = {}
    for upper in ACTOR_EPS:
        for lower in ACTOR_EPS:
            subset = frame.loc[(frame.upper_actor_ep == upper) & (frame.lower_actor_ep == lower)]
            panels[upper, lower] = subset.set_index(KEYS)[list(METRICS)].sort_index()
    early = original_early.set_index(KEYS)[list(METRICS)].sort_index()
    for panel in [*panels.values(), early]:
        if not panel.index.is_unique or not panel.index.equals(early.index):
            raise ValueError("actor cross has unmatched or duplicated contexts")
    a, b = panels[39, 39], panels[39, 199]
    c, d = panels[199, 39], panels[199, 199]
    effects = {
        "lower_with_upper40": b - a,
        "lower_with_upper200": d - c,
        "upper_with_lower40": c - a,
        "upper_with_lower200": d - b,
        "lower_averaged_effect": ((b - a) + (d - c)) / 2,
        "upper_averaged_effect": ((c - a) + (d - b)) / 2,
        "upper_lower_interaction": d - c - b + a,
        "actor_age_total": d - a,
        "non_actor_context_and_stage_residual": a - early,
        "observed_40_to_200": d - early,
    }
    return pd.concat([values.reset_index().assign(effect=name)
                      for name, values in effects.items()], ignore_index=True)


def aggregate_cross(out_root):
    frame = pd.concat([pd.read_csv(path) for path in sorted((out_root / "records").glob("*.csv"))],
                      ignore_index=True)
    columns = ["config", "train_seed", "eval_seed", "upper_actor_ep", "lower_actor_ep"]
    expected = pd.MultiIndex.from_product([CONFIGS, TRAIN_SEEDS, EVAL_SEEDS, ACTOR_EPS, ACTOR_EPS],
                                          names=columns)
    actual = pd.MultiIndex.from_frame(frame[columns])
    if actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError("actor cross grid is incomplete, duplicated, or outside the design")
    if not frame.groupby("eval_seed").scenario_tape_id.nunique().eq(1).all():
        raise ValueError("actor cross scenario tapes differ")
    if not frame.deployment_ep.eq(BASE_EP).all() or not frame.policy_episode.eq(BASE_EP).all():
        raise ValueError("actor cross deployment state or evaluation stage differs")
    archived = {}
    for ep in ACTOR_EPS:
        archived[ep] = pd.concat([pd.read_csv(TRAIN_ROOT / "records" /
            f"{config}_seed{seed}_ep{ep}.csv") for config in CONFIGS for seed in TRAIN_SEEDS],
            ignore_index=True)
    late = frame.loc[(frame.upper_actor_ep == BASE_EP) & (frame.lower_actor_ep == BASE_EP)]
    new = late.set_index(KEYS)[list(METRICS)].sort_index()
    old = archived[BASE_EP].set_index(KEYS)[list(METRICS)].sort_index()
    pd.testing.assert_index_equal(new.index, old.index)
    if not np.isfinite(new.to_numpy()).all() or not (new - old).abs().le(1e-9).all().all():
        raise ValueError("unchanged 200/200 control does not reproduce V34")
    effects = factorial_effects(frame, archived[39])
    sys.path.insert(0, str(MODEL_ROOT))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval

    cohorts = {
        "all_train_all_eval": (TRAIN_SEEDS, EVAL_SEEDS),
        "v9_train_v9_eval": (V9_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v33_train_v9_eval": (V33_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v9_train_v33_eval": (V9_TRAIN_SEEDS, V33_EVAL_SEEDS),
        "v33_train_v33_eval": (V33_TRAIN_SEEDS, V33_EVAL_SEEDS),
    }
    summaries = []
    for (config, effect), rows in effects.groupby(["config", "effect"]):
        for cohort, (train_seeds, eval_seeds) in cohorts.items():
            subset = rows.loc[rows.train_seed.isin(train_seeds) & rows.eval_seed.isin(eval_seeds)]
            for metric in PRIMARY:
                low, high = hierarchical_interval(subset, metric)
                summaries.append(dict(config=config, effect=effect, cohort=cohort, metric=metric,
                                      delta=float(subset[metric].mean()), ci95_low=low, ci95_high=high,
                                      positive_train_seed_fraction=float(
                                          subset.groupby("train_seed")[metric].mean().gt(0).mean())))
    target = out_root / "summary"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(target / "factorial_effects.csv", index=False)
    frame.groupby(["config", "upper_actor_ep", "lower_actor_ep"])[
        [*METRICS, *MECHANISM]].mean().to_csv(target / "cell_means.csv")
    effects.groupby(["config", "effect", "train_seed"])[list(METRICS)].mean().to_csv(
        target / "training_seed_effects.csv")
    result = dict(status="complete_exploratory_attribution", rollouts=len(frame),
                  training_runs=0, deployment_ep=BASE_EP, policy_episode=BASE_EP,
                  training_source_commit=SOURCE_COMMIT, runtime_optimization_commit=OPTIMIZATION_COMMIT,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
                  unchanged_control_matches_v34=True, independent_confirmation=False,
                  residual_scope="non-actor deployment/trainer context plus evaluation stage",
                  interaction_note="not added again to the averaged upper/lower effects")
    (target / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--upper-ep", type=int, choices=ACTOR_EPS)
    parser.add_argument("--lower-ep", type=int, choices=ACTOR_EPS)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate_cross(args.out_root)
    elif args.command == "smoke":
        if args.config is None:
            parser.error("smoke requires --config")
        smoke(args.out_root, args.config)
    else:
        if any(value is None for value in (args.config, args.train_seed, args.upper_ep, args.lower_ep)):
            parser.error("run requires --config, --train-seed, --upper-ep, --lower-ep")
        run_cell(args.out_root, args.config, args.train_seed, args.upper_ep, args.lower_ep)


if __name__ == "__main__":
    main()

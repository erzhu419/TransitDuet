#!/usr/bin/env python3
"""Matched-prefix lower-action values and deterministic deployment consequences."""

import argparse
import copy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd

from run_freqduet_v35_actor_cross import (
    ACTOR_EPS, BASE_EP, CONFIGS, CURRENT, METRICS, MODEL_ROOT, TRAIN_ROOT,
    TRAIN_SEEDS, assert_same_state, non_actor_state, optimized_runtime,
    runtime_config, seed_runtime,
)


RUN_NAME = "protocol_v6_v36_lower_action_credit"
SCENARIOS = (45007, 45013, 934011, 934029)
TARGET_TIMES = (7200, 21600, 36000)
OUTCOMES = (*METRICS, "holding_vehicle_seconds", "holding_passenger_seconds",
            "upper_interval_wait_cost_sum", "upper_interval_onboard_cost_sum",
            "upper_interval_dispatch_backlog_cost_sum")
RETURNS = ("trip_reward_return", "trip_cost_return")
KEYS = ["config", "train_seed", "eval_seed", "anchor_lower_ep", "target_time_s"]


def case_name(config, seed, scenario, anchor):
    return f"{config}_seed{seed}_eval{scenario}_L{anchor + 1}"


def rng_state(runner, torch):
    return copy.deepcopy(dict(
        python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.random.get_rng_state(),
        runtime={name: rng.get_state()
                 for name, rng in runner._runtime_numpy_streams().items()},
    ))


def action_values(trainer, state, torch):
    """Use exactly the categorical actor objective, without sampling an action."""
    state = torch.as_tensor(state, dtype=torch.float32, device=trainer.device)[None, :]
    with torch.no_grad():
        probs, log_probs, _ = trainer.policy_net.dist_info(state)
        values = trainer._discrete_q_values(trainer.q_net, state)
        q_lcb, q_mean, q_std = trainer._policy_q_value(values)
        cost = trainer._discrete_cost_values(trainer.cost_q_net, state)
        lam = float(trainer.lambda_param)
        terms = dict(probability=probs, q_mean=q_mean, q_std=q_std,
                     pessimism=q_mean - q_lcb, cost_penalty=lam * cost,
                     cost_q=cost, entropy_term=trainer.alpha * log_probs,
                     energy=-q_lcb + lam * cost)
        terms["actor_loss_term"] = terms["energy"] + terms["entropy_term"]
        rows = [{key: float(tensor[0, index]) for key, tensor in terms.items()}
                for index in range(probs.shape[1])]
        for row, action in zip(rows, trainer.policy_net.action_bins.flatten().tolist()):
            row.update(action_s=float(action), alpha=float(trainer.alpha), lambda_value=lam)
    return rows


def same_prefix(reference, candidate):
    for field in ("identity", "raw_state", "encoded_state", "previous_action_s",
                  "actor_action_s", "departure_context", "lower_prefix",
                  "upper_prefix", "rng"):
        assert_same_state(reference[field], candidate[field], field)


def check_unchanged(left, right):
    ignored = {"wall_env_s", "wall_train_s", "config_fingerprint_sha256", "policy_digest"}
    fields = [name for name in left if name not in ignored]
    assert_same_state({name: left[name] for name in fields},
                      {name: right[name] for name in fields}, "unchanged_episode")


def run_branch(out_root, config, seed, scenario, anchor, times, label,
               intervention=None, diagnose=False):
    _, Runner, load_config, torch = optimized_runtime(out_root)
    cell = out_root / "cells" / case_name(config, seed, scenario, anchor)
    seed_runtime(torch, seed)
    cfg = load_config(str(MODEL_ROOT / "configs_freqduet" / f"{config}.yaml"))
    runner = Runner(runtime_config(cfg, seed, cell / "logs" / label, 40), device="cpu")
    checkpoints = TRAIN_ROOT / "logs" / f"{config}_seed{seed}/checkpoints"
    runner.load_checkpoint(checkpoints, ep=BASE_EP, require_deployment_state=True)
    before = non_actor_state(runner)
    lower_states = {ep: torch.load(checkpoints / f"lower_ep{ep}.pt",
                                  map_location="cpu", weights_only=False) for ep in ACTOR_EPS}
    runner.lower_trainer.policy_net.load_state_dict(lower_states[anchor]["policy"])
    assert_same_state(before, non_actor_state(runner))
    del before
    models = {}
    if diagnose:
        for ep in ACTOR_EPS:
            models[ep] = copy.deepcopy(runner.lower_trainer)
            models[ep].load_training_state_dict(lower_states[ep])
    del lower_states
    captures, value_rows = {}, []
    lower_prefix, upper_prefix = [], []
    original_lower = runner._lower_action_for_agent
    original_upper = runner._upper_callback_v2
    original_transition = runner._record_lower_transition

    def upper_trace(state, trip):
        result = original_upper(state, trip)
        if len(captures) < len(times):
            upper_prefix.append(dict(
                time_s=float(runner.env.current_time), trip_id=int(trip.launch_turn),
                state=np.asarray(state).copy(), target_headway_s=float(result),
                decision=copy.deepcopy(runner._ep_trip_records[-1])))
        return result

    def lower_trace(obs, key, last_action=0.0, deterministic=False):
        action = original_lower(obs, key, last_action=last_action, deterministic=deterministic)
        if len(captures) == len(times):
            return action
        bus = runner._bus_for_agent(key)
        now = float(runner.env.current_time)
        raw = np.asarray(obs, dtype=np.float32).copy()
        event = dict(time_s=now, bus_id=int(key), trip_id=int(bus.trip_id),
                     raw_state=raw, previous_action_s=float(last_action),
                     actor_action_s=float(np.asarray(action).item()))
        lower_prefix.append(event)
        # Only actual, observable, nonterminal decisions can be intervened on.
        departure = runner._pending_lower_action_context[int(key)]
        eligible = (not runner._lower_terminal_action_masked(bus)
                    and not runner._fixed_expert_active and departure.evidence_valid)
        pending = [time for time in times if time not in captures and now >= time]
        if not eligible or not pending:
            return action
        target = pending[0]
        state = runner._augment_lower_state(obs, last_action)
        capture = dict(
            identity=dict(time_s=now, bus_id=int(key), trip_id=int(bus.trip_id),
                          station_id=int(raw[1]), direction=bool(bus.direction)),
            raw_state=raw, encoded_state=state.copy(), previous_action_s=float(last_action),
            actor_action_s=event["actor_action_s"], departure_context=asdict(departure),
            lower_prefix=copy.deepcopy(lower_prefix), upper_prefix=copy.deepcopy(upper_prefix),
            rng=rng_state(runner, torch), transitions=0, trip_done=False,
            trip_reward_return=0.0, trip_cost_return=0.0)
        captures[target] = capture
        if diagnose:
            for ep, model in models.items():
                model_rows = action_values(model, state, torch)
                greedy = max(model_rows, key=lambda row: row["probability"])["action_s"]
                if ep == anchor and greedy != event["actor_action_s"]:
                    raise ValueError("captured deployment action differs from the diagnosed actor")
                for row in model_rows:
                    penalty, load, person_delay = runner.lower_load_holding_penalty.evaluate(
                        raw, row["action_s"], base_state_dim=runner.env._base_state_dim,
                        context_features=runner.env.lower_context_features)
                    regularity = runner.lower_departure_regularity.evaluate(departure, row["action_s"])
                    row.update(model_ep=ep, target_time_s=target, load_penalty=penalty,
                               load_ratio=load, normalized_person_delay=person_delay,
                               regularity_reward=regularity.reward_adjustment,
                               regularity_cost=regularity.cost)
                    value_rows.append(row)
        if intervention is not None:
            return np.asarray([intervention], dtype=np.float32)
        return action

    def transition_trace(**event):
        result = original_transition(**event)
        bus = event.get("bus") or runner._bus_for_agent(event["key"])
        tid = event.get("trip_id")
        tid = int(bus.trip_id if tid is None else tid)
        for capture in captures.values():
            ident = capture["identity"]
            if (not capture["trip_done"] and int(event["key"]) == ident["bus_id"]
                    and tid == ident["trip_id"]):
                discount = runner.lower_trainer.gamma ** capture["transitions"]
                capture["trip_reward_return"] += discount * float(result[0])
                capture["trip_cost_return"] += discount * float(result[1])
                capture["transitions"] += 1
                capture["trip_done"] = bool(event["transition_done"])
        return result

    runner._lower_action_for_agent = lower_trace
    runner._upper_callback_v2 = upper_trace
    runner._record_lower_transition = transition_trace
    rows, _ = runner.evaluate([scenario], output_dir=cell / "evaluation" / label, policy_ep=BASE_EP)
    if set(captures) != set(times):
        raise ValueError(f"missing causal targets: {set(times) - set(captures)}")
    if any(not cap["upper_prefix"] for cap in captures.values()):
        raise ValueError("upper dispatch prefix was not observed")
    if any(not cap["trip_done"] or cap["transitions"] == 0 for cap in captures.values()):
        raise ValueError("target trip did not produce a complete lower return")
    # Compare CSV-normalized fields to the archived CSV (including empty cells).
    episode = pd.read_csv(cell / "evaluation" / label / "evaluation.csv").iloc[0].to_dict()
    result = dict(episode=episode, captures=captures, values=value_rows,
                  actions=runner.lower_trainer.policy_net.action_bins.flatten().tolist())
    del runner, models
    gc.collect()
    print(f"LOWER_CREDIT_BRANCH_DONE {case_name(config, seed, scenario, anchor)} {label}", flush=True)
    return result


def run_case(out_root, config, seed, scenario, anchor, times=TARGET_TIMES):
    name = case_name(config, seed, scenario, anchor)
    record = out_root / "records" / name
    ready = out_root / "ready" / name / "ready.json"
    if ready.is_file():
        if json.loads(ready.read_text())["target_times_s"] != list(times):
            raise ValueError("completed case has a different context design")
        return
    common = dict(out_root=out_root, config=config, seed=seed, scenario=scenario, anchor=anchor)
    reference = run_branch(**common, times=times, label="reference", diagnose=True)
    archive = TRAIN_ROOT.parent / "protocol_v6_v35_actor_cross_b200" / "cells" / (
        f"{config}_seed{seed}_U200_L{anchor + 1}/evaluation/evaluation.csv")
    old = pd.read_csv(archive).set_index("eval_seed").loc[scenario].to_dict()
    old["eval_seed"] = scenario
    check_unchanged(reference["episode"], old)
    repeat = run_branch(**common, times=times, label="repeat")
    check_unchanged(reference["episode"], repeat["episode"])
    for target in times:
        same_prefix(reference["captures"][target], repeat["captures"][target])
    labels = []
    for target in times:
        base = reference["captures"][target]
        for action in reference["actions"]:
            branch = run_branch(**common, times=(target,), label=f"t{target}_a{action:g}",
                                intervention=action)
            capture = branch["captures"][target]
            same_prefix(base, capture)
            if action == base["actor_action_s"]:
                check_unchanged(reference["episode"], branch["episode"])
                for field in (*RETURNS, "transitions", "trip_done"):
                    assert_same_state(base[field], capture[field], field)
            row = dict(target_time_s=target, action_s=action,
                       actor_action_s=base["actor_action_s"], **capture["identity"])
            row.update({metric: float(branch["episode"][metric]) for metric in OUTCOMES})
            row.update({metric: capture[metric] for metric in RETURNS})
            row.update(trip_transitions=capture["transitions"], trip_done=capture["trip_done"])
            labels.append(row)
    record.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("action_values.csv", reference["values"]), ("rollouts.csv", labels)):
        frame = pd.DataFrame(rows)
        for key, value in dict(config=config, train_seed=seed, eval_seed=scenario,
                               anchor_lower_ep=anchor).items():
            frame[key] = value
        frame.to_csv(record / filename, index=False)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps(dict(
        config=config, train_seed=seed, eval_seed=scenario, anchor_lower_ep=anchor,
        target_times_s=list(times), actions_s=reference["actions"],
        rollouts=2 + len(times) * len(reference["actions"]),
        unchanged_matches_v35=True, exact_prefixes=True,
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"LOWER_CREDIT_CASE_DONE {name}", flush=True)


def preference_pairs(values, rollouts):
    choices = []
    for keys, group in values.groupby(KEYS, sort=True):
        key = dict(zip(KEYS, keys))
        actions = {}
        for ep in ACTOR_EPS:
            model = group.loc[group.model_ep == ep].sort_values("action_s")
            actions[ep] = float(model.loc[model.probability.idxmax(), "action_s"])
            actions[f"energy{ep}"] = float(model.loc[model.energy.idxmin(), "action_s"])
        outcomes = rollouts
        for name, value in key.items():
            outcomes = outcomes.loc[outcomes[name] == value]
        outcomes = outcomes.set_index("action_s")
        for contrast, first, last in (
                ("policy200_minus_policy40", actions[39], actions[199]),
                ("energy200_minus_energy40", actions["energy39"], actions["energy199"]),
                ("max_hold_minus_zero", 0.0, float(outcomes.index.max()))):
            row = dict(key, contrast=contrast, action_delta_s=last - first)
            row.update({metric: float(outcomes.loc[last, metric] - outcomes.loc[first, metric])
                        for metric in (*OUTCOMES, *RETURNS)})
            choices.append(row)
    return pd.DataFrame(choices)


def aggregate(out_root):
    expected = {case_name(config, seed, scenario, anchor)
                for config in CONFIGS for seed in TRAIN_SEEDS
                for scenario in SCENARIOS for anchor in ACTOR_EPS}
    ready_paths = list((out_root / "ready").glob("*/ready.json"))
    if {path.parent.name for path in ready_paths} != expected:
        raise ValueError("lower-credit cases incomplete or outside design")
    values = pd.concat([pd.read_csv(out_root / "records" / name / "action_values.csv")
                        for name in sorted(expected)], ignore_index=True)
    rollouts = pd.concat([pd.read_csv(out_root / "records" / name / "rollouts.csv")
                          for name in sorted(expected)], ignore_index=True)
    if values.duplicated([*KEYS, "model_ep", "action_s"]).any() or rollouts.duplicated([*KEYS, "action_s"]).any():
        raise ValueError("duplicate lower-credit action records")
    pairs = preference_pairs(values, rollouts)
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval
    summary = []
    primary = ("action_delta_s", "headway_cv", "restricted_total_journey_horizon_min",
               "restricted_wait_horizon_min", "holding_vehicle_seconds_per_launched_trip",
               "fleet_readiness_delay_mean_s", *RETURNS)
    for (config, anchor, contrast), rows in pairs.groupby(["config", "anchor_lower_ep", "contrast"]):
        # Average contexts within train/scenario before the crossed bootstrap.
        contexts = rows.groupby(["train_seed", "eval_seed"])[list(primary)].mean().reset_index()
        for metric in primary:
            low, high = hierarchical_interval(contexts, metric)
            summary.append(dict(config=config, anchor_lower_ep=anchor, contrast=contrast,
                                metric=metric, delta=float(contexts[metric].mean()),
                                ci95_low=low, ci95_high=high))
    target = out_root / "summary"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(target / "preference_effects.csv", index=False)
    pairs.groupby(["config", "anchor_lower_ep", "contrast", "train_seed"])[
        list(primary)].mean().to_csv(target / "training_seed_effects.csv")
    numeric = ["probability", "q_mean", "q_std", "pessimism", "cost_penalty",
               "entropy_term", "energy", "actor_loss_term", "alpha", "lambda_value",
               "load_penalty", "regularity_reward", "regularity_cost"]
    zero = values.loc[values.action_s == 0, [*KEYS, "model_ep", *numeric]]
    relative = values.merge(zero, on=[*KEYS, "model_ep"], validate="many_to_one",
                            suffixes=("", "_zero"))
    for metric in ("q_mean", "pessimism", "cost_penalty", "entropy_term", "energy", "actor_loss_term"):
        relative[metric] -= relative[f"{metric}_zero"]
    relative.groupby(["config", "anchor_lower_ep", "model_ep", "action_s"])[
        numeric].mean().to_csv(target / "action_terms.csv")
    result = dict(status="complete_exploratory_lower_credit", cases=len(expected),
                  contexts=len(values[KEYS].drop_duplicates()),
                  rollouts=sum(json.loads(path.read_text())["rollouts"] for path in ready_paths),
                  training_runs=0, independent_confirmation=False,
                  unchanged_controls_match_v35=True, exact_prefixes=True,
                  return_scope="deterministic remaining target trip; not stochastic soft-Q calibration",
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (target / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "smoke", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config", choices=CONFIGS, default=CURRENT)
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS, default=TRAIN_SEEDS[0])
    parser.add_argument("--eval-seed", type=int, choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument("--anchor-lower-ep", type=int, choices=ACTOR_EPS, default=199)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    else:
        run_case(args.out_root, args.config, args.train_seed, args.eval_seed,
                 args.anchor_lower_ep, (TARGET_TIMES[0],) if args.command == "smoke" else TARGET_TIMES)


if __name__ == "__main__":
    main()

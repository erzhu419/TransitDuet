#!/usr/bin/env python3
"""Matched frozen/updated actor by deterministic/stochastic deployment diagnostic."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

from freqduet_system_credit import observable_decision
from freqduet_system_learner import ArrivalMoments, SystemStateEncoder, warm_start_actor
import run_freqduet_v37_joint_lower as joint
from run_freqduet_v58_system_credit import CASES, CreditObserver, case_name, reference_trace
from run_freqduet_v59_system_learning import csv_episode


MODES = [(actor, mode, rep) for actor in ("initial", "updated")
         for mode in ("deterministic", "stochastic")
         for rep in ((0,) if mode == "deterministic" else range(3))]
ITEMS = list(itertools.product(range(len(CASES)), MODES))
METRICS = ("journey_min", "waiting_min", "onboard_min", "cv", "cv_squared", "holding_s_per_trip",
           "readiness_delay_s", "passenger_hours", "backlog_trip_s")


def item_name(index):
    case, (actor, mode, rep) = ITEMS[index]
    return f"{case_name(*CASES[case])}_{actor}_{mode}_r{rep}"


def sampler_seed(train_seed, rep):
    return int(train_seed) + 59000 + int(rep) * 1009


def make_policy(old, encoder, seed, checkpoint=None):
    from lower.resac_lagrangian import CategoricalPolicy
    policy = CategoricalPolicy(encoder.dimension, old.action_bins,
                               hidden_dim=old.fc2.in_features, sample_seed=seed,
                               action_limit_feature_index=old.action_limit_feature_index, device="cpu")
    warm_start_actor(old, policy, encoder.local_dim)
    if checkpoint is not None:
        if checkpoint["encoder"] != encoder.__dict__ or checkpoint["steps"] != 64:
            raise ValueError("V60 requires the matched V59 encoder and post-64-update actor")
        policy.load_state_dict(checkpoint["policy"])
    return policy.eval()


def collect(root, v59_root, v56_root, index, decision_audit=None):
    label = item_name(index)
    marker = root / "ready" / f"{label}.json"
    if marker.exists():
        return label
    case, (actor, mode, rep) = ITEMS[index]
    seed, scenario = CASES[case]
    _, Runner, load_config, torch = joint.optimized_runtime(root)
    v59_marker = json.loads((v59_root / "exports" / case_name(seed, scenario) / "done.json").read_text())
    if v59_marker["status"] != "learning_smoke_pass":
        raise ValueError("V59 learning/checkpoint prerequisite is incomplete")
    payload = None
    if actor == "updated":
        payload = torch.load(v59_root / "cells" / case_name(seed, scenario) / "learning_smoke.pt",
                             map_location="cpu", weights_only=False)
    observer, moments = CreditObserver(), ArrivalMoments()
    encoder, policy = None, None
    entropy, expected_hold, zero_probability = [], [], []

    class FactorialRunner(Runner):
        def _lower_policy_action(self, obs, last_action=0.0, deterministic=False):
            nonlocal encoder, policy
            if encoder is None:
                env = self.env
                encoder = SystemStateEncoder(self.lower_state_dim, env.max_agent_num,
                                             len(env.timetables), len(env.stations),
                                             env.protocol.evaluation_end_time_s)
                policy = make_policy(self.lower_trainer.policy_net, encoder, sampler_seed(seed, rep), payload)
            key = self._factorial_agent
            self._capture_lower_action_context(key, self._bus_for_agent(key))
            decision = observable_decision(self, obs, key, last_action)
            decision = replace(decision, pending_commands=tuple(observer.stream.pending_commands))
            moments.observe(self.env.headway_events.headways_s)
            encoded = encoder.encode(decision, moments)
            with torch.no_grad():
                probs, logs, _ = policy.dist_info(torch.from_numpy(encoded)[None, :])
                if actor == "initial":
                    old_probs = self.lower_trainer.policy_net.dist_info(
                        torch.tensor(decision.local_state, dtype=torch.float32)[None, :])[0]
                    torch.testing.assert_close(probs, old_probs, atol=0, rtol=0)
                entropy.append(float(-(probs * logs).sum()))
                expected_hold.append(float((probs * policy.action_bins.flatten()).sum()))
                zero_probability.append(float(probs[0, 0]))
                action = policy.get_action(encoded, deterministic=(mode == "deterministic"))
            if decision_audit is not None:
                decision_audit.decision(encoded, decision, moments.cv_squared,
                                        float(np.asarray(action).item()), probs[0].cpu().numpy())
            if actor == "initial" and mode == "deterministic":
                original = super()._lower_policy_action(obs, last_action, deterministic=True)
                np.testing.assert_array_equal(np.asarray(action).reshape(-1), np.asarray(original).reshape(-1))
            return np.asarray([float(np.asarray(action).item())], dtype=np.float32)

        def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
            self._factorial_agent = key
            action = super()._lower_action_for_agent(obs, key, last_action, deterministic)
            observer.stream.decide(observable_decision(self, obs, key, last_action),
                                   float(np.asarray(action).item()))
            return action

    joint.seed_runtime(torch, seed)
    config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
    cell = root / "cells" / label
    runner = FactorialRunner(joint.runtime_config(config, seed, cell / "logs", 40), device="cpu")
    runner.load_checkpoint(joint.TRAIN_ROOT / "logs" / f"{joint.CURRENT}_seed{seed}/checkpoints",
                           ep=joint.BASE_EP, require_deployment_state=True)
    if runner.lower_terminal_action_mode != "transition" or runner.lower_causal_guard_policy_mask_enabled:
        raise ValueError("V60 requires the matched unmasked V59 main action contract")
    upper_before = copy.deepcopy(runner.upper_trainer.policy_net.state_dict())
    observer.attach(runner)
    episode = runner.run_episode(joint.BASE_EP, training=False, scenario_seed=scenario, record_diagnostics=False)
    for name, value in upper_before.items():
        torch.testing.assert_close(value, runner.upper_trainer.policy_net.state_dict()[name], atol=0, rtol=0)
    if len(runner.replay_buffer):
        raise AssertionError("deployment diagnostic inserted training replay")
    if any(h["action_s"] not in joint.ACTIONS for h in observer.holds):
        raise AssertionError("deployment emitted a nonregistered holding action")
    if actor == "initial" and mode == "deterministic":
        reference = reference_trace(v56_root, seed, scenario)
        joint.check_unchanged(csv_episode(episode), reference["snapshot"]["episode"])
        pd.testing.assert_frame_equal(pd.DataFrame(observer.holds), reference["holds"], check_exact=True)
    audit = observer.stream.audit()
    cv = float(runner.env.headway_events.summary()["headway_cv"])
    moments.observe(runner.env.headway_events.headways_s)
    cv_squared = moments.cv_squared
    np.testing.assert_allclose(cv_squared, cv ** 2, atol=1e-10, rtol=0)
    if actor == "initial" and mode == "deterministic":
        np.testing.assert_equal(cv_squared, v59_marker["episodes"][0]["cv_squared"])
    total = observer.stream.total
    row = dict(index=index, label=label, train_seed=seed, scenario_seed=scenario,
               actor=actor, mode=mode, rep=rep, sampler_seed=sampler_seed(seed, rep),
               actor_updates=64 if actor == "updated" else 0,
               journey_min=float(episode["restricted_total_journey_horizon_min"]),
               waiting_min=float(episode["restricted_wait_horizon_min"]),
               onboard_min=float(episode["restricted_in_vehicle_horizon_min"]), cv=cv, cv_squared=cv_squared,
               holding_s_per_trip=float(episode["holding_vehicle_seconds_per_launched_trip"]),
               readiness_delay_s=float(episode["fleet_readiness_delay_mean_s"]),
               passenger_hours=float((total[0] + total[1]) / 3600), backlog_trip_s=float(total[4]),
               fleet_denied_trip_rate=float(episode["fleet_denied_trip_rate"]),
               trips_unlaunched=int(runner.env._measurement_details["trips_unlaunched"]),
               trips_incomplete=int(runner.env._measurement_details["trips_incomplete"]),
               cv_budget_squared=v59_marker["episodes"][0]["cv_squared"],
               entropy_mean=float(np.mean(entropy)), expected_hold_mean_s=float(np.mean(expected_hold)),
               zero_probability_mean=float(np.mean(zero_probability)),
               action_counts={str(a): sum(h["action_s"] == a for h in observer.holds) for a in joint.ACTIONS},
               upper_actor_unchanged=True, training_updates=0, stream_audit=audit,
               diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    if decision_audit is not None:
        row["decision_audit"] = decision_audit.finish(row, observer.stream)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(row, indent=2) + "\n")
    print("V60_ITEM_DONE " + json.dumps({k: v for k, v in row.items() if k != "stream_audit"}), flush=True)
    return label


def factorial_summary(rows):
    cases = []
    for seed, scenario in CASES:
        selected = [r for r in rows if (r["train_seed"], r["scenario_seed"]) == (seed, scenario)]
        groups = {}
        for actor in ("initial", "updated"):
            for mode in ("deterministic", "stochastic"):
                group = [r for r in selected if (r["actor"], r["mode"]) == (actor, mode)]
                expected = {0} if mode == "deterministic" else {0, 1, 2}
                if len(group) != len(expected) or {r["rep"] for r in group} != expected:
                    raise ValueError("factorial group is missing or duplicates a sampling replicate")
                groups[actor, mode] = {m: float(np.mean([r[m] for r in group])) for m in METRICS}
        d0, d1 = groups["initial", "deterministic"], groups["updated", "deterministic"]
        s0, s1 = groups["initial", "stochastic"], groups["updated", "stochastic"]
        comparisons = {"learning_deterministic": (d1, d0), "learning_stochastic": (s1, s0),
                       "sampling_initial": (s0, d0), "sampling_updated": (s1, d1),
                       "combined_updated_stochastic": (s1, d0)}
        effects = {name: {m: a[m] - b[m] for m in METRICS} for name, (a, b) in comparisons.items()}
        effects["interaction"] = {m: (s1[m] - s0[m]) - (d1[m] - d0[m]) for m in METRICS}
        cases.append(dict(train_seed=seed, scenario_seed=scenario, effects=effects,
                          means={f"{a}_{mode}": v for (a, mode), v in groups.items()},
                          cv_budget_squared=selected[0]["cv_budget_squared"],
                          mean_cv_squared_feasible={f"{a}_{mode}": v["cv_squared"] <= selected[0]["cv_budget_squared"]
                                                    for (a, mode), v in groups.items()},
                          updated_cv_budget_violations=sum(r["actor"] == "updated"
                              and r["cv_squared"] > r["cv_budget_squared"] for r in selected)))
    return dict(cases=cases, mean_effects={name: {m: float(np.mean([
        c["effects"][name][m] for c in cases])) for m in METRICS} for name in cases[0]["effects"]})


def aggregate(root):
    rows = [json.loads((root / "ready" / f"{item_name(i)}.json").read_text()) for i in range(len(ITEMS))]
    result = dict(status="deployment_factorial_complete", rollouts=len(rows),
                  diagnostic_cases=len(CASES), stochastic_replicates_per_actor=3,
                  decisions=sum(r["stream_audit"]["decisions"] for r in rows),
                  training_updates=0, main_policy_changed=False,
                  conservation_max_error=max(r["stream_audit"]["conservation_max_error"] for r in rows),
                  **factorial_summary(rows))
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"stream_audit", "action_counts"}} for r in rows
                  ]).to_csv(output / "rollouts.csv", index=False)
    print("V60_AGGREGATE " + json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v56-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
        return
    if args.command == "smoke":
        collect(args.out_root, args.v59_root, args.v56_root, 0)
        output = args.out_root / "smoke_export"
        output.mkdir(exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(status="exact_initial_deployment_pass")) + "\n")
        return
    if not 0 <= args.start < args.end <= len(ITEMS):
        parser.error("batch range is outside the fixed 32-rollout factorial")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                             itertools.repeat(args.v56_root), range(args.start, args.end)))
    output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()

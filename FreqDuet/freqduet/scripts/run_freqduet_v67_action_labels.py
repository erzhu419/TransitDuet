#!/usr/bin/env python3
"""Matched seven-action full-day labels under frozen V59 stochastic actors."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, replace
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import run_freqduet_v66_crossfit_shrinkage as shrink


residual, expanded = shrink.residual, shrink.expanded
warmup, deployment, joint = shrink.warmup, shrink.deployment, shrink.deployment.joint
DEMAND_SEEDS = tuple(range(670001, 670009))
TARGETS = (7200, 21600, 36000)
REFERENCES = list(itertools.product(range(len(deployment.CASES)), range(len(DEMAND_SEEDS))))
CONTEXTS = list(itertools.product(range(len(REFERENCES)), range(len(TARGETS))))
BRANCHES = list(itertools.product(range(len(CONTEXTS)), joint.ACTIONS))
OUTCOMES = ("cv_squared", "passenger_person_s", "journey_min", "waiting_min", "onboard_min",
            "holding_s_per_trip", "holding_total_s", "readiness_delay_s", "backlog_trip_s",
            "trips_unlaunched", "trips_incomplete")
PREDICTORS = ("frozen", "linear_residual", "mlp_residual", "shrunk_linear", "shrunk_mlp")
PREFIX_FIELDS = ("identity", "observation", "encoded_state", "prefix_cv_squared", "actor_action_s",
                 "actor_probabilities", "exposure", "lower_prefix", "upper_prefix", "rng",
                 "policy_sampling", "upper_sampling", "scenario_streams")


def reference_spec(index):
    case, day = REFERENCES[index]
    seed, actor_scenario = deployment.CASES[case]
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=DEMAND_SEEDS[day], sampler_seed=6700000 + case * 1000 + day,
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{DEMAND_SEEDS[day]}")


def context_spec(index):
    reference, target = CONTEXTS[index]
    return dict(reference_spec(reference), reference_index=reference, context_index=index,
                target_time_s=TARGETS[target])


def same_prefix(left, right):
    for field in PREFIX_FIELDS:
        joint.assert_same_state(left[field], right[field], field)


def draw_action(policy, state, forced=None):
    proposed = float(np.asarray(policy.get_action(state, deterministic=False)).item())
    return proposed, proposed if forced is None else float(forced)


def run_branch(root, v59_root, reference_index, targets, label, forced=None, continuation=None):
    return run_spec(root, v59_root, reference_spec(reference_index), targets, label, forced, continuation)


def run_spec(root, v59_root, spec, targets, label, forced=None, continuation=None):
    _, Runner, load_config, _ = joint.optimized_runtime(root)
    source = torch.load(v59_root / "cells" / deployment.case_name(spec["train_seed"], spec["actor_scenario"])
                        / "learning_smoke.pt", map_location="cpu", weights_only=False)
    observer, moments = deployment.CreditObserver(), deployment.ArrivalMoments()
    captures, lower_prefix, upper_path = {}, [], []
    encoder, policy = None, None

    class LabelRunner(Runner):
        def _upper_callback_v2(self, state, trip):
            result = super()._upper_callback_v2(state, trip)
            upper_path.append(dict(time_s=float(self.env.current_time), trip_id=int(trip.launch_turn),
                                   state=np.asarray(state).copy(), action=float(result)))
            return result

        def _lower_policy_action(self, obs, last_action=0.0, deterministic=False):
            nonlocal encoder, policy
            if encoder is None:
                env = self.env
                encoder = deployment.SystemStateEncoder(self.lower_state_dim, env.max_agent_num,
                                                         len(env.timetables), len(env.stations),
                                                         env.protocol.evaluation_end_time_s)
                policy = deployment.make_policy(self.lower_trainer.policy_net, encoder, spec["sampler_seed"], source)
            key, bus = self._label_agent, self._bus_for_agent(self._label_agent)
            self._capture_lower_action_context(key, bus)
            observation = replace(deployment.observable_decision(self, obs, key, last_action),
                                  pending_commands=tuple(observer.stream.pending_commands))
            moments.observe(self.env.headway_events.headways_s)
            encoded = encoder.encode(observation, moments)
            tensor = torch.from_numpy(encoded)[None, :]
            with torch.no_grad():
                probabilities = policy.dist_info(tensor)[0][0].numpy()
                feasible = bool(policy.feasible_action_mask(tensor).all())
            pending = [target for target in targets if target not in captures and observation.time_s >= target]
            eligible = (pending and feasible and not self._fixed_expert_active
                        and not self._lower_terminal_action_masked(bus)
                        and self._pending_lower_action_context[int(key)].evidence_valid)
            target = pending[0] if eligible else None
            proposed, executed = draw_action(policy, encoded, forced if target is not None else None)
            if target is not None:
                captures[target] = dict(
                    identity=(observation.time_s, observation.bus_id, observation.trip_id,
                              observation.station_id, observation.direction),
                    observation=asdict(observation), encoded_state=encoded.copy(), prefix_cv_squared=moments.cv_squared,
                    actor_action_s=proposed, actor_probabilities=probabilities.copy(),
                    exposure=observer.stream.total.copy(), lower_prefix=list(lower_prefix), upper_prefix=copy.deepcopy(upper_path),
                    rng=joint.rng_state(self, torch), policy_sampling=policy.sampling_state().clone(),
                    upper_sampling=self.upper_trainer.policy_net.sampling_state(),
                    scenario_streams=copy.deepcopy({k: r.get_state() for k, r in self.env.scenario_tape._streams.items()}))
                self._label_capture = target
            return np.asarray([executed], dtype=np.float32)

        def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
            self._label_agent, self._label_capture = key, None
            action = super()._lower_action_for_agent(obs, key, last_action, deterministic)
            value = float(np.asarray(action).item())
            observation = deployment.observable_decision(self, obs, key, last_action)
            observer.stream.decide(observation, value)
            if self._label_capture is not None:
                capture = captures[self._label_capture]
                expected = capture["actor_action_s"] if forced is None else forced
                if value != expected:
                    raise AssertionError("requested holding label was changed by deployment projection")
                capture["executed_action_s"] = value
            lower_prefix.append((observation.time_s, int(key), observation.trip_id, observation.station_id, value))
            if self._label_capture is not None and continuation is not None:
                captures[self._label_capture]["continuation"] = continuation(self, policy)
            return action

    joint.seed_runtime(torch, spec["train_seed"])
    config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
    runner = LabelRunner(joint.runtime_config(config, spec["train_seed"], root / "cells" / label / "logs", 40), device="cpu")
    runner.load_checkpoint(joint.TRAIN_ROOT / "logs" / f"{joint.CURRENT}_seed{spec['train_seed']}/checkpoints",
                           ep=joint.BASE_EP, require_deployment_state=True)
    if runner.lower_terminal_action_mode != "transition" or runner.lower_causal_guard_policy_mask_enabled:
        raise ValueError("V67 requires the frozen unmasked V59 action contract")
    upper_before = copy.deepcopy(runner.upper_trainer.policy_net.state_dict())
    dual_before = runner.lower_trainer.log_lambda.detach().clone()
    observer.attach(runner)
    episode = runner.run_episode(joint.BASE_EP, training=False, scenario_seed=spec["scenario_seed"], record_diagnostics=False)
    if set(captures) != set(targets):
        raise ValueError("predeclared eligible holding windows were not observed")
    joint.assert_same_state(source["policy"], policy.state_dict(), "frozen_V59_actor")
    joint.assert_same_state(upper_before, runner.upper_trainer.policy_net.state_dict(), "frozen_upper_actor")
    torch.testing.assert_close(dual_before, runner.lower_trainer.log_lambda, atol=0, rtol=0)
    if len(runner.replay_buffer):
        raise AssertionError("counterfactual replay inserted training data")
    audit = observer.stream.audit()
    moments.observe(runner.env.headway_events.headways_s)
    cv = float(runner.env.headway_events.summary()["headway_cv"])
    np.testing.assert_allclose(moments.cv_squared, cv ** 2, atol=1e-10, rtol=0)
    total = observer.stream.total
    summary = dict(cv=cv, cv_squared=moments.cv_squared, passenger_person_s=float(total[0] + total[1]),
                   passenger_hours=float((total[0] + total[1]) / 3600), backlog_trip_s=float(total[4]),
                   journey_min=float(episode["restricted_total_journey_horizon_min"]), waiting_min=float(episode["restricted_wait_horizon_min"]),
                   onboard_min=float(episode["restricted_in_vehicle_horizon_min"]),
                   holding_s_per_trip=float(episode["holding_vehicle_seconds_per_launched_trip"]),
                   holding_total_s=float(episode["holding_vehicle_seconds"]), readiness_delay_s=float(episode["fleet_readiness_delay_mean_s"]),
                   fleet_denied_trip_rate=float(episode["fleet_denied_trip_rate"]),
                   trips_unlaunched=int(runner.env._measurement_details["trips_unlaunched"]),
                   trips_incomplete=int(runner.env._measurement_details["trips_incomplete"]),
                   action_counts={str(a): sum(h["action_s"] == a for h in observer.holds) for a in joint.ACTIONS},
                   scenario_seed=runner.env.scenario_tape.seed, scenario_version=runner.env.scenario_tape.version,
                   stream_audit=audit)
    if any(h["action_s"] not in joint.ACTIONS for h in observer.holds):
        raise AssertionError("counterfactual emitted an unregistered holding action")
    return dict(summary=summary, captures=captures, holds=observer.holds, upper_path=upper_path)


def reference_path(root, index):
    return root / "references" / (reference_spec(index)["label"] + ".pt")


def reference(root, v59_root, index):
    path = reference_path(root, index)
    if path.exists():
        return path.stem
    spec = reference_spec(index)
    result = run_branch(root, v59_root, index, TARGETS, spec["label"] + "_reference")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print("V67_REFERENCE_DONE " + json.dumps(spec), flush=True)
    return path.stem


def branch(root, v59_root, index):
    context, action = BRANCHES[index]
    spec = context_spec(context)
    label = f"c{context}_a{action}"
    path = root / "labels" / (label + ".json")
    if path.exists():
        return label
    base = torch.load(reference_path(root, spec["reference_index"]), map_location="cpu", weights_only=False)
    result = run_branch(root, v59_root, spec["reference_index"], (spec["target_time_s"],), label, action)
    capture = result["captures"][spec["target_time_s"]]
    original = base["captures"][spec["target_time_s"]]
    same_prefix(original, capture)
    for field in ("scenario_seed", "scenario_version"):
        joint.assert_same_state(base["summary"][field], result["summary"][field], field)
    noop = action == original["actor_action_s"]
    if noop:
        joint.assert_same_state(base["summary"], result["summary"], "noop_day")
        joint.assert_same_state(base["holds"], result["holds"], "noop_complete_holding_path")
        joint.assert_same_state(base["upper_path"], result["upper_path"], "noop_complete_upper_path")
    row = dict(**spec, action_s=action, actor_action_s=original["actor_action_s"],
               **{name: result["summary"][name] for name in OUTCOMES},
               **{f"actor_{name}": base["summary"][name] for name in OUTCOMES},
               exact_prefix=True, noop_matches=noop, conservation_max_error=result["summary"]["stream_audit"]["conservation_max_error"],
               diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n")
    print("V67_BRANCH_DONE " + json.dumps(dict(context_index=context, action_s=action, noop_matches=noop)), flush=True)
    return label


def action_predictions(capture, models, payload, alphas):
    day = dict(states=np.repeat(capture["encoded_state"][None, :], len(joint.ACTIONS), axis=0),
               actions=np.asarray(joint.ACTIONS, dtype=np.float32)[:, None],
               prefix=np.full(len(joint.ACTIONS), capture["prefix_cv_squared"]),
               returns=np.zeros(len(joint.ACTIONS), dtype=np.float32))
    inputs = residual.features(day, payload["encoder"])
    values = {name: warmup.predict(model, inputs).astype(float) for name, model in models.items() if name != "frozen"}
    values["frozen"] = warmup.predict(models["frozen"], day).astype(float)
    values.update(shrunk_linear=alphas["linear_residual"] * values["linear_residual"],
                  shrunk_mlp=alphas["mlp_residual"] * values["mlp_residual"])
    return {name: value - value[0] for name, value in values.items()}


def predictor_metrics(predicted, truth):
    # Zero action is the reference, not an extra easy label in the error average.
    differences = predicted[:, None] - predicted[None, :]
    actual = truth[:, None] - truth[None, :]
    valid = np.triu(np.ones(actual.shape, dtype=bool), 1) & (actual != 0)
    return dict(delta_mae=float(np.abs(predicted[1:] - truth[1:]).mean()),
                delta_rmse=float(np.sqrt(np.square(predicted[1:] - truth[1:]).mean())),
                rank_correct=int((np.sign(differences[valid]) == np.sign(actual[valid])).sum()),
                rank_pairs=int(valid.sum()), tied_truth_pairs=int((np.triu(np.ones(actual.shape, dtype=bool), 1) & (actual == 0)).sum()))


def aggregate(root, v59_root, v66_root):
    joint.optimized_runtime(root)
    labels = [json.loads((root / "labels" / f"c{context}_a{action}.json").read_text()) for context, action in BRANCHES]
    if sum(row["noop_matches"] for row in labels) != len(CONTEXTS):
        raise ValueError("each context must have one exact native-action reproduction")
    contexts, features, values = [], [], {}
    for case in range(len(deployment.CASES)):
        calibration = json.loads(shrink.calibration_path(v66_root, case).read_text())
        payload = torch.load(calibration["base_value_source"], map_location="cpu", weights_only=False)
        models = {}
        for name in residual.CANDIDATES:
            weights = payload["models"][name]
            model = residual.ResidualValue(name, weights["center"], weights["scale"], payload["hidden"])
            model.load_state_dict(weights)
            models[name] = model.eval()
        source = torch.load(v59_root / "cells" / deployment.case_name(*deployment.CASES[case]) / "learning_smoke.pt",
                            map_location="cpu", weights_only=False)
        models["frozen"] = warmup.load_cost_model(source["trainer"]["cost_q_net"]).eval()
        values[case] = (models, payload, {k: v["alpha"] for k, v in calibration["coefficients"].items()})
    effects = []
    for context in range(len(CONTEXTS)):
        spec = context_spec(context)
        selected = [row for row in labels if row["context_index"] == context]
        if [row["action_s"] for row in selected] != list(joint.ACTIONS):
            raise ValueError("counterfactual context lacks the exact seven-action grid")
        base = torch.load(reference_path(root, spec["reference_index"]), map_location="cpu", weights_only=False)
        capture = base["captures"][spec["target_time_s"]]
        predictions = action_predictions(capture, *values[spec["case_index"]])
        truth = np.asarray([row["cv_squared"] - selected[0]["cv_squared"] for row in selected])
        metrics = {name: predictor_metrics(predictions[name], truth) for name in PREDICTORS}
        metrics["zero_delta"] = predictor_metrics(np.zeros_like(truth), truth)
        eligible = [row for row in selected if row["cv_squared"] <= row["actor_cv_squared"]
                    and row["readiness_delay_s"] <= row["actor_readiness_delay_s"]
                    and row["trips_unlaunched"] <= row["actor_trips_unlaunched"]
                    and row["trips_incomplete"] <= row["actor_trips_incomplete"]]
        best = min(eligible, key=lambda row: row["passenger_person_s"])
        contexts.append(dict(**spec, actor_action_s=capture["actor_action_s"],
                             prefix_cv_squared=capture["prefix_cv_squared"], **metrics,
                             cost_range=float(np.ptp(truth)), oracle_person_s=best["passenger_person_s"] - best["actor_passenger_person_s"]))
        features.append(capture["encoded_state"])
        for row in selected:
            effect = dict(row)
            for name in OUTCOMES:
                effect[f"delta_actor_{name}"] = row[name] - row[f"actor_{name}"]
                effect[f"delta_zero_{name}"] = row[name] - selected[0][name]
            effect["delta_actor_reward"] = -effect["delta_actor_passenger_person_s"] / (3600 * 1000)
            effects.append(effect)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    pd.DataFrame(effects).to_csv(root / "labels" / "action_effects.csv", index=False)
    np.savez_compressed(root / "labels" / "observable_contexts.npz", states=np.asarray(features, dtype=np.float32),
                        prefix_cv_squared=np.asarray([c["prefix_cv_squared"] for c in contexts]),
                        context_indices=np.arange(len(CONTEXTS)), demand_seeds=np.asarray([c["scenario_seed"] for c in contexts]))
    (root / "labels" / "predictor_audit.json").write_text(json.dumps(contexts, indent=2) + "\n")
    contrasts = {}
    for i, name in enumerate(PREDICTORS):
        contrasts[name] = {}
        for metric in ("delta_mae", "delta_rmse"):
            deltas = [float(np.mean([c[name][metric] - c["zero_delta"][metric] for c in contexts if c["scenario_seed"] == seed]))
                      for seed in DEMAND_SEEDS]
            contrasts[name][metric] = expanded.paired_interval(deltas, 67000 + i)
    status = dict(status="matched_seven_action_labels_complete", references=len(REFERENCES), contexts=len(CONTEXTS),
                  action_branches=len(BRANCHES), formal_rollouts=len(REFERENCES) + len(BRANCHES), demand_days=len(DEMAND_SEEDS),
                  target_times_s=TARGETS, exact_prefixes=True, noop_matches=sum(row["noop_matches"] for row in labels),
                  conservation_max_error=max(row["conservation_max_error"] for row in labels),
                  mean_cost_range=float(np.mean([c["cost_range"] for c in contexts])),
                  mean_oracle_person_s=float(np.mean([c["oracle_person_s"] for c in contexts])),
                  oracle_improving_contexts=sum(c["oracle_person_s"] < 0 for c in contexts),
                  predictor_metrics={name: dict(delta_mae=float(np.mean([c[name]["delta_mae"] for c in contexts])),
                                                delta_rmse=float(np.mean([c[name]["delta_rmse"] for c in contexts])),
                                                rank_correct=sum(c[name]["rank_correct"] for c in contexts),
                                                rank_pairs=sum(c[name]["rank_pairs"] for c in contexts))
                                     for name in (*PREDICTORS, "zero_delta")}, paired_vs_zero=contrasts,
                  actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False,
                  hindsight_only=True, control_admitted=False, independent_controller_confirmation=False)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("V67_AGGREGATE " + json.dumps(status), flush=True)


def smoke(root, v59_root, workers):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v6[0-7]_*.py", "-v"], check=True)
    reference(root, v59_root, 0)
    base = torch.load(reference_path(root, 0), map_location="cpu", weights_only=False)
    spec = reference_spec(0)
    original = deployment.ITEMS.index((0, ("updated", "stochastic", 0)))
    label = deployment.collect(root, v59_root, None, original, scenario_seed=spec["scenario_seed"], private_sampler_seed=spec["sampler_seed"])
    old = json.loads((root / "ready" / f"{label}.json").read_text())
    for name in (*deployment.METRICS, "action_counts", "stream_audit", "trips_unlaunched", "trips_incomplete"):
        joint.assert_same_state(base["summary"][name], old[name], "native_V60_" + name)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(branch, itertools.repeat(root), itertools.repeat(v59_root), range(len(joint.ACTIONS))))
    output = root / "smoke_export"
    output.mkdir(exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(status="native_V60_and_seven_action_noop_smoke_pass", rollouts=9)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "reference-batch", "branch-batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v66-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v59_root, args.workers)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.v59_root, args.v66_root)
    else:
        reference_mode = args.command == "reference-batch"
        items, function = (REFERENCES, reference) if reference_mode else (BRANCHES, branch)
        end = len(items) if args.end is None else args.end
        if not 0 <= args.start < end <= len(items):
            parser.error("batch range is outside the fixed counterfactual design")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(function, itertools.repeat(args.out_root), itertools.repeat(args.v59_root), range(args.start, end)))
        output = args.out_root / ("reference_exports" if reference_mode else "branch_exports") / f"batch_{args.start}_{end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=len(done), start=args.start, end=end)) + "\n")


if __name__ == "__main__":
    main()

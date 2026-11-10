#!/usr/bin/env python3
"""Read-only full-day cost-critic calibration and episodic-dual timing audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_freqduet_v60_deployment_factorial as deployment


ITEMS = [i for i, (_, (actor, mode, _)) in enumerate(deployment.ITEMS)
         if (actor, mode) == ("updated", "stochastic")]


def remaining_cost(prefix, final):
    prefix = np.asarray(prefix, dtype=float)
    increments = np.diff(np.r_[prefix, float(final)])
    returns = float(final) - prefix
    np.testing.assert_allclose(np.cumsum(increments[::-1])[::-1], returns, rtol=0, atol=1e-12)
    return increments, returns


def calibration_metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    error = predicted - actual
    return dict(count=int(len(actual)), predicted_mean=float(predicted.mean()), actual_mean=float(actual.mean()),
                bias=float(error.mean()), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.square(error).mean())),
                zero_mae=float(np.abs(actual).mean()), zero_rmse=float(np.sqrt(np.square(actual).mean())),
                actual_negative_fraction=float((actual < 0).mean()))


def dual_timing(marker, updates, training_state):
    block = marker["updates_per_day"]
    if len(updates) != marker["updates"] or len(updates) != len(marker["episodes"]) * block:
        raise ValueError("V59 optimizer log does not match its episode/update contract")
    positive, same_day_response = [], 0
    for i, day in enumerate(marker["episodes"]):
        logged = [u["lambda_value"] for u in updates[i * block:(i + 1) * block]]
        if day["cv_squared"] > day["regularity_budget_cv_squared"]:
            positive.append(dict(day=i, gap=day["cv_squared"] - day["regularity_budget_cv_squared"],
                                 lambda_before=day["lambda_before"], lambda_after=day["lambda_after"]))
            same_day_response += sum(value == day["lambda_after"] for value in logged)
    first_after = positive[0]["lambda_after"] if positive else None
    steps = lambda name: sorted({int(v["step"]) for v in training_state[name]["state"].values()})
    return dict(actor_updates=len(updates), dual_updates=steps("lambda_optimizer"),
                actor_optimizer_steps=steps("policy_optimizer"),
                actor_learning_rate=training_state["policy_optimizer"]["param_groups"][0]["lr"],
                dual_learning_rate=training_state["lambda_optimizer"]["param_groups"][0]["lr"],
                positive_violation_days=positive, same_day_updates_using_post_violation_lambda=same_day_response,
                actor_updates_using_first_positive_dual=(sum(u["lambda_value"] >= first_after for u in updates)
                                                        if first_after is not None else 0),
                actor_update_lambda_values=sorted({u["lambda_value"] for u in updates}),
                checkpoint_lambda=float(training_state["log_lambda"].exp().item()))


class CostAudit:
    def __init__(self, root, expected, checkpoint):
        from lower.dsac_lagrangian import CostQNetwork
        self.root, self.expected = root, expected
        self.models = []
        for name in ("cost_q_net", "target_cost_q_net"):
            weights = checkpoint["trainer"][name]
            hidden, inputs = weights["net.0.weight"].shape
            model = CostQNetwork(inputs - 1, 1, hidden_dim=hidden)
            model.load_state_dict(weights)
            self.models.append(model.eval())
        self.events, self.prefix, self.predicted, self.state_value, self.target_value = [], [], [], [], []

    def decision(self, encoded, state, prefix, action, probabilities):
        states = torch.from_numpy(encoded)[None, :].expand(len(deployment.joint.ACTIONS), -1)
        actions = torch.tensor(deployment.joint.ACTIONS, dtype=torch.float32)[:, None]
        with torch.no_grad():
            current, target = [model(states, actions).flatten().numpy() for model in self.models]
        self.events.append((state.time_s, state.bus_id, state.trip_id, state.station_id, action))
        self.prefix.append(prefix)
        self.predicted.append(float(current[deployment.joint.ACTIONS.index(action)]))
        self.state_value.append(float(np.dot(probabilities, current)))
        self.target_value.append(float(np.dot(probabilities, target)))

    def finish(self, row, stream):
        fields = (*deployment.METRICS, "action_counts", "entropy_mean", "expected_hold_mean_s",
                  "zero_probability_mean", "fleet_denied_trip_rate", "trips_unlaunched", "trips_incomplete")
        deployment.joint.assert_same_state({k: row[k] for k in fields},
                                           {k: self.expected[k] for k in fields}, "V60_exact_replay")
        executed = [(t.state.time_s, t.state.bus_id, t.state.trip_id, t.state.station_id, t.action_s)
                    for t in stream.transitions]
        if self.events != executed:
            raise AssertionError("cost predictions do not align with the physically executed credit stream")
        increments, truth = remaining_cost(self.prefix, row["cv_squared"])
        predicted, state_value = np.asarray(self.predicted), np.asarray(self.state_value)
        next_target = np.r_[self.target_value[1:], 0.0]
        bellman_target = increments + next_target
        times = np.asarray([e[0] for e in self.events])
        thirds = np.minimum((3 * times / stream.clock_s).astype(int), 2)
        result = dict(selected=calibration_metrics(predicted, truth),
                      policy_value=calibration_metrics(state_value, truth),
                      one_step_target=calibration_metrics(predicted, bellman_target),
                      thirds={name: calibration_metrics(predicted[thirds == i], truth[thirds == i])
                              for i, name in enumerate(("early", "middle", "late")) if (thirds == i).any()},
                      final_decision=calibration_metrics(predicted[-1:], truth[-1:]),
                      exact_v60_replay=True, action_labels_only=True)
        records = self.root / "records"
        records.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(time_s=times, action_s=[e[4] for e in self.events],
                          cv_prefix=self.prefix, mc_remaining_cv_squared=truth,
                          cost_q=predicted, policy_cost_value=state_value,
                          next_target_value=next_target, immediate_cost=increments,
                          one_step_target=bellman_target)).to_csv(records / f"{row['label']}.csv", index=False)
        return result


def collect(root, v59_root, v60_root, index):
    original_index = ITEMS[index]
    case = deployment.ITEMS[original_index][0]
    seed, scenario = deployment.CASES[case]
    label = deployment.item_name(original_index)
    marker = root / "ready" / f"{label}.json"
    if marker.exists():
        return label
    deployment.joint.optimized_runtime(root)
    cell = v59_root / "cells" / deployment.case_name(seed, scenario)
    checkpoint = torch.load(cell / "learning_smoke.pt", map_location="cpu", weights_only=False)
    expected = json.loads((v60_root / "ready" / f"{label}.json").read_text())
    audit = CostAudit(root, expected, checkpoint)
    return deployment.collect(root, v59_root, None, original_index, decision_audit=audit)


def aggregate(root, v59_root):
    days = [json.loads((root / "ready" / f"{deployment.item_name(i)}.json").read_text()) for i in ITEMS]
    contexts, timing = [], []
    for seed, scenario in deployment.CASES:
        selected = [d for d in days if (d["train_seed"], d["scenario_seed"]) == (seed, scenario)]
        means = {channel: {metric: float(np.mean([d["decision_audit"][channel][metric] for d in selected]))
                           for metric in ("bias", "mae", "rmse", "zero_mae", "zero_rmse", "predicted_mean", "actual_mean")}
                 for channel in ("selected", "policy_value", "one_step_target")}
        contexts.append(dict(train_seed=seed, scenario_seed=scenario, means=means,
                             days_worse_than_zero=sum(d["decision_audit"]["selected"]["mae"]
                                                     > d["decision_audit"]["selected"]["zero_mae"] for d in selected)))
        cell = v59_root / "cells" / deployment.case_name(seed, scenario)
        checkpoint = torch.load(cell / "learning_smoke.pt", map_location="cpu", weights_only=False)
        marker = json.loads((v59_root / "exports" / deployment.case_name(seed, scenario) / "done.json").read_text())
        updates = json.loads((cell / "updates.json").read_text())
        timing.append(dict(train_seed=seed, scenario_seed=scenario, **dual_timing(marker, updates, checkpoint["trainer"])))
    result = dict(status="cost_calibration_complete", contexts=contexts, dual_timing=timing,
                  rollouts=len(days), decisions=sum(d["decision_audit"]["selected"]["count"] for d in days),
                  exact_v60_replays=all(d["decision_audit"]["exact_v60_replay"] for d in days),
                  training_updates=0, main_policy_changed=False,
                  mean_metrics={channel: {m: float(np.mean([c["means"][channel][m] for c in contexts]))
                                         for m in contexts[0]["means"][channel]}
                                for channel in contexts[0]["means"]},
                  days_worse_than_zero=sum(c["days_worse_than_zero"] for c in contexts))
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    (summary / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    (summary / "day_calibration.json").write_text(json.dumps([
        dict(label=d["label"], **d["decision_audit"]) for d in days], indent=2) + "\n")
    print("V61_AGGREGATE " + json.dumps({k: v for k, v in result.items() if k != "contexts"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path, required=True)
    parser.add_argument("--v60-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root, args.v59_root)
        return
    if args.command == "smoke":
        collect(args.out_root, args.v59_root, args.v60_root, 0)
        output = args.out_root / "smoke_export"
        output.mkdir(exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(status="cost_audit_exact_replay_pass")) + "\n")
        return
    if not 0 <= args.start < args.end <= len(ITEMS):
        parser.error("batch range is outside the fixed 12-rollout cost audit")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                             itertools.repeat(args.v60_root), range(args.start, args.end)))
    output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen-actor full-return cost fitting with complete-demand-day validation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import torch

import run_freqduet_v60_deployment_factorial as deployment
from run_freqduet_v61_cost_calibration import calibration_metrics, remaining_cost


TRAIN_DAYS = ((45007, 0), (45007, 1), (934011, 0), (934011, 1))
VALIDATION_DAYS = ((45107, 2), (934111, 2))
DAYS = TRAIN_DAYS + VALIDATION_DAYS
ITEMS = list(itertools.product(range(len(deployment.CASES)), range(len(DAYS))))
FIT_STEPS, FIT_BATCH = 2048, 256


def specification(index):
    case, day = ITEMS[index]
    seed, actor_scenario = deployment.CASES[case]
    scenario, rep = DAYS[day]
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=scenario, rep=rep, split="train" if day < len(TRAIN_DAYS) else "validation",
                label=f"{deployment.case_name(seed, actor_scenario)}_demand{scenario}_r{rep}")


def data_indices(case, split):
    return [i for i in range(len(ITEMS)) if specification(i)["case_index"] == case
            and specification(i)["split"] == split]


def calibration_gate(days):
    return all(day["fitted"][metric] < day[baseline][metric]
               for day in days for baseline in ("frozen", "zero", "causal_final_mean")
               for metric in ("mae", "rmse"))


def load_cost_model(weights):
    from lower.dsac_lagrangian import CostQNetwork
    hidden, inputs = weights["net.0.weight"].shape
    model = CostQNetwork(inputs - 1, 1, hidden_dim=hidden)
    model.load_state_dict(weights)
    return model


def predict(model, day):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(day["states"][i:i + 1024]),
                                     torch.from_numpy(day["actions"][i:i + 1024])).flatten().numpy()
                               for i in range(0, len(day["returns"]), 1024)])


def fit_cost(model, days, lr, seed, steps=FIT_STEPS, batch_size=FIT_BATCH):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tensors = [(torch.from_numpy(d["states"]), torch.from_numpy(d["actions"]),
                torch.as_tensor(d["returns"], dtype=torch.float32)) for d in days]
    losses = []
    model.train()
    for _ in range(steps):
        states, actions, returns = tensors[int(torch.randint(len(tensors), (), generator=generator))]
        indices = torch.randint(len(returns), (batch_size,), generator=generator)
        loss = (model(states[indices], actions[indices]).flatten() - returns[indices]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite full-return cost loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return dict(steps=steps, batch_size=batch_size, learning_rate=lr,
                first_block_mse=float(np.mean(losses[:min(128, steps)])),
                last_block_mse=float(np.mean(losses[-min(128, steps):])))


class FullDayData:
    def __init__(self, root, spec, expected):
        self.root, self.spec, self.expected = root, spec, expected
        self.states, self.events, self.prefix = [], [], []

    def decision(self, encoded, state, prefix, action, probabilities):
        self.states.append(encoded.copy())
        self.events.append((state.time_s, state.bus_id, state.trip_id, state.station_id, action))
        self.prefix.append(prefix)

    def finish(self, row, stream):
        executed = [(t.state.time_s, t.state.bus_id, t.state.trip_id, t.state.station_id, t.action_s)
                    for t in stream.transitions]
        if self.events != executed:
            raise AssertionError("full-return training states do not align with executed actions")
        if self.expected is not None:
            fields = (*deployment.METRICS, "action_counts", "entropy_mean", "expected_hold_mean_s",
                      "zero_probability_mean", "fleet_denied_trip_rate", "trips_unlaunched", "trips_incomplete")
            deployment.joint.assert_same_state({k: row[k] for k in fields},
                                               {k: self.expected[k] for k in fields}, "V62_exact_V60_replay")
        _, truth = remaining_cost(self.prefix, row["cv_squared"])
        output = self.root / "datasets"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{self.spec['label']}.npz"
        np.savez_compressed(path, states=np.asarray(self.states, dtype=np.float32),
                            actions=np.asarray([e[4] for e in self.events], dtype=np.float32)[:, None],
                            returns=truth, prefix=np.asarray(self.prefix),
                            final_cv_squared=row["cv_squared"], time_s=np.asarray([e[0] for e in self.events]))
        return dict(dataset=str(path), decisions=len(truth),
                    exact_v60_replay=self.expected is not None, signed_returns=True)


def collect(root, v59_root, v60_root, index):
    spec = specification(index)
    marker = root / "ready" / f"{spec['label']}.json"
    if marker.exists():
        return spec["label"]
    original = deployment.ITEMS.index((spec["case_index"], ("updated", "stochastic", spec["rep"])))
    expected = None
    if spec["scenario_seed"] == spec["actor_scenario"]:
        expected = json.loads((v60_root / "ready" / f"{deployment.item_name(original)}.json").read_text())
    audit = FullDayData(root, spec, expected)
    label = deployment.collect(root, v59_root, None, original, decision_audit=audit,
                               scenario_seed=spec["scenario_seed"],
                               private_sampler_seed=deployment.sampler_seed(spec["train_seed"], spec["rep"]))
    row = json.loads((root / "ready" / f"{label}.json").read_text())
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(dict(**spec, **row["decision_audit"], final_cv_squared=row["cv_squared"],
                                      training_updates=0, diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")),
                                 indent=2) + "\n")
    print("V62_DATA_DONE " + json.dumps(spec), flush=True)
    return spec["label"]


def load_days(root, case, split):
    days = []
    for index in data_indices(case, split):
        spec = specification(index)
        marker = json.loads((root / "ready" / f"{spec['label']}.json").read_text())
        with np.load(marker["dataset"]) as source:
            days.append(dict(label=spec["label"], **{name: source[name] for name in source.files}))
    return days


def fit(root, v59_root, case):
    deployment.joint.optimized_runtime(root)
    seed, scenario = deployment.CASES[case]
    cell = v59_root / "cells" / deployment.case_name(seed, scenario)
    checkpoint = torch.load(cell / "learning_smoke.pt", map_location="cpu", weights_only=False)
    weights = checkpoint["trainer"]["cost_q_net"]
    original, model = load_cost_model(weights), load_cost_model(weights)
    train = load_days(root, case, "train")
    lr = checkpoint["trainer"]["cost_q_optimizer"]["param_groups"][0]["lr"]
    training = fit_cost(model, train, lr, seed + scenario + 62000)
    destination = root / "fits" / deployment.case_name(seed, scenario)
    destination.mkdir(parents=True, exist_ok=True)
    # Save the fixed-budget fit before loading any validation labels.
    torch.save(dict(cost_q_net=model.state_dict(), target_cost_q_net=model.state_dict(),
                    source_checkpoint=str(cell / "learning_smoke.pt"), **training), destination / "cost_warmup.pt")
    final_mean = float(np.mean([d["final_cv_squared"] for d in train]))
    validation = []
    for day in load_days(root, case, "validation"):
        truth = day["returns"]
        predictions = dict(fitted=predict(model, day), frozen=predict(original, day),
                           zero=np.zeros_like(truth), causal_final_mean=final_mean - day["prefix"])
        validation.append(dict(label=day["label"], **{name: calibration_metrics(values, truth)
                                                      for name, values in predictions.items()}))
    result = dict(case_index=case, train_seed=seed, actor_scenario=scenario, training=training,
                  training_days=[d["label"] for d in train], train_final_cv_squared_mean=final_mean,
                  validation=validation, gate_pass=calibration_gate(validation),
                  actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (destination / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V62_FIT_DONE " + json.dumps({k: v for k, v in result.items() if k not in ("validation", "training_days")}), flush=True)


def aggregate(root):
    fits = [json.loads((root / "fits" / deployment.case_name(*case) / "done.json").read_text())
            for case in deployment.CASES]
    days = [json.loads((root / "ready" / f"{specification(i)['label']}.json").read_text()) for i in range(len(ITEMS))]
    validation = [day for case in fits for day in case["validation"]]
    result = dict(status="critic_full_return_warmup_complete", rollouts=len(days),
                  decisions=sum(d["decisions"] for d in days), exact_v60_replays=sum(d["exact_v60_replay"] for d in days),
                  passed_cases=sum(f["gate_pass"] for f in fits), control_admitted=all(f["gate_pass"] for f in fits),
                  validation_days=len(validation), actor_updates=0, dual_updates=0, main_policy_changed=False,
                  mean_metrics={name: {metric: float(np.mean([d[name][metric] for d in validation]))
                                       for metric in ("mae", "rmse", "bias")}
                                for name in ("fitted", "frozen", "zero", "causal_final_mean")}, cases=fits)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V62_AGGREGATE " + json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "fit", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v60-root", type=Path)
    parser.add_argument("--case-index", type=int, choices=range(len(deployment.CASES)), default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command == "fit":
        fit(args.out_root, args.v59_root, args.case_index)
    elif args.command == "smoke":
        collect(args.out_root, args.v59_root, args.v60_root, 0)
        output = args.out_root / "smoke_export"
        output.mkdir(exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(status="full_return_data_exact_smoke_pass")) + "\n")
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("range is outside the fixed 24-day collection")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(collect, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 itertools.repeat(args.v60_root), range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross frozen early/late mean action preferences and state-dependent logits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v78_system_policy_step as pilot


CAL_DAYS = tuple(range(1870001, 1870017))
EVAL_DAYS = tuple(range(1871001, 1871017))
CAL_ITEMS = list(itertools.product(pilot.SEEDS, CAL_DAYS))
ITEMS = list(itertools.product(pilot.SEEDS, EVAL_DAYS, range(3)))
POLICIES = ("late", "early", "late_earlymean", "early_latemean")
CONTRASTS = dict(early_mean_at_late_feedback=("late_earlymean", "late"),
    early_mean_at_early_feedback=("early", "early_latemean"),
    late_feedback_at_early_mean=("late_earlymean", "early"),
    late_feedback_at_late_mean=("late", "early_latemean"),
    early_vs_late=("early", "late"))
METRICS = (*pilot.METRICS, "waiting_person_s", "onboard_person_s", "backlog_trip_s", "holding_person_s")


def centered(logits):
    return logits - logits.mean(axis=-1, keepdims=True)


def mean_shift(days):
    # Each independent day has equal weight, not each visited decision.
    return centered(np.mean([d["logit_delta_mean"] for d in days], axis=0))


def factorial(values):
    result = {name: values[left] - values[right] for name, (left, right) in CONTRASTS.items()}
    result["interaction"] = result["early_mean_at_late_feedback"] - result["early_mean_at_early_feedback"]
    return result


def distribution(late, early, feasible, shift, label):
    logits = early if label in ("early", "early_latemean") else late
    if label in ("late_earlymean", "early_latemean"):
        sign = 1. if label == "late_earlymean" else -1.
        logits = logits + sign * torch.as_tensor(shift, dtype=logits.dtype)
    return torch.softmax(logits.masked_fill(~feasible, torch.finfo(logits.dtype).min), -1)


class LogitCross:
    def __init__(self, seed, label, shift):
        self.seed, self.label, self.shift = seed, label, shift
        self.early, self.count = None, 0
        self.delta_sum = np.zeros(7)
        self.expected_hold_sum = np.zeros(2)

    def attach(self, runner):
        self.runner = runner
        self.old = copy.deepcopy(runner.lower_trainer.policy_net)
        path = pilot.joint.TRAIN_ROOT / "logs" / f"{pilot.joint.CURRENT}_seed{self.seed}/checkpoints/lower_ep39.pt"
        self.old.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["policy"])

    def __call__(self, policy, states, features, theta):
        if self.early is None:
            env = self.runner.env
            encoder = pilot.SystemStateEncoder(self.runner.lower_state_dim, env.max_agent_num,
                len(env.timetables), len(env.stations), env.protocol.evaluation_end_time_s)
            # Constructing the diagnostic copy must not shift any existing sampler.
            with torch.random.fork_rng(devices=[]):
                self.early = pilot.deployment.make_policy(self.old, encoder, 0)
        with torch.no_grad():
            late_z, early_z = policy.forward(states), self.early.forward(states)
            feasible = policy.feasible_action_mask(states)
            self.delta_sum += centered(early_z.numpy().astype(float))[0] - centered(late_z.numpy().astype(float))[0]
            for i, label in enumerate(("late", "early")):
                p = distribution(late_z, early_z, feasible, self.shift, label)
                self.expected_hold_sum[i] += float(p[0].numpy() @ pilot.step.ACTIONS)
            result = distribution(late_z, early_z, feasible, self.shift, self.label)
        self.count += 1
        return result

    def finish(self, runner, records, holds):
        self.holding_person_s = float(sum(b.episode_hold_person_seconds for b in runner.env.bus_all))
        self.runner = None

    def detach(self):
        pass


def replay(root, seed, day, rep, label, shift):
    rule = LogitCross(seed, label, shift)
    credit = pilot.CreditObserver()
    sampler = 187000000 + seed + 100 * day + rep
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", sampler, [0.] * 4,
        f"seed{seed}_day{day}_r{rep}_{label}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),
        holding_person_s=rule.holding_person_s)
    row.update(logit_delta_mean=(rule.delta_sum / rule.count).tolist(), decisions=rule.count,
        expected_hold_on_visited_states_s=(rule.expected_hold_sum / rule.count).tolist())
    return row, events, holds


def calibrate(root, index):
    path = root / "calibration" / f"c{index}.json"
    if path.exists():
        return
    seed, day = CAL_ITEMS[index]
    row, _, _ = replay(root, seed, day, 0, "late", np.zeros(7))
    pilot.write_json(path, row)
    print("V87_CALIBRATION_DONE " + str(index), flush=True)


def freeze_means(root):
    for seed in pilot.SEEDS:
        days = [json.loads((root / "calibration" / f"c{i}.json").read_text())
                for i, (s, _) in enumerate(CAL_ITEMS) if s == seed]
        result = dict(seed=seed, shift=mean_shift(days).tolist(), calibration_days=list(CAL_DAYS),
            equal_day_weight=True, outcome_labels_used=False, actor_updates=0,
            mean_expected_hold_on_late_states_s=np.mean([d["expected_hold_on_visited_states_s"] for d in days], axis=0).tolist())
        pilot.write_json(root / "adapters" / f"seed{seed}.json", result)
        print("V87_MEANS_FROZEN " + json.dumps(result), flush=True)
    pilot.write_json(root / "means_ready/done.json", dict(status="calibration_means_frozen"))


def evaluate(root, index):
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return
    seed, day, rep = ITEMS[index]
    shift = json.loads((root / "adapters" / f"seed{seed}.json").read_text())["shift"]
    rows, events = {}, {}
    for label in POLICIES:
        row, event, _ = replay(root, seed, day, rep, label, shift)
        rows[label], events[label] = row, event
    effects = {name: {} for name in (*CONTRASTS, "interaction")}
    for metric in METRICS:
        for name, delta in factorial({label: r[metric] for label, r in rows.items()}).items():
            effects[name][metric] = delta
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, results=rows, effects=effects,
        alignment={label: pilot.event.alignment(events["late"], e) for label, e in events.items() if label != "late"}))
    print("V87_FACTORIAL_CELL_DONE " + str(index), flush=True)


def intervals(rows):
    result = {}
    for contrast in (*CONTRASTS, "interaction"):
        result[contrast] = {}
        for metric in METRICS:
            values = [float(np.mean([r["effects"][contrast][metric] for r in rows if r["day"] == day])) for day in EVAL_DAYS]
            summary = pilot.event.labels.expanded.paired_interval(values, 87000)
            result[contrast][metric] = {k: summary[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    result = dict(status="frozen_logit_factorial_complete", effects=intervals(rows),
        by_seed={str(seed): intervals([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        empirical_logit_means={str(seed): json.loads((root / "adapters" / f"seed{seed}.json").read_text()) for seed in pilot.SEEDS},
        evaluation_rollouts=4 * len(rows), calibration_rollouts=len(CAL_ITEMS),
        inference="paired demand days conditional on two original actors; not training-seed confirmation",
        scope="mean centered logits, not guaranteed identical marginal holding or probability response derivatives",
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False)
    pilot.write_json(root / "summary/status.json", result)
    print("V87_FACTORIAL_COMPLETE " + json.dumps(result["effects"]), flush=True)


def smoke(root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v87_*.py", "-v"], check=True)
    shift = None
    for seed in pilot.SEEDS:
        day = 1879901
        baseline = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day,
                                [0.] * 4, f"smoke_native_seed{seed}")
        native = replay(root, seed, day, 0, "late", np.zeros(7))
        pilot.joint.assert_same_state({m: baseline[0][m] for m in pilot.METRICS},
                                     {m: native[0][m] for m in pilot.METRICS}, "v87_native_metrics")
        pilot.joint.assert_same_state(baseline[1], native[1], "v87_native_event_path")
        pilot.joint.assert_same_state(baseline[3], native[2], "v87_native_physical_holding")
        if seed == pilot.SEEDS[0]:
            shift = native[0]["logit_delta_mean"]
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(replay, root, pilot.SEEDS[0], 1879901, 0, label, shift) for label in POLICIES[1:]]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="native_paths_and_four_logit_corners_passed",
        calibration_or_evaluation_days_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "calibrate", "freeze-means", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root)
    elif args.command == "freeze-means":
        freeze_means(args.out_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        function = calibrate if args.command == "calibrate" else evaluate
        end = args.end if args.end is not None else len(CAL_ITEMS if args.command == "calibrate" else ITEMS)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(function, args.out_root, i) for i in range(args.start, end)]:
                task.result()
        pilot.write_json(args.out_root / f"{args.command}_shards/{args.start}-{end}.json", dict(status="complete"))


if __name__ == "__main__":
    main()

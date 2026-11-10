#!/usr/bin/env python3
"""Prospective closed-loop test of V88's two observable extra-holding groups."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v88_response_context as audit


experiment, pilot = audit.experiment, audit.pilot
DAYS = tuple(range(1891001, 1891017))
ITEMS = list(itertools.product(pilot.SEEDS, DAYS, range(3)))
MODES = ("late", "no_gain_early", "gain_early", "both_early", "early")
CONTRASTS = dict(no_gain_vs_late=("no_gain_early", "late"), gain_vs_late=("gain_early", "late"),
    both_vs_late=("both_early", "late"), no_gain_with_gain=("both_early", "gain_early"),
    gain_with_no_gain=("both_early", "no_gain_early"), early_vs_late=("early", "late"),
    both_vs_early=("both_early", "early"))
RISK_KEYS = list(itertools.product(audit.GROUPS, (False, True), (False, True), (False, True)))


def use_early(mode, group):
    return (mode == "early" or
        (mode in ("no_gain_early", "both_early") and group == "more_hold_gain_not_up") or
        (mode in ("gain_early", "both_early") and group == "more_hold_gain_up"))


def effects(values):
    result = {name: values[a] - values[b] for name, (a, b) in CONTRASTS.items()}
    result["interaction"] = values["both_early"] - values["no_gain_early"] - values["gain_early"] + values["late"]
    return result


class ConditionalCross(experiment.LogitCross):
    def __init__(self, seed, mode):
        super().__init__(seed, "early" if mode == "early" else "late", np.zeros(7))
        self.mode, self.early_decisions = mode, 0
        self.selected_groups = {g: 0 for g in audit.GROUPS}
        self.risks = {k: np.zeros(2) for k in RISK_KEYS}
        self.max_selected_holding_increase = self.predicted_gain_change = 0.

    def __call__(self, policy, states, features, theta):
        native = super().__call__(policy, states, features, theta)
        with torch.no_grad():
            late_z, early_z = policy.forward(states), self.early.forward(states)
            mask = policy.feasible_action_mask(states)
            laws = [experiment.distribution(late_z, early_z, mask, self.shift, label) for label in audit.SOURCES]
            late, early = [law[0].numpy().astype(float) for law in laws]
        delta = float((late - early) @ pilot.step.ACTIONS)
        gain = float((late - early) @ features[:, 0])
        runner = self.runner
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        bus = runner._bus_for_agent(runner._pilot_agent)
        group = audit.context_group(context.evidence_valid and context.follower_evidence_valid, delta, gain)
        key = (group, len(bus.passengers) > 0, -features[-1, 2] > 0, -features[-1, 3] > 0)
        self.risks[key] += [1., delta]
        selected = use_early(self.mode, group)
        if selected:
            self.early_decisions += 1
            self.selected_groups[group] += 1
            self.max_selected_holding_increase = max(self.max_selected_holding_increase, -delta)
            self.predicted_gain_change -= gain
        return laws[1] if selected and self.mode != "early" else native

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.context_summary = dict(decisions=self.count, early_decisions=self.early_decisions,
            selected_groups=self.selected_groups, max_selected_holding_increase_s=self.max_selected_holding_increase,
            selected_predicted_gain_change_sum=self.predicted_gain_change,
            risk_bins=[dict(group=g, passengers_present=loaded, current_pressure_present=pressure,
                previous_holding_commanded=spent, count=int(self.risks[g, loaded, pressure, spent][0]),
                signed_extra_holding_contribution_s=float(self.risks[g, loaded, pressure, spent][1] / self.count))
                for g, loaded, pressure, spent in RISK_KEYS])


def replay(root, seed, day, rep, mode):
    rule, credit = ConditionalCross(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v89_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s)
    if mode == "no_gain_early":
        if rule.max_selected_holding_increase > 1e-10 or rule.predicted_gain_change < -1e-10:
            raise AssertionError("primary intervention violated its instantaneous holding/gain definition")
    return row, events, holds, rule.context_summary


def run_cell(root, index):
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return
    seed, day, rep = ITEMS[index]
    rows, contexts, events = {}, {}, {}
    for mode in MODES:
        row, event, _, context = replay(root, seed, day, rep, mode)
        rows[mode], events[mode], contexts[mode] = row, event, context
    contrasts = {name: {} for name in (*CONTRASTS, "interaction")}
    for metric in experiment.METRICS:
        for name, value in effects({mode: r[metric] for mode, r in rows.items()}).items():
            contrasts[name][metric] = value
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, results=rows, contexts=contexts,
        effects=contrasts, alignment={mode: pilot.event.alignment(events["late"], event)
            for mode, event in events.items() if mode != "late"}))
    print("V89_CONDITIONAL_CELL_DONE " + str(index), flush=True)


def intervals(rows):
    result = {}
    for contrast in (*CONTRASTS, "interaction"):
        result[contrast] = {}
        for metric in experiment.METRICS:
            values = [float(np.mean([r["effects"][contrast][metric] for r in rows if r["day"] == day])) for day in DAYS]
            stats = pilot.event.labels.expanded.paired_interval(values, 89000)
            result[contrast][metric] = {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    stats = intervals(rows)
    primary = stats["no_gain_vs_late"]
    joint = primary["passenger_person_s"]["ci95"][1] < 0 and all(primary[m]["ci95"][1] <= 0 for m in
        ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete"))
    risk = []
    for i, (group, loaded, pressure, spent) in enumerate(RISK_KEYS):
        counts = [r["contexts"]["late"]["risk_bins"][i] for r in rows]
        risk.append(dict(group=group, passengers_present=loaded, current_pressure_present=pressure,
            previous_holding_commanded=spent, count=sum(c["count"] for c in counts),
            mean_signed_extra_holding_contribution_s=float(np.mean([c["signed_extra_holding_contribution_s"] for c in counts]))))
    result = dict(status="conditional_actor_cross_complete", effects=stats,
        by_seed={str(seed): intervals([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        absolute={mode: {m: float(np.mean([r["results"][mode][m] for r in rows])) for m in experiment.METRICS} for mode in MODES},
        native_late_joint_contexts=risk, primary_contrast="no_gain_vs_late", primary_diagnostic_gate_passed=joint,
        original_actors=2, demand_days=len(DAYS), sampler_repeats=3, evaluation_rollouts=len(MODES) * len(rows),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope="prospective demand-day diagnostic conditional on two original actors; dynamic regimes, not natural mediation",
        hypothesis="V88 no-predicted-gain extra holding can be replaced without sacrificing full-day regularity/readiness",
        alternative_arms_are_mechanism_controls_not_a_winner_search=True)
    pilot.write_json(root / "summary/status.json", result)
    print("V89_CONDITIONAL_CROSS_COMPLETE " + json.dumps(dict(primary=primary, diagnostic_gate=joint)), flush=True)


def smoke_case(root, v88_root, seed, mode):
    day, rep = experiment.EVAL_DAYS[0], 0
    original = audit.replay(root, seed, day, rep, mode)
    candidate = replay(root, seed, day, rep, mode)
    index = experiment.ITEMS.index((seed, day, rep))
    old = json.loads((v88_root / "pairs" / f"p{index}.json").read_text())["metrics"][mode]
    pilot.joint.assert_same_state(old, {m: candidate[0][m] for m in experiment.METRICS}, "v89_original_v88_endpoints")
    pilot.joint.assert_same_state(original[1], candidate[1], "v89_native_events")
    pilot.joint.assert_same_state(original[2], candidate[2], "v89_native_holding")
    if seed == pilot.SEEDS[0] and mode == "late":
        changed = replay(root, seed, day, rep, "no_gain_early")
        if changed[3]["early_decisions"] == 0 or pilot.event.alignment(original[1], changed[1])["action_differences"] == 0:
            raise AssertionError("smoke must actually execute the primary conditional intervention")


def smoke(root, v88_root):
    for pattern in ("test_v87_*.py", "test_v88_*.py", "test_v89_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", pattern, "-v"], check=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(smoke_case, root, v88_root, seed, mode) for seed in pilot.SEEDS for mode in audit.SOURCES]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="native_paths_and_primary_conditional_intervention_passed",
        prospective_days_read=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v88-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v88_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

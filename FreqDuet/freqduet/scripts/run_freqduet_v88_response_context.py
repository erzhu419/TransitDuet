#!/usr/bin/env python3
"""Locate V87 common-state holding differences without changing either actor."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v87_logit_factorial as experiment
from run_freqduet_v87_factorial_review import holding_transport, interval


pilot = experiment.pilot
ITEMS = experiment.ITEMS
SOURCES = ("late", "early")
GROUPS = ("missing_bilateral", "more_hold_gain_up", "more_hold_gain_not_up", "no_more_hold")
FIELDS = ("late_hold_s", "early_hold_s", "holding_difference_s", "passenger_command_difference_s",
          "predicted_gain_difference", "return_pressure_weighted_difference_s", "spent_weighted_difference_s")


def context_group(bilateral, holding_difference, gain_difference):
    if not bilateral:
        return "missing_bilateral"
    if holding_difference <= 0:
        return "no_more_hold"
    return "more_hold_gain_up" if gain_difference > 0 else "more_hold_gain_not_up"


class ContextSums:
    def __init__(self):
        self.count = 0
        self.groups = {g: dict(count=0, sums=np.zeros(len(FIELDS))) for g in GROUPS}

    def add(self, late, early, gains, bilateral, passengers, pressure, spent):
        lh, eh = float(late @ pilot.step.ACTIONS), float(early @ pilot.step.ACTIONS)
        delta, gain = lh - eh, float((late - early) @ gains)
        group = self.groups[context_group(bilateral, delta, gain)]
        group["count"] += 1
        group["sums"] += [lh, eh, delta, passengers * delta, gain, pressure * delta, spent * delta]
        self.count += 1

    def summary(self):
        return dict(decisions=self.count, groups={g: dict(count=r["count"], fraction=r["count"] / self.count,
            contribution={k: float(v / self.count) for k, v in zip(FIELDS, r["sums"])}) for g, r in self.groups.items()})


def group_transport(late_states, early_states):
    result = {}
    for group in GROUPS:
        a, b = [s["groups"][group]["contribution"] for s in (late_states, early_states)]
        result[group] = holding_transport([a["late_hold_s"], a["early_hold_s"]],
                                         [b["late_hold_s"], b["early_hold_s"]])
    direct = holding_transport(*[[sum(s["groups"][g]["contribution"][k] for g in GROUPS)
                                 for k in ("late_hold_s", "early_hold_s")] for s in (late_states, early_states)])
    for key, value in direct.items():
        np.testing.assert_allclose(sum(r[key] for r in result.values()), value, atol=1e-10, rtol=0)
    return result


class ResponseContext(experiment.LogitCross):
    def __init__(self, seed, label):
        super().__init__(seed, label, np.zeros(7))
        self.sums = ContextSums()

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        with torch.no_grad():
            late_z, early_z = policy.forward(states), self.early.forward(states)
            mask = policy.feasible_action_mask(states)
            late, early = [experiment.distribution(late_z, early_z, mask, self.shift, name)[0].numpy().astype(float)
                           for name in SOURCES]
        runner = self.runner
        bus = runner._bus_for_agent(runner._pilot_agent)
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        self.sums.add(late, early, features[:, 0], context.evidence_valid and context.follower_evidence_valid,
                      len(bus.passengers), -features[-1, 2], -features[-1, 3])
        return law

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        if self.sums.count != self.count:
            raise AssertionError("context audit missed a lower decision")
        self.summary = self.sums.summary()


def replay(root, seed, day, rep, label):
    rule, credit = ResponseContext(seed, label), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v88_seed{seed}_day{day}_r{rep}_{label}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
               backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s)
    return row, events, holds, rule.summary


def run_cell(root, v87_root, index, check_paths=False):
    path = root / "pairs" / f"p{index}.json"
    if path.exists() and not check_paths:
        return
    seed, day, rep = ITEMS[index]
    old = json.loads((v87_root / "pairs" / f"p{index}.json").read_text())
    summaries, metrics = {}, {}
    for label in SOURCES:
        row, events, holds, summary = replay(root, seed, day, rep, label)
        metrics[label] = {m: row[m] for m in experiment.METRICS}
        pilot.joint.assert_same_state({m: old["results"][label][m] for m in experiment.METRICS},
                                     metrics[label], "v88_original_v87_endpoints")
        pilot.joint.assert_same_state(old["results"][label]["action_counts"], row["action_counts"], "v88_action_counts")
        expected = [sum(summary["groups"][g]["contribution"][k] for g in GROUPS)
                    for k in ("late_hold_s", "early_hold_s")]
        np.testing.assert_allclose(expected, old["results"][label]["expected_hold_on_visited_states_s"], atol=1e-10, rtol=0)
        if check_paths:
            native = experiment.replay(root, seed, day, rep, label, np.zeros(7))
            pilot.joint.assert_same_state(native[1], events, "v88_passive_native_events")
            pilot.joint.assert_same_state(native[2], holds, "v88_passive_native_holding")
        summaries[label] = summary
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, sources=summaries,
        metrics=metrics, transport=group_transport(summaries["late"], summaries["early"]),
        exact_v87_endpoints=True, native_paths_verified=check_paths))
    print("V88_RESPONSE_CONTEXT_DONE " + str(index), flush=True)


def day_interval(rows, getter):
    return interval([float(np.mean([getter(r) for r in rows if r["day"] == day])) for day in experiment.EVAL_DAYS])


def summarize(rows):
    sources = {source: {g: dict(
        fraction=day_interval(rows, lambda r: r["sources"][source]["groups"][g]["fraction"]),
        contribution={m: day_interval(rows, lambda r: r["sources"][source]["groups"][g]["contribution"][m]) for m in FIELDS})
        for g in GROUPS} for source in SOURCES}
    transport = {g: {m: day_interval(rows, lambda r: r["transport"][g][m]) for m in
        ("expected_command_s", "common_state_law_s", "visitation_s")} for g in GROUPS}
    return dict(sources=sources, transport=transport)


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    result = dict(status="response_context_audit_complete", **summarize(rows),
        by_seed={str(seed): summarize([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        labelled_paths=2 * len(rows), all_original_v87_endpoints=True,
        inference="16 consumed V87 demand days, three samplers, two fixed actors; no fresh-day or training-seed confirmation",
        predicted_gain_is_not_causal_action_value=True, actor_updates=0, main_policy_changed=False, controller_fits=0)
    pilot.write_json(root / "summary/status.json", result)
    print("V88_RESPONSE_CONTEXT_COMPLETE " + json.dumps({k: result[k] for k in ("sources", "transport")}), flush=True)


def smoke(root, v87_root):
    for pattern in ("test_v87_*.py", "test_v88_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", pattern, "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(run_cell, root, v87_root, ITEMS.index((seed, experiment.EVAL_DAYS[0], 0)), True)
                     for seed in pilot.SEEDS]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="context_accounting_native_paths_and_v87_endpoints_passed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v87-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v87_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, args.v87_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

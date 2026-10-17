#!/usr/bin/env python3
"""Compare frozen compensation selection with a reference-budget mixture."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v94_compensation_confirmation as confirmation


frozen, pilot = confirmation.frozen, confirmation.pilot
ITEMS = [(i, *item) for i, item in enumerate(confirmation.ITEMS) if item[0] in confirmation.VALIDATION_SEEDS]
CONTRASTS = dict(preserve_vs_late=("preserve", "late"), uniform_vs_late=("uniform", "late"),
    preserve_vs_uniform=("preserve", "uniform"))


def reference_probability(contexts):
    budgets = [contexts[g]["expected_holding_reduction_s"] for g in frozen.GROUPS]
    return float(budgets[0] / sum(budgets))


def mixture(late, early, probability):
    if probability == 0:
        return late
    if probability == 1:
        return early
    return (1 - probability) * late + probability * early


class UniformMixture(frozen.ReleaseHolding):
    def __init__(self, seed, probability):
        super().__init__(seed, "no_gain_early")
        self.probability = probability

    def __call__(self, policy, states, features, theta):
        previous = self.early_decisions
        early = super().__call__(policy, states, features, theta)
        if self.early_decisions == previous:
            return early
        with torch.no_grad():
            logits = policy.forward(states)
            mask = policy.feasible_action_mask(states)
            late = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), -1)
        # Compensation strata are logged by the parent, not used to select this law.
        return mixture(late, early, self.probability)

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        for group in self.group_summary.values():
            group["selected"] = self.probability * group["count"]


def replay(root, seed, day, rep, probability):
    rule, credit = UniformMixture(seed, probability), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v95_seed{seed}_day{day}_r{rep}_p{probability}", probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return row, events, holds, rule


def axis_review(rows):
    result = {}
    for contrast in confirmation.CONTRASTS:
        result[contrast] = {}
        for axis in ("seed", "day"):
            levels = sorted({r[axis] for r in rows})
            means = {str(level): {m: float(np.mean([r["effects"][contrast][m] for r in rows if r[axis] == level]))
                for m in ("journey_min", "cv_squared", "readiness_delay_s")} for level in levels}
            result[contrast][axis] = dict(means=means,
                negative_counts={m: sum(v[m] < 0 for v in means.values()) for m in next(iter(means.values()))},
                jointly_negative=sum(all(x < 0 for x in v.values()) for v in means.values()))
    return result


def prepare(root, v93_root, v94_root):
    old = json.loads((v93_root / "summary/status.json").read_text())
    probability = reference_probability(old["context_totals"]["late"])
    rows = [json.loads((v94_root / "pairs" / f"p{i}.json").read_text()) for i, *_ in ITEMS]
    pilot.write_json(root / "prepare/status.json", dict(status="v93_budget_fixed_v94_axes_reviewed",
        probability=probability, reference_contexts=old["context_totals"]["late"],
        probability_source="V93 B200 observable budgets only; no passenger/CV/readiness outcome fitting",
        v94_validation_axis_review=axis_review(rows), new_rollouts=0, main_policy_changed=False))
    print("V95_PREPARE_COMPLETE probability=" + str(probability), flush=True)


def dose(contexts, mode, probability):
    budgets = {g: contexts[g]["expected_holding_reduction_s"] for g in frozen.GROUPS}
    weights = dict(nonloss=1., compensation_loss=0.) if mode == "preserve" else dict.fromkeys(frozen.GROUPS, probability)
    total = sum(weights[g] * budgets[g] for g in frozen.GROUPS)
    return dict(expected_reduction_sum_s=total,
        eligible_count=sum(contexts[g]["count"] for g in frozen.GROUPS),
        effective_early_selections=sum(weights[g] * contexts[g]["count"] for g in frozen.GROUPS),
        shadow_preserve_reduction_sum_s=budgets["nonloss"],
        expected_reduction_minus_shadow_preserve_s=total - budgets["nonloss"],
        compensation_loss_expected_reduction_sum_s=weights["compensation_loss"] * budgets["compensation_loss"])


def saved_timing(path):
    with np.load(path) as data:
        events = [dict(zip(data["event_fields"].tolist(), values)) for values in data["arrivals"]]
    return SimpleNamespace(coordinates=frozen.coordinate_moments(events),
        summary=dict(arrival=frozen.timing.timing_series(events)))


def run_cell(root, v94_root, index, probability):
    if (root / "pairs" / f"p{index}.json").exists():
        return
    original, seed, day, rep = ITEMS[index]
    old = json.loads((v94_root / "pairs" / f"p{original}.json").read_text())
    if (old["seed"], old["day"], old["rep"]) != (seed, day, rep):
        raise AssertionError("V95 pairing differs from the completed V94 cell")
    row, _, _, rule = replay(root, seed, day, rep, probability)
    rule.save(root / "records" / f"p{index}_uniform.npz")
    rows = dict(old["results"], uniform=row)
    timelines = {mode: saved_timing(v94_root / "records" / f"p{original}_{mode}.npz") for mode in ("late", "preserve")}
    timelines["uniform"] = rule
    doses = dict(preserve=dose(old["contexts"]["preserve"], "preserve", probability),
        uniform=dose(rule.group_summary, "uniform", probability))
    result = dict(index=index, original_v94_index=original, seed=seed, day=day, rep=rep,
        effects={name: {m: rows[a][m] - rows[b][m] for m in frozen.METRICS} for name, (a, b) in CONTRASTS.items()},
        timing_effects={name: confirmation.timing_effect(timelines[b], timelines[a]) for name, (a, b) in CONTRASTS.items()},
        doses=doses, dose_effects=dict(uniform_vs_preserve={m: doses["uniform"][m] - doses["preserve"][m] for m in doses["uniform"]}),
        contexts=rule.group_summary, probability=probability)
    pilot.write_json(root / "pairs" / f"p{index}.json", result)
    print("V95_SPECIFICITY_CELL_DONE " + str(index), flush=True)


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    if [(r["original_v94_index"], r["seed"], r["day"], r["rep"]) for r in rows] != ITEMS:
        raise AssertionError("V95 aggregate differs from the fixed paired grid")
    prepare_summary = json.loads((root / "prepare/status.json").read_text())
    effects = confirmation.describe(rows, "effects")
    result = dict(status="compensation_specificity_complete", effects=effects,
        timing_effects=confirmation.describe(rows, "timing_effects"),
        dose_effects=confirmation.describe(rows, "dose_effects"),
        absolute_expected_doses={mode: {m: float(np.mean([r["doses"][mode][m] for r in rows])) for m in rows[0]["doses"][mode]}
            for mode in ("preserve", "uniform")},
        by_seed={str(seed): confirmation.day_stats([r for r in rows if r["seed"] == seed]) for seed in confirmation.VALIDATION_SEEDS},
        preserve_joint_gate_passed=confirmation.joint_gate(effects),
        uniform_joint_gate_passed=confirmation.joint_gate(dict(preserve_vs_late=effects["uniform_vs_late"])),
        probability=prepare_summary["probability"], additional_rollouts=len(ITEMS),
        archived_training_seeds=list(confirmation.VALIDATION_SEEDS), demand_days=list(confirmation.DAYS), sampler_repeats=3,
        mechanism_contrast="preserve_vs_uniform", original_gate_changed=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        scope="mechanism follow-up on consumed V94 days; reference-budget targeted, not exact closed-loop dose matching")
    pilot.write_json(root / "summary/status.json", result)
    print("V95_SPECIFICITY_COMPLETE " + json.dumps(effects["preserve_vs_uniform"]), flush=True)


def smoke_case(root, seed, probability):
    day = frozen.DAYS[0]
    for p, mode in ((0., "late"), (1., "no_gain_early")):
        current, native = replay(root, seed, day, 0, p), frozen.replay(root, seed, day, 0, mode)
        for m in (*frozen.METRICS, "action_counts"):
            pilot.joint.assert_same_state(current[0][m], native[0][m], "v95_endpoint")
        pilot.joint.assert_same_state(current[1], native[1], "v95_sampler_path")
        pilot.joint.assert_same_state(current[2], native[2], "v95_executed_holding")
    mixed = replay(root, seed, day, 0, probability)[3]
    if any(mixed.group_summary[g]["count"] == 0 for g in frozen.GROUPS):
        raise AssertionError("mixture smoke must exercise both compensation strata")


def smoke(root, probability):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v95_*.py", "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(smoke_case, root, seed, probability) for seed in (12037, 933031)]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_and_exact_mixture_endpoint_paths_passed",
        formal_mixture_cells_read=False, probability=probability))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v93-root", type=Path)
    parser.add_argument("--v94-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_root, args.v93_root, args.v94_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        probability = json.loads((args.out_root / "prepare/status.json").read_text())["probability"]
        if args.command == "smoke":
            smoke(args.out_root, probability)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
                for task in [pool.submit(run_cell, args.out_root, args.v94_root, i, probability) for i in range(args.start, args.end)]:
                    task.result()
            pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

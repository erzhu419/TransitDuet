#!/usr/bin/env python3
"""Cross frozen upper regimes with preserve versus zero holding."""

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

import run_freqduet_v97_benchmark_qualification as prior


pilot, frozen, confirmation = prior.pilot, prior.frozen, prior.confirmation
ZERO_ITEMS = list(itertools.product(confirmation.VALIDATION_SEEDS, prior.DAYS))
ITEMS = [(s, d, 0, "learned_zero") for s, d in ZERO_ITEMS] + [
    (s, d, r, "fixed_preserve") for s, d, r in prior.ITEMS]
CONTRASTS = dict(holding_at_learned=("learned_preserve", "learned_zero"),
    holding_at_fixed=("fixed_preserve", "fixed_zero"),
    upper_with_preserve=("learned_preserve", "fixed_preserve"),
    upper_without_holding=("learned_zero", "fixed_zero"),
    full_gap=("learned_preserve", "fixed_zero"))


def zero_law(policy, states):
    if not policy.feasible_action_mask(states)[:, 0].all():
        raise AssertionError("zero holding must remain a feasible native action")
    law = torch.zeros((len(states), len(pilot.step.ACTIONS)), dtype=states.dtype)
    law[:, 0] = 1.
    return law


def contrasts(rows):
    effects = {name: {m: rows[a][m] - rows[b][m] for m in frozen.METRICS}
               for name, (a, b) in CONTRASTS.items()}
    effects["interaction"] = {m: effects["holding_at_learned"][m] - effects["holding_at_fixed"][m]
                              for m in frozen.METRICS}
    for m in frozen.METRICS:
        np.testing.assert_allclose(effects["full_gap"][m],
            effects["holding_at_learned"][m] + effects["upper_without_holding"][m], atol=1e-8, rtol=0)
    return effects


class FactorialAudit(frozen.ReleaseHolding):
    def __init__(self, seed, mode):
        super().__init__(seed, "preserve")
        self.arm = mode

    def attach(self, runner):
        super().attach(runner)
        self.original_initialize = runner.env.initialize_state
        if self.arm.startswith("fixed_"):
            from scripts.run_freqduet_external_baselines import install_exact_dispatch_schedule
            def initialize():
                install_exact_dispatch_schedule(runner.env, lambda *_: 360.,
                    projection_mode="exact_fixed_headway_schedule_v6")
                def upper_callback(state, trip):
                    trip._upper_queried = True
                    return float(trip.target_headway)
                runner.env._upper_policy_callback = upper_callback
                return self.original_initialize()
            runner.env.initialize_state = initialize

    def __call__(self, policy, states, features, theta):
        if self.arm.endswith("_zero"):
            return zero_law(policy, states)
        return super().__call__(policy, states, features, theta)

    def detach(self):
        self.env.initialize_state = self.original_initialize
        super().detach()

    def finish(self, runner, records, holds):
        self.contract = prior.physical_contract(runner.env, runner.cfg)
        self.dispatch = prior.dispatch_metrics(runner.env)
        super().finish(runner, records, holds)


def replay(root, seed, day, rep, mode):
    rule, credit = FactorialAudit(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v98_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return row, events, holds, rule


def run_item(root, index):
    path = root / "pairs" / f"p{index}.json"
    if not path.exists():
        seed, day, rep, mode = ITEMS[index]
        row, _, _, rule = replay(root, seed, day, rep, mode)
        rule.save(root / "records" / f"p{index}.npz")
        pilot.write_json(path, dict(seed=seed, day=day, rep=rep, mode=mode, results=row,
            contract=rule.contract, groups=rule.group_summary))
    print("V98_FACTORIAL_ITEM_DONE " + str(index), flush=True)


def aggregate(root, v96_root, v97_root):
    new = {}
    for i, item in enumerate(ITEMS):
        row = json.loads((root / "pairs" / f"p{i}.json").read_text())
        if tuple(row[k] for k in ("seed", "day", "rep", "mode")) != item:
            raise AssertionError("factorial row differs from the frozen grid")
        new[item] = row
    paired, absolutes = [], []
    for i, (seed, day, rep) in enumerate(prior.ITEMS):
        preserve = json.loads((v96_root / "pairs" / f"p{i}.json").read_text())
        fixed = json.loads((v97_root / "fixed" / f"d{day}.json").read_text())
        zero = new[seed, day, 0, "learned_zero"]
        fixed_preserve = new[seed, day, rep, "fixed_preserve"]
        for row in (zero, fixed_preserve):
            pilot.joint.assert_same_state(row["contract"], fixed["contract"], "v98_shared_physical_contract")
        rows = dict(learned_preserve=preserve["results"]["preserve"], learned_zero=zero["results"],
            fixed_preserve=fixed_preserve["results"], fixed_zero=fixed["results"])
        paired.append(dict(seed=seed, day=day, rep=rep, effects=contrasts(rows)))
        absolutes.append(rows)
    stats = confirmation.describe(paired, "effects")
    result = dict(status="frozen_upper_holding_factorial_complete", effects=stats,
        by_seed={str(s): confirmation.day_stats([r for r in paired if r["seed"] == s])
                 for s in confirmation.VALIDATION_SEEDS},
        effect_axes=prior.prior.axis_summary(paired, "effects", prior.prior.KEY_METRICS),
        absolute={mode: {m: float(np.mean([r[mode][m] for r in absolutes])) for m in frozen.METRICS}
                  for mode in absolutes[0]},
        new_rollouts=len(ITEMS), reused_preserve_rollouts=len(prior.ITEMS), reused_fixed_rollouts=len(prior.DAYS),
        archived_training_seeds=list(confirmation.VALIDATION_SEEDS), demand_days=list(prior.DAYS),
        sampler_repeats=3, zero_paths_per_actor_day=1, fixed_paths_per_day=1,
        bootstrap="5000 crossed actor/day draws after sampler averaging; deterministic corners shared, not independent repeats",
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        candidate_retuned=False, original_gate_changed=False,
        scope="consumed-day controlled regime factorial, not mediation, new confirmation or repaired training",
        next_decision="use passenger, CV, readiness and completion effects to locate objective/coordination work; no winner selection")
    pilot.write_json(root / "summary/status.json", result)
    print("V98_UPPER_HOLDING_FACTORIAL_COMPLETE " + json.dumps(stats), flush=True)


def smoke(root, v96_root, v97_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v98_*.py", "-v"], check=True)
    seed, day, rep = prior.ITEMS[0]
    native = replay(root, seed, day, rep, "learned_preserve")
    old = json.loads((v96_root / "pairs/p0.json").read_text())
    for m in (*frozen.METRICS, "action_counts"):
        pilot.joint.assert_same_state(native[0][m], old["results"]["preserve"][m], "v98_cached_preserve")
    zero = replay(root, seed, day, rep, "learned_zero")
    repeated_zero = replay(root, seed, day, rep + 1, "learned_zero")
    for m in (*frozen.METRICS, "action_counts"):
        pilot.joint.assert_same_state(zero[0][m], repeated_zero[0][m], "v98_zero_sampler_invariant")
    pilot.joint.assert_same_state(zero[2], repeated_zero[2], "v98_zero_holding_path")
    fixed = replay(root, seed, day, rep, "fixed_zero")
    old_fixed = json.loads((v97_root / "fixed" / f"d{day}.json").read_text())
    for m in frozen.METRICS:
        np.testing.assert_allclose(fixed[0][m], old_fixed["results"][m], atol=1e-8, rtol=0, err_msg=m)
    changed = replay(root, seed, day, rep, "fixed_preserve")
    if changed[3].early_decisions == 0:
        raise AssertionError("fixed-upper smoke must exercise preserve, not silently keep late")
    for row in (native, zero, repeated_zero, fixed, changed):
        pilot.joint.assert_same_state(row[3].contract, old_fixed["contract"], "v98_smoke_physical_contract")
    if zero[0]["holding_s_per_trip"] != 0 or fixed[0]["holding_s_per_trip"] != 0:
        raise AssertionError("zero arm executed holding")
    pilot.write_json(root / "smoke/done.json", dict(status="tests_cached_preserve_external_fixed_and_deterministic_zero_passed",
        new_training_runs=0, main_policy_changed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v96-root", type=Path)
    parser.add_argument("--v97-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v96_root, args.v97_root)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.v96_root, args.v97_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_item, args.out_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

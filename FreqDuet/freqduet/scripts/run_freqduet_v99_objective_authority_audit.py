#!/usr/bin/env python3
"""Passive audit of effective upper authority and lower reward ownership."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v98_upper_holding_factorial as prior


pilot, frozen, confirmation = prior.pilot, prior.frozen, prior.confirmation
DAYS = (prior.prior.DAYS[0], prior.prior.DAYS[-1])
MODES = ("learned_preserve", "learned_zero")
ITEMS = list(itertools.product(confirmation.VALIDATION_SEEDS, DAYS, MODES))


def amplitude(values, reference=0.):
    a = np.asarray(values, float) - reference
    if a.size == 0:
        raise AssertionError("authority audit requires observed upper values")
    return dict(mean=float(a.mean()), rms=float(np.sqrt(np.mean(a * a))), max_abs=float(np.abs(a).max()))


def reward_curves(runner, raw, context):
    regularity = np.asarray([runner.lower_departure_regularity.evaluate(context, a).reward_adjustment
                             for a in pilot.step.ACTIONS])
    load = np.asarray([runner.lower_load_holding_penalty.evaluate(raw, a,
        base_state_dim=runner.env._base_state_dim, context_features=runner.env.lower_context_features)[0]
        for a in pilot.step.ACTIONS])
    return regularity, load


def shadow_summary(probabilities, regularity, load, actions, feasible):
    p, r, c, mask = [np.asarray(v) for v in (probabilities, regularity, load, feasible)]
    net = r - c
    best = np.argmax(np.where(mask, net, -np.inf), axis=1)
    expected_r, expected_c = (p * r).sum(1), (p * c).sum(1)
    chosen = net[np.arange(len(actions)), np.asarray(actions, int)] - net[:, 0]
    return dict(decisions=len(actions), expected_holding_s=float(np.mean(p @ pilot.step.ACTIONS)),
        greedy_local_holding_s=float(pilot.step.ACTIONS[best].mean()),
        greedy_local_nonzero_fraction=float(np.mean(best != 0)),
        expected_regularity_reward_vs_zero=float(np.mean(expected_r - r[:, 0])),
        expected_load_penalty_vs_zero=float(np.mean(expected_c - c[:, 0])),
        expected_local_net_vs_zero=float(np.mean(expected_r - expected_c - net[:, 0])),
        chosen_nonzero_fraction=float(np.mean(np.asarray(actions) != 0)),
        chosen_negative_local_net_fraction=float(np.mean(chosen < 0)))


class ObjectiveAudit(prior.FactorialAudit):
    def __init__(self, seed, mode):
        super().__init__(seed, mode)
        self.targets, self.phases, self.curves, self.probabilities, self.masks = [], [], [], [], []
        self.upper_calls = 0

    def attach(self, runner):
        super().attach(runner)
        self.saved_upper = runner._upper_callback_v2
        self.saved_lower = runner._lower_action_for_agent
        self.saved_launch = runner.env.launch_bus
        def upper(state, trip):
            self.upper_calls += 1
            return self.saved_upper(state, trip)
        def launch(trip, actual_launch=None):
            bus = self.saved_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                self.targets.append(float(trip.target_headway))
                self.phases.append(float(trip._freqduet_phase_displacement_s))
            return bus
        def lower(obs, key, last_action=0., deterministic=False):
            self.raw_observation = np.asarray(obs, dtype=np.float32).copy()
            return self.saved_lower(obs, key, last_action=last_action, deterministic=deterministic)
        runner._upper_callback_v2, runner.env.launch_bus = upper, launch
        runner._lower_action_for_agent = lower

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        self.curves.append(reward_curves(self.runner, self.raw_observation, context))
        self.probabilities.append(law[0].numpy().copy())
        self.masks.append(policy.feasible_action_mask(states)[0].numpy().copy())
        return law

    def detach(self):
        self.runner._upper_callback_v2 = self.saved_upper
        self.runner._lower_action_for_agent = self.saved_lower
        super().detach()

    def finish(self, runner, records, holds):
        r, c = [np.stack([row[i] for row in self.curves]) for i in (0, 1)]
        if len(records) != len(self.curves):
            raise AssertionError("shadow credit curves do not align with actual decisions")
        self.shadow = shadow_summary(self.probabilities, r, c, [row["action"] for row in records], self.masks)
        self.authority = dict(upper_callback_count=self.upper_calls, plan_decisions=runner._ep_upper_plan_decisions,
            target_minus_360_s=amplitude(self.targets, 360.), phase_displacement_s=amplitude(self.phases),
            effective_delta_s=amplitude(runner._ep_upper_deltas))
        reg = np.asarray(runner._ep_lower_departure_regularity_rewards)
        load = np.asarray(runner._ep_lower_load_hold_penalties)
        total = float(np.sum(runner._ep_lower_rewards))
        self.credit = dict(completed_transitions=len(runner._ep_lower_rewards),
            recorded_shaped_reward_sum=total, regularity_reward_sum=float(reg.sum()), load_penalty_sum=float(load.sum()),
            other_reward_sum=total - float(reg.sum()) + float(load.sum()),
            configuration={k: runner.cfg[k] for k in ("reward_attribution", "leakage")},
            lower_regularity=dict(runner.cfg["lower"]["causal_departure_regularity"]),
            lower_load_holding=dict(runner.cfg["lower"]["load_weighted_holding"]),
            upper_interval_credit=dict(runner.cfg["upper"]["interval_credit"]))
        super().finish(runner, records, holds)


def replay(root, seed, day, mode):
    rule, credit = ObjectiveAudit(seed, mode), pilot.CreditObserver()
    row, _, _, _ = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day,
        [0.] * 4, f"v99_seed{seed}_day{day}_{mode}", probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return dict(seed=seed, day=day, mode=mode, results=row, authority=rule.authority,
        shadow=rule.shadow, credit=rule.credit, contract=rule.contract), rule


def match_cached(result, v96_root, v98_root):
    seed, day, mode = [result[k] for k in ("seed", "day", "mode")]
    if mode == "learned_preserve":
        index = prior.prior.ITEMS.index((seed, day, 0))
        old = json.loads((v96_root / "pairs" / f"p{index}.json").read_text())["results"]["preserve"]
        records = v96_root / "records" / f"p{index}_preserve.npz"
    else:
        index = prior.ITEMS.index((seed, day, 0, mode))
        old = json.loads((v98_root / "pairs" / f"p{index}.json").read_text())["results"]
        records = v98_root / "records" / f"p{index}.npz"
    for metric in (*frozen.METRICS, "action_counts"):
        pilot.joint.assert_same_state(old[metric], result["results"][metric], "v99_passive_cached_" + metric)
    return records


def run_item(root, v96_root, v98_root, index, full_path=False):
    path = root / "pairs" / f"p{index}.json"
    if not path.exists():
        result, rule = replay(root, *ITEMS[index])
        cached = match_cached(result, v96_root, v98_root)
        if full_path:
            current = root / "smoke_records" / f"p{index}.npz"
            rule.save(current)
            with np.load(cached) as old, np.load(current) as new:
                for key in old.files:
                    np.testing.assert_array_equal(old[key], new[key], err_msg=key)
        result["cached_endpoints_exact"] = True
        pilot.write_json(path, result)
    print("V99_PASSIVE_AUDIT_ITEM_DONE " + str(index), flush=True)


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    if [tuple(r[k] for k in ("seed", "day", "mode")) for r in rows] != ITEMS:
        raise AssertionError("passive audit differs from the fixed consumed-day grid")
    flat = [{**{k: r[k] for k in ("seed", "day", "mode", "shadow", "credit")},
        **{f"{name}_{stat}": value for name, d in r["authority"].items() if isinstance(d, dict)
           for stat, value in d.items()}, upper_callback_count=r["authority"]["upper_callback_count"],
        plan_decisions=r["authority"]["plan_decisions"]} for r in rows]
    keys = [k for k, v in flat[0].items() if isinstance(v, (int, float)) and k not in ("seed", "day")]
    result = dict(status="passive_objective_authority_audit_complete", rows=flat,
        upper_by_mode={mode: {k: dict(mean=float(np.mean([r[k] for r in flat if r["mode"] == mode])),
            min=float(min(r[k] for r in flat if r["mode"] == mode)),
            max=float(max(r[k] for r in flat if r["mode"] == mode))) for k in keys} for mode in MODES},
        shadow_by_mode={mode: {k: float(np.mean([r["shadow"][k] for r in rows if r["mode"] == mode]))
            for k in rows[0]["shadow"]} for mode in MODES},
        archived_training_seeds=list(confirmation.VALIDATION_SEEDS), consumed_days=list(DAYS), rollouts=len(ITEMS),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        cached_endpoints_exact=all(r["cached_endpoints_exact"] for r in rows),
        scope="passive source/objective audit on two consumed days; not another efficacy confirmation",
        shadow_scope="regularity adjustment minus observed-load holding penalty only, not total shaped reward or action value")
    pilot.write_json(root / "summary/status.json", result)
    print("V99_OBJECTIVE_AUTHORITY_AUDIT_COMPLETE", flush=True)


def smoke(root, v96_root, v98_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v99_*.py", "-v"], check=True)
    for i in (0, 1):
        run_item(root, v96_root, v98_root, i, full_path=True)
    pilot.write_json(root / "smoke/done.json", dict(status="tests_both_cached_regimes_and_full_timing_paths_exact",
        main_policy_changed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v96-root", type=Path)
    parser.add_argument("--v98-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v96_root, args.v98_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_item, args.out_root, args.v96_root, args.v98_root, i)
                         for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

#!/usr/bin/env python3
"""Qualify frozen compensation against matched lower-40 and fixed baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_system_credit import SystemCreditStream
import run_freqduet_v96_frozen_compensation as prior


frozen, pilot, confirmation = prior.frozen, prior.pilot, prior.confirmation
ITEMS, DAYS = prior.ITEMS, prior.DAYS
TOTAL_ITEMS = len(ITEMS) + len(DAYS)
CONTRASTS = dict(late_vs_lower40=("late", "lower40"), preserve_vs_lower40=("preserve", "lower40"),
    preserve_vs_fixed=("preserve", "fixed"), late_vs_fixed=("late", "fixed"), lower40_vs_fixed=("lower40", "fixed"))


def physical_contract(env, cfg):
    return dict(protocol_version=cfg["protocol"]["version"], clock=asdict(env.protocol),
        scenario_tape_id=str(env.scenario_tape.identifier), passengers_generated=int(env._measurement_details["passengers_generated"]),
        fleet_target=int(env._n_fleet_target), physical_vehicle_count=len(env.bus_all),
        fleet_inventory_mode=env.fleet_inventory_mode, dt_s=float(env.time_step))


def dispatch_metrics(env):
    gaps = np.asarray(env._actual_dispatch_gaps_s)
    return dict(dispatch_gap_mean_s=float(gaps.mean()), dispatch_gap_variance_s2=float(gaps.var()),
                dispatch_cv_squared=float(gaps.var() / gaps.mean() ** 2))


class BenchmarkAudit(frozen.timing.TimingAudit):
    def finish(self, runner, records, holds):
        self.contract = physical_contract(runner.env, runner.cfg)
        self.dispatch = dispatch_metrics(runner.env)
        super().finish(runner, records, holds)


def lower_replay(root, seed, day, rep, mode="early"):
    rule, credit = BenchmarkAudit(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v97_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return row, events, holds, rule


class FixedExposure:
    def __init__(self):
        self.stream, self.rates = SystemCreditStream(), None

    def attach(self, env, tracker):
        original = tracker.record_step
        def record(**kwargs):
            original(**kwargs)
            self.rates = [sum(kwargs[k].values()) for k in ("waiting_by_direction", "onboard_by_direction",
                "waiting_low_by_direction", "onboard_low_by_direction", "dispatch_backlog_by_direction")]
            self.stream.advance(float(env.current_time), float(kwargs["dt_s"]), *self.rates)
        tracker.record_step = record
        env._upper_interval_outcome_tracker = tracker

    def finish(self, env):
        details = env._measurement_details
        cleared = env._done_reason == "service_cleared" and details["trips_unlaunched"] == details["trips_incomplete"] == 0
        if self.rates[0] != details["passengers_unserved"]:
            raise AssertionError("fixed exposure and terminal passenger ledger differ")
        self.stream.finish(env.protocol.evaluation_end_time_s, service_cleared=cleared,
            unserved=self.rates[0], lf_unserved=self.rates[2])
        return self.stream.audit()


def fixed_metrics(raw, exposure, gaps):
    total = exposure.stream.total
    return dict(passenger_person_s=float(total[0] + total[1]),
        journey_min=round(raw["restricted_total_journey_horizon_min"], 6), cv_squared=float(raw["headway_cv"] ** 2),
        readiness_delay_s=round(raw["fleet_readiness_delay_mean_s"], 6),
        holding_s_per_trip=round(raw["holding_vehicle_seconds_per_launched_trip"], 6),
        trips_unlaunched=int(raw["trips_unlaunched"]), trips_incomplete=int(raw["trips_incomplete"]),
        waiting_person_s=float(total[0]), onboard_person_s=float(total[1]), backlog_trip_s=float(total[4]),
        holding_person_s=float(raw["holding_passenger_seconds"]), **gaps)


def fixed_replay(root, day, exposure):
    _, _, _, torch = pilot.joint.optimized_runtime(root)
    from scripts import run_freqduet_external_baselines as external
    from upper.interval_credit import UpperIntervalOutcomeTracker
    pilot.joint.seed_runtime(torch, day)
    path = pilot.joint.MODEL_ROOT / "configs_freqduet" / f"{pilot.joint.CURRENT}.yaml"
    env, cfg = external.make_env_from_config(str(path))
    if cfg["protocol"]["version"] != "freqduet-eval-v6":
        raise AssertionError("fixed benchmark must use the same V6 contract")
    env._freqduet_training, env._freqduet_episode, env.scenario_seed = False, pilot.joint.BASE_EP, day
    if exposure is not None:
        exposure.attach(env, UpperIntervalOutcomeTracker.from_config(cfg["upper"]["interval_credit"]))
    raw = external.run_episode_external(env, variant="fixed_headway", n_fleet=cfg["upper"]["N_fleet"],
        rng=np.random.RandomState(day), demand_noise=cfg["env"]["demand_noise"],
        exact_schedule=True, schedule_contract_version="v6")
    contract = physical_contract(env, cfg)
    if exposure is None:
        return raw, contract
    audit = exposure.finish(env)
    row = fixed_metrics(raw, exposure, dispatch_metrics(env))
    if row["holding_s_per_trip"] != 0 or raw["baseline_schedule_projection_mode"] != "exact_fixed_headway_schedule_v6":
        raise AssertionError("fixed benchmark changed the registered 360-second/no-holding policy")
    return dict(results=row, contract=contract, stream_audit=audit, native_results=raw)


def run_item(root, index):
    if index < len(ITEMS):
        path = root / "lower40" / f"p{index}.json"
        if not path.exists():
            row, _, _, rule = lower_replay(root, *ITEMS[index])
            rule.save(root / "records" / f"p{index}_lower40.npz")
            pilot.write_json(path, dict(index=index, seed=ITEMS[index][0], day=ITEMS[index][1], rep=ITEMS[index][2],
                results=row, contract=rule.contract))
    else:
        day = DAYS[index - len(ITEMS)]
        path = root / "fixed" / f"d{day}.json"
        if not path.exists():
            pilot.write_json(path, fixed_replay(root, day, FixedExposure()))
    print("V97_BENCHMARK_ITEM_DONE " + str(index), flush=True)


def comparisons(rows):
    return {name: {m: rows[a][m] - rows[b][m] for m in frozen.METRICS} for name, (a, b) in CONTRASTS.items()}


def aggregate(root, v96_root):
    paired, absolutes = [], []
    for index, (seed, day, rep) in enumerate(ITEMS):
        old = json.loads((v96_root / "pairs" / f"p{index}.json").read_text())
        early = json.loads((root / "lower40" / f"p{index}.json").read_text())
        fixed = json.loads((root / "fixed" / f"d{day}.json").read_text())
        if any((r["seed"], r["day"], r["rep"]) != (seed, day, rep) for r in (old, early)):
            raise AssertionError("benchmark rows differ from the paired V96 grid")
        pilot.joint.assert_same_state(early["contract"], fixed["contract"], "same_day_physical_contract")
        rows = dict(old["results"], lower40=early["results"], fixed=fixed["results"])
        effects = comparisons(rows)
        for m in frozen.METRICS:
            np.testing.assert_allclose(effects["preserve_vs_lower40"][m],
                effects["late_vs_lower40"][m] + old["effects"]["preserve_vs_late"][m], atol=1e-8, rtol=0)
        paired.append(dict(seed=seed, day=day, rep=rep, effects=effects))
        absolutes.append(rows)
    stats = confirmation.describe(paired, "effects")
    result = dict(status="frozen_benchmark_qualification_complete", effects=stats,
        by_seed={str(seed): confirmation.day_stats([r for r in paired if r["seed"] == seed]) for seed in confirmation.VALIDATION_SEEDS},
        effect_axes=prior.axis_summary(paired, "effects", prior.KEY_METRICS),
        absolute={mode: {m: float(np.mean([r[mode][m] for r in absolutes])) for m in frozen.METRICS}
                  for mode in ("late", "preserve", "uniform", "lower40", "fixed")},
        joint_gates={name: confirmation.joint_gate(dict(preserve_vs_late=stats[name])) for name in stats},
        archived_training_seeds=list(confirmation.VALIDATION_SEEDS), demand_days=list(DAYS), sampler_repeats=3,
        new_lower40_rollouts=len(ITEMS), new_fixed_rollouts=len(DAYS), reused_v96_rollouts=len(ITEMS) * 3,
        lower40_context="upper actor, critics, dual, non-actor deployment and episode stage all remain B200; not full B40",
        fixed_context="shared deterministic 360-second/no-holding external policy; same hard fleet/scenario/horizon",
        bootstrap="5000 crossed actor/day draws after repeat averaging; one shared fixed result per day",
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, candidate_retuned=False,
        scope="benchmark follow-up on consumed V96 days; no new confirmation of V96 or complete training-convergence claim")
    pilot.write_json(root / "summary/status.json", result)
    print("V97_BENCHMARK_QUALIFICATION_COMPLETE " + json.dumps({k: stats[k] for k in
        ("preserve_vs_lower40", "preserve_vs_fixed")}), flush=True)


def smoke_actor(root, v96_root):
    seed, day, rep = ITEMS[0]
    early = lower_replay(root, seed, day, rep)
    native_early = frozen.source.replay(root, seed, day, rep, "early")
    for m in (*pilot.METRICS, "waiting_person_s", "onboard_person_s", "backlog_trip_s", "holding_person_s"):
        pilot.joint.assert_same_state(early[0][m], native_early[0][m], "v97_original_lower40")
    pilot.joint.assert_same_state(early[1], native_early[1], "v97_lower40_event_sampler")
    pilot.joint.assert_same_state(early[2], native_early[2], "v97_lower40_executed_holding")
    native = lower_replay(root, seed, day, rep, "late")
    old = json.loads((v96_root / "pairs/p0.json").read_text())
    for m in (*frozen.METRICS, "action_counts"):
        pilot.joint.assert_same_state(native[0][m], old["results"]["late"][m], "v97_cached_v96_native")
    pilot.joint.assert_same_state(early[3].contract, native[3].contract, "v97_same_day_lower_contract")
    return native[3].contract


def smoke_fixed(root):
    observed = fixed_replay(root, DAYS[0], FixedExposure())
    raw, contract = fixed_replay(root, DAYS[0], None)
    pilot.joint.assert_same_state(observed["native_results"], raw, "v97_passive_fixed_observer")
    pilot.joint.assert_same_state(observed["contract"], contract, "v97_fixed_contract")
    return contract


def smoke(root, v96_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v97_*.py", "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        actor, fixed = pool.submit(smoke_actor, root, v96_root), pool.submit(smoke_fixed, root)
        pilot.joint.assert_same_state(actor.result(), fixed.result(), "v97_fixed_vs_learned_physical_contract")
    pilot.write_json(root / "smoke/done.json", dict(status="tests_native_actor_and_fixed_contracts_passed",
        training_runs=0, formal_benchmark_cells_read=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v96-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=TOTAL_ITEMS)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v96_root)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.v96_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_item, args.out_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

#!/usr/bin/env python3
"""Diagnose V89's unresolved primary CV effect without changing action laws."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from audit_freqduet_v81_projection_tradeoff import repeat_variance
from audit_freqduet_v82_mechanism import variance_effect
import run_freqduet_v82_local_global_regularity as local
import run_freqduet_v89_conditional_cross as source


pilot, experiment = source.pilot, source.experiment
ITEMS, MODES = source.ITEMS, ("late", "no_gain_early")
NOISE_METRICS = ("journey_min", "cv_squared", "readiness_delay_s", "holding_s_per_trip")


def interval(values):
    stats = pilot.event.labels.expanded.paired_interval(values, 90000)
    return {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}


def day_interval(rows, getter):
    return interval([float(np.mean([getter(r) for r in rows if r["day"] == day])) for day in source.DAYS])


def review(root, original_root):
    rows = [json.loads((original_root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    original = json.loads((original_root / "summary/status.json").read_text())
    primary = {m: original["effects"]["no_gain_vs_late"][m] for m in NOISE_METRICS}
    for m in NOISE_METRICS:
        np.testing.assert_allclose(np.mean([r["effects"]["no_gain_vs_late"][m] for r in rows]),
                                   primary[m]["mean"], atol=1e-12, rtol=0)
    noise = {}
    for seed in (None, *pilot.SEEDS):
        chosen = [r for r in rows if seed is None or r["seed"] == seed]
        values = np.asarray([[np.mean([[r["effects"]["no_gain_vs_late"][m] for m in NOISE_METRICS]
            for r in chosen if (r["day"], r["rep"]) == (day, rep)], axis=0)
            for rep in range(3)] for day in source.DAYS])
        noise[str(seed) if seed is not None else "combined"] = repeat_variance(values)
    specificity = {m: day_interval(rows, lambda r: r["effects"]["no_gain_vs_late"][m] -
                                  r["effects"]["gain_vs_late"][m]) for m in NOISE_METRICS}
    result = dict(status="v89_primary_noise_review_complete", primary=primary,
        noise_metrics=NOISE_METRICS, sampling_noise=noise, no_gain_minus_gain=specificity,
        scope="consumed V89 days; conditional sampling variation and day heterogeneity, not training-seed inference",
        specificity_has_different_intervention_doses=True, original_primary_means_exact=True,
        new_rollouts=0, actor_updates=0, main_policy_changed=False, original_gate_changed=False)
    pilot.write_json(root / "review/status.json", result)
    print("V90_PRIMARY_NOISE_REVIEW " + json.dumps(result), flush=True)


def follower_label(capture, record, holding):
    key = (record["trip_id"], record["station_id"], record["direction"])
    follower = (capture.followers[key], record["station_id"], record["direction"])
    ready = capture.ready.get(follower)
    if ready is None or key not in capture.departures:
        return None
    context = record["context"]
    gap = ready - capture.ready[key]
    return dict(resolved=True, actual_follower_gap_s=gap,
        raw_gap_prediction_error_s=context.follower_departure_gap_s - gap if context.follower_evidence_valid else None,
        follower_action_s=holding.get(follower[:2]), follower_departure_time_s=capture.departures.get(follower),
        decision_time_s=capture.ready[key])


def paired_timing(base, candidate):
    def keyed(rows):
        return {(r["trip_id"], r["station_id"], r["direction"]): r for r in rows}

    b, c = keyed(base["events"]), keyed(candidate["events"])
    groups = defaultdict(lambda: defaultdict(list))
    for key in b.keys() & c.keys():
        group = groups[str(key[1:])]
        group["arrival_delta_s"].append(c[key]["arrival_time_s"] - b[key]["arrival_time_s"])
        if b[key]["headway_s"] is not None and c[key]["headway_s"] is not None:
            group["headway_delta_s"].append(c[key]["headway_s"] - b[key]["headway_s"])
        if key in base["departures"] and key in candidate["departures"]:
            group["departure_delta_s"].append(candidate["departures"][key] - base["departures"][key])
    return dict(unmatched_base=len(b.keys() - c.keys()), unmatched_candidate=len(c.keys() - b.keys()),
        groups={g: {m: dict(count=len(v), mean=float(np.mean(v)), rms=float(np.sqrt(np.mean(np.square(v)))))
                    for m, v in fields.items()} for g, fields in groups.items()})


class RegularityAudit(source.ConditionalCross):
    def __init__(self, seed, mode):
        super().__init__(seed, mode)
        self.capture, self.laws = local.LocalGlobalAudit("base"), []

    def attach(self, runner):
        super().attach(runner)
        self.capture.attach(runner)

    def detach(self):
        self.capture.detach()

    def __call__(self, policy, states, features, theta):
        actual = super().__call__(policy, states, features, theta)
        with torch.no_grad():
            late_z, early_z = policy.forward(states), self.early.forward(states)
            mask = policy.feasible_action_mask(states)
            laws = [experiment.distribution(late_z, early_z, mask, self.shift, name)[0].numpy().astype(float)
                    for name in source.audit.SOURCES]
        late, early = laws
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        group = source.audit.context_group(context.evidence_valid and context.follower_evidence_valid,
            float((late - early) @ pilot.step.ACTIONS), float((late - early) @ features[:, 0]))
        self.laws.append((late / late.sum(), early / early.sum(), group))
        return actual

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        if len(records) != len(self.laws):
            raise AssertionError("completed labels must align with the original decisions")
        holding = {(r["trip_id"], r["station_id"]): r["action_s"] for r in holds}
        groups = defaultdict(list)
        for record, (late, early, group) in zip(records, self.laws):
            groups[group].append(local.local_effect(record["context"], record["features"], late, early,
                follower_label(self.capture, record, holding)))
        events = runner.env.headway_events.events
        self.arrival = dict(events=events, departures=self.capture.departures)
        self.audit_summary = dict(
            early_minus_late_local_formula={g: local.summarize_local(rows) for g, rows in groups.items()},
            arrival=dict(stop_direction=local.arrival_decomposition(events, lambda e: (e["station_id"], e["direction"])),
                four_hour_band=local.arrival_decomposition(events, lambda e: int(e["arrival_time_s"] // 14400))))


def replay(root, seed, day, rep, mode):
    rule, credit = RegularityAudit(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v90_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s)
    return row, events, holds, rule


def run_cell(root, original_root, index, check_paths=False):
    path = root / "pairs" / f"p{index}.json"
    if path.exists() and not check_paths:
        return
    seed, day, rep = ITEMS[index]
    old = json.loads((original_root / "pairs" / f"p{index}.json").read_text())
    audits, arrivals = {}, {}
    for mode in MODES:
        row, events, holds, rule = replay(root, seed, day, rep, mode)
        for metric in (*experiment.METRICS, "action_counts"):
            pilot.joint.assert_same_state(old["results"][mode][metric], row[metric], "v90_original_v89_endpoints")
        pilot.joint.assert_same_state(old["contexts"][mode], rule.context_summary, "v90_original_v89_contexts")
        if check_paths:
            native = source.replay(root, seed, day, rep, mode)
            pilot.joint.assert_same_state(native[1], events, "v90_passive_event_path")
            pilot.joint.assert_same_state(native[2], holds, "v90_passive_holding_path")
        np.testing.assert_allclose(row["cv_squared"], rule.audit_summary["arrival"]["stop_direction"]["cv_squared"], atol=1e-12, rtol=0)
        covered = rule.audit_summary["early_minus_late_local_formula"]["more_hold_gain_not_up"]
        if not covered["observed"]["completed_departure_gain_delta"]:
            raise AssertionError("primary group has no completed follower departure labels")
        audits[mode], arrivals[mode] = rule.audit_summary, rule.arrival
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, audits=audits,
        timing=paired_timing(arrivals["late"], arrivals["no_gain_early"]), original_endpoints_and_contexts_exact=True,
        full_paths_checked=check_paths))
    print("V90_REGULARITY_CELL_DONE " + str(index), flush=True)


def cv_parts(base, candidate):
    parts = variance_effect(base["stop_direction"], candidate["stop_direction"])
    for partition in ("stop_direction", "four_hour_band"):
        for key in ("within_group_cv_squared", "between_group_cv_squared"):
            parts[partition + "_" + key] = candidate[partition][key] - base[partition][key]
    return parts


def pooled_local(rows, mode):
    result = {}
    for group in source.audit.GROUPS:
        entries = [r["audits"][mode]["early_minus_late_local_formula"][group] for r in rows
                   if group in r["audits"][mode]["early_minus_late_local_formula"]]
        n = sum(e["decisions"] for e in entries)
        counts = {m: sum(e["observed"][m] for e in entries) for m in local.LOCAL_METRICS}
        result[group] = dict(decisions=n, observed=counts, means={m:
            sum(e["means"][m] * e["observed"][m] for e in entries if e["observed"][m]) / counts[m]
            if counts[m] else None for m in local.LOCAL_METRICS},
            predicted_nonharm_but_completed_ready_harm=sum(e["predicted_nonharm_but_completed_ready_harm"] for e in entries))
    return result


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    for r in rows:
        r["parts"] = cv_parts(r["audits"]["late"]["arrival"], r["audits"]["no_gain_early"]["arrival"])
    def describe(chosen):
        partitions = {}
        for partition in ("stop_direction", "four_hour_band"):
            def contributions(r, mode):
                return r["audits"][mode]["arrival"][partition]["group_cv_contributions"]
            keys = sorted(set().union(*(contributions(r, mode) for r in chosen for mode in MODES)))
            partitions[partition] = {k: day_interval(chosen, lambda r: contributions(r, "no_gain_early").get(k, 0.) -
                contributions(r, "late").get(k, 0.)) for k in keys}
        return dict(cv_parts={k: day_interval(chosen, lambda r: r["parts"][k]) for k in chosen[0]["parts"]},
            group_cv_contribution_changes=partitions, local={m: pooled_local(chosen, m) for m in MODES})
    timing_keys = sorted(set().union(*(r["timing"]["groups"] for r in rows)))
    timing = {g: {m: dict(cell_means_mean=float(np.mean([r["timing"]["groups"][g][m]["mean"] for r in rows
        if g in r["timing"]["groups"] and m in r["timing"]["groups"][g]])),
        cell_rms_mean=float(np.mean([r["timing"]["groups"][g][m]["rms"] for r in rows
        if g in r["timing"]["groups"] and m in r["timing"]["groups"][g]])))
        for m in ("arrival_delta_s", "headway_delta_s", "departure_delta_s")
        if any(g in r["timing"]["groups"] and m in r["timing"]["groups"][g] for r in rows)} for g in timing_keys}
    result = dict(status="conditional_regularity_diagnosis_complete", combined=describe(rows),
        by_seed={str(seed): describe([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        matched_stop_timing=timing, unmatched_arrivals={key: sum(r["timing"][key] for r in rows)
            for key in ("unmatched_base", "unmatched_candidate")}, rollouts=2 * len(rows),
        original_endpoints_and_contexts_exact=True, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, original_gate_changed=False, controller_admitted=False,
        scope="same consumed V89 days; additive metric decomposition and completed-label formula audit, not causal mediation or action values")
    pilot.write_json(root / "summary/status.json", result)
    print("V90_REGULARITY_DIAGNOSIS_COMPLETE " + json.dumps(result["combined"]["cv_parts"]), flush=True)


def smoke(root, original_root):
    for pattern in ("test_v82_local_global_regularity.py", "test_v90_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", pattern, "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(run_cell, root, original_root, ITEMS.index((seed, source.DAYS[0], 0)), True)
                     for seed in pilot.SEEDS]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_exact_native_and_conditional_paths_completed_labels_passed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review", "smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v89-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "review":
        review(args.out_root, args.v89_root)
    elif args.command == "smoke":
        smoke(args.out_root, args.v89_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, args.v89_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

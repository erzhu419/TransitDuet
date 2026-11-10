#!/usr/bin/env python3
"""Frozen-law release-gap/realized-holding counterfactual regimes."""

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

import run_freqduet_v91_timing_components as timing
from run_freqduet_v92_elapsed_review import covariance_moments


source, pilot = timing.source, timing.pilot
DAYS = tuple(range(1931001, 1931017))
ITEMS = list(itertools.product(pilot.SEEDS, DAYS, range(3)))
MODES = ("late", "preserve", "loss_only", "no_gain_early")
GROUPS = ("nonloss", "compensation_loss")
CONTRASTS = dict(preserve_vs_late=("preserve", "late"), loss_vs_late=("loss_only", "late"),
    all_vs_late=("no_gain_early", "late"), loss_added_to_preserve=("no_gain_early", "preserve"),
    preserve_added_to_loss=("no_gain_early", "loss_only"))
GROUP_FIELDS = ("count", "expected_holding_reduction_s", "early_minus_late_loss",
    "release_gap_s", "realized_holding_gap_s", "other_elapsed_gap_s")
METRICS = (*source.experiment.METRICS, "dispatch_gap_mean_s", "dispatch_gap_variance_s2", "dispatch_cv_squared")


def compensation_delta(late, early, release_gap, holding_gap, target):
    if target <= 0:
        raise ValueError("release compensation requires an executable positive target")
    late, early = [p / p.sum() for p in (np.asarray(late, float), np.asarray(early, float))]
    losses = ((release_gap + holding_gap + pilot.step.ACTIONS - target) / target) ** 2
    return float((early - late) @ losses)


def selected(mode, group):
    return mode == "no_gain_early" or (mode == "preserve" and group == "nonloss") or (
        mode == "loss_only" and group == "compensation_loss")


def contrasts(values):
    result = {name: values[a] - values[b] for name, (a, b) in CONTRASTS.items()}
    result["interaction"] = values["no_gain_early"] - values["preserve"] - values["loss_only"] + values["late"]
    return result


def coordinate_moments(events):
    previous, headways, components = {}, [], []
    for row in events:
        group = row["station_id"], row["direction"]
        if group in previous:
            before = previous[group]
            headways.append(row["time_s"] - before["time_s"])
            components.append([row[k] - before[k] for k in timing.COMPONENTS])
        previous[group] = row
    result = covariance_moments(headways, components)
    result["covariance"] = result["covariance"].tolist()
    return result


class ReleaseHolding(timing.TimingAudit):
    def __init__(self, seed, mode):
        super().__init__(seed, mode)
        self.completed, self.cursor = {}, 0
        self.groups = {g: np.zeros(len(GROUP_FIELDS)) for g in GROUPS}
        self.selected_counts = dict.fromkeys(GROUPS, 0)

    def observable_coordinates(self, bus, context):
        for row in self.departures[self.cursor:]:
            self.completed[row["trip_id"], row["station_id"], row["direction"]] = row
        self.cursor = len(self.departures)
        before = self.completed[int(bus.forward_predecessor_trip_id), int(bus.last_station.station_id), bool(bus.direction)]
        # Another bus may have overwritten the recorder's latest departure since this context was captured.
        ready_time = before["time_s"] + context.forward_headway_s
        if not before["time_s"] <= ready_time <= self.env.current_time:
            raise AssertionError("release compensation used a future departure or pre-action context")
        launch = self.launches[int(bus.trip_id)]
        held = float(bus.episode_hold_vehicle_seconds) - launch["holding_counter_at_launch"]
        if held < 0:
            raise AssertionError("realized holding was not measured since this trip's launch")
        release_gap = launch["time_s"] - before["release"]
        holding_gap = held - before["realized_holding"]
        other_gap = context.forward_headway_s - release_gap - holding_gap
        return release_gap, holding_gap, other_gap

    def __call__(self, policy, states, features, theta):
        native = source.experiment.LogitCross.__call__(self, policy, states, features, theta)
        with torch.no_grad():
            z = policy.forward(states), self.early.forward(states)
            mask = policy.feasible_action_mask(states)
            laws = [torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), -1) for logits in z]
            late, early = [law[0].numpy().astype(float) for law in laws]
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        holding_difference = float((late - early) @ pilot.step.ACTIONS)
        group = source.audit.context_group(context.evidence_valid and context.follower_evidence_valid,
            holding_difference, float((late - early) @ features[:, 0]))
        key = group, len(bus.passengers) > 0, -features[-1, 2] > 0, -features[-1, 3] > 0
        self.risks[key] += [1., holding_difference]
        if group != "more_hold_gain_not_up":
            return native
        release, held, other = self.observable_coordinates(bus, context)
        delta = compensation_delta(late, early, release, held, context.target_headway_s)
        split = "compensation_loss" if delta > 0 else "nonloss"
        self.groups[split] += [1., holding_difference, delta, release, held, other]
        if not selected(self.mode, split):
            return native
        self.early_decisions += 1
        self.selected_groups[group] += 1
        self.selected_counts[split] += 1
        self.max_selected_holding_increase = max(self.max_selected_holding_increase,
            float((early - late) @ pilot.step.ACTIONS))
        self.predicted_gain_change += float((early - late) @ features[:, 0])
        return laws[1]

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.group_summary = {g: dict(zip(GROUP_FIELDS, row.tolist()), selected=self.selected_counts[g])
                              for g, row in self.groups.items()}
        self.coordinates = coordinate_moments(self.arrivals)
        gaps = np.asarray(runner.env._actual_dispatch_gaps_s)
        self.dispatch = dict(dispatch_gap_mean_s=float(gaps.mean()), dispatch_gap_variance_s2=float(gaps.var()),
            dispatch_cv_squared=float(gaps.var() / gaps.mean() ** 2))
        np.testing.assert_allclose(self.coordinates["cv_squared"], self.summary["arrival"]["cv_squared"], atol=1e-10, rtol=0)
        if self.max_selected_holding_increase > 1e-10 or self.predicted_gain_change < -1e-10:
            raise AssertionError("replacement left V89's lower-holding/nonworse-local-gain group")


def replay(root, seed, day, rep, mode):
    rule, credit = ReleaseHolding(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v93_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    return row, events, holds, rule


def run_cell(root, index):
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return
    seed, day, rep = ITEMS[index]
    rows, contexts, moments, series, events = {}, {}, {}, {}, {}
    for mode in MODES:
        row, event, _, rule = replay(root, seed, day, rep, mode)
        rows[mode], contexts[mode] = row, rule.group_summary
        moments[mode], series[mode], events[mode] = rule.coordinates, rule.summary["arrival"], event
        rule.save(root / "records" / f"p{index}_{mode}.npz")
    effects = {name: {} for name in (*CONTRASTS, "interaction")}
    for metric in METRICS:
        for name, value in contrasts({mode: row[metric] for mode, row in rows.items()}).items():
            effects[name][metric] = value
    covariance_effects = {}
    for name, (candidate, base) in CONTRASTS.items():
        delta = (np.asarray(moments[candidate]["covariance"]) - moments[base]["covariance"]) / moments[base]["mean"] ** 2
        parts = {"release_diagonal": float(delta[0, 0]), "holding_diagonal": float(delta[1, 1]),
            "other_diagonal": float(delta[2, 2]), "release_holding_cross": float(2 * delta[0, 1]),
            "release_other_cross": float(2 * delta[0, 2]), "holding_other_cross": float(2 * delta[1, 2])}
        measured = timing.component_effect(series[base], series[candidate])
        np.testing.assert_allclose(sum(parts.values()) + measured["mean_normalization"],
                                   effects[name]["cv_squared"], atol=1e-10, rtol=0)
        covariance_effects[name] = dict(**parts, **measured)
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, results=rows, contexts=contexts,
        moments=moments, effects=effects, timing_effects=covariance_effects,
        alignment={mode: pilot.event.alignment(events["late"], event) for mode, event in events.items() if mode != "late"}))
    print("V93_RELEASE_HOLDING_CELL_DONE " + str(index), flush=True)


def intervals(rows, field):
    result = {}
    for contrast in rows[0][field]:
        result[contrast] = {}
        for metric in rows[0][field][contrast]:
            values = [float(np.mean([r[field][contrast][metric] for r in rows if r["day"] == day])) for day in DAYS]
            stats = pilot.event.labels.expanded.paired_interval(values, 93000)
            result[contrast][metric] = {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    stats = intervals(rows, "effects")
    primary = stats["preserve_vs_late"]
    joint = primary["passenger_person_s"]["ci95"][1] < 0 and all(primary[m]["ci95"][1] <= 0
        for m in ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete"))
    result = dict(status="release_holding_counterfactual_complete", effects=stats, timing_effects=intervals(rows, "timing_effects"),
        by_seed={str(seed): dict(effects=intervals([r for r in rows if r["seed"] == seed], "effects"),
            timing_effects=intervals([r for r in rows if r["seed"] == seed], "timing_effects")) for seed in pilot.SEEDS},
        absolute={mode: {m: float(np.mean([r["results"][mode][m] for r in rows])) for m in METRICS} for mode in MODES},
        context_totals={mode: {g: {k: float(sum(r["contexts"][mode][g][k] for r in rows))
            for k in (*GROUP_FIELDS, "selected")} for g in GROUPS} for mode in MODES},
        primary_contrast="preserve_vs_late", mechanism_contrast="loss_added_to_preserve",
        original_joint_diagnostic_gate_passed=joint, evaluation_rollouts=len(ITEMS) * len(MODES),
        original_actors=2, demand_days=len(DAYS), sampler_repeats=3,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, alternative_arms_are_not_a_winner_search=True,
        scope="new-day dynamic-regime intervention conditional on two actors, not natural mediation or training-seed confirmation",
        classifier="expected squared release-plus-executed-holding gap error; excludes other elapsed, no future labels",
        unequal_regime_intervention_doses=True)
    pilot.write_json(root / "summary/status.json", result)
    print("V93_RELEASE_HOLDING_COMPLETE " + json.dumps(dict(primary=primary, joint_gate=joint)), flush=True)


def smoke_case(root, seed):
    day = source.DAYS[0]
    summaries = {}
    for mode in MODES:
        current = replay(root, seed, day, 0, mode)
        summaries[mode] = current[3].group_summary
        if mode in ("late", "no_gain_early"):
            original = source.replay(root, seed, day, 0, mode)
            for metric in (*source.experiment.METRICS, "action_counts"):
                pilot.joint.assert_same_state(original[0][metric], current[0][metric], "v93_native_endpoints")
            pilot.joint.assert_same_state(original[1], current[1], "v93_native_event_path")
            pilot.joint.assert_same_state(original[2], current[2], "v93_native_executed_holding")
            pilot.joint.assert_same_state(original[3], current[3].context_summary, "v93_native_conditional_counts")
        if mode != "late" and current[3].early_decisions == 0:
            raise AssertionError("smoke must exercise every intervention, not silently retain native")
    if any(summaries["late"][g]["count"] == 0 for g in GROUPS):
        raise AssertionError("smoke must observe both release compensation strata")


def smoke(root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v93_*.py", "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(smoke_case, root, seed) for seed in pilot.SEEDS]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_native_and_all_replacement_paths_and_both_strata_passed",
        prospective_days_read=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))

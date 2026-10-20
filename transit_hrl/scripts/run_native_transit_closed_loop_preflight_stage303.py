#!/usr/bin/env python3
"""Two-decision on-policy software preflight, with frozen Stage301 controllers."""

import argparse
import copy
from functools import partial
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_comparison_stage301 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_closed_loop_preflight_stage303_v1"
SOURCE = "native_transit_service_comparison_stage301_20261020_r1"
ARMS = ("raw", "service_only", "joint_budgeted")
PROBES = (8, 9)
PROFILE = 0
parent = source.parent


def directory(run): return ROOT/"results"/run


def scene_seed(root, scenario):
    return 2090000000+source.ROOTS.index(root)*100000+source.SCENARIOS.index(scenario)*1000


def future_seed(root, scenario, probe, draw):
    return 2140000000+source.ROOTS.index(root)*1000000+source.SCENARIOS.index(scenario)*100000+PROBES.index(probe)*1000+draw


def contract():
    return {"source": SOURCE, "locked_policy": source.read(source.directory(SOURCE)/"inputs/locked_policy.json"),
        "scope": "software_preflight_only; all_pilot_scenes_excluded_from_future_performance_claims",
        "arms": list(ARMS), "profile": PROFILE, "probes": list(PROBES),
        "cohort": "10_fresh_pilot_scenes; both_roots_all5regimes; one_fixed_profile_not_a_training_seed",
        "history": "replay_each_controller_actual_prior_actions; rebuild_physical_state_context_and_P8S8_at_second_probe",
        "proposal_screen": "unchanged_P8_S8_all6actions; frozen_raw_fleet_critic_and_cutoffs; no_reranking",
        "future": "actual_scene2090000000_namespace; independent_inner_future2140000000_namespace; no_E32_selection",
        "qualification": "unchanged_weights; full_native_ledgers; exact_nontrivial_history_identity; registered_prior_overrides",
        "budget": {"tasks": 30, "native_episodes": 4590},
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def replay_plan_class(base, history, continuation_seed=None):
    history = dict(history)
    class ReplayPlan(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.applied_choices = {}

        def select_candidate(self, candidates):
            index = len(self.decisions)
            selected = super().select_candidate(candidates)
            if index in history: selected = history[index]
            self.applied_choices[index] = selected
            if index == self.probe_index:
                self.probe["selected"] = selected
                if continuation_seed is not None:
                    tape = self.env.scenario_tape
                    self.env.scenario_tape = type(tape)(continuation_seed, version=tape.version)
                    self.probe.update(continuation_seed=continuation_seed,
                        continuation_start_time_s=float(self.env.current_time))
            return selected
    return ReplayPlan


def choose(run, arm, row, first, screen):
    raw, _, logits = parent.native.controls([row])
    if arm == "raw": return int(raw[PROFILE, 0])
    proposal = parent.native.proposals([row], raw, logits, first)
    _, keep = parent.native.rule.screen(proposal, raw, screen)
    service = parent.native.rule.paired_values(screen, proposal, raw)[..., 2].mean(axis=2)
    locked = source.read(directory(run)/"inputs/preregistration.json")["contract"]["locked_policy"]
    if arm == "service_only": accepted = keep & (service <= locked["service"]["threshold"])
    else:
        model = source.read(parent.directory(source.previous.selection.PARENT)/"fit_raw/model_before_calibration.json")
        fleet = parent.learning.predict(model, parent.learning.features([row], raw, proposal, first, screen))
        accepted = keep & (fleet+service <= locked["joint"]["threshold"])
    return int(np.where(accepted, proposal, raw)[PROFILE, 0])


def run_pilot(run, root, scenario, arm):
    import torch
    from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit, qualify_timetable
    native, model, checkpoint = parent.native.native_inputs(root)
    shared = native.shared
    output = directory(run)/"cells"/str(root)/scenario/arm
    scene, episodes = scene_seed(root, scenario), 0

    def episode(probe, history, label, *, candidate=None, reference=None, future=None):
        nonlocal episodes
        runner = shared.native.make_runner(root, scenario, output/"raw"/label, checkpoint)
        before = shared.native.model_arrays(runner)
        plan = replay_plan_class(NativeTimetableContextAudit, history, future)(runner.env, model, probe,
            forced_candidate=candidate, reference=reference)
        runner._upper_callback_v2 = plan
        result = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
        after = shared.native.model_arrays(runner)
        if any(not np.array_equal(before[k], after[k]) for k in before): raise ValueError("Closed-loop preflight updated learned weights")
        trips = [{"trip_id": int(t), **r} for t, r in sorted(runner.env.completed_trip_observations.items())]
        if len(trips) != result["trips_completed"] or len(trips) != 262 or plan.probe is None:
            raise ValueError("Incomplete closed-loop native episode")
        compact = {"condition": "observer" if candidate is None else f"candidate_{candidate}",
            "scenario": scenario, "scene_seed": scene,
            **shared.native.source.source_spec.authority.routing.compact_row(result),
            "schedule": plan.summarize(), "supply": plan.supply_summary(), "probe": plan.probe,
            "window_ledger_qualified": True}
        compact["window_panels"] = shared.source.validate_audits({**compact, "audits": plan.audits, "completed_trips": trips})
        overrides = dict(history)
        if candidate is not None: overrides[probe] = candidate
        shared.validate_scoring({**compact, "condition": "prefix_wait_full_fleet"}, model,
            strengths=shared.source.incumbent.CANDIDATES["rolling_plus_incumbent"], choice_overrides=overrides)
        native.qualify_observer(compact)
        if any(plan.applied_choices[p] != action for p, action in history.items()): raise ValueError("Past controller choices were not replayed")
        episodes += 1
        return compact, plan

    def context(plan, probe):
        c = next(c for c in plan.contexts if c["index"] == probe)
        row = {"root": root, "scenario": scenario, "replica": 0, "scene_seed": scene, "probe_index": probe,
            **{k: copy.deepcopy(c[k]) for k in ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")},
            "original_choice": c["greedy"], "context": native.labels.training_context(plan.snapshot),
            "occupancy_context": {k: copy.deepcopy(c[k]) for k in native.context_source.OCCUPANCY_KEYS},
            "timetable_context": copy.deepcopy(c["timetable_context"])}
        qualify_timetable(c, row)
        return row

    def panel(row, reference, history, draws):
        records = []
        probe = row["probe_index"]
        for draw in draws:
            seed = future_seed(root, scenario, probe, draw)
            torch.set_rng_state(torch.tensor(reference.snapshot["rng"]["torch"], dtype=torch.uint8))
            baseline, current = episode(probe, history, f"probe{probe}/draw{draw}/observer", reference=reference, future=seed)
            values = {}
            for candidate in [row["original_choice"], *[a for a in range(6) if a != row["original_choice"]]]:
                torch.set_rng_state(torch.tensor(current.snapshot["rng"]["torch"], dtype=torch.uint8))
                branch, branch_plan = episode(probe, history, f"probe{probe}/draw{draw}/candidate{candidate}",
                    candidate=candidate, reference=current, future=seed)
                shared.qualify(branch, baseline, candidate=candidate)
                if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
                    raise ValueError("Closed-loop branch lost its independent inner future")
                cost = branch["service_cost_restricted"]-baseline["service_cost_restricted"]
                fleet = max(branch["peak_fleet"]-12, 0)**2/12-max(baseline["peak_fleet"]-12, 0)**2/12
                values[str(candidate)] = [cost, fleet, cost-fleet,
                    branch["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"]]
            if values[str(row["original_choice"])] != [0., 0., 0., 0.]: raise ValueError("Closed-loop identity differs")
            records.append({"draw": draw, "future_seed": seed, "values": values})
        write_json(output/f"probe{probe}/draws{draws[0]}to{draws[-1]}.json", records)
        return np.asarray([[[r["values"][str(a)] for a in range(6)] for r in records]])

    history, decisions, witness = {}, [], None
    for probe in PROBES:
        _, plan = episode(probe, history, f"actual/probe{probe}")
        row = context(plan, probe)
        if arm == "raw" and probe == PROBES[0]:
            original_curve = row["candidates"][row["original_choice"]]["commitment"]["prefix_planned_s"]
            forced = next((i for i, c in enumerate(row["candidates"]) if c["commitment"]["prefix_planned_s"] != original_curve), None)
            if forced is None: raise ValueError("Pilot did not exercise a nontrivial prior timetable action")
            baseline, ref = episode(PROBES[1], {probe: forced}, "witness/observer")
            torch.set_rng_state(torch.tensor(ref.snapshot["rng"]["torch"], dtype=torch.uint8))
            identity, _ = episode(PROBES[1], {probe: forced}, "witness/identity",
                candidate=ref.probe["greedy"], reference=ref)
            shared.qualify(identity, baseline, candidate=ref.probe["greedy"])
            witness = {"prior_probe": probe, "forced_candidate": forced, "exact_identity": True}
        if arm == "raw": first = screen = None
        else:
            first = panel(row, plan, history, list(range(8)))
            screen = panel(row, plan, history, list(range(8, 16)))
        selected = choose(run, arm, row, first, screen)
        decisions.append({"probe": probe, "history_before": dict(history), "selected": selected, "row": row})
        history[probe] = selected
        print(f"CLOSED_LOOP_DECISION_QUALIFIED {root} {scenario} {arm} probe={probe} action={selected}", flush=True)
    final, _ = episode(PROBES[-1], history, "actual/final")
    if episodes != (5 if arm == "raw" else 227): raise ValueError("Closed-loop preflight budget differs")
    write_json(output/"result.json", {"protocol": PROTOCOL, "contract": contract(), "software_qualified": True,
        "root": root, "scenario": scenario, "arm": arm, "native_episodes": episodes,
        "profile": PROFILE, "decisions": decisions, "nontrivial_history_witness": witness,
        "performance_claim": False,
        "actual_metrics": {k: final[k] for k in ("service_cost_restricted", "restricted_wait_horizon_min", "peak_fleet")}})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    p.add_argument("--root", type=int, required=True, choices=source.ROOTS)
    p.add_argument("--scenario", required=True, choices=source.SCENARIOS); p.add_argument("--arm", required=True, choices=ARMS)
    a = p.parse_args()
    if source.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract(): p.error("Changed closed-loop pilot registration")
    run_pilot(a.run_name, a.root, a.scenario, a.arm)


if __name__ == "__main__": main()

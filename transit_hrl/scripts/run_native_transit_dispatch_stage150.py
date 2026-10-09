#!/usr/bin/env python3
"""Qualify causal signed departures with frozen native lower weights."""

import argparse
import copy
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_frontier_stage149 as frontier
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_dispatch_stage150_v1"
ROOTS = frontier.ROOTS
CONDITIONS = {
    "hiro_baseline": ("hiro", 0, None),
    "legacy_channels_minus60": ("channels", 0, -60),
    "signed_minus60": ("channels", 120, -60),
    "signed_zero": ("channels", 120, 0),
    "signed_plus60": ("channels", 120, 60),
}


def contract():
    return {"source_run": frontier.SOURCE_RUN, "source_method": "physical_lower",
        "checkpoint_ep": 299, "roots": list(ROOTS), "scenario": "low_noise",
        "conditions": {key: {"coupling": mode, "lookahead_s": lead, "fixed_command_s": command}
                       for key, (mode, lead, command) in CONDITIONS.items()},
        "training_updates": 0, "episode_clock_s": 61380, "fleet": 12,
        "scene_seeds": "reuse_stage148_registered_low_noise_scene",
        "baseline_gate": "exact_HIRO_source_and_zero_dispatch_neutral_source_reproduction",
        "commitment": "current_clock_observation_once_before_earliest_signed_release",
        "unchanged": ["networks", "RE_SAC", "frequency", "lower_reward", "fleet_cap"],
        "stage": "execution_qualification_not_channels_trained_policy_evidence",
        "artifacts": "compact_JSON_no_checkpoint_download"}


def dispatch_ledger(env, queries):
    times = dict(queries)
    if len(times) != len(queries) or set(times) != {trip.launch_turn for trip in env.timetables}:
        raise RuntimeError("Dispatch must query upper exactly once per trip")
    advances, delays, shifts, late, leads, commitment_clipped = [], [], [], [], [], 0
    for trip in env.timetables:
        nominal = float(trip.launch_time)
        decision = times[trip.launch_turn]
        expected = max(0.0, nominal - env._upper_dispatch_lookahead_s)
        if decision != expected:
            raise RuntimeError("Dispatch queried outside its causal commitment clock")
        leads.append(nominal - decision)
        planned = float(trip._original_launch + trip._delta_t)
        due = max(decision, planned)
        commitment_clipped += int(planned < decision)
        if not trip.launched:
            continue
        actual = float(trip._actual_launch_time)
        if actual < due:
            raise RuntimeError("Departure preceded its committed release")
        shift = actual - nominal
        shifts.append(shift)
        late.append(actual - due)
        if shift < 0:
            advances.append(shift)
        elif shift > 0:
            delays.append(shift)
    if not shifts:
        raise RuntimeError("Signed dispatch qualification launched no trips")
    return {"queries": len(queries), "launched": len(shifts),
        "advance_count": len(advances), "delay_count": len(delays),
        "nominal_count": len(shifts) - len(advances) - len(delays),
        "actual_shift_mean_s": float(np.mean(shifts)),
        "actual_shift_min_s": min(shifts), "actual_shift_max_s": max(shifts),
        "release_lateness_mean_s": float(np.mean(late)),
        "release_lateness_p95_s": float(np.quantile(late, .95)),
        "decision_lead_median_s": float(np.median(leads)),
        "commitment_clipped_count": commitment_clipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)
    source_path, source = frontier.load_source("physical_lower", args.seed)
    checkpoint = source_path.parent.with_name(source_path.parent.name + "_raw") / "training" / (
        f"F_freqduet_harmonic_hiro_seed{args.seed}") / "checkpoints"
    references = {row["condition"]: row for row in source["evaluation"] if row["scenario"] == "low_noise"}
    raw = raw_directory(args.output)
    rows = []
    for condition, (mode, lookahead, command) in CONDITIONS.items():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cfg = frontier.source_spec.configure(
            load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
            "physical_lower", args.seed, preflight=False)
        cfg["coupling"]["coupling_mode"] = mode
        cfg["env"].update(copy.deepcopy(frontier.source_spec.routing.SCENARIOS["low_noise"]))
        cfg["logging"] = {"logs_dir": str(raw / condition)}
        runner = TransitDuetV2Runner(cfg, device="cpu")
        runner.load_checkpoint(checkpoint, 299, require_deployment_state=True)
        if mode == "channels" and lookahead == 0:
            runner.dispatch_lookahead_s = 0.0  # Reproduce the original execution defect.
        before = model_arrays(runner)
        probe = NativePolicyProbe(runner.upper_trainer.policy_net, "upper", fixed_action_s=command)
        probe.policy.get_action = probe.get_action
        callback, queries = runner._upper_callback_v2, []
        def record_query(state, trip):
            queries.append((trip.launch_turn, float(runner.env.current_time)))
            return callback(state, trip)
        runner._upper_callback_v2 = record_query
        reference = references["baseline" if condition == "hiro_baseline" else "neutral_upper"]
        row = runner.run_episode(300, training=False, N_fleet_override=12,
            scenario_seed=reference["scene_seed"], record_diagnostics=False)
        exact = condition in {"hiro_baseline", "signed_zero"}
        check_episode(row, reference, baseline=exact)
        if exact and row["lower_action_mean"] != reference["lower_action_mean"]:
            raise RuntimeError("Dispatch extraction changed preserved lower actions")
        if condition == "hiro_baseline" and row["upper_delta_mean"] != reference["upper_delta_mean"]:
            raise RuntimeError("Dispatch extraction changed preserved HIRO actions")
        ledger = dispatch_ledger(runner.env, queries)
        if ledger["advance_count"] != 0 and condition in {"hiro_baseline", "legacy_channels_minus60", "signed_zero"}:
            raise RuntimeError("Uncommanded advance in preserved/zero dispatch")
        if condition == "signed_minus60" and not ledger["advance_count"]:
            raise RuntimeError("Negative dispatch command still cannot advance a trip")
        if condition == "signed_plus60" and not ledger["delay_count"]:
            raise RuntimeError("Positive dispatch command did not delay any trip")
        after = model_arrays(runner)
        if any(not np.array_equal(before[key], after[key]) for key in before):
            raise RuntimeError("Frozen dispatch qualification changed a network")
        rows.append({"condition": condition, "scenario": "low_noise", "scene_seed": reference["scene_seed"],
            **frontier.source_spec.routing.compact_row(row), "dispatch": ledger,
            **{key: row[key] for key in ("lower_action_mean", "upper_delta_mean")}})
        print(f"root={args.seed} {condition} cost={row['service_cost_restricted']} "
              f"advance={ledger['advance_count']} delay={ledger['delay_count']}", flush=True)
        probe.policy.get_action = probe.original_get_action
        del runner, probe, before, after
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "seed": args.seed, "software_qualified": True, "baseline_reproduced": True,
        "zero_dispatch_reproduced": True, "training_updates": 0,
        "native_steps": len(rows) * 61380, "evaluation": rows})
    print("NATIVE_SIGNED_DISPATCH_QUALIFIED", flush=True)


if __name__ == "__main__":
    main()

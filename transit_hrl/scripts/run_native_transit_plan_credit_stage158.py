#!/usr/bin/env python3
"""Freeze Stage157 and measure single-plan authority and common-mode credit."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_residual_plan_stage157 as source
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from freq_hrl.rl.offpolicy_actor_critic import FlatOffPolicyActorCritic, OffPolicyConfig

EXPERIMENT_PROTOCOL = "native_transit_plan_credit_stage158_v1"
SOURCE_RUN = "native_transit_learned_residual_plan_stage157_development_20261010_r2"
ROOTS = source.ROOTS
SCENARIOS = tuple(source.contract()["scenarios"])
PROBES = (8, 22, 36)
ALTERNATIVES = ("zero", "opposite")
CONTROLS = ("learned", "forecast", "constant_residual")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "controls": list(CONTROLS),
        "scene": "first_registered_stage157_scene_per_root_and_regime_no_selection",
        "probe_decision_indices": list(PROBES), "alternatives": list(ALTERNATIVES),
        "intervention": "one_macro_action_only_then_same_frozen_deterministic_actor",
        "prefix": "identical_causal_states_actions_cost_and_exogenous_seed_before_intervention",
        "training_updates": 0, "episodes_per_cell": 9, "clock_s": 61380,
        "credit": "raw_and_forecast_paired_prefix_differences_same_terminal_physical_cost",
        "critic": "min_twin_Q_action_margin_at_same_preintervention_state",
        "boundary": "soft_stochastic_Q_vs_deterministic_deployment_rank_diagnostic_not_Bellman_calibration"}


def source_cell(root):
    path = ROOT / "results" / SOURCE_RUN / "cells" / source.METHOD / f"seed_{root}/result.json"
    cell = json.loads(path.read_text())
    if not (cell["protocol"] == source.EXPERIMENT_PROTOCOL and cell["contract"] == source.contract()
            and cell["seed"] == root and cell["software_qualified"] and cell["training"]["updates"] == 5500):
        raise ValueError("Frozen diagnostics require the completed registered Stage157 actor")
    return path, cell


class SinglePlanIntervention:
    def __init__(self, agent, reference, index, action):
        self.agent, self.reference, self.index = agent, reference, index
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def __call__(self, state):
        index = self.calls
        self.calls += 1
        action = self.agent.act(state, sample=False)
        if index <= self.index:
            ref = self.reference[index]
            if not (np.array_equal(state, ref["state"]) and np.array_equal(action, ref["action"])):
                raise RuntimeError("Single-plan intervention has a different causal prefix")
        return self.action.copy() if index == self.index else action


def trace(plan):
    if len(plan.decisions) != len(plan.credit.transitions):
        raise RuntimeError("Every executed macro action needs exactly one credit transition")
    return [{"state": d["state"].tolist(), "action": d["action"].tolist(), "time_s": d["time_s"],
        **{k: t[k] for k in ("reward", "cost_before", "cost_after", "duration_s", "done")}}
        for d, t in zip(plan.decisions, plan.credit.transitions)]


def moments(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
        "abs_mean": float(np.abs(values).mean()), "abs_max": float(np.abs(values).max())}


def paired_credit(learned, forecast):
    if not (len(learned) == len(forecast) and len(learned) > 0
            and [r["time_s"] for r in learned] == [r["time_s"] for r in forecast]
            and learned[0]["cost_before"] == forecast[0]["cost_before"]):
        raise ValueError("Forecast credit pairing requires the same decision clocks and initial cost")
    left = np.asarray([r["reward"] for r in learned])
    right = np.asarray([r["reward"] for r in forecast])
    delta = left - right
    expected = 100 * (forecast[-1]["cost_after"] - learned[-1]["cost_after"])
    if not np.isclose(delta.sum(), expected, rtol=1e-10, atol=1e-8):
        raise RuntimeError("Paired credit no longer telescopes to forecast-relative physical cost")
    return {"raw_learned": moments(left), "raw_forecast": moments(right), "paired": moments(delta),
        "paired_reward_sum": float(delta.sum()), "terminal_advantage": float(expected),
        "paired_rewards": delta.tolist()}


def critic_values(agent, state, actions):
    actions = np.asarray(actions, dtype=np.float32)
    states = np.repeat(np.asarray(state, dtype=np.float32)[None], len(actions), axis=0)
    with torch.no_grad():
        q1, q2 = agent.critic(torch.as_tensor(states, device=agent.device),
                             torch.as_tensor(actions, device=agent.device))
    values = np.stack([q1.cpu().numpy().ravel(), q2.cpu().numpy().ravel()], axis=1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Frozen upper critic produced nonfinite action values")
    return values


def probe_result(agent, reference, alternative, index, action):
    base, changed = reference["trace"][index], alternative["trace"][index]
    if not (base["time_s"] == changed["time_s"] and base["cost_before"] == changed["cost_before"]
            and np.array_equal(changed["action"], action)):
        raise RuntimeError("Intervention did not preserve the pre-action cost or execute its registered action")
    values = critic_values(agent, base["state"], [base["action"], action])
    deltas = {k: alternative[k] - reference[k] for k in source.source.source_spec.authority.routing.METRICS}
    return {"decision_index": index, "time_s": base["time_s"], "state": base["state"],
        "reference_action": base["action"], "intervention_action": np.asarray(action).tolist(),
        "prefix_matched": True, "critic_twin_values": values.tolist(),
        "critic_margin": float(values[1].min() - values[0].min()),
        "physical_return_delta": -100 * (alternative["credit"]["final_cost"] - reference["credit"]["final_cost"]),
        "physical_cost_delta": alternative["credit"]["final_cost"] - reference["credit"]["final_cost"],
        "outcome_delta": deltas, "credit": alternative["credit"], "execution": alternative["execution"],
        "N_fleet": alternative["N_fleet"], "simulation_end_time_s": alternative["simulation_end_time_s"],
        "passengers_generated": alternative["passengers_generated"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = source.NativeRoutingTracker
    torch.set_num_threads(1)
    path, registered = source_cell(args.seed)
    payload = torch.load(raw_directory(path) / "upper_final.pt", map_location="cpu", weights_only=False)
    agent = FlatOffPolicyActorCritic(OffPolicyConfig(**payload["config"]))
    agent.load_state_dict(payload["state_dict"])
    if payload["training"] != source.contract():
        raise RuntimeError("Upper checkpoint does not belong to the registered training protocol")
    agent.eval()
    before = {k: t.detach().clone() for k, t in agent.state_dict().items()}
    _, lower = source.load_source(args.seed)
    scene = source.source.source_spec.scene_seeds(source.LOWER_ROOT[args.seed], args.scenario, preflight=False)[0]
    raw = raw_directory(args.output)

    def episode(label, action_fn):
        row, plan, credit, execution = source.episode(args.seed, args.scenario, scene, raw / label, lower, action_fn)
        return {**source.source.source_spec.authority.routing.compact_row(row),
            "credit": credit, "execution": execution, "trace": trace(plan)}

    controls = {}
    for condition in CONTROLS:
        fn = ((lambda s: agent.act(s, sample=False)) if condition == "learned" else
              (lambda s: np.asarray(registered["training"]["constant_action"], dtype=np.float32))
              if condition == "constant_residual" else (lambda s: np.zeros(2, dtype=np.float32)))
        controls[condition] = episode(condition, fn)
        ref = next(r for r in registered["evaluation"] if r["condition"] == condition
            and r["scenario"] == args.scenario and r["scene_seed"] == scene)
        check_episode(controls[condition], ref, baseline=True)
        print(f"PLAN_CREDIT_SOURCE_REPRODUCED root={args.seed} {args.scenario}/{condition}", flush=True)
    learned = controls["learned"]
    probes = []
    for index in PROBES:
        for alternative in ALTERNATIVES:
            action = (np.zeros(2, dtype=np.float32) if alternative == "zero" else
                      -np.asarray(learned["trace"][index]["action"], dtype=np.float32))
            callback = SinglePlanIntervention(agent, learned["trace"], index, action)
            row = episode(f"probe_{index}_{alternative}", callback)
            check_episode(row, learned, baseline=False)
            if callback.calls != len(learned["trace"]):
                raise RuntimeError("Intervention changed the macro-decision budget")
            probes.append({"alternative": alternative, **probe_result(agent, learned, row, index, action)})
            print(f"PLAN_CREDIT_PROBE root={args.seed} {args.scenario}/{index}/{alternative} "
                  f"cost_delta={probes[-1]['physical_cost_delta']}", flush=True)
    if any(not torch.equal(t, before[k]) for k, t in agent.state_dict().items()):
        raise RuntimeError("Frozen upper actor or critic changed during diagnostics")
    states = np.asarray([r["state"] for r in learned["trace"]])
    actions = np.asarray([r["action"] for r in learned["trace"]])
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "scenario": args.scenario, "scene_seed": scene, "software_qualified": True,
        "training_updates": 0, "native_steps": 9 * 61380, "source_reproduced": list(CONTROLS),
        "controls": controls, "probes": probes, "alpha": float(agent.alpha),
        "credit_diagnostics": paired_credit(learned["trace"], controls["forecast"]["trace"]),
        "state_min": states.min(axis=0).tolist(), "state_max": states.max(axis=0).tolist(),
        "action_abs_gt_0_95_fraction": (np.abs(actions) > .95).mean(axis=0).tolist()})
    print("NATIVE_PLAN_CREDIT_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

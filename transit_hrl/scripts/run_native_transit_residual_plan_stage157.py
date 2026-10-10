#!/usr/bin/env python3
"""Learn upper service-plan residuals against the unchanged physical objective."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_residual_plan import NativeResidualPlan, STATE_DIM, RESIDUAL_SCALE_S
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from freq_hrl.rl.offpolicy_actor_critic import FlatOffPolicyActorCritic, OffPolicyConfig, ReplayBuffer
from scripts import run_native_transit_service_plan_stage156 as source
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_learned_residual_plan_stage157_v1"
SOURCE_RUN = "native_transit_service_allocation_stage156_frozen_20261010_r1"
ROOTS = (397, 401)
LOWER_ROOT = dict(zip(ROOTS, source.ROOTS))
METHOD = "residual_sac"
CONDITIONS = ("learned", "forecast", "nominal", "constant_residual")


def contract(preflight=False):
    return {"roots": list(ROOTS), "lower_roots": {str(k): v for k, v in LOWER_ROOT.items()},
        "source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "source_checkpoint": "stage155_one_step_299_no_selection", "preflight": preflight,
        "train_episodes": 2 if preflight else 120, "warmup": 1 if preflight else 10,
        "updates_per_episode": 2 if preflight else 50, "batch_size": 8 if preflight else 64,
        "state_dim": STATE_DIM, "action_dim": 2, "residual_scale_s": RESIDUAL_SCALE_S,
        "gamma": 1.0, "reward_scale": 100.0,
        "credit": "causal_prefix_physical_cost_differences_telescope_to_same_terminal_cost",
        "plan": "forecast_plus_centered_linear_and_curvature_residual_committed_six_trips",
        "learner": "shared_FlatOffPolicyActorCritic_SAC_default_64x64_init_alpha_0.05",
        "lower": "source_native_weights_deployment_state_and_deterministic_actions_frozen",
        "training_clock_s": 5400 if preflight else 61380, "fleet": 12,
        "scenarios": list(source.contract()["scenarios"]), "conditions": list(CONDITIONS),
        "training_scenes": "600000000_plus_upper_root_times1000_plus_episode" if preflight else
            "700000000_plus_upper_root_times1000_plus_episode_disjoint_from_eval",
        "evaluation_scenes": "twenty_registered_source_scenes_per_lower_root",
        "constant_control": "last_actor_mean_on_training_states_only_before_evaluation",
        "selection": "last_episode_only", "primary": "own_learned_minus_forecast_and_constant_physical_cost",
        "statistics": "two_root_development_not_confirmation_or_joint_HRL",
        "qualification": "three_full_source_reproductions_then_independent_two_short_learning_episodes"}


def load_source(root):
    lower_root = LOWER_ROOT[root]
    path = ROOT / "results" / SOURCE_RUN / "cells/one_step" / f"seed_{lower_root}/result.json"
    result = json.loads(path.read_text())
    if not (result["software_qualified"] and result["baseline_reproduced"] and result["neutral_reproduced"]
            and result["protocol"] == source.EXPERIMENT_PROTOCOL and result["contract"] == source.contract()
            and result["seed"] == lower_root and result["training_updates"] == 0):
        raise ValueError("Residual learning requires qualified native service-plan source")
    original_path, _ = source.load_source(lower_root)
    checkpoint = original_path.parent.with_name(original_path.parent.name + "_raw") / "full/training" / (
        f"F_freqduet_harmonic_hiro_seed{lower_root}") / "checkpoints"
    return result, checkpoint


def make_runner(root, scenario, raw, checkpoint, *, preflight=False):
    import torch
    from runner_v3 import TransitDuetV2Runner, load_config
    lower_root = LOWER_ROOT[root]
    # Repeated native constructors must not reset the learned actor's noise stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(lower_root)
        np.random.seed(lower_root)
        random.seed(lower_root)
        cfg = source.source_spec.configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
            "one_step", lower_root, preflight=preflight)
        cfg["env"].update(copy.deepcopy(source.source_spec.authority.routing.SCENARIOS[scenario]))
        cfg["logging"] = {"logs_dir": str(raw)}
        runner = TransitDuetV2Runner(cfg, device="cpu")
        runner.load_checkpoint(checkpoint, 299, require_deployment_state=True)
    for level in ("upper", "lower"):
        getattr(runner, f"{level}_trainer").update = source.reject_training
    return runner


def episode(root, scenario, scene, raw, checkpoint, action_fn, *, nominal=False, baseline=False, preflight=False):
    runner = make_runner(root, scenario, raw, checkpoint, preflight=preflight)
    before = model_arrays(runner)
    plan = None if baseline else (NativeServicePlan(runner.env, "nominal_plan") if nominal
                                  else NativeResidualPlan(runner.env, action_fn))
    if plan is not None:
        runner._upper_callback_v2 = plan
    row = runner.run_episode(300, training=False, N_fleet_override=12, scenario_seed=scene, record_diagnostics=False)
    after = model_arrays(runner)
    if any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError("Frozen native weights changed during upper-only learning")
    if (row["simulation_end_time_s"] != contract(preflight)["training_clock_s"] or row["N_fleet"] != 12
            or not all(np.isfinite(row[k]) for k in source.source_spec.authority.routing.METRICS)):
        raise RuntimeError("Invalid native clock/fleet/outcome in residual-plan experiment")
    credit = plan.finish(row) if isinstance(plan, NativeResidualPlan) else None
    execution = plan.summarize() if plan is not None else None
    return row, plan, credit, execution


def train(root, raw, checkpoint, *, preflight):
    import torch
    torch.manual_seed(root)
    settings = contract(preflight)
    agent = FlatOffPolicyActorCritic(OffPolicyConfig(STATE_DIM, 2, gamma=1, init_alpha=.05))
    replay = ReplayBuffer(20000, STATE_DIM, 2)
    rng = np.random.default_rng(root)
    initial = {k: t.detach().clone() for k, t in agent.actor.state_dict().items()}
    curve, states, actions, seeds, sums = [], [], [], [], {}
    transitions, stats = 0, {}
    for ep in range(settings["train_episodes"]):
        scenario = settings["scenarios"][ep % len(settings["scenarios"])]
        scene = (600000000 if preflight else 700000000) + root * 1000 + ep
        action_fn = ((lambda s: rng.uniform(-1, 1, 2).astype(np.float32)) if ep < settings["warmup"]
                     else (lambda s: agent.act(s, sample=True)))
        row, plan, credit, execution = episode(root, scenario, scene, raw / f"episode_{ep}", checkpoint,
            action_fn, preflight=preflight)
        for transition in plan.credit.transitions:
            replay.add(*(transition[k] for k in ("state", "action", "reward", "next_state", "done")))
        states.extend(d["state"] for d in plan.decisions)
        actions.extend(d["action"] for d in plan.decisions)
        transitions += credit["decisions"]
        seeds.append(scene)
        if ep >= settings["warmup"]:
            for _ in range(settings["updates_per_episode"]):
                stats = agent.update(replay.sample(settings["batch_size"], rng, agent.device))
                if not all(np.isfinite(v) for v in stats.values()):
                    raise RuntimeError("Nonfinite upper SAC update")
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.) + value
        if ep % 10 == 0 or ep == settings["train_episodes"] - 1:
            curve.append({"episode": ep, "scenario": scenario,
                **source.source_spec.authority.routing.compact_row(row), "credit": credit,
                "execution": execution, "learning": stats})
            print(f"residual root={root} preflight={preflight} train={ep+1}/{settings['train_episodes']} "
                f"cost={row['service_cost_restricted']} upper_updates={agent.update_step}", flush=True)
    change = max(float(torch.max(torch.abs(t - initial[k]))) for k, t in agent.actor.state_dict().items())
    if agent.update_step != (settings["train_episodes"] - settings["warmup"]) * settings["updates_per_episode"] or change <= 0:
        raise RuntimeError("Residual policy did not complete registered actor-critic learning")
    with torch.no_grad():
        deterministic, _ = agent.actor.sample(torch.as_tensor(np.asarray(states)), deterministic=True)
    mean = deterministic.numpy().mean(axis=0).astype(np.float32)
    summary = {"updates": agent.update_step, "actor_change_max_abs": change, "transitions": transitions,
        "training_scene_seeds": seeds, "training_curve": curve, "constant_action": mean.tolist(),
        "final_actor_training_state_std": deterministic.numpy().std(axis=0).tolist(),
        "training_action_std": np.std(actions, axis=0).tolist(),
        "learning_mean": {k: v / agent.update_step for k, v in sums.items()}}
    return agent, summary


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
    torch.set_num_threads(1)
    source_result, checkpoint = load_source(args.seed)
    raw = raw_directory(args.output)
    qualification = []
    scenario = "low_noise"
    scene = source.source_spec.scene_seeds(LOWER_ROOT[args.seed], scenario, preflight=False)[0]
    for condition in ("source_baseline", "nominal_plan", "causal_forecast"):
        row, plan, credit, execution = episode(args.seed, scenario, scene, raw / "qualification" / condition,
            checkpoint, lambda s: np.zeros(2, dtype=np.float32),
            baseline=condition == "source_baseline", nominal=condition == "nominal_plan")
        reference = next(r for r in source_result["evaluation"] if r["condition"] == condition
            and r["scenario"] == scenario and r["scene_seed"] == scene)
        check_episode(row, reference, baseline=True)
        qualification.append({"condition": condition, "native_steps": row["simulation_end_time_s"], "reproduced": True})
        print(f"RESIDUAL_SOURCE_REPRODUCED root={args.seed} {condition}", flush=True)
    _, preflight = train(args.seed, raw / "qualification/learning", checkpoint, preflight=True)
    print("RESIDUAL_PLAN_SOURCE_AND_SHORT_LEARNING_QUALIFIED", flush=True)
    agent, training = train(args.seed, raw / "training", checkpoint, preflight=False)
    raw.mkdir(parents=True, exist_ok=True)
    torch.save({"config": agent.config.to_dict(), "state_dict": agent.state_dict(), "training": contract()},
               raw / "upper_final.pt")
    before = {k: t.detach().clone() for k, t in agent.state_dict().items()}
    evaluations = []
    for scenario in contract()["scenarios"]:
        for scene in source.source_spec.scene_seeds(LOWER_ROOT[args.seed], scenario, preflight=False):
            reference = None
            for condition in CONDITIONS:
                action_fn = ((lambda s: agent.act(s, sample=False)) if condition == "learned" else
                    (lambda s: np.asarray(training["constant_action"], dtype=np.float32)) if condition == "constant_residual"
                    else (lambda s: np.zeros(2, dtype=np.float32)))
                row, plan, credit, execution = episode(args.seed, scenario, scene,
                    raw / condition / scenario / str(scene), checkpoint, action_fn, nominal=condition == "nominal")
                if reference is None:
                    reference = row
                check_episode(row, reference, baseline=False)
                if condition in {"forecast", "nominal"}:
                    original = next(r for r in source_result["evaluation"]
                        if r["condition"] == ("causal_forecast" if condition == "forecast" else "nominal_plan")
                        and r["scenario"] == scenario and r["scene_seed"] == scene)
                    check_episode(row, original, baseline=True)
                evaluations.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                    **source.source_spec.authority.routing.compact_row(row), "credit": credit, "execution": execution,
                    "source_control_reproduced": condition in {"forecast", "nominal"},
                    "residual_action_mean": np.mean([d["action"] for d in plan.decisions], axis=0).tolist()
                        if isinstance(plan, NativeResidualPlan) else [0., 0.],
                    "residual_action_std": np.std([d["action"] for d in plan.decisions], axis=0).tolist()
                        if isinstance(plan, NativeResidualPlan) else [0., 0.]})
                print(f"residual root={args.seed} {condition}/{scenario}/{scene} "
                    f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']}", flush=True)
    if any(not torch.equal(t, before[k]) for k, t in agent.state_dict().items()):
        raise RuntimeError("Frozen learned upper changed during evaluation")
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "lower_root": LOWER_ROOT[args.seed], "software_qualified": True,
        "qualification": qualification, "preflight_learning": preflight,
        "worker_preflight_native_steps": 3 * 61380 + 2 * 5400,
        "native_steps": (contract()["train_episodes"] + len(evaluations)) * 61380,
        "native_training_updates": 0, "training": training, "evaluation": evaluations})
    print("NATIVE_RESIDUAL_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

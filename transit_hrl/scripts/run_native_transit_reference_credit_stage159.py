#!/usr/bin/env python3
"""Change only native upper replay credit using a same-scene forecast reference."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.core.reference_credit import paired_prefix_credit
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from freq_hrl.rl.offpolicy_actor_critic import FlatOffPolicyActorCritic, OffPolicyConfig, ReplayBuffer
from scripts import run_native_transit_residual_plan_stage157 as source
from scripts import run_native_transit_plan_credit_stage158 as diagnosis
from scripts.run_native_transit_diagnostics_stage147 import check_episode

EXPERIMENT_PROTOCOL = "native_transit_reference_credit_stage159_v1"
SOURCE_RUN = diagnosis.SOURCE_RUN
DIAGNOSTIC_RUN = "native_transit_plan_credit_stage158_frozen_20261011_r1"
ROOTS = source.ROOTS
METHOD = "paired_reference"
CONDITIONS = source.CONDITIONS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "source_training_contract": source.contract(), "diagnostic_run": DIAGNOSTIC_RUN,
        "roots": list(ROOTS), "method": METHOD, "state_dim": 34, "action_dim": 2,
        "change": "replay_reward_raw_minus_same_scene_action_independent_forecast_reward_only",
        "reference": "one_full_frozen_forecast_episode_per_training_episode_same_seed_and_clock",
        "actor_inputs": "unchanged_causal_34_no_reference_or_future_inputs",
        "raw_control": "registered_stage157_same_initial_seed_training_scenes_and_SAC_budget",
        "qualification": "one_full_forecast_reproduction_raw_short_with_discarded_reference_then_paired_short",
        "training_episodes": 120, "reference_episodes": 120, "upper_updates": 5500,
        "evaluation_episodes": 80, "native_updates": 0, "main_native_ticks_per_root": 320 * 61380,
        "worker_preflight_ticks_per_root": 61380 + 8 * 5400,
        "selection": "last_actor_only_constant_from_training_states_before_evaluation",
        "boundary": "two_root_development_reuses_stage157_scenes_not_confirmation_or_joint_HRL"}


def train(root, raw, checkpoint, *, credit_mode, preflight):
    if credit_mode not in {"raw", "paired"}:
        raise ValueError("The credit ablation is raw or paired")
    torch.manual_seed(root)
    settings = source.contract(preflight)
    agent = FlatOffPolicyActorCritic(OffPolicyConfig(34, 2, gamma=1, init_alpha=.05))
    replay = ReplayBuffer(20000, 34, 2)
    rng = np.random.default_rng(root)
    initial = {k: t.detach().clone() for k, t in agent.actor.state_dict().items()}
    curve, states, actions, seeds, sums, pairs = [], [], [], [], {}, []
    transitions, stats = 0, {}
    for ep in range(settings["train_episodes"]):
        scenario = settings["scenarios"][ep % len(settings["scenarios"])]
        scene = (600000000 if preflight else 700000000) + root * 1000 + ep
        ref, reference, _, _ = source.episode(root, scenario, scene, raw / f"reference_{ep}", checkpoint,
            lambda s: np.zeros(2, dtype=np.float32), preflight=preflight)
        action_fn = ((lambda s: rng.uniform(-1, 1, 2).astype(np.float32)) if ep < settings["warmup"]
                     else (lambda s: agent.act(s, sample=True)))
        row, plan, credit, execution = source.episode(root, scenario, scene, raw / f"episode_{ep}", checkpoint,
            action_fn, preflight=preflight)
        check_episode(row, ref, baseline=False)
        paired, summary = paired_prefix_credit(plan.credit, reference.credit)
        actual = paired if credit_mode == "paired" else plan.credit.transitions
        for t in actual:
            replay.add(*(t[k] for k in ("state", "action", "reward", "next_state", "done")))
        states.extend(d["state"] for d in plan.decisions)
        actions.extend(d["action"] for d in plan.decisions)
        transitions += credit["decisions"]
        seeds.append(scene)
        pairs.append({"episode": ep, "scenario": scenario, "scene_seed": scene, **summary})
        if ep >= settings["warmup"]:
            for _ in range(settings["updates_per_episode"]):
                stats = agent.update(replay.sample(settings["batch_size"], rng, agent.device))
                if not all(np.isfinite(v) for v in stats.values()):
                    raise RuntimeError("Nonfinite reference-credit SAC update")
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.) + value
        if ep % 10 == 0 or ep == settings["train_episodes"] - 1:
            curve.append({"episode": ep, "scenario": scenario,
                **source.source.source_spec.authority.routing.compact_row(row), "credit": credit,
                "execution": execution, "learning": stats})
            print(f"reference_credit root={root} mode={credit_mode} preflight={preflight} "
                f"train={ep+1}/{settings['train_episodes']} cost={row['service_cost_restricted']} "
                f"upper_updates={agent.update_step} raw_sd={summary['raw_reward_std']} "
                f"paired_sd={summary['paired_reward_std']}", flush=True)
    change = max(float(torch.max(torch.abs(t - initial[k]))) for k, t in agent.actor.state_dict().items())
    if agent.update_step != (settings["train_episodes"] - settings["warmup"]) * settings["updates_per_episode"] or change <= 0:
        raise RuntimeError("Reference-credit actor-critic did not complete its learning budget")
    with torch.no_grad():
        deterministic, _ = agent.actor.sample(torch.as_tensor(np.asarray(states)), deterministic=True)
    result = {"updates": agent.update_step, "actor_change_max_abs": change, "transitions": transitions,
        "training_scene_seeds": seeds, "training_curve": curve,
        "constant_action": deterministic.numpy().mean(axis=0).astype(np.float32).tolist(),
        "final_actor_training_state_std": deterministic.numpy().std(axis=0).tolist(),
        "training_action_std": np.std(actions, axis=0).tolist(),
        "learning_mean": {k: v / agent.update_step for k, v in sums.items()},
        "credit_mode": credit_mode, "reference_episodes": len(pairs), "credit_pairs": pairs}
    return agent, result


def check_raw_reproduction(actual, expected):
    for key in ("updates", "transitions", "training_scene_seeds"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"Raw short reproduction changed {key}")
    for key in ("actor_change_max_abs", "constant_action", "final_actor_training_state_std", "training_action_std"):
        if not np.allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Reference rollouts interfered with raw short learning: {key}")
    for key in expected["learning_mean"]:
        if not np.isclose(actual["learning_mean"][key], expected["learning_mean"][key], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"Reference rollouts changed raw learning statistic {key}")
    if len(actual["training_curve"]) != len(expected["training_curve"]):
        raise RuntimeError("Changed raw short training curve")
    for row, ref in zip(actual["training_curve"], expected["training_curve"]):
        check_episode(row, ref, baseline=True)


def evaluate(root, agent, training, raw, checkpoint, source_result):
    before = {k: t.detach().clone() for k, t in agent.state_dict().items()}
    rows = []
    for scenario in source.contract()["scenarios"]:
        for scene in source.source.source_spec.scene_seeds(source.LOWER_ROOT[root], scenario, preflight=False):
            reference = None
            for condition in CONDITIONS:
                action_fn = ((lambda s: agent.act(s, sample=False)) if condition == "learned" else
                    (lambda s: np.asarray(training["constant_action"], dtype=np.float32)) if condition == "constant_residual"
                    else (lambda s: np.zeros(2, dtype=np.float32)))
                row, plan, credit, execution = source.episode(root, scenario, scene,
                    raw / condition / scenario / str(scene), checkpoint, action_fn, nominal=condition == "nominal")
                reference = row if reference is None else reference
                check_episode(row, reference, baseline=False)
                if condition in {"forecast", "nominal"}:
                    original = next(r for r in source_result["evaluation"] if r["condition"] == condition
                        and r["scenario"] == scenario and r["scene_seed"] == scene)
                    check_episode(row, original, baseline=True)
                rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                    **source.source.source_spec.authority.routing.compact_row(row), "credit": credit,
                    "execution": execution, "source_control_reproduced": condition in {"forecast", "nominal"},
                    "residual_action_mean": np.mean([d["action"] for d in plan.decisions], axis=0).tolist()
                        if credit is not None else [0., 0.],
                    "residual_action_std": np.std([d["action"] for d in plan.decisions], axis=0).tolist()
                        if credit is not None else [0., 0.]})
                print(f"reference_credit root={root} {condition}/{scenario}/{scene} "
                      f"cost={row['service_cost_restricted']}", flush=True)
    if any(not torch.equal(t, before[k]) for k, t in agent.state_dict().items()):
        raise RuntimeError("Frozen paired actor changed during evaluation")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = source.NativeRoutingTracker
    torch.set_num_threads(1)
    _, registered = diagnosis.source_cell(args.seed)
    _, checkpoint = source.load_source(args.seed)
    raw = raw_directory(args.output)
    scenario = "low_noise"
    scene = source.source.source_spec.scene_seeds(source.LOWER_ROOT[args.seed], scenario, preflight=False)[0]
    row, _, _, _ = source.episode(args.seed, scenario, scene, raw / "qualification/forecast", checkpoint,
        lambda s: np.zeros(2, dtype=np.float32))
    check_episode(row, next(r for r in registered["evaluation"] if r["condition"] == "forecast"
        and r["scenario"] == scenario and r["scene_seed"] == scene), baseline=True)
    _, raw_short = train(args.seed, raw / "qualification/raw", checkpoint, credit_mode="raw", preflight=True)
    check_raw_reproduction(raw_short, registered["preflight_learning"])
    _, paired_short = train(args.seed, raw / "qualification/paired", checkpoint, credit_mode="paired", preflight=True)
    print("REFERENCE_CREDIT_RAW_REPRODUCTION_AND_PAIRED_LEARNING_QUALIFIED", flush=True)
    agent, training = train(args.seed, raw / "training", checkpoint, credit_mode="paired", preflight=False)
    torch.save({"config": agent.config.to_dict(), "state_dict": agent.state_dict(), "training": contract()},
               raw / "upper_final.pt")
    rows = evaluate(args.seed, agent, training, raw / "evaluation", checkpoint, registered)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "software_qualified": True, "native_training_updates": 0, "native_steps": 320 * 61380,
        "worker_preflight_native_steps": contract()["worker_preflight_ticks_per_root"],
        "forecast_source_reproduced": True, "raw_short_reproduced": True,
        "raw_short": raw_short, "paired_short": paired_short, "training": training, "evaluation": rows})
    print("NATIVE_REFERENCE_CREDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

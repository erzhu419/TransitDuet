#!/usr/bin/env python3
"""Test complete-episode on-policy upper credit with the existing PPO core."""

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
from freq_hrl.rl.smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch, SMDPPPOConfig, concat_level_batches,
)
from scripts import run_native_transit_reference_credit_stage159 as previous

source = previous.source
EXPERIMENT_PROTOCOL = "native_transit_mc_upper_stage160_v1"
SOURCE_RUN = "native_transit_reference_credit_stage159_development_20261011_r1"
ROOTS, CONDITIONS = source.ROOTS, source.CONDITIONS
METHOD = "mc_ppo"


def contract(preflight=False):
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "source_physical_contract": source.contract(preflight), "roots": list(ROOTS),
        "method": METHOD, "preflight": preflight, "training_episodes": 2 if preflight else 120,
        "reference_episodes": 2 if preflight else 120, "batch_episodes": 2 if preflight else 5,
        "epochs": 2 if preflight else 8, "minibatch_size": 8 if preflight else 64,
        "upper_actor_optimizer_steps": 2 if preflight else 768,
        "gamma": 1.0, "gae_lambda": 1.0, "entropy_coef": 0.0,
        "upper_learning_rate": 3e-4, "hidden_dim": 64, "clip_ratio": 0.2,
        "learner": "existing_FrequencySeparatedActorCriticPPO_upper_only_complete_episode_MC",
        "actor": "existing_GaussianActor_two_tanh_layers_state_independent_log_std_init_minus1",
        "execution": "tanh_latent_action_two_unchanged_20_second_residual_coefficients",
        "credit": "same_scene_reference_paired_prefix_return_to_go_no_bootstrapped_Q_or_entropy_bonus",
        "collection": "all_on_policy_no_uniform_warmup_one_transition_per_macro_no_replay",
        "value": "MC_regression_only_previous_frozen_value_as_action_independent_baseline",
        "native_updates": 0, "evaluation_episodes": 80, "clock_s": 5400 if preflight else 61380,
        "selection": "last_actor_only_constant_from_training_states_before_evaluation",
        "boundary": "two_root_development_trainer_replacement_not_single_factor_Q_causality_or_confirmation"}


def source_cell(root):
    path = ROOT / "results" / SOURCE_RUN / "cells" / previous.METHOD / f"seed_{root}/result.json"
    cell = json.loads(path.read_text())
    if not (cell["protocol"] == previous.EXPERIMENT_PROTOCOL and cell["contract"] == previous.contract()
            and cell["software_qualified"] and cell["seed"] == root and cell["training"]["updates"] == 5500):
        raise ValueError("MC training requires completed qualified reference-credit SAC")
    return cell


def learner(root, preflight):
    settings = contract(preflight)
    torch.manual_seed(root)
    np.random.seed(root)
    return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
        upper_state_dim=34, upper_action_dim=2, lower_state_dim=1, lower_action_dim=1,
        upper_cost_critic=False, lower_cost_critic=False, hidden_dim=64,
        upper_learning_rate=3e-4, gamma=1, gae_lambda=1, entropy_coef=0,
        epochs=settings["epochs"], minibatch_size=settings["minibatch_size"], init_log_std=-1))


def episode_batch(plan, draws, transitions):
    if len(draws) != len(transitions):
        raise RuntimeError("PPO must record each upper action exactly once")
    for decision, draw in zip(plan.decisions, draws):
        if not (np.array_equal(decision["state"], draw["state"])
                and np.array_equal(decision["action"], np.tanh(draw["action"]).astype(np.float32))):
            raise RuntimeError("PPO likelihood does not describe the executed macro action")
    batch = LevelTrajectoryBatch(
        state=np.asarray([d["state"] for d in draws], dtype=np.float32),
        action=np.asarray([d["action"] for d in draws], dtype=np.float32),
        reward=np.asarray([t["reward"] for t in transitions], dtype=np.float32),
        duration=np.asarray([t["duration_s"] for t in transitions], dtype=np.int64),
        done=np.asarray([t["done"] for t in transitions], dtype=np.float32),
        old_logp=np.asarray([d["logp"] for d in draws], dtype=np.float32),
        old_value=np.asarray([d["value"] for d in draws], dtype=np.float32))
    if not (batch.done[-1] == 1 and batch.done.sum() == 1):
        raise RuntimeError("Monte Carlo credit requires complete episodes including the terminal tail")
    return batch


def train(root, raw, checkpoint, *, preflight):
    settings, agent = contract(preflight), learner(root, preflight)
    initial = {k: t.detach().clone() for k, t in agent.upper_actor.state_dict().items()}
    lower = {k: t.detach().clone() for k, t in agent.lower_actor.state_dict().items()}
    batches, pairs, curve, update_rows, states, seeds = [], [], [], [], [], []
    steps = transitions_count = 0
    for ep in range(settings["training_episodes"]):
        scenario = source.contract()["scenarios"][ep % 5]
        scene = (600000000 if preflight else 700000000) + root * 1000 + ep
        draws = []

        def sample(state):
            draw = agent.act_upper(state, sample=True)
            draws.append({"state": state.copy(), **draw})
            return np.tanh(draw["action"]).astype(np.float32)

        # Native construction resets NumPy; preserve the learner's minibatch stream.
        numpy_state = np.random.get_state()
        try:
            ref, reference, _, _ = source.episode(root, scenario, scene, raw / f"reference_{ep}", checkpoint,
                lambda s: np.zeros(2, dtype=np.float32), preflight=preflight)
            row, plan, credit, execution = source.episode(root, scenario, scene, raw / f"episode_{ep}",
                checkpoint, sample, preflight=preflight)
        finally:
            np.random.set_state(numpy_state)
        previous.check_episode(row, ref, baseline=False)
        paired, summary = paired_prefix_credit(plan.credit, reference.credit)
        batch = episode_batch(plan, draws, paired)
        _, returns = agent._gae(batch.reward, batch.done, batch.duration, batch.old_value)
        expected = np.cumsum(batch.reward.astype(np.float64)[::-1])[::-1]
        if not np.allclose(returns, expected, rtol=1e-5, atol=1e-4):
            raise RuntimeError("PPO targets are not complete-episode Monte Carlo returns")
        batches.append(batch)
        states.extend(batch.state)
        seeds.append(scene)
        transitions_count += batch.size
        pairs.append({"episode": ep, "scenario": scenario, "scene_seed": scene,
            "mc_returns_qualified": True, "mc_return_std": float(np.std(returns)), **summary})
        curve.append({"episode": ep, "scenario": scenario,
            **source.source.source_spec.authority.routing.compact_row(row), "credit": credit})
        if (ep + 1) % settings["batch_episodes"] == 0:
            stats = agent._update_level(level="upper", batch=concat_level_batches(batches),
                actor=agent.upper_actor, value_net=agent.upper_value,
                actor_optimizer=agent.upper_actor_optimizer, value_optimizer=agent.upper_value_optimizer)
            if not all(np.isfinite(v) for v in stats.values()):
                raise RuntimeError("Nonfinite on-policy MC update")
            steps += int(stats["upper_actor_optimizer_steps"])
            update_rows.append({"last_episode": ep, **stats})
            batches.clear()
            print(f"mc_upper root={root} preflight={preflight} train={ep+1}/{settings['training_episodes']} "
                  f"cost={row['service_cost_restricted']} actor_steps={steps}", flush=True)
    change = max(float((t - initial[k]).abs().max()) for k, t in agent.upper_actor.state_dict().items())
    if batches or steps != settings["upper_actor_optimizer_steps"] or change <= 0:
        raise RuntimeError("MC upper did not complete its registered learning budget")
    if any(not torch.equal(t, lower[k]) for k, t in agent.lower_actor.state_dict().items()):
        raise RuntimeError("Upper-only PPO changed its unused lower actor")
    with torch.no_grad():
        actions = torch.tanh(agent.upper_actor.distribution(torch.as_tensor(np.asarray(states))).mean).numpy()
    return agent, {"updates": steps, "actor_change_max_abs": change, "transitions": transitions_count,
        "training_scene_seeds": seeds, "reference_episodes": len(pairs), "credit_pairs": pairs,
        "training_curve": curve, "update_curve": update_rows, "constant_action": actions.mean(axis=0).tolist(),
        "final_actor_training_state_std": actions.std(axis=0).tolist(), "mc_targets_qualified": True,
        "unused_lower_actor_unchanged": True}


def evaluate(root, agent, training, raw, checkpoint, registered):
    before = {k: t.detach().clone() for k, t in agent.upper_actor.state_dict().items()}
    rows = []
    for scenario in source.contract()["scenarios"]:
        for scene in source.source.source_spec.scene_seeds(source.LOWER_ROOT[root], scenario, preflight=False):
            for condition in CONDITIONS:
                action_fn = ((lambda s: np.tanh(agent.act_upper(s, sample=False)["action"]).astype(np.float32))
                    if condition == "learned" else
                    (lambda s: np.asarray(training["constant_action"], dtype=np.float32))
                    if condition == "constant_residual" else (lambda s: np.zeros(2, dtype=np.float32)))
                row, plan, credit, execution = source.episode(root, scenario, scene,
                    raw / condition / scenario / str(scene), checkpoint, action_fn, nominal=condition == "nominal")
                ref = next(r for r in registered["evaluation"] if r["condition"] == "forecast"
                           and r["scenario"] == scenario and r["scene_seed"] == scene)
                previous.check_episode(row, ref, baseline=False)
                if condition in {"forecast", "nominal"}:
                    original = next(r for r in registered["evaluation"] if r["condition"] == condition
                                    and r["scenario"] == scenario and r["scene_seed"] == scene)
                    previous.check_episode(row, original, baseline=True)
                rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                    **source.source.source_spec.authority.routing.compact_row(row), "credit": credit,
                    "execution": execution, "source_control_reproduced": condition in {"forecast", "nominal"},
                    "residual_action_mean": np.mean([d["action"] for d in plan.decisions], axis=0).tolist()
                        if credit is not None else [0., 0.],
                    "residual_action_std": np.std([d["action"] for d in plan.decisions], axis=0).tolist()
                        if credit is not None else [0., 0.]})
                print(f"mc_upper root={root} {condition}/{scenario}/{scene} cost={row['service_cost_restricted']}", flush=True)
    if any(not torch.equal(t, before[k]) for k, t in agent.upper_actor.state_dict().items()):
        raise RuntimeError("Frozen MC actor changed during evaluation")
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
    registered = source_cell(args.seed)
    _, checkpoint = source.load_source(args.seed)
    raw = raw_directory(args.output)
    scenario = "low_noise"
    scene = source.source.source_spec.scene_seeds(source.LOWER_ROOT[args.seed], scenario, preflight=False)[0]
    row, _, _, _ = source.episode(args.seed, scenario, scene, raw / "qualification/forecast", checkpoint,
        lambda s: np.zeros(2, dtype=np.float32))
    previous.check_episode(row, next(r for r in registered["evaluation"] if r["condition"] == "forecast"
                           and r["scenario"] == scenario and r["scene_seed"] == scene), baseline=True)
    _, short = train(args.seed, raw / "qualification/learning", checkpoint, preflight=True)
    print("MC_UPPER_FORECAST_AND_COMPLETE_EPISODE_LEARNING_QUALIFIED", flush=True)
    agent, training = train(args.seed, raw / "training", checkpoint, preflight=False)
    torch.save(agent.state_dict(), raw / "upper_final.pt")
    rows = evaluate(args.seed, agent, training, raw / "evaluation", checkpoint, registered)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "seed": args.seed,
        "software_qualified": True, "native_training_updates": 0, "native_steps": 320 * 61380,
        "worker_preflight_native_steps": 61380 + 4 * 5400, "forecast_source_reproduced": True,
        "short_learning": short, "training": training, "evaluation": rows})
    print("NATIVE_MC_UPPER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

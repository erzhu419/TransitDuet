"""Inspect saved critics and option credit without sampling or optimization."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeObservation
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from . import pointmaze_joint_renewal as joint
from . import pointmaze_update_diagnostics as diagnostics
from .pointmaze_critic_calibration import monte_carlo_returns
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from .pointmaze_root_response import write_json
from scripts import pointmaze_credit_diagnostics_stage62_spec as spec


def features(raw, args, period, gamma):
    horizon = len(raw["reward"])
    decisions = np.arange(0, horizon, period)
    if horizon != args.horizon:
        raise ValueError("Stage62 archive horizon changed")
    np.testing.assert_array_equal(raw["decision_steps"], decisions)
    history = PointMazeRegimeFeatureBuilder(time_scale=joint.scale_for(args))
    upper, lower = [], []
    for step in range(horizon):
        sample = raw["measurement"][step]
        obs = PointMazeRegimeObservation(physical=raw["physical"][step], achieved_goal=raw["achieved_before"][step],
            target=raw["target_before"][step], task_measurement=sample, force=sample[2:4], distractor=sample[4:6])
        history.reset(obs) if step == 0 else history.update(obs)
        if step % period == 0:
            upper.append(history.upper_state(obs, oracle_context=None))
        base = history.lower_state(obs, subgoal=raw["lower_reference"][step])
        lower.append(np.concatenate((base, raw["lower_value_context"][step])).astype(np.float32))
    rewards = np.asarray(raw["reward"], dtype=np.float64).reshape(-1, period).copy()
    rewards[:, 0] -= joint.spec.CALL_COST
    discounts = np.power(gamma, np.arange(period, dtype=np.float64))
    charged = np.asarray([np.dot(discounts, row) for row in rewards], dtype=np.float32)
    return {"upper_state": np.asarray(upper, dtype=np.float32), "lower_value_state": np.asarray(lower),
            "reward": np.asarray(raw["reward"], dtype=np.float32), "upper_reward": charged}


def predict(value_net, states, model):
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(states), spec.VALUE_BATCH_SIZE):
            tensor = torch.as_tensor(states[offset:offset + spec.VALUE_BATCH_SIZE], dtype=torch.float32, device=model.device)
            outputs.append(value_net(tensor).cpu().numpy().reshape(-1))
    return np.concatenate(outputs), len(outputs)


def mc(reward, done, duration, gamma):
    return monte_carlo_returns(SimpleNamespace(size=len(reward), reward=reward, done=done, duration=duration), gamma)


def episode_terms(model, data, period):
    reward = data["reward"]
    n = len(reward)
    episode_done = np.zeros(n, dtype=np.float32)
    episode_done[-1] = 1.
    option_done = episode_done.copy()
    option_done[period - 1::period] = 1.
    duration = np.ones(n, dtype=np.int64)
    lower, lower_passes = predict(model.lower_value, data["lower_value_state"], model)
    upper, upper_passes = predict(model.upper_value, data["upper_state"], model)
    next_values = np.concatenate((lower[1:], [0.])).astype(np.float32)
    option_adv, option_target = model._gae(reward, option_done, duration, lower)
    bootstrap_adv, _ = model._gae(reward, option_done, duration, lower, next_values, episode_done)
    episode_adv, _ = model._gae(reward, episode_done, duration, lower)
    upper_done = np.zeros(len(upper), dtype=np.float32)
    upper_done[-1] = 1.
    upper_duration = np.full(len(upper), period, dtype=np.int64)
    _, upper_target = model._gae(data["upper_reward"], upper_done, upper_duration, upper)
    boundary = np.flatnonzero(option_done[:-1])
    near = np.zeros(n, dtype=bool)
    for step in boundary:
        near[max(0, step - spec.BOUNDARY_WINDOW + 1):step + 1] = True
    return {"lower_value": lower, "option_mc": mc(reward, option_done, duration, model.config.gamma),
        "episode_mc": mc(reward, episode_done, duration, model.config.gamma), "option_gae_target": option_target,
        "option_adv": option_adv, "bootstrap_adv": bootstrap_adv, "episode_adv": episode_adv,
        "boundary_value": lower[boundary], "boundary_bootstrap": model.config.gamma * next_values[boundary],
        "boundary_td": reward[boundary] - lower[boundary],
        "near_option_adv": option_adv[near], "near_bootstrap_adv": bootstrap_adv[near], "near_episode_adv": episode_adv[near],
        "upper_value": upper, "upper_gae_target": upper_target,
        "upper_mc": mc(data["upper_reward"], upper_done, upper_duration, model.config.gamma),
        "cost": {"archive_episodes": 1, "feature_lower_rows": n, "feature_upper_rows": len(upper),
            "lower_value_rows": n, "upper_value_rows": len(upper),
            "lower_value_forward_batches": lower_passes, "upper_value_forward_batches": upper_passes,
            "gae_calls": 4, "mc_return_calls": 3}}


def value_metrics(prediction, target):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return {**diagnostics.value_terms(p, t), "prediction_mean": float(p.mean()), "prediction_std": float(p.std()),
            "target_mean": float(t.mean()), "target_std": float(t.std()), "bias": float((p - t).mean())}


def alignment(a, b):
    a, b = FrequencySeparatedActorCriticPPO._normalize(a), FrequencySeparatedActorCriticPPO._normalize(b)
    return {"normalized_sign_disagreement": float(np.mean(np.sign(a) != np.sign(b))),
            "normalized_rms_difference": float(np.sqrt(np.mean(np.square(a - b)))),
            "correlation": None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])}


def summarize(episodes):
    arrays = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0] if key != "cost"}
    v, u = arrays["lower_value"], arrays["upper_value"]
    return {"lower_option_mc": value_metrics(v, arrays["option_mc"]),
        "lower_episode_mc": value_metrics(v, arrays["episode_mc"]),
        "lower_option_gae": value_metrics(v, arrays["option_gae_target"]),
        "upper_episode_mc": value_metrics(u, arrays["upper_mc"]),
        "upper_episode_gae": value_metrics(u, arrays["upper_gae_target"]),
        "advantage_alignment": {"option_vs_bootstrap": alignment(arrays["option_adv"], arrays["bootstrap_adv"]),
            "option_vs_episode": alignment(arrays["option_adv"], arrays["episode_adv"]),
            "bootstrap_vs_episode": alignment(arrays["bootstrap_adv"], arrays["episode_adv"])},
        "boundary": {"count": len(arrays["boundary_value"]),
            "value_mean": float(arrays["boundary_value"].mean()),
            "suppressed_bootstrap_mean": float(arrays["boundary_bootstrap"].mean()),
            "option_terminal_td_mean": float(arrays["boundary_td"].mean()),
            "continuing_td_mean": float((arrays["boundary_td"] + arrays["boundary_bootstrap"]).mean()),
            **{key + "_mean": float(arrays[key].mean()) for key in
                ("near_option_adv", "near_bootstrap_adv", "near_episode_adv")}}}


def diagnose(root, *, preflight, output):
    file = spec.source_result(root, preflight=preflight)
    source = json.loads(file.read_text())
    training_file = spec.training_source.source_result(root, preflight=preflight)
    training = json.loads(training_file.read_text())
    if any((c["status"], c["root"], c["preflight"]) != ("complete", root, preflight) for c in (source, training)):
        raise ValueError("Stage62 requires completed Stage57/61 for the same root")
    if (source["protocol"] != spec.source.EXPERIMENT_PROTOCOL or source["contract"] != spec.source.contract()
            or training["protocol"] != spec.source.deployment.EXPERIMENT_PROTOCOL
            or training["contract"] != spec.source.deployment.contract()):
        raise ValueError("Stage62 source protocol changed")
    roles = spec.seed_roles(root, preflight=preflight)
    if set(roles[spec.SPLITS[0]]).intersection(roles[spec.SPLITS[1]]):
        raise ValueError("Stage62 held-out paths overlap fitting batch")
    args, groups, checks = spec.arguments(root, preflight=preflight), {}, {}
    cost, started = dict.fromkeys(spec.budget(preflight=preflight), 0), time.monotonic()
    training_raw = training_file.parent.with_name(training_file.parent.name + "_raw")
    for period in spec.PERIODS:
        p = str(period)
        groups[p], checks[p] = {}, {}
        for arm in spec.TRAIN_POLICIES:
            checkpoint = Path(source["checkpoints"][p][arm][spec.TREATMENT])
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["treatment"]) != (
                    spec.source.EXPERIMENT_PROTOCOL, root, period, arm, spec.TREATMENT):
                raise ValueError("Stage62 candidate checkpoint identity changed")
            model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**saved["state_dict"]["config"]))
            model.load_state_dict(saved["state_dict"])
            cost["checkpoint_loads"] += 1
            groups[p][arm] = {}
            for split in spec.SPLITS:
                rows = (training["training"][p][arm]["history"][0]["rows"] if split == spec.SPLITS[0]
                        else source["evaluation_rows"][p][arm][spec.TREATMENT])
                if [r["seed"] for r in rows] != roles[split]:
                    raise ValueError("Stage62 archive seed roster changed")
                directory = (training_raw / p / arm / "train" / "1" / "training"
                    if split == spec.SPLITS[0] else checkpoint.parent)
                episodes = []
                for row in rows:
                    with np.load(directory / f"episode_{row['seed']}.npz") as archive:
                        raw = {key: archive[key] for key in ("physical", "measurement", "achieved_before", "target_before",
                            "lower_reference", "lower_value_context", "reward", "decision_steps")}
                    if float(np.sum(raw["reward"])) != row["episode_return"]:
                        raise ValueError("Stage62 archived reward differs from source")
                    episode = episode_terms(model, features(raw, args, period, model.config.gamma), period)
                    for key, count in episode["cost"].items():
                        cost[key] += count
                    episodes.append(episode)
                groups[p][arm][split] = {"seeds": roles[split], "episodes": len(episodes), "metrics": summarize(episodes)}
            current, expected = model.state_dict(), dict(saved["state_dict"])
            if current.pop("config") != expected.pop("config"):
                raise ValueError("Stage62 configuration changed during diagnosis")
            torch.testing.assert_close(current, expected, atol=0, rtol=0)
            checks[p][arm] = "passed"
            print(f"credit/critic diagnosis {root}/period{period}/{arm}: frozen state passed", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "options": spec.options(preflight=preflight), "seed_roles": roles,
        "budget": spec.budget(preflight=preflight), "cost": cost, "groups": groups, "frozen_state_checks": checks,
        "source_result": str(file), "training_result": str(training_file), "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["root"] not in spec.roots(preflight=preflight) or c["preflight"] != preflight
            or c["options"] != spec.options(preflight=preflight) or c["seed_roles"] != spec.seed_roles(c["root"], preflight=preflight)
            or c["budget"] != spec.budget(preflight=preflight) or c["cost"] != c["budget"]
            or set(c["groups"]) != {str(p) for p in spec.PERIODS}
            or c["frozen_state_checks"] != {str(p): dict.fromkeys(spec.TRAIN_POLICIES, "passed") for p in spec.PERIODS}):
        raise ValueError("Stage62 protocol, frozen state or accounting changed")
    for p, arms in c["groups"].items():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage62 arm roster changed")
        for splits in arms.values():
            if set(splits) != set(spec.SPLITS):
                raise ValueError("Stage62 split roster changed")
            for split, row in splits.items():
                if row["seeds"] != c["seed_roles"][split] or row["episodes"] != len(row["seeds"]):
                    raise ValueError("Stage62 archive pairing changed")
                expected_boundaries = row["episodes"] * (spec.arguments(c["root"], preflight=preflight).horizon // int(p) - 1)
                if row["metrics"]["boundary"]["count"] != expected_boundaries:
                    raise ValueError("Stage62 artificial boundary count changed")
    return {"root": c["root"], "groups": {p: {a: {split: row["metrics"] for split, row in splits.items()}
        for a, splits in arms.items()} for p, arms in c["groups"].items()}}


def mean_tree(values):
    if isinstance(values[0], dict):
        return {key: mean_tree([v[key] for v in values]) for key in values[0]}
    observed = [v for v in values if v is not None]
    return None if not observed else float(np.mean(observed))


def aggregate(cells, *, preflight):
    by_root = {c["root"]: c for c in cells}
    if len(by_root) != len(cells) or set(by_root) != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage62 complete root roster required")
    rows = [qualify(by_root[root], preflight=preflight) for root in spec.roots(preflight=preflight)]
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "equal_root_descriptive_means": mean_tree([r["groups"] for r in rows]),
        "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget(preflight=preflight)},
        "performance_claim": "none_diagnostic_only"}

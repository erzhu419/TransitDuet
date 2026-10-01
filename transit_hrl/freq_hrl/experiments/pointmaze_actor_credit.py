"""Diagnose the actual pre-update actor signal without another policy update."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_normalized_update as normalized
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_credit_diagnostics as credit
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import write_json
from scripts import pointmaze_actor_credit_stage66_spec as spec


def actor_gradients(actor, lower, advantages, *, clip_ratio, chunk_size):
    named = list(actor.named_parameters())
    params = [p for _, p in named]
    mask = np.concatenate([np.full(p.numel(), name == "log_std", dtype=bool) for name, p in named])
    gradients = {k: np.zeros(len(mask), dtype=np.float64) for k in (*advantages, "entropy")}
    device = params[0].device
    state, action, old_logp = [torch.as_tensor(v, dtype=torch.float32, device=device)
        for v in (lower.state, lower.action, lower.old_logp)]
    signals = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in advantages.items()}
    count, logp_error = 0, 0.
    for start in range(0, lower.size, chunk_size):
        stop = min(start + chunk_size, lower.size)
        logp, entropy = actor.log_prob_entropy(state[start:stop], action[start:stop])
        logp_error = max(logp_error, float((logp.detach() - old_logp[start:stop]).abs().max()))
        ratio = torch.exp((logp - old_logp[start:stop]).clamp(-20., 20.))
        clipped = ratio.clamp(1. - clip_ratio, 1. + clip_ratio)
        losses = {k: -torch.minimum(ratio * v[start:stop], clipped * v[start:stop]).sum() / lower.size
            for k, v in signals.items()}
        losses["entropy"] = -entropy.sum() / lower.size
        for index, (key, loss) in enumerate(losses.items()):
            g = torch.autograd.grad(loss, params, retain_graph=index < len(losses) - 1, allow_unused=True)
            gradients[key] += np.concatenate([np.zeros(p.numel()) if v is None else v.detach().double().cpu().numpy().reshape(-1)
                for p, v in zip(params, g)])
        count += 1
    return gradients, mask, {"actor_score_forward_batches": count,
        "actor_score_backward_batches": count * len(gradients), "max_abs_old_logp_difference": logp_error}


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(np.clip(np.dot(a, b) / denominator, -1., 1.))


def gradient_summary(gradients, sigma_mask, entropy_coef):
    result = {}
    for name, mask in (("all", np.ones(len(sigma_mask), dtype=bool)), ("mean", ~sigma_mask), ("log_std", sigma_mask)):
        gae, mc, entropy = [gradients[k][mask] for k in ("gae", "mc", "entropy")]
        result[name] = {"gae_credit_norm": float(np.linalg.norm(gae)), "mc_credit_norm": float(np.linalg.norm(mc)),
            "entropy_norm": float(np.linalg.norm(entropy)), "credit_cosine": cosine(gae, mc),
            "full_objective_cosine": cosine(gae + entropy_coef * entropy, mc + entropy_coef * entropy)}
    result["log_std_loss_gradient"] = {k: v[sigma_mask].tolist() for k, v in gradients.items()}
    return result


def diagnose(model, lower, mc, *, horizon):
    before = copy.deepcopy(model.state_dict())
    advantage, _ = model._gae(lower.reward, lower.done, lower.duration, lower.old_value, lower.next_value, lower.terminal)
    mc_advantage = mc - lower.old_value
    gradients, mask, cost = actor_gradients(model.lower_actor, lower,
        {"gae": model._normalize(advantage), "mc": model._normalize(mc_advantage)},
        clip_ratio=model.config.clip_ratio, chunk_size=spec.SCORE_CHUNK_SIZE)
    after = model.state_dict()
    if before.pop("config") != after.pop("config"):
        raise ValueError("Stage66 configuration changed during diagnosis")
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    phase, width = np.arange(lower.size) % horizon, max(1, horizon // 10)
    windows = {}
    for name, selected in (("first_decile", phase < width), ("last_decile", phase >= horizon - width)):
        windows[name] = {"value_mc": credit.value_metrics(lower.old_value[selected], mc[selected]),
            "gae_mean": float(advantage[selected].mean()), "mc_advantage_mean": float(mc_advantage[selected].mean())}
    return {"value_mc": credit.value_metrics(lower.old_value, mc), "gae_advantage_std": float(np.std(advantage)),
        "mc_advantage_std": float(np.std(mc_advantage)), "credit_alignment": credit.alignment(advantage, mc_advantage),
        "gradient": gradient_summary(gradients, mask, model.config.entropy_coef), "windows": windows,
        "model_and_Adam_unchanged": "passed", "score_cost": cost}


def replay(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    source_row, _, _, _ = normalized.qualify(source, preflight=preflight)
    original = json.loads(spec.source.source.training_result(root, preflight=preflight).read_text())
    critics = json.loads(spec.source.source_result(root, preflight=preflight).read_text())
    normalized.values.qualify(critics, preflight=preflight)
    clones, _, initialization = normalized.native.load_source(root, preflight=preflight)
    args = spec.source.arguments(root, preflight=preflight)
    roles = spec.source.seed_roles(root, preflight=preflight)
    archive = spec.source.source.training_result(root, preflight=preflight).parent
    archive = archive.with_name(archive.name + "_raw")
    started, cost, groups = time.monotonic(), dict.fromkeys(spec.budget(preflight=preflight), 0), {}
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"), initializer=continuing.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                fits, weights = {}, {}
                for treatment in spec.TREATMENTS:
                    path = critics["groups"][p][arm]["treatments"][treatment]["checkpoint"]
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    fits[treatment] = normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment=treatment)
                    weights[treatment] = joint.inference_weights(clone)
                    weights[treatment]["lower_value"] = fits[treatment].public_state()
                    cost["critic_checkpoint_loads"] += 1
                first = original["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training"]:
                    raise ValueError("Stage66 first-batch roster changed")
                path = archive / p / arm / "train" / "1" / "training"
                pairs = list(pool.map(continuing.worker_pair, [(weights["gae_raw"], weights[spec.source.CANDIDATE],
                    str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in first["rows"]]))
                for (_, _, row), old in zip(pairs, first["rows"]):
                    if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                        raise ValueError("Stage66 archived action or reward changed")
                    for key, field in (("archive_episodes", None), ("reconstructed_lower_calls", "lower_calls"),
                            ("reconstructed_upper_calls", "upper_calls"), ("extra_critic_scalar_calls", "episode_value_calls"),
                            ("archive_network_checks", "network_checks")):
                        cost[key] += 1 if field is None else row[field]
                batch = concat_hierarchical_batches([b for b, _, _ in pairs])
                lower = {"gae_raw": continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower,
                    spec.source.CANDIDATE: continuing.episode_batch(batch, np.concatenate([v for _, v, _ in pairs]), args.horizon).lower}
                mc = normalized.values.monte_carlo_returns(lower["gae_raw"], clone.config.gamma)
                cost["mc_calls"] += 1
                rows = {}
                for treatment, fit in fits.items():
                    row = diagnose(fit.model, lower[treatment], mc, horizon=args.horizon)
                    expected = source["groups"][p][arm]["updates"][treatment]
                    if row["value_mc"] != expected["source_probe"] or row["gae_advantage_std"] != expected["actor"]["advantage_std"]:
                        raise ValueError("Stage66 does not reproduce the actual Stage65 pre-update critic/GAE")
                    for key in ("source_probe_checks", "gae_calls", "frozen_model_snapshots", "frozen_model_checks"):
                        cost[key] += 1
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += row["score_cost"][key]
                    rows[treatment] = row
                groups[p][arm] = rows
                print(f"pre-update credit {root}/period{period}/{arm}: actual Stage65 signal exact; model/Adam frozen", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "cost": cost, "source_initialization": initialization, "groups": groups,
        "source_Stage65_endpoints": source_row["endpoints"], "new_native_steps": 0, "optimizer_steps": 0,
        "critic_fits": 0, "checkpoint_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return result


def qualify(cell, *, preflight):
    if (cell["root"] not in spec.roots(preflight=preflight) or cell["status"] != "complete"
            or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract() or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or any(cell[k] for k in ("new_native_steps", "optimizer_steps", "critic_fits", "checkpoint_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage66 frozen diagnosis or cost changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage66 arm roster changed")
        for rows in arms.values():
            if set(rows) != set(spec.TREATMENTS) or any(r["model_and_Adam_unchanged"] != "passed" for r in rows.values()):
                raise ValueError("Stage66 critic roster or frozen model changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage66 needs the complete frozen root roster")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "new_native_steps": 0, "optimizer_steps": 0, "critic_fits": 0, "checkpoint_writes": 0}

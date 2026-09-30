"""Native positive controls and independent-batch task-credit directions."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
from scipy.linalg import solve_continuous_are
import torch
from freq_hrl.rl.dual_actor_critic import GaussianActor
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_update_direction import load_pair, score_targets
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_lower_learnability_stage51_spec as spec


def feedback_gain():
    a, b = np.zeros((4, 4)), np.vstack((np.zeros((2, 2)), np.eye(2)))
    a[:2, 2:] = np.eye(2)
    p = solve_continuous_are(a, b, np.diag(spec.Q_DIAGONAL), np.diag(spec.R_DIAGONAL))
    return np.linalg.solve(np.diag(spec.R_DIAGONAL), b.T @ p)


def feedback_mean(state, gain, goal):
    error = state[..., 4:6] if goal == "waypoint" else state[..., -6:-4] - state[..., :2]
    command = error @ gain[:, :2].T - state[..., 2:4] @ gain[:, 2:].T - state[..., -4:-2]
    return torch.atanh(command.clamp(-spec.ACTION_LIMIT, spec.ACTION_LIMIT))


class FeedbackActor(GaussianActor):
    def __init__(self, source, gain, goal):
        torch.nn.Module.__init__(self)
        self.register_buffer("log_std", source.log_std.detach().clone())
        self.register_buffer("gain", torch.as_tensor(gain, dtype=source.log_std.dtype, device=source.log_std.device))
        self.goal = goal

    def distribution(self, state):
        return torch.distributions.Normal(feedback_mean(state, self.gain, self.goal), self.log_std.exp().clamp(1e-4, 3.))


def clone_mean(model, states, labels, *, root, goal, epochs, sham):
    result = copy.deepcopy(model)
    x = torch.as_tensor(states, dtype=torch.float32, device=model.device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=model.device)
    if sham:
        order = np.random.default_rng(spec.label_seed(root, goal)).permutation(len(x))
        y = y[torch.as_tensor(order, device=model.device)]
    optimizer = torch.optim.Adam(result.lower_actor.net.parameters(), lr=spec.BC_LR)
    rng, steps = np.random.default_rng(spec.shuffle_seed(root, goal)), 0
    with torch.no_grad():
        initial = float((result.lower_actor.net(x) - y).square().mean())
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), spec.BC_MINIBATCH):
            indices = torch.as_tensor(order[start:start + spec.BC_MINIBATCH], device=model.device)
            loss = (result.lower_actor.net(x[indices]) - y[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    with torch.no_grad():
        final = float((result.lower_actor.net(x) - y).square().mean())
    before, after = model.state_dict(), result.state_dict()
    for name in before:
        if name not in ("lower_actor", "config"):
            torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
    torch.testing.assert_close(result.lower_actor.log_std, model.lower_actor.log_std, atol=0, rtol=0)
    return result, {"supervised_optimizer_steps": steps, "initial_label_mse": initial, "final_label_mse": final,
                    "sham": sham, "shuffle_seed": spec.shuffle_seed(root, goal),
                    "label_seed": spec.label_seed(root, goal) if sham else None, "unchanged_other_state": "passed"}


def score_gradient(actor, batch, weight, *, entropy_coef=0., scale=1.):
    actor = copy.deepcopy(actor).double()
    parameters = tuple(actor.parameters())
    state, action = torch.as_tensor(batch.state, dtype=torch.float64), torch.as_tensor(batch.action, dtype=torch.float64)
    logp, entropy = actor.log_prob_entropy(state, action)
    objective = scale * (logp * torch.as_tensor(weight, dtype=torch.float64)).mean() + entropy_coef * entropy.mean()
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(objective, parameters)]).detach().numpy()


def alignment(left, right):
    norms = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    dot = float(left @ right)
    return {"dot": dot, "cosine": dot / (norms[0] * norms[1]) if all(norms) else 0.,
            "A_norm": norms[0], "B_reward_norm": norms[1], "zero_norm": not all(norms)}


def credit_directions(model, batch_a, rewards_a, batch_b, rewards_b, labels):
    reference = score_gradient(model.lower_actor, batch_b, score_targets(rewards_b).reshape(-1),
                               scale=batch_b.size / len(rewards_b))
    gae, _ = model._gae(batch_a.reward, batch_a.done, batch_a.duration, batch_a.old_value, batch_a.next_value, batch_a.terminal)
    weights = {"mc": score_targets(rewards_a).reshape(-1).astype(np.float32), "gae": gae}
    vectors = {k: score_gradient(model.lower_actor, batch_a, model._normalize(w), entropy_coef=model.config.entropy_coef)
               for k, w in weights.items()}
    for goal, target in labels.items():
        actor = copy.deepcopy(model.lower_actor).double()
        objective = -(actor.net(torch.as_tensor(batch_a.state, dtype=torch.float64)) - torch.as_tensor(target, dtype=torch.float64)).square().mean()
        params = tuple(actor.parameters())
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        vectors["bc_" + goal] = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)]).detach().numpy()
    return {k: alignment(v, reference) for k, v in vectors.items()}, {"B_raw_mc": reference, **vectors}


def worker_rollout(job):
    weights, seed, policy, phase, mode, gain, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    original = model.lower_actor
    if policy.startswith("teacher_"):
        model.lower_actor = FeedbackActor(original, gain, policy.removeprefix("teacher_"))
    try:
        batch, row, raw = joint.rollout(model, args, "learned_history", seed=seed, capture=True,
            lower_credit="task_option", lower_value_context_builder=clocks.context_builder("task_clock"),
            **spec.rollout_arguments(args.optimizer_seed, seed, phase=phase, mode=mode))
    finally:
        model.lower_actor = original
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), deployment_mode=mode)
    lower = None if batch is None else batch.lower
    clocks.audit_context(lower, row, raw["lower_value_context"], clock=True)
    if lower is not None:
        np.testing.assert_array_equal(lower.reward, raw["reward"].astype(np.float32))
        np.testing.assert_array_equal(lower.state[:, :2], raw["achieved_before"].astype(np.float32))
        np.testing.assert_array_equal(lower.state[:, -6:], raw["measurement"].astype(np.float32))
    np.savez_compressed(path, **raw)
    return lower, row, raw["reward"] if lower is not None else None


def train(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles = spec.seed_roles(root, preflight=preflight)
    source = json.loads(spec.source_result(root, "task_clock", preflight=preflight).read_text())
    model = load_pair(source, root=root, method="task_clock", preflight=preflight)[0]
    if (model.config.lower_state_dim, model.config.lower_action_dim) != (390, 2):
        raise ValueError("Stage51 requires the registered PointMaze lower feature layout")
    raw, started, gain = raw_directory(output), time.monotonic(), feedback_gain()
    keys = ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")
    counts = {phase: dict.fromkeys(keys, 0) for phase in ("A", "B", "eval")}
    batch_rows, evaluation, training, checkpoints = {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(model.config, args, "learned_history", "task_option")) as pool:
        def episodes(controller, policy, phase, mode, seeds):
            directory = raw / phase / policy / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(controller)
            outputs = list(pool.map(worker_rollout, [(weights, seed, policy, phase, mode, gain,
                           str(directory / f"episode_{seed}.npz")) for seed in seeds]))
            rows = [r for _, r, _ in outputs]
            joint.audit_trajectories(rows, args=args, method="learned_history", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in keys[1:]:
                    counts[phase][key] += row[key]
            return outputs

        batches, rewards = {}, {}
        for phase in ("A", "B"):
            outputs = episodes(model, "frozen", phase, "fitting", roles[phase])
            batches[phase] = concat_level_batches([b for b, _, _ in outputs])
            rewards[phase] = np.stack([r for _, _, r in outputs])
            batch_rows[phase] = [row for _, row, _ in outputs]
        x = torch.as_tensor(batches["A"].state)
        labels = {goal: feedback_mean(x, torch.as_tensor(gain, dtype=x.dtype), goal).numpy() for goal in ("waypoint", "task")}
        credit, vectors = credit_directions(model, batches["A"], rewards["A"], batches["B"], rewards["B"], labels)
        np.savez_compressed(raw / "credit_directions.npz", **vectors)
        policies = {p: model for p in spec.POLICIES[:3]}
        for goal in ("waypoint", "task"):
            for sham in (False, True):
                policy = ("sham_" if sham else "clone_") + goal
                controller, training[policy] = clone_mean(model, batches["A"].state, labels[goal], root=root,
                    goal=goal, epochs=opt["bc_epochs"], sham=sham)
                policies[policy] = controller
                path = raw / (policy + ".pt")
                torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "policy": policy,
                            "epochs": opt["bc_epochs"], "state_dict": controller.state_dict()}, path)
                checkpoints[policy] = str(path)
                print(f"fit {root}/{policy}: {training[policy]}", flush=True)
        for policy in spec.POLICIES:
            evaluation[policy] = {mode: [row for _, row, _ in episodes(policies[policy], policy, "eval", mode, roles["evaluation"])]
                                  for mode in spec.MODES}
            print(f"evaluated {root}/{policy}", flush=True)
    budget = spec.budget(preflight=preflight)
    if (sum(counts[p]["primitive_steps"] for p in ("A", "B")) != budget["fitting_primitive_steps"]
            or counts["eval"]["primitive_steps"] != budget["evaluation_primitive_steps"]):
        raise ValueError("Stage51 native accounting changed")
    warm = spec.warmup_iterations(preflight=preflight)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget,
        "inference_counts": counts, "batch_rows": batch_rows, "evaluation_rows": evaluation, "training": training,
        "credit": credit, "feedback_gain": gain.tolist(), "checkpoints": checkpoints,
        "source_checkpoint": source["snapshots"][str(warm)]["checkpoint"], "source_checkpoint_iteration": warm,
        "native_trace_audits": budget["native_trace_audits"], "wall_seconds": time.monotonic() - started,
        "computation": {"riccati_solves": 1, "score_backward_calls": 5,
                        "teacher_label_states": 2 * len(x), "supervised_optimizer_steps": sum(r["supervised_optimizer_steps"] for r in training.values()),
                        "original_actor_optimizer_steps": 0, "original_value_optimizer_steps": 0}}
    write_json(output, result)
    return result


def aggregate(results, *, preflight):
    roots, opt, budget = spec.roots(preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("Stage51 root roster incomplete")
    root_rows, totals, computation, audits = [], {}, {}, 0
    for root in roots:
        cell, roles = cells[root], spec.seed_roles(root, preflight=preflight)
        horizon = spec.arguments(root, preflight=preflight).horizon
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
                or cell["preflight"] != preflight or cell["options"] != opt or cell["budget"] != budget
                or cell["seed_roles"] != roles or cell["native_trace_audits"] != budget["native_trace_audits"]
                or cell["source_checkpoint_iteration"] != spec.warmup_iterations(preflight=preflight)):
            raise ValueError("Stage51 result violates frozen protocol")
        clones = set(spec.POLICIES[3:])
        if (set(cell["evaluation_rows"]) != set(spec.POLICIES) or set(cell["training"]) != clones
                or set(cell["checkpoints"]) != clones or set(cell["batch_rows"]) != {"A", "B"}
                or set(cell["credit"]) != {"mc", "gae", "bc_waypoint", "bc_task"}):
            raise ValueError("Stage51 policy or batch roster incomplete")
        expected_steps = opt["bc_epochs"] * math.ceil(opt["batch_paths"] * horizon / spec.BC_MINIBATCH)
        for policy, row in cell["training"].items():
            goal, sham = policy.split("_")[1], policy.startswith("sham_")
            if (row["supervised_optimizer_steps"] != expected_steps or row["unchanged_other_state"] != "passed"
                    or row["sham"] != sham or row["shuffle_seed"] != spec.shuffle_seed(root, goal)
                    or row["label_seed"] != (spec.label_seed(root, goal) if sham else None)):
                raise ValueError("Stage51 cloning cost or label pairing changed")
        expected_computation = {"riccati_solves": 1, "score_backward_calls": 5,
            "teacher_label_states": 2 * opt["batch_paths"] * horizon, "supervised_optimizer_steps": 4 * expected_steps,
            "original_actor_optimizer_steps": 0, "original_value_optimizer_steps": 0}
        if cell["computation"] != expected_computation:
            raise ValueError("Stage51 computation accounting changed")
        for key, value in cell["computation"].items():
            computation[key] = computation.get(key, 0) + value
        observed = {p: dict.fromkeys(cell["inference_counts"][p], 0) for p in ("A", "B", "eval")}

        def inspect(rows, phase, mode, seeds):
            if [r["seed"] for r in rows] != seeds:
                raise ValueError("Stage51 paired path roster changed")
            for row in rows:
                kwargs = spec.rollout_arguments(root, row["seed"], phase=phase, mode=mode)
                if (row["episode_length"] != horizon or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                        or row["deployment_mode"] != mode or row["lower_inference_calls"] != horizon
                        or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
                    raise ValueError("Stage51 native sampling or horizon changed")
                observed[phase]["primitive_steps"] += horizon
                for key in observed[phase]:
                    if key != "primitive_steps":
                        observed[phase][key] += row[key]

        for phase in ("A", "B"):
            inspect(cell["batch_rows"][phase], phase, "fitting", roles[phase])
        means = {m: {} for m in spec.MODES}
        for policy, stages in cell["evaluation_rows"].items():
            if set(stages) != set(spec.MODES):
                raise ValueError("Stage51 deployment modes incomplete")
            for mode, rows in stages.items():
                inspect(rows, "eval", mode, roles["evaluation"])
                means[mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
        if observed != cell["inference_counts"]:
            raise ValueError("Stage51 inference accounting changed")
        for phase in observed.values():
            for key, value in phase.items():
                totals[key] = totals.get(key, 0) + value
        audits += cell["native_trace_audits"]
        root_rows.append({"root": root, "means": means, "credit": cell["credit"], "training": cell["training"],
                          "endpoints": spec.contrasts(means, cell["credit"])})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": root_rows, "method_cost": totals, "computation": computation,
        "native_trace_audits": audits, "verification_primitive_steps": 0}
    if not preflight:
        x = np.asarray([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in root_rows])
        indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"} for i, k in enumerate(spec.ENDPOINTS)}
    return summary

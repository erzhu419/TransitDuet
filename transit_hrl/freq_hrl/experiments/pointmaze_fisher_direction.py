"""Full-task score ascent with matrix-free Fisher geometry and exact episode KL."""

import copy
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from . import pointmaze_episode_credit as episodes
from . import pointmaze_episode_kl as bounded
from scripts import pointmaze_fisher_direction_stage50_spec as spec


def score_geometry(actor, batch, advantage, entropy_coef):
    actor = copy.deepcopy(actor).double()
    parameters = tuple(actor.parameters())
    state = torch.as_tensor(batch.state, dtype=torch.float64, device=parameters[0].device)
    action = torch.as_tensor(batch.action, dtype=torch.float64, device=state.device)
    weight = torch.as_tensor(advantage, dtype=torch.float64, device=state.device)
    logp, entropy = actor.log_prob_entropy(state, action)
    objective = (logp * weight).mean() + entropy_coef * entropy.mean()
    gradient = torch.cat([g.reshape(-1) for g in torch.autograd.grad(objective, parameters)]).detach().cpu().numpy()
    old = actor.distribution(state)
    old = torch.distributions.Normal(old.mean.detach(), old.stddev.detach())
    kl = torch.distributions.kl_divergence(old, actor.distribution(state)).sum(dim=-1).mean()
    first = torch.cat([g.reshape(-1) for g in torch.autograd.grad(kl, parameters, create_graph=True)])

    def fisher(vector):
        v = torch.as_tensor(vector, dtype=torch.float64, device=state.device)
        result = torch.autograd.grad(torch.dot(first, v), parameters, retain_graph=True)
        return torch.cat([g.reshape(-1) for g in result]).detach().cpu().numpy()

    return gradient, fisher


def direction(actor, batch, advantage, entropy_coef, *, natural, horizon):
    gradient, fisher = score_geometry(actor, batch, advantage, entropy_coef)
    counts = {"score_backward_calls": 1, "fisher_vector_products": 0, "cg_iterations": 0}

    def multiply(vector):
        counts["fisher_vector_products"] += 1
        return fisher(vector)

    def iteration(_):
        counts["cg_iterations"] += 1

    vector, info, residual = gradient.copy(), 0, 0.
    if natural and np.linalg.norm(gradient) > 0:
        operator = LinearOperator((len(gradient), len(gradient)),
            matvec=lambda v: multiply(v) + spec.FISHER_DAMPING * v, dtype=np.float64)
        vector, info = cg(operator, gradient, maxiter=spec.CG_ITERATIONS, rtol=spec.CG_RTOL, atol=0., callback=iteration)
        residual = float(np.linalg.norm(operator @ vector - gradient) / np.linalg.norm(gradient))
    quadratic = float(vector @ multiply(vector))
    slope = float(gradient @ vector)
    if not np.isfinite(vector).all() or not math.isfinite(quadratic) or not math.isfinite(slope) or info < 0:
        raise ValueError("nonfinite or failed Fisher score direction")
    if np.linalg.norm(gradient) > 0 and (quadratic <= 0 or slope <= 0):
        raise ValueError("Fisher direction has nonpositive curvature or score ascent")
    coefficient = math.sqrt(2 * spec.KL_BUDGET / (horizon * quadratic)) if quadratic > 0 else 0.
    return vector * coefficient, {**counts, "cg_info": int(info), "cg_relative_residual": residual,
        "gradient_norm": float(np.linalg.norm(gradient)), "direction_norm": float(np.linalg.norm(vector)),
        "score_direction_dot": slope, "mean_fisher_quadratic": quadratic, "initial_coefficient": coefficient,
        "predicted_mean_episode_kl": .5 * horizon * quadratic * coefficient ** 2}


def update(model, batch, task_rewards, treatment, *, root, iteration):
    if treatment not in spec.TREATMENTS:
        raise ValueError("unregistered Fisher direction treatment")
    if treatment in ("gae", "episode_mc"):
        return bounded.update(model, batch, task_rewards, treatment, root=root, iteration=iteration, specification=spec)
    reference = copy.deepcopy(model.lower_actor)
    advantage = episodes.score_targets(task_rewards).reshape(-1).astype(np.float32)
    before = bounded.policy_terms(model, batch, reference, advantage, len(task_rewards))
    np.random.seed(spec.shuffle_seed(root, iteration))
    metrics = episodes.update(model, batch, task_rewards, treatment, specification=spec, actor_updates_enabled=False)
    proposal, geometry = direction(model.lower_actor, batch, model._normalize(advantage), model.config.entropy_coef,
        natural=treatment == "fisher_mc", horizon=batch.size // len(task_rewards))
    parameters = tuple(model.lower_actor.parameters())
    initial = parameters_to_vector(parameters).detach().clone()
    delta = torch.as_tensor(proposal, dtype=initial.dtype, device=initial.device)
    trials, accepted, selected = [], False, 0.
    for backtrack in range(spec.MAX_BACKTRACKS + 1):
        scale = 2. ** -backtrack
        with torch.no_grad():
            vector_to_parameters(initial + scale * delta, parameters)
        terms = bounded.policy_terms(model, batch, reference, advantage, len(task_rewards))
        trials.append({"scale": scale, **terms, "actor_optimizer_steps": 0, "value_optimizer_steps": 0})
        if terms["max_episode_kl"] <= spec.KL_BUDGET:
            accepted, selected = True, scale
            break
    if not accepted:
        with torch.no_grad():
            vector_to_parameters(initial.clone(), parameters)
    deployed = bounded.policy_terms(model, batch, reference, advantage, len(task_rewards))
    return {**metrics, "retained_actor_steps": 0, "retained_value_steps": metrics["value_optimizer_steps"],
        "accepted": accepted, "selected_scale": selected, "before_terms": before, "deployed_terms": deployed,
        "trials": trials, "kl_check_calls": len(trials) + 2, "geometry": geometry,
        "parameter_proposals": len(trials), "retained_parameter_updates": int(accepted and np.linalg.norm(proposal) > 0)}


def worker_rollout(job):
    return episodes.worker_rollout(job, specification=spec)


def train(root, *, preflight, output):
    return episodes.train(root, preflight=preflight, output=output, specification=spec,
                          rollout_worker=worker_rollout, update_fn=update)


def optimizer_steps(row, steps):
    if row["actor_credit"] in ("gae", "episode_mc"):
        return bounded.optimizer_steps(row, steps, specification=spec)
    trials, geometry = row["trials"], row["geometry"]
    if not 1 <= len(trials) <= spec.MAX_BACKTRACKS + 1:
        raise ValueError("direct direction trial roster changed")
    for index, trial in enumerate(trials):
        if (trial["scale"] != 2. ** -index or trial["actor_optimizer_steps"] != 0 or trial["value_optimizer_steps"] != 0
                or len(trial["episode_kl"]) != row["native_episodes"]
                or trial["max_episode_kl"] != max(trial["episode_kl"])
                or trial["mean_episode_kl"] != float(np.mean(trial["episode_kl"]))
                or (index < len(trials) - 1 and trial["max_episode_kl"] <= spec.KL_BUDGET)):
            raise ValueError("direct direction trial or first feasible selection changed")
    accept = trials[-1]["max_episode_kl"] <= spec.KL_BUDGET
    expected_terms = {k: trials[-1][k] for k in row["deployed_terms"]} if accept else row["before_terms"]
    natural = row["actor_credit"] == "fisher_mc"
    expected_products = geometry["cg_iterations"] + 2 if natural and geometry["gradient_norm"] > 0 else 1
    if (row["accepted"] != accept or row["selected_scale"] != (trials[-1]["scale"] if accept else 0.)
            or (not accept and len(trials) != spec.MAX_BACKTRACKS + 1)
            or row["deployed_terms"] != expected_terms or row["deployed_terms"]["max_episode_kl"] > spec.KL_BUDGET
            or row["retained_actor_steps"] != 0 or row["retained_value_steps"] != steps
            or row["kl_check_calls"] != len(trials) + 2 or row["parameter_proposals"] != len(trials)
            or row["retained_parameter_updates"] != int(accept and geometry["direction_norm"] > 0)
            or geometry["score_backward_calls"] != 1 or geometry["fisher_vector_products"] != expected_products
            or not 0 <= geometry["cg_iterations"] <= (spec.CG_ITERATIONS if natural else 0)
            or geometry["cg_info"] not in (0, spec.CG_ITERATIONS)
            or not math.isfinite(geometry["cg_relative_residual"])
            or (geometry["gradient_norm"] > 0 and (geometry["score_direction_dot"] <= 0 or geometry["mean_fisher_quadratic"] <= 0))
            or not math.isclose(geometry["predicted_mean_episode_kl"], spec.KL_BUDGET if geometry["gradient_norm"] > 0 else 0., abs_tol=1e-12)):
        raise ValueError("direct direction deployed policy or computation accounting changed")
    return {"actor_optimizer_steps": 0, "value_optimizer_steps": steps}


def aggregate(results, *, preflight):
    summary = episodes.aggregate(results, preflight=preflight, specification=spec, step_counts=optimizer_steps)
    totals = dict.fromkeys(("score_backward_calls", "fisher_vector_products", "cg_iterations", "parameter_proposals",
                           "retained_parameter_updates", "retained_actor_steps", "retained_value_steps", "kl_check_calls"), 0)
    for cell, root_row in zip(sorted(results, key=lambda r: r["root"]), summary["root_rows"]):
        if cell["root"] != root_row["root"]:
            raise ValueError("Fisher diagnostic root pairing changed")
        pairs = cell["first_batch_pair_details"][spec.METHODS[0]]
        if set(pairs) != set(spec.TREATMENTS[1:]) or any(p["status"] != "passed" or p["first_critic_update"] != "passed"
                or p["native_task_rewards"] != "passed" for p in pairs.values()):
            raise ValueError("Fisher first batch pair incomplete")
        root_row["direction_diagnostics"] = {}
        for treatment, history in cell["training"][spec.METHODS[0]].items():
            for row in history:
                for key in totals:
                    totals[key] += row.get(key, row.get("geometry", {}).get(key, 0))
            root_row["direction_diagnostics"][treatment] = {
                "accepted_updates": sum(r["accepted"] for r in history),
                "selected_scales": [r["selected_scale"] for r in history],
                "max_deployed_episode_kl": max(r["deployed_terms"]["max_episode_kl"] for r in history),
                "first_geometry": history[0].get("geometry"),
                "max_cg_relative_residual": max(r.get("geometry", {}).get("cg_relative_residual", 0.) for r in history)}
    summary["direction_computation"] = totals
    return summary

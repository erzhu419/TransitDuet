"""Regularized safe-action imitation and budget-matched causal wait heads."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_pair_ridge as ridge
from freq_hrl.domains.transit import native_nonlinear_wait as neural
from freq_hrl.domains.transit import native_wait_feasibility as wait

STEPS, SEED = neural.STEPS, neural.SEED
WAIT_VARIANTS = ("linear", "nonlinear_pair", "state_conditioned_pair")
VARIANTS = {f"{price}_{suffix}": (price, gate)
    for price in ("ridge", "best") for suffix, gate in (
        ("glm", "glm"), ("linear", "linear"), ("nonlinear", "nonlinear_pair"), ("conditioned", "state_conditioned_pair"))}
VARIANTS["best_unfiltered"] = ("best", None)
FIELDS = ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")


def state_key(row):
    return row["root"], row["scene_seed"], row["probe_index"]


def half_targets(rows, panels, indices):
    replaced, masses = [], {}
    for row in rows:
        values = panels[state_key(row)]["values"][list(indices)]
        if np.any(values[:, row["original_choice"]] != 0):
            raise ValueError("Training half changed the exact original-action identity")
        means = values.mean(axis=0)
        labels = [{"candidate": j, **{f: float(v) for f, v in zip(FIELDS, means[j])}}
            for j in range(6)]
        replaced.append({**row, "labels": labels})
        positive, negative = wait.wait_masses(values[..., 3])
        masses[state_key(row)] = {"positive_wait_min": positive.tolist(),
            "negative_wait_min": negative.tolist(), "draw_count": len(indices)}
    return replaced, masses


def best_choices(values, originals):
    return np.asarray([paired.choice_from_cost(np.where(v[:, 3] <= 0, v[:, 0], np.inf), original)
        for v, original in zip(values, originals)])


def centered_cost(raw, rows, model):
    x = (raw[..., :98]-model["center"])/model["scale"]
    return x-x[np.arange(len(rows)), [r["original_choice"] for r in rows]][:, None].copy()


def fit_heads(rows, masses, *, batch=None):
    batch = neural.batches(rows) if batch is None else batch
    price = ridge.fit_value(rows)
    reference = wait.fit_wait(rows, masses, "symmetric_feasibility")
    context_center, context_scale = batch[1].mean(axis=0), batch[1].std(axis=0)
    context_scale[context_scale < 1e-8] = 1.
    originals = [r["original_choice"] for r in rows]
    means = np.asarray([[[l[f] for f in FIELDS] for l in r["labels"]] for r in rows])
    targets = best_choices(means, originals)
    keys = [state_key(r) for r in rows]
    mask = np.arange(6)[None, :] != np.asarray(originals)[:, None]
    design = wait.design(batch[0], originals, reference["center"], reference["scale"])
    offset = design@np.asarray(reference["coefficients"])
    positive, negative = [np.asarray([masses[k][name] for k in keys])[mask]/reference["target_scale_min"]
        for name in ("positive_wait_min", "negative_wait_min")]
    positive, negative, weights, offset = [torch.as_tensor(a, dtype=torch.float64)
        for a in (positive, negative, np.ones(mask.sum()), offset[mask])]
    x_cost = torch.as_tensor(centered_cost(batch[0], rows, price), dtype=torch.float64)
    target = torch.as_tensor(targets, dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        best = nn.Linear(98, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(best.weight)
        networks, inputs = {}, {}
        for variant in WAIT_VARIANTS:
            torch.manual_seed(SEED)
            if variant == "linear":
                network = nn.Linear(199, 1, bias=False, dtype=torch.float64)
                nn.init.zeros_(network.weight)
                x = design
            else:
                network = neural.network(variant)
                x = neural.inputs(rows, batch, reference, variant, context_center, context_scale)[0]
            networks[variant] = network
            inputs[variant] = torch.as_tensor(x[mask], dtype=torch.float64)
        modules = {"best": best, **networks}
        optimizers = {k: torch.optim.AdamW(m.parameters(), lr=neural.LEARNING_RATE,
            weight_decay=neural.WEIGHT_DECAY) for k, m in modules.items()}
        initial = {k: [p.detach().clone() for p in m.parameters()] for k, m in modules.items()}
        def losses():
            return {"best": F.cross_entropy(-best(x_cost).squeeze(-1), target),
                **{v: neural.mass_loss(offset+networks[v](inputs[v]).squeeze(-1), positive, negative, weights)
                    for v in WAIT_VARIANTS}}
        initial_loss = {k: float(v.detach()) for k, v in losses().items()}
        for _ in range(STEPS):
            for name, loss in losses().items():
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
        final_loss = {k: float(v.detach()) for k, v in losses().items()}
        fits = {k: {"parameter_count": sum(p.numel() for p in m.parameters()), "steps": STEPS,
            "initial_loss": initial_loss[k], "final_loss": final_loss[k],
            "parameter_update_l2": float(sum((p-old).square().sum() for p, old in zip(m.parameters(), initial[k])).detach().sqrt())}
            for k, m in modules.items()}
        best_model = {"center": price["center"], "scale": price["scale"],
            "coefficients": best.weight.detach().numpy()[0].tolist(), "meaning": "decision_score_not_cost_forecast"}
        wait_models = {"glm": reference, "linear": {"coefficients": networks["linear"].weight.detach().numpy()[0].tolist()}}
        for variant in WAIT_VARIANTS[1:]:
            wait_models[variant] = {"variant": variant, "reference": reference,
                "context_center": context_center.tolist(), "context_scale": context_scale.tolist(),
                "weights": {k: v.detach().numpy().tolist() for k, v in networks[variant].state_dict().items()}}
        fits["state_conditioned_pair"]["context_weight_l2"] = float(networks["state_conditioned_pair"][0].weight[:, 198:].detach().norm())
    if not all(np.isfinite(f["final_loss"]) for f in fits.values()):
        raise ValueError("Nonfinite supervised objective")
    return {"ridge": price, "best": best_model, "wait": wait_models}, {"training_states": len(rows),
        "training_groups": reference["training_groups"], "nonidentity_targets": int(np.sum(targets != originals)),
        "target_draw_count": sorted({masses[k]["draw_count"] for k in keys}), "fits": fits,
        "glm_iterations": reference["optimizer_iterations"], "glm_final_loss": reference["final_loss"],
        "supervised_decision_updates": STEPS, "wait_residual_updates": 3*STEPS,
        "new_native_episodes": 0, "rl_actor_updates": 0}


def predict(rows, models, *, batch=None):
    batch = neural.batches(rows) if batch is None else batch
    x = centered_cost(batch[0], rows, models["ridge"])
    heads = (x@np.asarray(models["ridge"]["coefficients"]))*models["ridge"]["target_scale"]
    costs = {"ridge": heads[..., :2].sum(axis=-1),
        "best": centered_cost(batch[0], rows, models["best"])@np.asarray(models["best"]["coefficients"])}
    originals = [r["original_choice"] for r in rows]
    reference = models["wait"]["glm"]
    design = wait.design(batch[0], originals, reference["center"], reference["scale"])
    logits = {"glm": design@np.asarray(reference["coefficients"])}
    logits["linear"] = logits["glm"]+design@np.asarray(models["wait"]["linear"]["coefficients"])
    for variant in WAIT_VARIANTS[1:]:
        logits[variant] = neural.logits_many(rows, models["wait"][variant], batch=batch)
    for value in logits.values(): value[np.arange(len(rows)), originals] = 0.
    choices = {name: np.asarray([paired.choice_from_cost(np.where(logits[gate][i] <= 0, costs[price][i], np.inf)
        if gate is not None else costs[price][i], original) for i, original in enumerate(originals)])
        for name, (price, gate) in VARIANTS.items()}
    return {"costs": costs, "wait_logits": logits, "choices": choices}


def records(rows, panels, indices, alternate, prediction):
    values = np.stack([panels[state_key(r)]["values"][list(indices)] for r in rows])
    other = np.stack([panels[state_key(r)]["values"][list(alternate)].mean(axis=0) for r in rows])
    mean = values.mean(axis=1)
    originals = [r["original_choice"] for r in rows]
    target, other_target = best_choices(mean, originals), best_choices(other, originals)
    positive, negative = wait.wait_masses(values.transpose(1, 0, 2, 3)[..., 3])
    output = []
    for i, row in enumerate(rows):
        base = {k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index", "original_choice")}
        decisions = {}
        for name, selected in prediction["choices"].items():
            j = int(selected[i])
            decisions[name] = {"selected": j, "cost": float(mean[i, j, 0]), "wait": float(mean[i, j, 3]),
                "best_match": bool(j == target[i]), "unsafe": bool(mean[i, j, 3] > 0), "changed": j != originals[i]}
        nonidentity = np.arange(6) != originals[i]
        gates = {}
        for gate, logits in prediction["wait_logits"].items():
            p, n, z = positive[i, nonidentity], negative[i, nonidentity], logits[i, nonidentity]
            gates[gate] = {"mass_loss_min": float(np.mean(p*np.logaddexp(0., -z)+n*np.logaddexp(0., z))),
                "false_safe": int(np.sum((z <= 0) & (mean[i, nonidentity, 3] > 0))),
                "false_unsafe": int(np.sum((z > 0) & (mean[i, nonidentity, 3] <= 0)))}
        score = prediction["costs"]["best"][i]
        decision_loss = float(np.logaddexp.reduce(-score)+score[target[i]])
        output.append({**base, "disputed": bool(target[i] != other_target[i]), "teacher": int(target[i]),
            "best_action_cross_entropy": decision_loss, "decisions": decisions, "gates": gates})
    return output

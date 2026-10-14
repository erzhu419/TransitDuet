"""Joint physical cost/wait increments with clean inner raw references."""

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.transit import native_refit_binding as binding

CHECKPOINTS = (0, 50, 100, 200, 400, 600)
ANCHORS = ("original", "raw")
PARAMETERS = 3410
CONTROLS = ("glm", "raw", "c2v4")
VARIANTS = (*CONTROLS, *(f"pair_{a}_{mode}" for a in ANCHORS for mode in ("zero", "last", "mse", "selected")))


def physical_values(rows, panels, indices):
    return np.stack([panels[binding.learned.state_key(r)]["values"][list(indices)].mean(axis=0)[:, (0, 3)] for r in rows])


def fit_reference(rows, panels, indices, *, batch=None):
    targets, masses = binding.learned.half_targets(rows, panels, indices)
    ridge = binding.learned.ridge.fit_value(targets)
    reference = binding.learned.wait.fit_wait(targets, masses, "symmetric_feasibility")
    wait = binding.learned.neural.fit_wait(targets, masses, reference, "state_conditioned_pair", batch=batch)
    return {"ridge": ridge, "wait": wait}, {"draws": list(indices), "training_states": len(rows),
        "training_groups": ridge["training_groups"], "wait_training_groups": wait["training_groups"],
        "raw_steps": wait["steps"], "raw_parameters": wait["parameter_count"]}


def causal_data(rows, reference, *, batch=None):
    batch = binding.learned.neural.batches(rows) if batch is None else batch
    originals = np.asarray([r["original_choice"] for r in rows])
    x, _ = binding.learned.neural.inputs(rows, batch, reference["wait"]["reference"], "state_conditioned_pair",
        reference["wait"]["context_center"], reference["wait"]["context_scale"])
    z = binding.learned.neural.logits_many(rows, reference["wait"], batch=batch)
    z[np.arange(len(rows)), originals] = 0.
    heads = (binding.learned.centered_cost(batch[0], rows, reference["ridge"])@np.asarray(reference["ridge"]["coefficients"]))*reference["ridge"]["target_scale"]
    physical = np.stack((heads[..., :2].sum(axis=-1), heads[..., 2]), axis=-1)
    return {"x": x, "heads": physical, "raw_logits": z, "originals": originals,
        "roots": np.asarray([r["root"] for r in rows]),
        "raw_choices": binding.policy.selected(physical[..., 0], z, originals)}


def anchors(data, kind):
    return data["originals"] if kind == "original" else data["raw_choices"]


def relative(values, indices):
    return values-values[np.arange(len(values)), indices][:, None]


def increments(data, model, step):
    w = model["weights"][str(step)]
    residual = np.tanh(data["x"]@np.asarray(w["0.weight"]).T+w["0.bias"])@np.asarray(w["2.weight"]).T+w["2.bias"]
    return relative(data["heads"]+residual*model["scales"], anchors(data, model["anchor"]))


def choose(data, model, step):
    delta = increments(data, model, step)
    return binding.policy.selected(delta[..., 0], delta[..., 1], anchors(data, model["anchor"]))


def validation_stats(data, values, model, step):
    anchor = anchors(data, model["anchor"])
    mask = np.arange(6)[None, :] != anchor[:, None]
    errors = ((increments(data, model, step)-relative(values, anchor))/model["scales"])**2
    selected = choose(data, model, step)
    delta = values[np.arange(len(values)), selected]-values[np.arange(len(values)), data["raw_choices"]]
    scaled = delta/model["scales"]
    return {"error_sum": float(errors[mask].sum()), "error_cases": int(mask.sum()*2),
        "changed_count": int((selected != data["raw_choices"]).sum()),
        "by_root": {str(root): {"cases": int((data["roots"] == root).sum()),
            "cost_sum": float(delta[data["roots"] == root, 0].sum()), "wait_sum_min": float(delta[data["roots"] == root, 1].sum()),
            "scaled_cost_sum": float(scaled[data["roots"] == root, 0].sum()),
            "scaled_wait_sum": float(scaled[data["roots"] == root, 1].sum())} for root in sorted(set(data["roots"]))}}


def fit_path(rows, data, values, kind, *, validation=None):
    anchor = anchors(data, kind)
    mask = np.arange(6)[None, :] != anchor[:, None]
    targets = relative(values, anchor)
    scales = np.maximum(np.sqrt((targets[mask]**2).mean(axis=0)), 1e-6)
    tx, offset, target = [torch.as_tensor(a, dtype=torch.float64) for a in (data["x"], data["heads"]/scales, targets/scales)]
    ix, ta, tm = torch.arange(len(rows)), torch.as_tensor(anchor), torch.as_tensor(mask)
    model = {"anchor": kind, "scales": scales.tolist(), "weights": {}}
    curves, ledger = {}, []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(binding.learned.neural.SEED)
        network = binding.learned.neural.network("state_conditioned_pair")
        network[2] = nn.Linear(binding.learned.neural.WIDTH, 2, dtype=torch.float64)
        nn.init.zeros_(network[2].weight)
        nn.init.zeros_(network[2].bias)
        initial = [p.detach().clone() for p in network.parameters()]
        optimizer = torch.optim.AdamW(network.parameters(), lr=binding.learned.neural.LEARNING_RATE,
            weight_decay=binding.learned.neural.WEIGHT_DECAY)
        def loss():
            point = offset+network(tx)
            delta = point-point[ix, ta][:, None]
            return ((delta-target)[tm]**2).mean()
        for step in range(max(CHECKPOINTS)+1):
            if step in CHECKPOINTS:
                model["weights"][str(step)] = {k: v.detach().numpy().tolist() for k, v in network.state_dict().items()}
                ledger.append({"step": step, "training_mse": float(loss().detach())})
                if validation is not None: curves[str(step)] = validation_stats(*validation, model, step)
            if step == max(CHECKPOINTS): break
            objective = loss()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        update = float(sum((p-old).square().sum() for p, old in zip(network.parameters(), initial)).detach().sqrt())
    if not all(np.isfinite(x["training_mse"]) for x in ledger): raise ValueError("Nonfinite joint-advantage objective")
    fit = {"anchor": kind, "training_states": len(rows), "training_groups": sorted(map(list, {(r["root"], r["scene_seed"]) for r in rows})),
        "parameter_count": sum(p.numel() for p in network.parameters()), "steps": max(CHECKPOINTS),
        "scales": model["scales"], "parameter_update_l2": update, "loss_ledger": ledger}
    return model, fit, curves


def select_checkpoints(curves):
    pooled = {}
    for step in CHECKPOINTS:
        xs = [c[str(step)] for c in curves]
        by_root = {}
        for root in sorted({r for x in xs for r in x["by_root"]}):
            pieces = [x["by_root"][root] for x in xs if root in x["by_root"]]
            n = sum(x["cases"] for x in pieces)
            by_root[root] = {"cases": n, **{name: sum(x[field] for x in pieces)/n for name, field in (
                ("mean_delta_cost", "cost_sum"), ("mean_delta_wait_min", "wait_sum_min"),
                ("mean_scaled_cost", "scaled_cost_sum"), ("mean_scaled_wait", "scaled_wait_sum"))}}
        score = max(x[k] for x in by_root.values() for k in ("mean_scaled_cost", "mean_scaled_wait"))
        pooled[str(step)] = {"mse": sum(x["error_sum"] for x in xs)/sum(x["error_cases"] for x in xs),
            "joint_score": score, "by_root": by_root, "changed_count": sum(x["changed_count"] for x in xs)}
    mse_step = min(CHECKPOINTS, key=lambda s: (pooled[str(s)]["mse"], s))
    joint_step = min(CHECKPOINTS, key=lambda s: (pooled[str(s)]["joint_score"], s))
    # Keeping the frozen raw action has exactly zero validation increment.
    if pooled[str(joint_step)]["joint_score"] >= 0: joint_step = None
    return {"mse_step": mse_step, "joint_step": joint_step, "curves": pooled}


def collect(train, panels, indices, reused_oof):
    batch = binding.learned.neural.batches(train)
    curves, ledger, predictions = {a: [] for a in ANCHORS}, [], []
    halves = (tuple(indices[:2]), tuple(indices[2:]))
    for group, training, held in binding.policy.prefix_folds(train):
        rows, validation = [train[i] for i in training], [train[i] for i in held]
        train_batch, val_batch = tuple(x[training] for x in batch), tuple(x[held] for x in batch)
        for direction in (0, 1):
            a, b = halves[direction], halves[1-direction]
            reference, ref_fit = fit_reference(rows, panels, a, batch=train_batch)
            data, val_data = causal_data(rows, reference, batch=train_batch), causal_data(validation, reference, batch=val_batch)
            fits = {}
            for kind in ANCHORS:
                _, fits[kind], curve = fit_path(rows, data, physical_values(rows, panels, a), kind,
                    validation=(val_data, physical_values(validation, panels, b)))
                curves[kind].append(curve)
            predictions.append(val_data)
            ledger.append({"held_group": list(group), "direction": direction, "training_draws": list(a), "validation_draws": list(b),
                "reference": ref_fit, "fits": fits, "held_states": len(validation)})
        print(f"JOINT_ADVANTAGE_INNER_COMPLETE {group[0]}/{group[1]}", flush=True)
    costs = np.concatenate([p["heads"][..., 0] for p in predictions])
    logits = np.concatenate([p["raw_logits"] for p in predictions])
    originals = np.concatenate([p["originals"] for p in predictions])
    roots = np.concatenate([p["roots"] for p in predictions])
    difference = max(float(np.abs(costs-reused_oof["costs"]).max()), float(np.abs(logits-reused_oof["state_conditioned_pair"]).max()))
    changed = int(np.sum(np.concatenate([p["raw_choices"] for p in predictions]) != binding.policy.selected(
        reused_oof["costs"], reused_oof["state_conditioned_pair"], reused_oof["originals"])))
    if difference > 1e-12 or changed or not np.array_equal(originals, reused_oof["originals"]) or not np.array_equal(roots, reused_oof["roots"]):
        raise ValueError("Clean inner raw references failed Stage223 OOF reproduction")
    return {a: select_checkpoints(curves[a]) for a in ANCHORS}, ledger, {"max_abs_difference": difference, "changed_choices": changed}


def fit_final(train, panels, indices, bundle, selection):
    reference = {"ridge": bundle["ridge"], "wait": bundle["wait"]["state_conditioned_pair"]}
    data, values = causal_data(train, reference), physical_values(train, panels, indices)
    models, fits = {}, {}
    for kind in ANCHORS: models[kind], fits[kind], _ = fit_path(train, data, values, kind)
    return {"reference": reference, "paths": models, "selection": selection}, fits


def predict(rows, model, bundle, calibration):
    control = binding.predict(rows, bundle, calibration)
    gate_mapping = {"glm": "glm", "raw": "raw_state_conditioned_pair", "c2v4": "oof_state_conditioned_pair"}
    prediction = {"costs": control["costs"], "wait_logits": {g: control["wait_logits"][old] for g, old in gate_mapping.items()},
        "choices": {g: control["choices"][old] for g, old in gate_mapping.items()}, "advantages": {}}
    data = causal_data(rows, model["reference"])
    for kind in ANCHORS:
        path, selection = model["paths"][kind], model["selection"][kind]
        for mode, step in (("zero", 0), ("last", max(CHECKPOINTS)), ("mse", selection["mse_step"]), ("selected", selection["joint_step"])):
            name = f"pair_{kind}_{mode}"
            prediction["choices"][name] = data["raw_choices"].copy() if step is None else choose(data, path, step)
            if step is not None:
                prediction["advantages"][name] = {"delta": increments(data, path, step), "scales": path["scales"], "anchor": anchors(data, kind)}
    return prediction

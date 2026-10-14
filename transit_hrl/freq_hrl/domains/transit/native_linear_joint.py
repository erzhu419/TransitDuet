"""Low-capacity paired residuals with the same physical targets and raw references."""

import numpy as np
import torch

from freq_hrl.domains.transit import native_joint_advantage as joint
from freq_hrl.domains.transit.native_joint_selection import repair_selection

FEATURES = 198
PARAMETERS = FEATURES*2
VARIANTS = (*joint.VARIANTS, *(f"linear_{a}_{mode}" for a in joint.ANCHORS for mode in ("zero", "last", "mse", "selected")))


def increments(data, model, step):
    residual = data["x"][..., :FEATURES]@np.asarray(model["weights"][str(step)]).T
    return joint.relative(data["heads"]+residual*model["scales"], joint.anchors(data, model["anchor"]))


def fit_path(rows, data, values, kind, *, validation=None):
    anchor = joint.anchors(data, kind)
    mask = np.arange(6)[None, :] != anchor[:, None]
    targets = joint.relative(values, anchor)
    scales = np.maximum(np.sqrt((targets[mask]**2).mean(axis=0)), 1e-6)
    # Context-only terms and a bias cancel in a linear paired difference.
    x = joint.relative(data["x"][..., :FEATURES], anchor)
    offset = joint.relative(data["heads"], anchor)/scales
    tx, to, target = [torch.as_tensor(a, dtype=torch.float64) for a in (x[mask], offset[mask], (targets/scales)[mask])]
    weight = torch.zeros((2, FEATURES), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([weight], lr=joint.binding.learned.neural.LEARNING_RATE,
        weight_decay=joint.binding.learned.neural.WEIGHT_DECAY)
    model = {"anchor": kind, "scales": scales.tolist(), "weights": {}}
    curves, ledger = {}, []
    def loss(): return ((to+tx@weight.T-target)**2).mean()
    for step in range(max(joint.CHECKPOINTS)+1):
        if step in joint.CHECKPOINTS:
            model["weights"][str(step)] = weight.detach().numpy().tolist()
            ledger.append({"step": step, "training_mse": float(loss().detach())})
            if validation is not None:
                vd, vv = validation
                curves[str(step)] = joint.prediction_stats(vd, vv, increments(vd, model, step), scales, joint.anchors(vd, kind))
        if step == max(joint.CHECKPOINTS): break
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()
    if not all(np.isfinite(x["training_mse"]) for x in ledger): raise ValueError("Nonfinite linear paired objective")
    fit = {"anchor": kind, "training_states": len(rows), "training_groups": sorted(map(list, {(r["root"],r["scene_seed"]) for r in rows})),
        "parameter_count": weight.numel(), "steps": max(joint.CHECKPOINTS), "scales": model["scales"],
        "parameter_update_l2": float(weight.detach().norm()), "loss_ledger": ledger}
    return model, fit, curves


def collect(train, panels, indices, reused_oof):
    selected, ledger, reproduction = joint.collect(train, panels, indices, reused_oof, path_fitter=fit_path)
    cell = {"selection": selected, "inner_fits": ledger}
    return {a: repair_selection(cell, a) for a in joint.ANCHORS}, ledger, reproduction


def predict(rows, model, nonlinear, bundle, calibration):
    prediction = joint.predict(rows, nonlinear, bundle, calibration)
    data = joint.causal_data(rows, nonlinear["reference"])
    for kind in joint.ANCHORS:
        path, selection = model["paths"][kind], model["selection"][kind]
        for mode, step in (("zero",0),("last",max(joint.CHECKPOINTS)),("mse",selection["mse_step"]),("selected",selection["joint_step"])):
            name = f"linear_{kind}_{mode}"
            anchor = joint.anchors(data, kind)
            if step is None:
                prediction["choices"][name] = data["raw_choices"].copy()
            else:
                delta = increments(data, path, step)
                prediction["choices"][name] = joint.binding.policy.selected(delta[...,0], delta[...,1], anchor)
                prediction["advantages"][name] = {"delta": delta, "scales": path["scales"], "anchor": anchor}
    return prediction

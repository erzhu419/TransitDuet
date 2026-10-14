"""Explicit ridge shrinkage of the same paired physical residual design."""

import numpy as np

from freq_hrl.domains.transit import native_linear_joint as linear

joint = linear.joint
GAMMAS = (1000., 100., 10., 1., .1, .01)
LEVELS = ("zero", *(str(g) for g in GAMMAS))
VARIANTS = (*linear.VARIANTS, *(f"ridge_{a}_{mode}" for a in joint.ANCHORS for mode in ("zero", "strong", "weak", "mse", "selected")))


def fit_path(rows, data, values, kind, *, validation=None):
    anchor = joint.anchors(data, kind)
    mask = np.arange(6)[None,:] != anchor[:,None]
    target = joint.relative(values, anchor)
    scales = np.maximum(np.sqrt((target[mask]**2).mean(axis=0)), 1e-6)
    x = joint.relative(data["x"][...,:linear.FEATURES], anchor)[mask]
    offset = joint.relative(data["heads"], anchor)[mask]/scales
    y = target[mask]/scales-offset
    u,s,vt = np.linalg.svd(x, full_matrices=False)
    energy = s*s
    mean_energy = float(energy.sum()/(len(x)*linear.FEATURES))
    rank = int((s > s[0]*max(x.shape)*np.finfo(float).eps).sum())
    spectral = {"numerical_rank":rank, "effective_rank":float(energy.sum()**2/(energy@energy)),
        "energy95_rank":int(np.searchsorted(np.cumsum(energy),.95*energy.sum())+1),
        "mean_feature_energy":mean_energy, "paired_rows":len(x), "feature_count":linear.FEATURES}
    model = {"anchor":kind, "scales":scales.tolist(), "weights":{"zero":np.zeros((2,linear.FEATURES)).tolist()}}
    degrees, penalties = {"zero":0.}, {"zero":0.}
    uy = u.T@y
    for gamma,level in zip(GAMMAS,LEVELS[1:]):
        penalty = gamma*mean_energy
        coefficients = vt.T@((s/(energy+len(x)*penalty))[:,None]*uy)
        model["weights"][level] = coefficients.T.tolist()
        degrees[level] = float((energy/(energy+len(x)*penalty)).sum())
        penalties[level] = penalty
    curves, ledger = {}, []
    for level in LEVELS:
        coefficients = np.asarray(model["weights"][level]).T
        errors = ((offset+x@coefficients-target[mask]/scales)**2).mean(axis=0)
        ledger.append({"level":level,"training_mse":float(errors.mean()),"cost_mse":float(errors[0]),"wait_mse":float(errors[1]),
            "ridge":penalties[level],"ridge_loss":float(penalties[level]*np.sum(coefficients**2)/2),
            "effective_df_per_head":degrees[level],"coefficient_l2":float(np.linalg.norm(coefficients))})
        if validation is not None:
            vd,vv = validation
            curves[level] = joint.prediction_stats(vd,vv,linear.increments(vd,model,level),scales,joint.anchors(vd,kind))
    fit = {"anchor":kind,"training_states":len(rows),"training_groups":sorted(map(list,{(r["root"],r["scene_seed"]) for r in rows})),
        "parameter_count":linear.PARAMETERS,"steps":0,"solver":"training_pair_SVD_ridge","scales":model["scales"],
        "spectrum":spectral,"loss_ledger":ledger}
    return model,fit,curves


def select(curves, ledger, kind, *, levels=LEVELS):
    roots = sorted({str(f["held_group"][0]) for f in ledger})
    scales = {r:np.average([f["fits"][kind]["scales"] for f in ledger if str(f["held_group"][0])==r],axis=0,
        weights=[f["held_states"] for f in ledger if str(f["held_group"][0])==r]).tolist() for r in roots}
    pooled = {}
    for level in levels:
        xs = [c[level] for c in curves]
        by_root = {}
        for root in roots:
            pieces = [x["by_root"][root] for x in xs if root in x["by_root"]]
            n = sum(x["cases"] for x in pieces)
            cost,wait = sum(x["cost_sum"] for x in pieces)/n, sum(x["wait_sum_min"] for x in pieces)/n
            by_root[root] = {"cases":n,"mean_delta_cost":cost,"mean_delta_wait_min":wait,
                "mean_scaled_cost":cost/scales[root][0],"mean_scaled_wait":wait/scales[root][1]}
        pooled[level] = {"mse":sum(x["error_sum"] for x in xs)/sum(x["error_cases"] for x in xs),
            "joint_score":max(x[k] for x in by_root.values() for k in ("mean_scaled_cost","mean_scaled_wait")),
            "by_root":by_root,"changed_count":sum(x["changed_count"] for x in xs)}
    mse = min(levels,key=lambda l:pooled[l]["mse"])
    chosen = min(levels,key=lambda l:pooled[l]["joint_score"])
    return {"mse_level":mse,"joint_level":chosen if pooled[chosen]["joint_score"]<0 else None,
        "curves":pooled,"source_scales":scales}


def collect(train, panels, indices, reused_oof):
    curves,ledger,reproduction = joint.collect_paths(train,panels,indices,reused_oof,path_fitter=fit_path)
    return {a:select(curves[a],ledger,a) for a in joint.ANCHORS},ledger,reproduction


def predict(rows, model, frozen_linear, nonlinear, bundle, calibration):
    prediction = linear.predict(rows,frozen_linear,nonlinear,bundle,calibration)
    data = joint.causal_data(rows,nonlinear["reference"])
    for kind in joint.ANCHORS:
        path,selection = model["paths"][kind],model["selection"][kind]
        anchor = joint.anchors(data,kind)
        for mode,level in (("zero","zero"),("strong",LEVELS[1]),("weak",LEVELS[-1]),("mse",selection["mse_level"]),("selected",selection["joint_level"])):
            name = f"ridge_{kind}_{mode}"
            if level is None: prediction["choices"][name] = data["raw_choices"].copy()
            else:
                delta = linear.increments(data,path,level)
                prediction["choices"][name] = joint.binding.policy.selected(delta[...,0],delta[...,1],anchor)
                prediction["advantages"][name] = {"delta":delta,"scales":path["scales"],"anchor":anchor}
    return prediction

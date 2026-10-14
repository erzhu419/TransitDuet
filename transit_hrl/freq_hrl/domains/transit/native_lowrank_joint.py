"""Matched low-rank wait-only and dual-head physical residuals."""

import numpy as np

from freq_hrl.domains.transit import native_ridge_joint as ridge

joint = ridge.joint
RANKS = (0, 1, 3, 8, 16)
LEVELS = ("zero", "1", "3", "8", "16")
ARMS = ("wait", "joint")
MODES = ("zero", "rank1", "rank3", "rank8", "rank16", "mse", "selected")
VARIANTS = (*ridge.VARIANTS, *(f"lowrank_{a}_{arm}_{mode}" for a in joint.ANCHORS for arm in ARMS for mode in MODES))


def increments(data, model, level, arm):
    delta = ridge.linear.increments(data,model,level)
    if arm == "wait": delta[...,0] = joint.relative(data["heads"],joint.anchors(data,model["anchor"]))[...,0]
    return delta


def fit_path(rows, data, values, kind, *, validation=None):
    anchor = joint.anchors(data,kind)
    mask = np.arange(6)[None,:]!=anchor[:,None]
    target = joint.relative(values,anchor)
    scales = np.maximum(np.sqrt((target[mask]**2).mean(axis=0)),1e-6)
    x = joint.relative(data["x"][...,:ridge.linear.FEATURES],anchor)[mask]
    offset = joint.relative(data["heads"],anchor)[mask]/scales
    y = target[mask]/scales-offset
    u,s,vt = np.linalg.svd(x,full_matrices=False)
    numerical = int((s>s[0]*max(x.shape)*np.finfo(float).eps).sum())
    model = {"anchor":kind,"scales":scales.tolist(),"weights":{}}
    ledgers,curves = {arm:[] for arm in ARMS},{arm:{} for arm in ARMS}
    for requested,level in zip(RANKS,LEVELS):
        k = min(requested,numerical)
        coefficients = vt[:k].T@((u[:,:k].T@y)/s[:k,None])
        model["weights"][level] = coefficients.T.tolist()
        # The shared solve makes the wait predictor identical across both arms.
        for arm in ARMS:
            active = coefficients.copy()
            if arm == "wait": active[:,0] = 0.
            errors = ((offset+x@active-target[mask]/scales)**2).mean(axis=0)
            ledgers[arm].append({"level":level,"requested_rank":requested,"retained_rank":k,"training_mse":float(errors.mean()),
                "cost_mse":float(errors[0]),"wait_mse":float(errors[1]),"cost_df":0 if arm=="wait" else k,"wait_df":k})
            if validation is not None:
                vd,vv = validation
                va = joint.anchors(vd,kind)
                delta = increments(vd,model,level,arm)
                stats = joint.prediction_stats(vd,vv,delta,scales,va)
                e = ((delta-joint.relative(vv,va))/scales)**2
                stats["error_sum_by_head"] = e[np.arange(6)[None,:]!=va[:,None]].sum(axis=0).tolist()
                curves[arm][level] = stats
    fit = {"anchor":kind,"training_states":len(rows),"training_groups":sorted(map(list,{(r["root"],r["scene_seed"]) for r in rows})),
        "steps":0,"solver":"training_pair_TSVD","scales":model["scales"],"numerical_rank":numerical,
        "paired_rows":len(x),"feature_count":198,"arms":{a:{"parameter_count":198 if a=="wait" else 396,"loss_ledger":ledgers[a]} for a in ARMS}}
    return model,fit,curves


def collect(train,panels,indices,reused_oof):
    curves,ledger,reproduction = joint.collect_paths(train,panels,indices,reused_oof,path_fitter=fit_path)
    selections = {}
    for kind in joint.ANCHORS:
        selections[kind] = {}
        for arm in ARMS:
            xs = [c[arm] for c in curves[kind]]
            selected = ridge.select(xs,ledger,kind,levels=LEVELS)
            for level in LEVELS:
                head_sums = np.sum([c[level]["error_sum_by_head"] for c in xs],axis=0)
                n = sum(c[level]["error_cases"] for c in xs)/2
                selected["curves"][level].update(cost_mse=float(head_sums[0]/n),wait_mse=float(head_sums[1]/n))
            selections[kind][arm] = selected
    return selections,ledger,reproduction


def predict(rows,model,frozen_ridge,frozen_linear,nonlinear,bundle,calibration):
    prediction = ridge.predict(rows,frozen_ridge,frozen_linear,nonlinear,bundle,calibration)
    data = joint.causal_data(rows,nonlinear["reference"])
    for kind in joint.ANCHORS:
        path = model["paths"][kind]
        anchor = joint.anchors(data,kind)
        for arm in ARMS:
            selection = model["selection"][kind][arm]
            levels = (*LEVELS,selection["mse_level"],selection["joint_level"])
            for mode,level in zip(MODES,levels):
                name = f"lowrank_{kind}_{arm}_{mode}"
                if level is None: prediction["choices"][name] = data["raw_choices"].copy()
                else:
                    delta = increments(data,path,level,arm)
                    prediction["choices"][name] = joint.binding.policy.selected(delta[...,0],delta[...,1],anchor)
                    prediction["advantages"][name] = {"delta":delta,"scales":path["scales"],"anchor":anchor}
    return prediction

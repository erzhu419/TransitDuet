"""Prefix/future rank selection for cost residuals under the raw wait gate."""

import numpy as np

from freq_hrl.domains.transit import native_cost_gate as base

lowrank,joint = base.lowrank,base.joint
NEW_MODES = ("cost_mse", "mean_selected", "safe_selected")
FROZEN_SOURCES = {f"raw_gate_{a}_{m}":(a,m) for a in joint.ANCHORS for m in lowrank.MODES}
FROZEN = ("raw", *FROZEN_SOURCES)
VARIANTS = (*FROZEN, *(f"cost_rank_{a}_{m}" for a in joint.ANCHORS for m in NEW_MODES))


def stats(data, values, path, level):
    anchor = joint.anchors(data,path["anchor"])
    cost = lowrank.increments(data,path,level,"joint")[...,0]
    chosen = joint.binding.policy.selected(cost,data["raw_logits"],anchor)
    actual = values[np.arange(len(values)),chosen]
    raw = values[np.arange(len(values)),data["raw_choices"]]
    delta = actual-raw
    risk = (actual[:,1]>0).astype(int)-(raw[:,1]>0).astype(int)
    mask = np.arange(6)[None,:]!=anchor[:,None]
    errors = ((cost-joint.relative(values,anchor)[...,0])/path["scales"][0])**2
    return {"error_sum":float(errors[mask].sum()),"error_cases":int(mask.sum()),
        "changed_count":int((chosen!=data["raw_choices"]).sum()),
        "by_root":{str(r):{"cases":int((data["roots"]==r).sum()),
            "cost_sum":float(delta[data["roots"]==r,0].sum()),"wait_sum_min":float(delta[data["roots"]==r,1].sum()),
            "unsafe_delta_count":int(risk[data["roots"]==r].sum())} for r in sorted(set(data["roots"]))}}


def fit_path(rows, data, values, kind, *, validation=None):
    path,fit,_ = lowrank.fit_path(rows,data,values,kind)
    # Reuse the shared solver's cost column; its wait residual never controls actions.
    curves = {l:stats(*validation,path,l) for l in lowrank.LEVELS} if validation is not None else {}
    return path,fit,curves


def select(curves, ledger, kind):
    chosen = lowrank.ridge.select(curves,ledger,kind,levels=lowrank.LEVELS)
    for level,c in chosen["curves"].items():
        for root,x in c["by_root"].items():
            pieces = [f[level]["by_root"][root] for f in curves if root in f[level]["by_root"]]
            x["mean_delta_unsafe"] = sum(p["unsafe_delta_count"] for p in pieces)/x["cases"]
    zero = chosen["curves"]["zero"]
    if zero["changed_count"] or any(x[k]!=0 for x in zero["by_root"].values()
            for k in ("mean_delta_cost","mean_delta_wait_min","mean_delta_unsafe")):
        raise ValueError("Inner zero residual did not reproduce raw")
    eligible = [l for l in lowrank.LEVELS if chosen["curves"][l]["joint_score"]<0 and
        all(x["mean_delta_unsafe"]<=0 for x in chosen["curves"][l]["by_root"].values())]
    safe = min(eligible,key=lambda l:chosen["curves"][l]["joint_score"]) if eligible else None
    return {"mse_level":chosen["mse_level"],"mean_level":chosen["joint_level"],"safe_level":safe,
        "curves":chosen["curves"],"source_scales":chosen["source_scales"]}


def collect(train, panels, indices, reused_oof):
    curves,ledger,reproduction = joint.collect_paths(train,panels,indices,reused_oof,path_fitter=fit_path)
    return {a:select(curves[a],ledger,a) for a in joint.ANCHORS},ledger,reproduction


def predict(data, model, selection):
    frozen = base.predict(data,model)
    choices = {"raw":data["raw_choices"].copy()}
    choices.update({f"raw_gate_{a}_{m}":frozen["choices"][a][m]["raw_cost"] for a in joint.ANCHORS for m in lowrank.MODES})
    levels = {}
    for kind in joint.ANCHORS:
        s = selection[kind]
        levels[kind] = dict(zip(NEW_MODES,(s["mse_level"],s["mean_level"],s["safe_level"])))
        for mode,level in levels[kind].items():
            name = f"cost_rank_{kind}_{mode}"
            if level is None: choices[name] = data["raw_choices"].copy()
            else:
                cost = lowrank.increments(data,model["paths"][kind],level,"joint")[...,0]
                choices[name] = joint.binding.policy.selected(cost,data["raw_logits"],joint.anchors(data,kind))
    return {"choices":choices,"levels":levels}


def records(rows, values, prediction, source):
    output,changed,maximum = [],0,0.
    for i,r in enumerate(rows):
        decisions = {}
        for name,selected in prediction["choices"].items():
            j = int(selected[i])
            decisions[name] = {"selected":j,"cost":float(values[i,j,0]),"wait":float(values[i,j,1]),
                "changed":j!=r["original_choice"],"unsafe":bool(values[i,j,1]>0)}
        for name in FROZEN:
            if name=="raw": old = source[i]["raw"]
            else:
                kind,mode = FROZEN_SOURCES[name]
                old = source[i]["gates"][kind][mode]["decisions"]["raw_cost"]
            new = decisions[name]
            changed += int(new["selected"]!=old["selected"])
            maximum = max(maximum,abs(new["cost"]-old["cost"]),abs(new["wait"]-old["wait"]))
        output.append({**{k:r[k] for k in ("root","scenario","scene_seed","probe_index","original_choice")},
            "decisions":decisions,"levels":prediction["levels"]})
    if changed or maximum>1e-12: raise ValueError("Frozen Stage232 raw-gate controls changed")
    return output,{"choice_comparisons":len(rows)*len(FROZEN),"changed_choices":changed,
        "score_comparisons":len(rows)*len(FROZEN)*2,"max_abs_difference":maximum}

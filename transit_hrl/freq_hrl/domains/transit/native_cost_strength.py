"""Joint training-only selection of frozen cost rank and residual strength."""

import numpy as np

from freq_hrl.domains.transit import native_cost_rank_selection as base

joint,lowrank = base.joint,base.lowrank
ALPHAS = (0., .25, .5, .75, 1.)
CANDIDATES = {"zero":{"level":"zero","alpha":0.},
    **{f"{l}@{a:g}":{"level":l,"alpha":a} for l in lowrank.LEVELS[1:] for a in ALPHAS[1:]}}
UNIT_KEYS = {l:"zero" if l=="zero" else f"{l}@1" for l in lowrank.LEVELS}
NEW_MODES = base.NEW_MODES
FROZEN = base.VARIANTS
VARIANTS = (*FROZEN, *(f"cost_strength_{a}_{m}" for a in joint.ANCHORS for m in NEW_MODES))


def fit_path(rows, data, values, kind, *, validation=None):
    path,fit,_ = lowrank.fit_path(rows,data,values,kind)
    curves = {k:base.stats(*validation,path,c["level"],alpha=c["alpha"]) for k,c in CANDIDATES.items()} if validation is not None else {}
    return path,fit,curves


def collect(train, panels, indices, reused_oof):
    curves,ledger,reproduction = joint.collect_paths(train,panels,indices,reused_oof,path_fitter=fit_path)
    selection = {a:base.select(curves[a],ledger,a,levels=tuple(CANDIDATES)) for a in joint.ANCHORS}
    unit = {a:base.select([{l:c[k] for l,k in UNIT_KEYS.items()} for c in curves[a]],ledger,a) for a in joint.ANCHORS}
    return selection,ledger,reproduction,unit


def predict(data, model, selection, unit):
    frozen = base.predict(data,model,unit)
    choices = dict(frozen["choices"])
    candidates = {}
    for kind in joint.ANCHORS:
        s = selection[kind]
        candidates[kind] = dict(zip(NEW_MODES,(s["mse_level"],s["mean_level"],s["safe_level"])))
        for mode,key in candidates[kind].items():
            name = f"cost_strength_{kind}_{mode}"
            if key is None: choices[name] = data["raw_choices"].copy()
            else:
                c = CANDIDATES[key]
                cost = base.costs(data,model["paths"][kind],c["level"],c["alpha"])
                choices[name] = joint.binding.policy.selected(cost,data["raw_logits"],joint.anchors(data,kind))
    return {"choices":choices,"candidates":candidates}


def records(rows, values, prediction, source):
    output,changed,maximum = [],0,0.
    for i,r in enumerate(rows):
        decisions = {}
        for name,selected in prediction["choices"].items():
            j = int(selected[i])
            decisions[name] = {"selected":j,"cost":float(values[i,j,0]),"wait":float(values[i,j,1]),
                "changed":j!=r["original_choice"],"unsafe":bool(values[i,j,1]>0)}
        for name in FROZEN:
            old,new = source[i]["decisions"][name],decisions[name]
            changed += int(old["selected"]!=new["selected"])
            maximum = max(maximum,abs(new["cost"]-old["cost"]),abs(new["wait"]-old["wait"]))
        output.append({**{k:r[k] for k in ("root","scenario","scene_seed","probe_index","original_choice")},
            "decisions":decisions,"candidates":prediction["candidates"]})
    if changed or maximum>1e-12: raise ValueError("Frozen Stage233 controls changed")
    return output,{"choice_comparisons":len(rows)*len(FROZEN),"changed_choices":changed,
        "score_comparisons":len(rows)*len(FROZEN)*2,"max_abs_difference":maximum}

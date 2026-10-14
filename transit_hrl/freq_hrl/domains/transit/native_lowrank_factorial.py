"""Frozen predictor-head interventions at identical low-rank levels."""

import numpy as np

from freq_hrl.domains.transit import native_lowrank_joint as lowrank

joint = lowrank.joint
ARMS = ("base", "cost", "wait", "joint")


def predict(data, model):
    choices, menus, levels = {}, {}, {}
    for kind in joint.ANCHORS:
        anchor = joint.anchors(data,kind)
        base = joint.relative(data["heads"],anchor)
        selection = model["selection"][kind]["joint"]
        levels[kind] = dict(zip(lowrank.MODES,(*lowrank.LEVELS,selection["mse_level"],selection["joint_level"])))
        choices[kind],menus[kind] = {},{}
        for mode,level in levels[kind].items():
            if level is None:
                choices[kind][mode] = {arm:data["raw_choices"].copy() for arm in ARMS}
                menus[kind][mode] = {"added":np.zeros(len(anchor),dtype=int),"removed":np.zeros(len(anchor),dtype=int)}
                continue
            full = lowrank.increments(data,model["paths"][kind],level,"joint")
            cost,wait = full.copy(),full.copy()
            cost[...,1],wait[...,0] = base[...,1],base[...,0]
            predictions = {"base":base,"cost":cost,"wait":wait,"joint":full}
            choices[kind][mode] = {arm:joint.binding.policy.selected(p[...,0],p[...,1],anchor) for arm,p in predictions.items()}
            old,new = base[...,1]<=0,full[...,1]<=0
            menus[kind][mode] = {"added":((~old)&new).sum(axis=1),"removed":(old&(~new)).sum(axis=1)}
    return {"choices":choices,"menus":menus,"levels":levels,"raw_choices":data["raw_choices"]}


def reproduce_choices(prediction, source):
    differences = int(np.sum(prediction["raw_choices"]!=np.asarray([r["decisions"]["raw"]["selected"] for r in source])))
    comparisons = len(source)
    for kind in joint.ANCHORS:
        for mode in lowrank.MODES[:len(lowrank.LEVELS)]:
            for arm in ("base","wait","joint"):
                old = f"lowrank_{kind}_joint_zero" if arm=="base" else f"lowrank_{kind}_{arm}_{mode}"
                expected = np.asarray([r["decisions"][old]["selected"] for r in source])
                differences += int(np.sum(prediction["choices"][kind][mode][arm]!=expected))
                comparisons += len(source)
    if differences: raise ValueError("Frozen raw or low-rank choices changed")
    return {"comparisons":comparisons,"changed_choices":differences}


def reproduce_scores(values, prediction, source):
    predictions = [(prediction["raw_choices"],"raw")]
    for kind in joint.ANCHORS:
        for mode in lowrank.MODES[:len(lowrank.LEVELS)]:
            for arm in ("base","wait","joint"):
                old = f"lowrank_{kind}_joint_zero" if arm=="base" else f"lowrank_{kind}_{arm}_{mode}"
                predictions.append((prediction["choices"][kind][mode][arm],old))
    differences = []
    for selected,name in predictions:
        actual = values[np.arange(len(values)),selected]
        expected = np.asarray([[r["decisions"][name]["cost"],r["decisions"][name]["wait"]] for r in source])
        differences.extend(np.abs(actual-expected).ravel().tolist())
    if max(differences)>1e-12: raise ValueError("Frozen scoring panels or future indices changed")
    return {"comparisons":len(differences),"max_abs_difference":max(differences)}


def records(rows, values, prediction, source):
    output = []
    for i,r in enumerate(rows):
        entry = {k:r[k] for k in ("root","scenario","scene_seed","probe_index","original_choice")}
        entry["raw"] = source[i]["decisions"]["raw"]
        entry["factorial"] = {}
        for kind in joint.ANCHORS:
            entry["factorial"][kind] = {}
            for mode,arms in prediction["choices"][kind].items():
                decisions = {}
                for arm,selected in arms.items():
                    j = int(selected[i])
                    decisions[arm] = {"selected":j,"cost":float(values[i,j,0]),"wait":float(values[i,j,1]),
                        "changed":j!=r["original_choice"],"unsafe":bool(values[i,j,1]>0)}
                entry["factorial"][kind][mode] = {"level":prediction["levels"][kind][mode],"decisions":decisions,
                    "added":int(prediction["menus"][kind][mode]["added"][i]),"removed":int(prediction["menus"][kind][mode]["removed"][i])}
        output.append(entry)
    return output


def effects(decisions):
    vector = lambda arm:np.asarray([decisions[arm]["cost"],decisions[arm]["wait"]])
    b,c,w,j = [vector(a) for a in ARMS]
    return {"cost":c-b,"wait":w-b,"joint":j-b,"interaction":j-c-w+b,
        "cost_after_wait":j-w,"wait_after_cost":j-c}

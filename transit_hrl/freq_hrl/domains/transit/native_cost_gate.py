"""Frozen cost residuals crossed with physical and raw-logit wait menus."""

import numpy as np

from freq_hrl.domains.transit import native_lowrank_joint as lowrank

joint = lowrank.joint
ARMS = ("physical_base", "physical_cost", "raw_base", "raw_cost")


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
                choices[kind][mode] = {a:data["raw_choices"].copy() for a in ARMS}
                menus[kind][mode] = {"added":np.zeros(len(anchor),dtype=int),"removed":np.zeros(len(anchor),dtype=int)}
                continue
            cost = lowrank.increments(data,model["paths"][kind],level,"joint")[...,0]
            # Logits encode feasibility versus the original action, not additive wait.
            predictors = ((base[...,0],base[...,1]),(cost,base[...,1]),
                (base[...,0],data["raw_logits"]),(cost,data["raw_logits"]))
            choices[kind][mode] = {arm:joint.binding.policy.selected(c,w,anchor) for arm,(c,w) in zip(ARMS,predictors)}
            old,new = base[...,1]<=0,data["raw_logits"]<=0
            menus[kind][mode] = {"added":((~old)&new).sum(axis=1),"removed":(old&(~new)).sum(axis=1)}
    return {"choices":choices,"menus":menus,"levels":levels,"raw_choices":data["raw_choices"]}


def reproduce(prediction, source, values):
    controls = [(prediction["raw_choices"],None,None,None)]
    for kind in joint.ANCHORS:
        for mode in lowrank.MODES:
            controls.extend((prediction["choices"][kind][mode][new],kind,mode,old)
                for new,old in (("physical_base","base"),("physical_cost","cost")))
            controls.append((prediction["choices"][kind][mode]["raw_base"],None,None,None))
            if mode=="zero": controls.append((prediction["choices"][kind][mode]["raw_cost"],None,None,None))
    changed,maximum = 0,0.
    for selected,kind,mode,arm in controls:
        expected = [r["raw"] if kind is None else r["factorial"][kind][mode]["decisions"][arm] for r in source]
        changed += int(np.sum(selected!=np.asarray([d["selected"] for d in expected])))
        actual = values[np.arange(len(values)),selected]
        truth = np.asarray([[d["cost"],d["wait"]] for d in expected])
        maximum = max(maximum,float(np.max(np.abs(actual-truth))))
    if changed or maximum>1e-12: raise ValueError("Frozen physical controls or raw-gate identity changed")
    return {"choice_comparisons":len(source)*len(controls),"changed_choices":changed,
        "score_comparisons":len(source)*len(controls)*2,"max_abs_difference":maximum}


def records(rows, values, prediction, source):
    output = []
    for i,r in enumerate(rows):
        entry = {k:r[k] for k in ("root","scenario","scene_seed","probe_index","original_choice")}
        entry.update(raw=source[i]["raw"],gates={})
        for kind in joint.ANCHORS:
            entry["gates"][kind] = {}
            for mode,arms in prediction["choices"][kind].items():
                decisions = {}
                for arm,selected in arms.items():
                    j = int(selected[i])
                    decisions[arm] = {"selected":j,"cost":float(values[i,j,0]),"wait":float(values[i,j,1]),
                        "changed":j!=r["original_choice"],"unsafe":bool(values[i,j,1]>0)}
                entry["gates"][kind][mode] = {"level":prediction["levels"][kind][mode],"decisions":decisions,
                    "added":int(prediction["menus"][kind][mode]["added"][i]),"removed":int(prediction["menus"][kind][mode]["removed"][i])}
        output.append(entry)
    return output


def effects(decisions):
    b,c,r,j = [np.asarray([decisions[a]["cost"],decisions[a]["wait"]]) for a in ARMS]
    return {"cost_on_physical":c-b,"cost_on_raw":j-r,"gate_on_base":r-b,"gate_on_cost":j-c,
        "interaction":j-r-c+b,"total":j-b}

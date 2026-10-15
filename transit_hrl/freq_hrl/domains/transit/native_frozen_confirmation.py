"""Frozen policy contrasts: replay divergent actions, retain zero-effect states."""

import numpy as np

from freq_hrl.domains.transit.native_frozen_future import conditional_mc, state_key

VARIANTS = ("raw", "all_mse_pooled_safe", "pair_gls_pooled_safe", "state_gls_pooled_safe")
PRIMARY = VARIANTS[-1]
FIELDS = ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")


def action_union(cells):
    choices, active = {}, set()
    for cell in cells:
        for block in cell["blocks"]:
            n = len(block["states"])
            if set(block["choices"]) != set(VARIANTS) or any(len(x) != n for x in block["choices"].values()):
                raise ValueError("Changed frozen action coverage")
            for i, row in enumerate(block["states"]):
                key = state_key(row)
                cs = {block["choices"][v][i] for v in VARIANTS}
                if not cs <= set(range(6)): raise ValueError("Invalid frozen candidate")
                choices.setdefault(key, set()).update(cs)
                if len(cs) > 1: active.add(key)
    return {key: sorted(choices[key]) for key in sorted(active)}


def sparse_panels(cells, draws, load):
    union = action_union(cells)
    keys = {state_key(r) for c in cells for b in c["blocks"] for r in b["states"]}
    panels = {k: {"values": np.zeros((len(draws), 6, 4))} for k in keys}
    for key, candidates in union.items():
        for j, draw in enumerate(draws):
            labels = {r["candidate"]: r for r in load(key, draw)}
            if not set(candidates) <= labels.keys(): raise ValueError("Missing frozen-action outcome")
            for candidate in candidates:
                values = [labels[candidate][f] for f in FIELDS]
                if not np.all(np.isfinite(values)): raise ValueError("Nonfinite replay label")
                panels[key]["values"][j, candidate] = values
    # Unsampled states only occur in exact same-action contrasts, not absolute outcomes.
    return panels


def summarize(cells, panels, *, baseline="raw"):
    paired = [{"blocks": [{"states": b["states"],
        "choices": {"raw": b["choices"][baseline], **{v: b["choices"][v] for v in VARIANTS if v != "raw"}}}
        for b in c["blocks"]]} for c in cells]
    result = conditional_mc(paired, panels, variants=VARIANTS)
    result["baseline"] = baseline
    if baseline != "raw": del result["variants"]["raw"]
    for variant in result["variants"].values():
        for field in variant["fields"].values():
            mean, se = field["mean"], field["conditional_mc_se"]
            field["conditional_mc_normal_interval95"] = [mean-1.96*se, mean+1.96*se]
    return result


def subset(cells, root=None, scenario=None):
    output = []
    for cell in cells:
        blocks = []
        for block in cell["blocks"]:
            indices = [i for i, r in enumerate(block["states"]) if (root is None or r["root"] == root)
                and (scenario is None or r["scenario"] == scenario)]
            if indices: blocks.append({"states": [block["states"][i] for i in indices],
                "choices": {v: [js[i] for i in indices] for v, js in block["choices"].items()}})
        if blocks: output.append({"key": cell["key"], "blocks": blocks})
    return output


def confirm(primary, halves):
    return {f: primary["fields"][f]["conditional_mc_normal_interval95"][1] < 0
        and all(h["variants"][PRIMARY]["fields"][f]["mean"] < 0 for h in halves)
        for f in ("cost", "wait_min")}

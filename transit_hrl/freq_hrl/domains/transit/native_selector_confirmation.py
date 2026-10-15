"""Paired confirmation and prefix attribution for frozen OOF selectors."""

import numpy as np

from freq_hrl.domains.transit import native_frozen_confirmation as frozen
from freq_hrl.domains.transit.native_selector_information import MODELS, VARIANTS


def contrast(cells, panels, target, baseline):
    return frozen.summarize(cells, panels, baseline=baseline, variants=("raw", target))["variants"][target]


def summarize(cells, panels):
    return {"versus_raw": frozen.summarize(cells, panels, variants=VARIANTS),
        "primary": {m: contrast(cells, panels, f"{m}_expanded8_pooled_safe", f"{m}_matched2_pooled_safe") for m in MODELS},
        "secondary": {f"{m}_expanded8-{s}": contrast(cells, panels, f"{m}_expanded8_pooled_safe", baseline)
            for m in MODELS for s, baseline in (("old_A2", f"{m}_old_A2_pooled_safe"), ("Stage247", "old_pooled_safe"))}}


def confirm(banks):
    """Require the paired improvement and raw utility, not a weak-control win."""
    families = {}
    for m in MODELS:
        checks = {}
        for comparison in ("primary", "versus_raw"):
            def stats(b, ds):
                x = banks[b][ds][comparison]
                return x[m] if comparison == "primary" else x["variants"][f"{m}_expanded8_pooled_safe"]
            checks[comparison] = {f: stats("pooled", "fresh64")["fields"][f]["conditional_mc_normal_interval95"][1] < 0
                and all(stats("pooled", h)["fields"][f]["mean"] < 0 for h in ("half0_32", "half1_32"))
                and all(stats(b, "fresh64")["fields"][f]["mean"] < 0 for b in ("bank0", "bank1"))
                for f in ("cost", "wait_min")}
        families[m] = checks
    return {"families": families, "joint_pass": all(ok for checks in families.values() for fields in checks.values() for ok in fields.values()),
        "rule": "all_families_cost_and_wait_paired_and_vs_raw_upper95_negative_both_halves_and_banks_negative; conjunction_not_best_arm_selection"}


def attribution(cells, panels, target, baseline):
    """Count shared state/draw events once, after combining exposure weights."""
    effects, weights = {}, {}
    for cell in cells:
        for block in cell["blocks"]:
            w = 1./(len(cells)*len(cell["blocks"])*len(block["states"]))
            for row, chosen, raw in zip(block["states"], block["choices"][target], block["choices"][baseline]):
                key = frozen.state_key(row)
                values = panels[key]["values"]
                effects.setdefault(key, np.zeros((len(values), 4)))
                weights[key] = weights.get(key, 0.)+w
                effects[key] += w*(values[:, chosen]-values[:, raw])
    prefixes = []
    for prefix in sorted({k[:2] for k in effects}):
        keys = [k for k in effects if k[:2] == prefix]
        values = np.stack([effects[k] for k in keys])
        means, weight = values.mean(axis=1).sum(axis=0), sum(weights[k] for k in keys)
        prefixes.append({"prefix": list(prefix), "weight": weight,
            "contribution": dict(zip(("cost", "fleet", "nonfleet", "wait_min"), map(float, means))),
            "fleet_events_beneficial": int((values[..., 1] < 0).sum()),
            "fleet_events_harmful": int((values[..., 1] > 0).sum())})
    concentration = {}
    for f in ("cost", "wait_min"):
        xs = [p["contribution"][f] for p in prefixes]
        benefit = [max(-x, 0.) for x in xs]
        concentration[f] = {"improving_prefixes": sum(x < 0 for x in xs), "worsening_prefixes": sum(x > 0 for x in xs),
            "top1_benefit_share": max(benefit)/sum(benefit) if sum(benefit) else None}
    return {"target": target, "baseline": baseline, "prefixes": prefixes, "concentration": concentration,
        "event_unit": "unique_state_draw_net_weighted_fleet_contrast_not_model_exposures"}

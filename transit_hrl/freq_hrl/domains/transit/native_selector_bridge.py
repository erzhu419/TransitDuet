"""Separate validation optimism, correction transfer and reference-action shift."""

import numpy as np

FIELDS = ("cost", "fleet", "nonfleet", "wait_min")
STRATA = ("all", "raw_same", "raw_different", "gate_same", "gate_different")


def compare(validation, evaluation, inner, full, inner_raw, full_raw, inner_gate, full_gate):
    validation, evaluation = np.asarray(validation).mean(axis=1), np.asarray(evaluation).mean(axis=1)
    inner, full, inner_raw, full_raw = (np.asarray(x) for x in (inner, full, inner_raw, full_raw))
    ix = np.arange(len(inner))
    val = validation[ix, inner]-validation[ix, inner_raw]
    oof = evaluation[ix, inner]-evaluation[ix, inner_raw]
    deployed = evaluation[ix, full]-evaluation[ix, full_raw]
    reference = evaluation[ix, full_raw]-evaluation[ix, inner_raw]
    selected = evaluation[ix, full]-evaluation[ix, inner]
    values = {"validation": val, "oof_future": oof, "full_future": deployed,
        "optimism": oof-val, "correction_transfer": deployed-oof,
        "raw_reference": reference, "selected_action_transfer": selected}
    residual = float(np.max(np.abs(selected-(deployed-oof+reference))))
    if residual > 1e-12: raise ValueError("Changed bridge decomposition")
    raw_same = inner_raw == full_raw
    gate_same = np.all(inner_gate == full_gate, axis=1)
    rates = {"raw_action_disagreement": ~raw_same, "feasible_mask_disagreement": ~gate_same,
        "selected_action_disagreement": inner != full,
        "inner_changed": inner != inner_raw, "full_changed": full != full_raw,
        "inner_choice_infeasible_under_full": ~full_gate[ix, inner],
        "full_choice_infeasible_under_inner": ~inner_gate[ix, full],
        "feasible_action_disagreement": (inner_gate != full_gate).mean(axis=1)}
    masks = dict(zip(STRATA, (np.ones(len(inner), dtype=bool), raw_same, ~raw_same, gate_same, ~gate_same)))
    return {"decomposition_max_abs_difference": residual, "strata": {s: {"cases": int(mask.sum()),
        "sums": {k: dict(zip(FIELDS, map(float, v[mask].sum(axis=0)))) for k, v in values.items()},
        "counts": {k: float(v[mask].sum()) for k, v in rates.items()}} for s, mask in masks.items()}}


def combine(blocks):
    result = {}
    for s in STRATA:
        xs = [b["strata"][s] for b in blocks]
        n = sum(x["cases"] for x in xs)
        result[s] = {"cases": n,
            "means": {k: {f: sum(x["sums"][k][f] for x in xs)/n if n else None for f in FIELDS} for k in xs[0]["sums"]},
            "rates": {k: sum(x["counts"][k] for x in xs)/n if n else None for k in xs[0]["counts"]}}
    return result

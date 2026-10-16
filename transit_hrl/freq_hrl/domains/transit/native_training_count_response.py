"""Separate nested training-future count from the remaining prefix-fit gap."""

from freq_hrl.domains.transit import native_old_key_response as response

FIELDS, PHYSICAL = response.FIELDS, response.PHYSICAL
PATHS = ("inner8", "inner16", "full_seen")
PAIRS = {"count16_minus8": ("inner16", "inner8"), "full_minus16": ("full_seen", "inner16"),
    "full_minus8": ("full_seen", "inner8")}


def contrasts(paths):
    effects = {name: {f: paths[a]["all"][f]-paths[b]["all"][f] for f in FIELDS} for name, (a,b) in PAIRS.items()}
    residual = max(abs(effects["count16_minus8"][f]+effects["full_minus16"][f]-effects["full_minus8"][f]) for f in FIELDS)
    if residual > 1e-12: raise ValueError("Training-count decomposition does not telescope")
    return effects


def summarize(pieces, denominator):
    if not pieces: return {"bank_cell_exposures": 0, "conditional": None,
        "full_denominator_count_effect": dict.fromkeys(PHYSICAL, 0.)}
    n = len(pieces)
    mean = lambda xs: {f: sum(x[f] for x in xs)/len(xs) for f in xs[0]}
    paths = {p: mean([{f: x["paths"][p]["all"][f] for f in FIELDS} for x in pieces]) for p in PATHS}
    effects = {c: mean([{f: x["effects"][c][f] for f in PHYSICAL} for x in pieces]) for c in PAIRS}
    return {"bank_cell_exposures": n, "conditional": {"path_means": paths, "contrasts": effects,
        "opposite_halves": [{p: mean([{f: x["evaluation_halves"][h][p]["all"][f] for f in PHYSICAL} for x in pieces])
            for p in PATHS} for h in (0,1)],
        "training_half_count_effects": [mean([{f: x["paths"]["inner16"]["all"][f]-x["training_halves"][h]["all"][f]
            for f in PHYSICAL} for x in pieces]) for h in (0,1)],
        "action_disagreement": mean([x["action_disagreement"] for x in pieces]),
        "absolute_full_response_gap": {p: {f: sum(abs(x["paths"]["full_seen"]["all"][f]-x["paths"][p]["all"][f]) for x in pieces)/n
            for f in PHYSICAL} for p in PATHS[:2]},
        "full_held_anchor": mean([x["full_held_anchor"] for x in pieces])},
        "full_denominator_count_effect": {f: effects["count16_minus8"][f]*n/denominator for f in PHYSICAL}}

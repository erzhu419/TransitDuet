"""Matched versus expanded independent futures with frozen critic structure."""

from functools import partial

import numpy as np

from freq_hrl.domains.transit import native_balanced_critic as previous

ARMS = ("old", "matched", "expanded")
VARIANTS = ("raw", *(f"{a}_{f}" for a in ARMS for f in previous.FILTERS))
aggregate = partial(previous.previous.previous.aggregate, variants=VARIANTS)


def draw_assignment(bank, arm, half=None):
    if bank not in (0, 1) or arm not in ("matched", "expanded") or half not in (None, 0, 1):
        raise ValueError("Invalid training-information assignment")
    blocks = [list(range(16*bank+8*h, 16*bank+8*(h+1))) for h in (0, 1)]
    selected = blocks if half is None else [blocks[half]]
    training = [d for block in selected for d in (block[:2] if arm == "matched" else block)]
    evaluation = list(range(16*(1-bank), 16*(2-bank)))
    return training, evaluation


def select(blocks):
    # Keep the existing per-arm OOF selector; never select between information arms.
    aligned = [{**b, "curves": {"all_mse": b["curves"]["old"], **b["curves"]}} for b in blocks]
    return previous.previous.previous.select(aligned, objectives=ARMS)


def predict(paths, selection):
    return {"raw": paths["old"]["zero"], **{f"{arm}_{f}": paths[arm][selection[arm][f] or "zero"]
        for arm in ARMS for f in previous.FILTERS}}


def information_error(train_draws, evaluation_draws, data, predictions):
    """Unshrunk target and predictor errors under independent held-out innovations."""
    anchor = data["originals"]
    ix = np.arange(len(anchor))
    def relative(draws):
        return draws-draws[ix, :, anchor][:, :, None]
    train_paired, evaluation_paired = relative(train_draws), relative(evaluation_draws)
    target, truth = train_paired.mean(axis=1), evaluation_paired.mean(axis=1)
    eligible = data["raw_logits"] <= 0
    lo, hi = previous.previous.LO, previous.previous.HI
    feasible = eligible[:, lo] & eligible[:, hi]
    pair_weights = feasible/np.maximum(feasible.sum(axis=1, keepdims=True), 1)
    nonanchor = np.arange(6)[None, :] != anchor[:, None]
    def errors(predicted, expected=truth):
        result = {}
        for name, column in (("cost", 0), ("fleet", 1), ("service", 2)):
            a, b = predicted[..., column], expected[..., column]
            margin_a, margin_b = a[:, hi]-a[:, lo], b[:, hi]-b[:, lo]
            result[name] = {"nonanchor_mse": float(((a-b)**2)[nonanchor].mean()),
                "feasible_pair_mse": float((pair_weights*(margin_a-margin_b)**2).sum(axis=1).mean()),
                "feasible_pair_sign_error": float((pair_weights*(np.sign(margin_a) != np.sign(margin_b))).sum(axis=1).mean())}
        return result
    predicted = {name: np.stack((p["cost"], p["components"][..., 1], p["components"][..., 0],
        np.zeros_like(p["cost"])), axis=-1) for name, p in predictions.items()}
    noise = {}
    for name, column in (("cost", 0), ("fleet", 1), ("service", 2)):
        v = evaluation_paired[..., column]
        variance = v.var(axis=1, ddof=1)/v.shape[1]
        pair_variance = (v[:, :, hi]-v[:, :, lo]).var(axis=1, ddof=1)/v.shape[1]
        noise[name] = {"nonanchor_mse": float(variance[nonanchor].mean()),
            "feasible_pair_mse": float((pair_weights*pair_variance).sum(axis=1).mean())}
    return {"training_draws": train_draws.shape[1], "evaluation_draws": evaluation_draws.shape[1],
        "target_mean_error": errors(target), "predictor_error": {name: errors(p) for name, p in predicted.items()},
        "predictor_error_on_training_bank": {name: errors(p, target) for name, p in predicted.items()},
        "zero_prediction_error": errors(np.zeros_like(truth)), "evaluation_mc_noise": noise,
        "training_fleet_events": int(np.count_nonzero(train_draws[..., 1])),
        "evaluation_fleet_events": int(np.count_nonzero(evaluation_draws[..., 1]))}

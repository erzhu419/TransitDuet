"""Seven-bin prefix-only softmax and discrete closed-loop forecast ensembles."""

import numpy as np

from freqduet_closed_loop_forecast import ACTIONS, fluid_rollout, holding_features


METRICS = ("waiting_person_seconds", "onboard_person_seconds", "backlog_trip_seconds",
           "holding_vehicle_seconds")
DRAWS = 64


def fit_distribution(rows, now):
    import torch

    past = [row for row in rows if row[0] < now]
    labels = np.array([int(np.argmin(abs(ACTIONS - row[2] * 45))) for row in past])
    if any(abs(ACTIONS[label] - row[2] * 45) > 1e-5 for row, label in zip(past, labels)):
        raise ValueError("observed holding label is outside the seven deployed bins")
    counts = np.bincount(labels, minlength=len(ACTIONS))
    prior = (counts + .5) / (len(past) + .5 * len(ACTIONS))
    result = dict(samples=len(past), latest_time_s=max((row[0] for row in past), default=None),
                  prior=prior.tolist(), coef=None, scale=None)
    if len(past) < 32:
        return result
    design = np.stack([row[1] for row in past])
    scale = np.maximum(np.sqrt(np.mean(design * design, axis=0)), 1e-8)
    x = torch.as_tensor(design / scale, dtype=torch.float64)
    center = x.mean(dim=0)
    y = torch.as_tensor(labels, dtype=torch.long)
    weights = torch.zeros((x.shape[1], len(ACTIONS)), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        weights[0] = torch.as_tensor(np.log(prior))
    optimizer = torch.optim.LBFGS([weights], max_iter=100, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        with torch.enable_grad():
            logits = x @ weights
            # Unweighted labels preserve the first moment; half a pseudo-observation
            # per bin keeps finite probabilities without inventing balanced classes.
            loss = (torch.nn.functional.cross_entropy(logits, y, reduction="sum")
                    + 5 * weights[1:].square().sum()
                    - .5 * torch.log_softmax(center @ weights, dim=0).sum())
            loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    result.update(coef=weights.detach().numpy().tolist(), scale=scale.tolist(),
                  gradient_max=float(weights.grad.abs().max()))
    return result


def probabilities_from_features(model, features):
    if model["coef"] is None:
        return np.asarray(model["prior"])
    logits = np.asarray(features) / np.asarray(model["scale"]) @ np.asarray(model["coef"])
    mass = np.exp(logits - np.max(logits))
    return mass / mass.sum()


def probabilities(model, values):
    return probabilities_from_features(model, holding_features(values))


def sample_action(probs, uniform):
    return float(ACTIONS[min(np.searchsorted(np.cumsum(probs), uniform, side="right"), len(ACTIONS) - 1)])


def ensemble(observed, horizon, model, draws=DRAWS):
    metrics = {metric: [] for metric in METRICS}
    returns = {str(vehicle["trip_id"]): [] for vehicle in observed["vehicles"] if vehicle["on_route"]}
    for draw in range(draws):
        # Never consume the simulator/actor RNG or use demand seeds as features.
        rng = np.random.default_rng(20261015 + draw)
        forecast = fluid_rollout(observed, horizon, "response",
                                 lambda values: sample_action(probabilities(model, values), rng.random()))
        for metric in metrics:
            metrics[metric].append(forecast[metric])
        times = {str(row["trip_id"]): row["time_s"] for row in forecast["returns"]}
        for trip in returns:
            returns[trip].append(times.get(trip))
    return dict(mean={metric: float(np.mean(values)) for metric, values in metrics.items()},
                draws=metrics, returns=returns)

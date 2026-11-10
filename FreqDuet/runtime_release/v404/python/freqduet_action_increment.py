"""Day-balanced ridge models of signed full-day holding-action differences."""

from dataclasses import dataclass

import numpy as np


ACTIONS = (0, 5, 10, 15, 20, 30, 45)
METRICS = ("passenger_person_s", "cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete")
KINDS = ("action_only", "state_quadratic")
RIDGE = 1.0


def observable_features(capture):
    return np.concatenate((np.asarray(capture["encoded_state"], dtype=float),
                           [capture["prefix_cv_squared"]],
                           np.asarray(capture["actor_probabilities"], dtype=float)))


def day_weights(days):
    unique, inverse, counts = np.unique(days, return_inverse=True, return_counts=True)
    return 1. / (len(unique) * counts[inverse])


def mean_differences(outcomes, native):
    reference = np.asarray([row[:, n, :] for row, n in zip(outcomes, native)])
    return (outcomes - reference[:, :, None, :]).mean(axis=1)


def design(states, native, center, scale, conditioned):
    features = np.column_stack((np.ones(len(states)), (states - center) / scale)) if conditioned else np.ones((len(states), 1))
    action = np.asarray(ACTIONS, dtype=float) / max(ACTIONS)
    reference = action[native, None]
    linear, quadratic = action[None, :] - reference, action[None, :] ** 2 - reference ** 2
    return np.concatenate((linear[:, :, None] * features[:, None, :],
                           quadratic[:, :, None] * features[:, None, :]), axis=2)


@dataclass
class IncrementModel:
    kind: str
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    effective_parameters_per_output: float

    def predict(self, states, native):
        return design(states, native, self.center, self.scale, self.kind == "state_quadratic") @ self.coefficients


def fit_model(states, native, targets, days, kind):
    if kind not in KINDS:
        raise ValueError("unknown fixed V72 model")
    weights = day_weights(days)
    center = (weights[:, None] * states).sum(axis=0)
    scale = np.sqrt((weights[:, None] * (states - center) ** 2).sum(axis=0))
    scale[scale < 1e-6] = 1.
    x = design(states, native, center, scale, kind == "state_quadratic")
    mask = np.arange(len(ACTIONS))[None, :] != native[:, None]
    row_weights = np.broadcast_to(weights[:, None] / (len(ACTIONS) - 1), mask.shape)[mask]
    x, y = x[mask], targets[mask]
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ (row_weights[:, None] * x)
        regularized = gram + RIDGE * np.eye(gram.shape[0])
        solved = np.linalg.solve(regularized, np.column_stack((x.T @ (row_weights[:, None] * y), gram)))
        coefficients = solved[:, :y.shape[1]]
    else:
        # Full fleet/plan inputs have more columns than training contrasts.
        # The dual solve is the same ridge objective without a huge P x P matrix.
        weighted = np.sqrt(row_weights)[:, None] * x
        gram = weighted @ weighted.T
        regularized = gram + RIDGE * np.eye(gram.shape[0])
        solved = np.linalg.solve(regularized, np.column_stack((np.sqrt(row_weights)[:, None] * y, gram)))
        coefficients = weighted.T @ solved[:, :y.shape[1]]
    return IncrementModel(kind, center, scale, coefficients, float(np.trace(solved[:, y.shape[1]:])))


def fit_excluding_day(states, native, targets, days, held_day, kind):
    training = days != held_day
    return fit_model(states[training], native[training], targets[training], days[training], kind)


def choose_action(predictions, native):
    feasible = (predictions[:, 1:] <= 0).all(axis=1)
    costs = np.where(feasible, predictions[:, 0], np.inf)
    best = int(np.argmin(costs))
    return best if costs[best] < 0 else int(native)


def error_statistics(predicted, truth, native):
    nonnative = np.arange(len(ACTIONS)) != native
    difference = predicted[nonnative] - truth[nonnative]
    nonzero = truth[nonnative] != 0
    pair = np.triu_indices(len(ACTIONS), 1)
    true_order = truth[pair[0]] - truth[pair[1]]
    predicted_order = predicted[pair[0]] - predicted[pair[1]]
    informative = true_order != 0
    return dict(mae=float(np.abs(difference).mean()), mse=float(np.square(difference).mean()),
                sign_correct=int(((np.sign(predicted[nonnative]) == np.sign(truth[nonnative])) & nonzero).sum()),
                sign_pairs=int(nonzero.sum()),
                order_correct=int(((np.sign(predicted_order) == np.sign(true_order)) & informative).sum()),
                order_pairs=int(informative.sum()))

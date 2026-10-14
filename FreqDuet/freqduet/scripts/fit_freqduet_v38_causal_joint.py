#!/usr/bin/env python3
"""Grouped development replay of an observable two-stage lower scorer."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_freqduet_v37_joint_lower import ACTIONS, PRIMARY, assert_same_state, pilot_items, shard_name


RIDGE = 10.0
STATE_FIELDS = ("clock_hours", "forward_gap_s", "follower_gap_s", "follower_valid",
                "load_ratio", "ready_return", "overdue_return_count",
                "return_margin_proxy_s", "past_mean_hold_s")


def state_vector(observed):
    target = max(observed["target_headway_s"], 1.0)
    valid = observed["follower_valid"]
    return np.array([observed["clock_hours"] / 14.0, observed["forward_gap_s"] / target - 1,
                     valid * (observed["follower_gap_s"] / target - 1), valid,
                     observed["load_ratio"], observed["ready_return"] / 12.0,
                     observed["overdue_return_count"] / 12.0,
                     observed["return_margin_proxy_s"] / 3600.0,
                     observed["past_mean_hold_s"] / 45.0])


def first_basis(observed, action):
    state, a = state_vector(observed), action / 45.0
    return np.r_[1.0, state, a, a * a, a * state, a * a * state]


def second_basis(first_observed, first_action, observed, action):
    a, incumbent = action / 45.0, observed["actor_hold_s"] / 45.0
    state = np.r_[1.0, state_vector(first_observed), state_vector(observed), first_action / 45.0]
    # Exact zero at the incumbent, conditional on this actual first-action prefix.
    return np.r_[(a - incumbent) * state, (a * a - incumbent * incumbent) * state]


class Ridge:
    def fit(self, design, outcomes, intercept=False):
        design, outcomes = np.asarray(design), np.asarray(outcomes)
        self.scale = np.maximum(np.sqrt(np.mean(design * design, axis=0)), 1e-8)
        design = design / self.scale
        penalty = np.eye(design.shape[1]) * RIDGE
        if intercept:
            penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ outcomes)
        return self

    def predict(self, design):
        return np.asarray(design) / self.scale @ self.coef


def feasible_choice(predictions, incumbent):
    feasible = np.flatnonzero((predictions[:, 0] < -1e-12)
                              & (predictions[:, 1:] <= 1e-12).all(axis=1))
    if not len(feasible):
        return ACTIONS.index(int(incumbent)), False
    return min(feasible, key=lambda i: (predictions[i, 0], abs(ACTIONS[i] - incumbent))), True


class CausalJointScorer:
    def __init__(self, first_model, second_model):
        self.first_model, self.second_model = first_model, second_model

    def choose_first(self, observed):
        predictions = self.first_model.predict([first_basis(observed, a) for a in ACTIONS])
        index, active = feasible_choice(predictions, observed["actor_hold_s"])
        return ACTIONS[index], active, predictions[index]

    def choose_second(self, first_observed, first_action, observed):
        predictions = self.second_model.predict([
            second_basis(first_observed, first_action, observed, a) for a in ACTIONS])
        index, _ = feasible_choice(predictions, observed["actor_hold_s"])
        return ACTIONS[index], predictions[index]


def pair_row(context, first, second):
    return context["pairs"].loc[(context["pairs"].first_action_s == first)
                                 & (context["pairs"].second_action_s == second)].iloc[0]


def endpoints(row):
    return row[list(PRIMARY)].to_numpy(dtype=float)


def load_contexts(root, v37):
    contexts = {}
    for seed, scenario, target, action in pilot_items():
        key = (seed, scenario, target)
        name = shard_name(seed, scenario, target, action)
        snapshot = json.loads((root / "observed" / name / "snapshot.json").read_text())
        if not snapshot["matches_v37"]:
            raise ValueError("snapshot lacks a matched V37 rollout")
        first, second = (snapshot["decisions"][stage]["features"] for stage in ("first", "second"))
        if key not in contexts:
            contexts[key] = dict(key=key, first=first, second={}, frames=[])
        context = contexts[key]
        assert_same_state(context["first"], first, "first features cannot depend on future intervention")
        context["second"][action] = second
        context["frames"].append(pd.read_csv(v37 / "records" / f"{name}.csv"))
    for context in contexts.values():
        context["pairs"] = pd.concat(context.pop("frames"), ignore_index=True)
        first_actor = int(context["first"]["actor_hold_s"])
        second_actor = int(context["second"][first_actor]["actor_hold_s"])
        context["reference"] = endpoints(pair_row(context, first_actor, second_actor))
    return list(contexts.values())


def fit_second(contexts):
    design, labels = [], []
    for context in contexts:
        for first_action in ACTIONS:
            observed = context["second"][first_action]
            reference = endpoints(pair_row(context, first_action, observed["actor_hold_s"]))
            for second_action in ACTIONS:
                design.append(second_basis(context["first"], first_action, observed, second_action))
                labels.append(endpoints(pair_row(context, first_action, second_action)) - reference)
    return Ridge().fit(design, labels)


def fit_joint(contexts):
    design, labels = [], []
    # Inner context exclusion prevents in-sample oracle continuation targets.
    for held, context in enumerate(contexts):
        continuation = CausalJointScorer(None, fit_second(contexts[:held] + contexts[held + 1:]))
        for first_action in ACTIONS:
            second, _ = continuation.choose_second(context["first"], first_action, context["second"][first_action])
            design.append(first_basis(context["first"], first_action))
            labels.append(endpoints(pair_row(context, first_action, second)) - context["reference"])
    return CausalJointScorer(Ridge().fit(design, labels, intercept=True), fit_second(contexts))


def evaluate_context(scorer, context):
    # Neither second snapshots nor rollout labels enter first-stage selection.
    first, active, predicted_first = scorer.choose_first(context["first"])
    original_first = int(context["first"]["actor_hold_s"])
    observed_second = context["second"][first]
    second, predicted_second = scorer.choose_second(context["first"], first, observed_second)
    if not active:
        second = int(observed_second["actor_hold_s"])
    second_only, _ = scorer.choose_second(context["first"], original_first, context["second"][original_first])
    choices = dict(causal_joint=(first, second), first_only=(first, int(observed_second["actor_hold_s"])),
                   second_only=(original_first, second_only))
    rows = []
    for policy, (a1, a2) in choices.items():
        actual = endpoints(pair_row(context, a1, a2)) - context["reference"]
        row = dict(train_seed=context["key"][0], eval_seed=context["key"][1], target_time_s=context["key"][2],
                   policy=policy, first_action_s=a1, second_action_s=a2,
                   first_stage_active=bool(active) if policy != "second_only" else False,
                   actual_joint_improvement=bool((actual <= 1e-12).all() and actual[0] < -1e-12),
                   actual_constraint_violation=bool((actual[1:] > 1e-12).any()))
        for index, metric in enumerate(PRIMARY):
            row[f"delta_{metric}"] = actual[index]
            if policy == "causal_joint":
                row[f"predicted_first_{metric}"] = predicted_first[index]
                row[f"predicted_second_increment_{metric}"] = predicted_second[index] if active else 0.0
        rows.append(row)
    return rows


def develop(root, v37):
    contexts = load_contexts(root, v37)
    rows = []
    models = root / "scorer_models"
    models.mkdir(parents=True, exist_ok=True)
    for held_seed in sorted({context["key"][0] for context in contexts}):
        train = [context for context in contexts if context["key"][0] != held_seed]
        test = [context for context in contexts if context["key"][0] == held_seed]
        scorer = fit_joint(train)
        np.savez(models / f"held_seed{held_seed}.npz", first_coef=scorer.first_model.coef,
                 first_scale=scorer.first_model.scale, second_coef=scorer.second_model.coef,
                 second_scale=scorer.second_model.scale)
        for context in test:
            rows.extend(evaluate_context(scorer, context))
    frame = pd.DataFrame(rows)
    summary = root / "scorer_summary"
    summary.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary / "selections.csv", index=False)
    columns = [name for name in frame if name.startswith("delta_")] + [
        "actual_joint_improvement", "actual_constraint_violation", "first_stage_active"]
    means = frame.groupby("policy")[columns].mean()
    means.to_csv(summary / "means.csv")
    frame.groupby(["train_seed", "policy"])[columns].mean().to_csv(summary / "fold_means.csv")
    status = dict(status="complete_grouped_development_replay", contexts=len(contexts), pairs=len(contexts) * 49,
                  outer_split="leave one training seed out", inner_split="leave one context out for continuation labels",
                  independent_confirmation=False, deployable_causal_inputs=True, promoted_to_main=False,
                  source="V37 development counterfactual bank and matched V38 observable snapshots",
                  ridge=RIDGE, features=[*STATE_FIELDS, "target_headway_s", "actor_hold_s"], endpoints=list(PRIMARY),
                  limitations="Two training seeds; correlated contexts; predicted constraints are not guarantees; no fixed comparison.")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(means.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path, required=True)
    args = parser.parse_args()
    develop(args.out_root, args.v37_root)

"""Test one-time actor Adam reset under a common episode-KL budget."""

from . import pointmaze_episode_credit as episodes
from . import pointmaze_episode_kl as bounded
from scripts import pointmaze_adam_initialization_stage49_spec as spec


def optimizer_steps(model):
    return [int(state["step"].item()) for state in model.lower_actor_optimizer.state.values()]


def update(model, batch, task_rewards, treatment, *, root, iteration):
    if treatment not in spec.TREATMENTS:
        raise ValueError("unregistered Adam initialization treatment")
    source = optimizer_steps(model)
    reset = iteration == 1 and treatment in spec.FRESH_TREATMENTS
    if reset:
        model.lower_actor_optimizer.state.clear()
    starting = optimizer_steps(model)
    result = bounded.update(model, batch, task_rewards, treatment, root=root, iteration=iteration, specification=spec)
    result.update(credit_estimator=spec.CREDITS[treatment], optimizer_initialization={
        "reset": reset, "source_steps": source, "starting_steps": starting,
        "ending_steps": optimizer_steps(model)})
    return result


def worker_rollout(job):
    return episodes.worker_rollout(job, specification=spec)


def train(root, *, preflight, output):
    return episodes.train(root, preflight=preflight, output=output, specification=spec,
                          rollout_worker=worker_rollout, update_fn=update)


def aggregate(results, *, preflight):
    summary = bounded.aggregate(results, preflight=preflight, specification=spec)
    cells = {r["root"]: r for r in results}
    for root_row in summary["root_rows"]:
        histories = cells[root_row["root"]]["training"][spec.METHODS[0]]
        source = histories["gae"][0]["optimizer_initialization"]["source_steps"]
        if not source or any(step <= 0 for step in source):
            raise ValueError("Adam initialization source is not warmed")
        for treatment, history in histories.items():
            previous = None
            for row in history:
                init = row["optimizer_initialization"]
                first, fresh = row["iteration"] == 1, treatment in spec.FRESH_TREATMENTS
                reset = first and fresh
                expected_start = [] if reset else init["source_steps"]
                start_steps = init["starting_steps"] or [0] * len(source)
                expected_end = [step + row["retained_actor_steps"] for step in start_steps]
                if (init["reset"] != reset or init["starting_steps"] != expected_start
                        or (first and init["source_steps"] != source)
                        or (not first and init["source_steps"] != previous)
                        or init["ending_steps"] != (expected_end if row["retained_actor_steps"] else init["starting_steps"])
                        or row["credit_estimator"] != spec.CREDITS[treatment]):
                    raise ValueError("Adam initialization or continuity changed")
                previous = init["ending_steps"]
            root_row["kl_diagnostics"][treatment]["initialization"] = history[0]["optimizer_initialization"]
            root_row["kl_diagnostics"][treatment]["final_optimizer_steps"] = previous
    return summary

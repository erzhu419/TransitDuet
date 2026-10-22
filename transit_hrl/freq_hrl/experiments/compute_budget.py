"""CPU-only accounting and validation-only selection under lifecycle budgets."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.metadata import version
import math
import platform
from pathlib import Path
import time


_ACTIVE = ContextVar("freq_hrl_compute_ledger", default=None)


@dataclass
class ComputeLedger:
    phases: dict = field(default_factory=dict)
    _stack: list = field(default_factory=list)

    @contextmanager
    def phase(self, name):
        path = "/".join([frame["name"] for frame in self._stack] + [name])
        frame = {"name": name, "wall": time.perf_counter(), "cpu": time.process_time(),
                 "child_wall": 0.0, "child_cpu": 0.0}
        row = self.phases.setdefault(path, {"calls": 0, "failures": 0,
            "cpu_seconds": 0.0, "wall_seconds": 0.0, "counts": {}})
        self._stack.append(frame)
        try:
            yield
        except BaseException:
            row["failures"] += 1
            raise
        finally:
            wall, cpu = time.perf_counter() - frame["wall"], time.process_time() - frame["cpu"]
            self._stack.pop()
            row["calls"] += 1
            row["cpu_seconds"] += max(0.0, cpu - frame["child_cpu"])
            row["wall_seconds"] += max(0.0, wall - frame["child_wall"])
            if self._stack:
                self._stack[-1]["child_cpu"] += cpu
                self._stack[-1]["child_wall"] += wall

    @contextmanager
    def recording(self, name="offline"):
        token = _ACTIVE.set(self)
        try:
            with self.phase(name):
                yield self
        finally:
            _ACTIVE.reset(token)

    def count(self, name, amount=1):
        path = "/".join(frame["name"] for frame in self._stack)
        counts = self.phases[path]["counts"]
        counts[name] = counts.get(name, 0) + int(amount)

    def summary(self):
        counts = {}
        for row in self.phases.values():
            for name, value in row["counts"].items():
                counts[name] = counts.get(name, 0) + value
        return {"timing_basis": "exclusive_nested_process_cpu_and_elapsed_wall",
                "cpu_seconds": sum(row["cpu_seconds"] for row in self.phases.values()),
                "wall_seconds": sum(row["wall_seconds"] for row in self.phases.values()),
                "counts": counts, "phases": self.phases}


def active_ledger():
    return _ACTIVE.get()


def count_work(name, amount=1):
    ledger = active_ledger()
    if ledger is not None:
        ledger.count(name, amount)


@contextmanager
def measured_phase(name):
    ledger = active_ledger()
    if ledger is None:
        yield
    else:
        with ledger.phase(name):
            yield


@contextmanager
def single_thread_runtime():
    import numpy as np
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            cpu_model = next(line.split(":", 1)[1].strip() for line in
                Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
            yield {"cpu_model": cpu_model, "machine": platform.machine(),
                   "python": platform.python_version(), "numpy": np.__version__,
                   "torch": str(torch.__version__), "torch_threads": torch.get_num_threads(),
                   "gymnasium": version("gymnasium"), "mujoco": version("mujoco"),
                   "threadpoolctl": version("threadpoolctl"),
                   "blas": sorted((p["internal_api"], p["version"], p["num_threads"])
                                  for p in threadpool_info())}
    finally:
        torch.set_num_threads(old_threads)


@dataclass
class ControlLatency:
    wall: list = field(default_factory=list)
    cpu: list = field(default_factory=list)
    episode_first_wall: list = field(default_factory=list)

    def add(self, wall, cpu, *, first_step):
        self.wall.append(float(wall))
        self.cpu.append(float(cpu))
        if first_step:
            self.episode_first_wall.append(float(wall))

    def summary(self, *, episodes, deadline_seconds):
        import numpy as np

        if not self.wall or episodes < 1:
            raise ValueError("latency requires measured validation decisions")
        return {"scope": "observation_history_update_through_actuator_action_excluding_simulator",
                "decisions": len(self.wall), "episodes": episodes,
                "cpu_seconds_per_episode": float(sum(self.cpu) / episodes),
                "wall_p50_seconds": float(np.quantile(self.wall, 0.5)),
                "wall_p95_seconds": float(np.quantile(self.wall, 0.95)),
                "wall_p99_seconds": float(np.quantile(self.wall, 0.99)),
                "wall_max_seconds": float(max(self.wall)),
                "episode_first_wall_max_seconds": float(max(self.episode_first_wall)),
                "deadline_seconds": deadline_seconds,
                "deadline_misses": sum(value > deadline_seconds for value in self.wall),
                "process_cold_start_measured": False}


def select_with_budget(records, *, cpu_cap_seconds, environment_step_cap,
                       deployment_episodes, latency_p99_cap_seconds, expected_runs,
                       shared_records=(), shared_expected_runs=()):
    """All trials are charged; ranking reads validation ISE, never test outcomes."""
    caps = (cpu_cap_seconds, environment_step_cap, latency_p99_cap_seconds)
    if not all(math.isfinite(v) and v > 0 for v in caps) or deployment_episodes < 0:
        raise ValueError("budget caps must be finite and positive")
    if len({row["run_id"] for row in records}) != len(records):
        raise ValueError("duplicate run IDs would double-count search costs")
    expected_by_id = {row["run_id"]: row for row in expected_runs}
    if not expected_runs or len(expected_by_id) != len(expected_runs):
        raise ValueError("declare a nonempty, unique run roster before measurement")
    by_id = {row["run_id"]: row for row in records}
    if set(by_id) - set(expected_by_id):
        raise ValueError("measurement outside the declared run roster")
    groups = {}
    for row in expected_runs:
        groups.setdefault((row["task"], row["optimizer_seed"], row["method"]), []).append(row["run_id"])
    for row in records:
        if any(row[key] != expected_by_id[row["run_id"]][key] for key in ("task", "optimizer_seed", "method")):
            raise ValueError("measurement task/root/method differs from its planned run")
    shared_by_id = {row["run_id"]: row for row in shared_records}
    shared_plan = {row["run_id"]: row for row in shared_expected_runs}
    if (len(shared_by_id) != len(shared_records) or len(shared_plan) != len(shared_expected_runs)
        or set(shared_by_id) - set(shared_plan) or set(shared_plan).intersection(expected_by_id)):
        raise ValueError("shared development costs need unique, separate planned run IDs")
    shared_groups = {}
    cohort_roots = {}
    for task, root, method in groups:
        cohort_roots.setdefault((task, method), set()).add(root)
    for row in shared_expected_runs:
        key = row["task"], row["method"]
        if key not in cohort_roots:
            raise ValueError("shared development method/task is outside the confirmation roster")
        shared_groups.setdefault(key, []).append(row["run_id"])
    for row in shared_records:
        if any(row[key] != shared_plan[row["run_id"]][key] for key in ("task", "optimizer_seed", "method")):
            raise ValueError("shared measurement differs from its planned run")
    output = []
    for (task, root, method), ids in groups.items():
        trials = [by_id[run_id] for run_id in ids if run_id in by_id]
        shared_ids = shared_groups.get((task, method), [])
        overhead = [shared_by_id[run_id] for run_id in shared_ids if run_id in shared_by_id]
        base = {"task": task, "optimizer_seed": root, "method": method,
                "trial_count": len(trials), "selected_run_id": None}
        missing = ([run_id for run_id in ids if run_id not in by_id] +
                   [run_id for run_id in shared_ids if run_id not in shared_by_id])
        if missing:
            output.append({**base, "status": "missing_measurement", "missing_run_ids": missing})
            continue
        if any(not row.get("accounting_complete") for row in (*trials, *overhead)):
            output.append({**base, "status": "incomplete_accounting"})
            continue
        profile = trials[0]["hardware_profile"]
        if any(row["hardware_profile"] != profile for row in (*trials, *overhead)):
            raise ValueError("budget matching requires the same measured hardware/thread profile")
        share = len(cohort_roots[(task, method)])
        shared_cpu = sum(row["offline_cost"]["cpu_seconds"] for row in overhead)
        shared_steps = sum(row["offline_cost"]["counts"].get("environment_steps", 0) for row in overhead)
        local_cpu = sum(row["offline_cost"]["cpu_seconds"] for row in trials)
        local_steps = sum(row["offline_cost"]["counts"].get("environment_steps", 0) for row in trials)
        search_cpu = local_cpu + shared_cpu / share
        search_steps = local_steps + shared_steps / share
        if not math.isfinite(search_cpu) or search_cpu < 0 or search_steps < 0:
            raise ValueError("invalid measured search cost")
        feasible = []
        for row in trials:
            if row["status"] != "validation_complete":
                continue
            latency = row["validation_latency"]
            if (not all(math.isfinite(v) for v in (latency["cpu_seconds_per_episode"],
                latency["wall_p99_seconds"], row["validation_ise"]))
                or min(latency["cpu_seconds_per_episode"], latency["wall_p99_seconds"]) < 0):
                raise ValueError("invalid measured validation cost or outcome")
            lifecycle_cpu = search_cpu + deployment_episodes * latency["cpu_seconds_per_episode"]
            if (lifecycle_cpu <= cpu_cap_seconds and search_steps <= environment_step_cap
                and latency["wall_p99_seconds"] <= latency_p99_cap_seconds):
                feasible.append((row, lifecycle_cpu))
        base.update(search_cpu_seconds=search_cpu, search_environment_steps=search_steps,
                    local_search_cpu_seconds=local_cpu, local_search_environment_steps=local_steps,
                    shared_development_cpu_seconds=shared_cpu,
                    shared_development_environment_steps=shared_steps, shared_allocation_roots=share,
                    shared_allocation_cpu_seconds=shared_cpu / share,
                    shared_allocation_environment_steps=shared_steps / share, shared_run_ids=shared_ids,
                    shared_failed_trial_count=sum(row["status"] == "failed" for row in overhead),
                    deployment_episodes=deployment_episodes,
                    failed_trial_count=sum(row["status"] == "failed" for row in trials))
        if not feasible:
            output.append({**base, "status": "no_budget_feasible_candidate"})
            continue
        winner, lifecycle = min(feasible, key=lambda pair: (
            pair[0]["validation_ise"], pair[1], pair[0]["run_id"]))
        output.append({**base, "status": "selected_without_test_access",
                       "selected_run_id": winner["run_id"], "lifecycle_cpu_seconds": lifecycle,
                       "validation_ise": winner["validation_ise"]})
    # Different methods may not be compared across different CPU/runtime profiles.
    profiles = [row["hardware_profile"] for row in (*records, *shared_records) if row.get("accounting_complete")]
    if profiles and any(profile != profiles[0] for profile in profiles):
        raise ValueError("cross-method budget comparisons require matched hardware profiles")
    return {"selections": output, "performance_claim_supported": False,
            "budget_caps": {"cpu_seconds": cpu_cap_seconds,
                            "environment_steps": environment_step_cap,
                            "deployment_episodes": deployment_episodes,
                            "latency_p99_seconds": latency_p99_cap_seconds},
            "all_groups_budget_feasible": all(row["status"] == "selected_without_test_access" for row in output),
            "budget_basis": "all_local_search_plus_amortized_shared_development_plus_future_control"}

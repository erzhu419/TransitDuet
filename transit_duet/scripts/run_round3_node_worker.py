#!/usr/bin/env python3
"""Run TransitDuet round-3 extra seed tasks concurrently on one CPU node."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PYBIN_DEFAULT = "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python"


@dataclass(frozen=True)
class Task:
    idx: int
    tag: str
    cmd: list[str]
    result_dir: Path


def parse_seeds(raw: str) -> list[int]:
    if ":" in raw:
        start, end = [int(x) for x in raw.split(":", 1)]
        return list(range(start, end + 1))
    return [int(x) for x in raw.replace(",", " ").split()]


def build_tasks(root: Path, pybin: str, seeds: list[int], episodes: int) -> list[Task]:
    main = ["H_timetable_v4"]
    timetable = ["H_hiro", "H_timetable_continuous", "H_timetable_no_context"]
    fixed = [
        "H_fixed_timetable_300",
        "H_fixed_timetable_330",
        "H_fixed_timetable",
        "H_fixed_timetable_390",
        "H_fixed_timetable_420",
    ]
    coupling = ["H_tpc", "H_haar"]
    baselines = ["fixed", "ga", "cmaes"]
    rules = ["rule_daganzo", "rule_xuan"]

    tasks: list[Task] = []
    idx = 0
    for seed in seeds:
        for exp in [*main, *timetable, *fixed, *coupling]:
            result = root / "logs" / f"{exp}_seed{seed}"
            cmd = [
                pybin,
                "-u",
                "runner_v3.py",
                "--config",
                f"configs_ablation/{exp}.yaml",
                "--episodes",
                str(episodes),
                "--seed",
                str(seed),
            ]
            tasks.append(Task(idx, f"train_{exp}_seed{seed}", cmd, result))
            idx += 1
        for method in baselines:
            lower_warmup = "0" if method == "fixed" else "20"
            result = root / "logs" / f"upper_{method}_seed{seed}"
            cmd = [
                pybin,
                "-u",
                "run_upper_comparison.py",
                "--method",
                method,
                "--episodes",
                str(episodes),
                "--lower_warmup",
                lower_warmup,
                "--seed",
                str(seed),
            ]
            tasks.append(Task(idx, f"baseline_{method}_seed{seed}", cmd, result))
            idx += 1
        for variant in rules:
            result = root / "logs" / f"baseline_{variant}_seed{seed}"
            cmd = [
                pybin,
                "-u",
                "run_baseline_rule.py",
                "--upper",
                variant,
                "--episodes",
                str(episodes),
                "--seed",
                str(seed),
            ]
            tasks.append(Task(idx, f"rule_{variant}_seed{seed}", cmd, result))
            idx += 1
    return tasks


def looks_complete(task: Task) -> bool:
    if not task.result_dir.exists():
        return False
    if (task.result_dir / "checkpoints" / "upper_ep299.pt").exists():
        return True
    if (task.result_dir / "history.json").exists() and any(task.result_dir.glob("*.csv")):
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--node-count", type=int, default=6)
    parser.add_argument("--seeds", default="2002:2060")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-procs", type=int, default=150)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--pybin", default=PYBIN_DEFAULT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log_dir = root / "logs" / "node_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / f"worker_node{args.node_index}_{int(time.time())}.jsonl"

    seeds = parse_seeds(args.seeds)
    all_tasks = build_tasks(root, args.pybin, seeds, args.episodes)
    tasks = [t for t in all_tasks if t.idx % args.node_count == args.node_index]

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )

    running: list[tuple[subprocess.Popen, Task, object]] = []
    failures: list[str] = []
    pending = list(tasks)
    last_heartbeat = 0.0

    def record(event: str, task: Task, **extra: object) -> None:
        row = {
            "time": time.time(),
            "event": event,
            "task": task.tag,
            "idx": task.idx,
            "cmd": " ".join(shlex.quote(x) for x in task.cmd),
            "result_dir": str(task.result_dir),
        }
        row.update(extra)
        with state_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        f"node-worker index={args.node_index}/{args.node_count} "
        f"tasks={len(tasks)} max_procs={args.max_procs} root={root}",
        flush=True,
    )

    while pending or running:
        while pending and len(running) < args.max_procs:
            task = pending.pop(0)
            if looks_complete(task):
                record("skip_complete", task)
                continue
            task.result_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{task.tag}.log"
            log_fh = log_path.open("w")
            record("start", task, log=str(log_path))
            proc = subprocess.Popen(
                task.cmd,
                cwd=root,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, task, log_fh))

        next_running: list[tuple[subprocess.Popen, Task, object]] = []
        for proc, task, log_fh in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((proc, task, log_fh))
                continue
            log_fh.close()
            record("done" if rc == 0 else "failed", task, returncode=rc)
            if rc != 0:
                failures.append(task.tag)
        running = next_running
        if pending or running:
            now = time.time()
            if now - last_heartbeat >= 60:
                last_heartbeat = now
                print(
                    f"node-worker heartbeat pending={len(pending)} "
                    f"running={len(running)} failures={len(failures)}",
                    flush=True,
                )
            time.sleep(5)

    print(f"node-worker complete failures={len(failures)}", flush=True)
    if failures:
        print("failed tasks: " + ", ".join(failures[:50]), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

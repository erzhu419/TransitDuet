#!/usr/bin/env python3
"""
launcher.py
===========
Smart experiment launcher for TransitDuet paper experiments.

Features:
  - Auto-detect available CPU/GPU resources
  - Dynamic worker scheduling
  - Resumable (skips runs with existing history.json)
  - Aggregates results on completion

Usage:
    python scripts/launcher.py                 # run all experiments
    python scripts/launcher.py --quick         # quick sanity run (50 eps)
    python scripts/launcher.py --tier 1        # only Tier 1
    python scripts/launcher.py --dry-run       # print plan without running
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
os.chdir(str(PROJECT_DIR))

# ═══════════════════════════════════════════════════════════════
#  Experiment plan
# ═══════════════════════════════════════════════════════════════

ABLATIONS = ['A_full', 'B_no_holding_feedback', 'C_no_csbapr',
             'D_no_hindsight', 'E_no_morl', 'F_fixed_fleet',
             'G_no_demand_noise']

BASELINES = ['cmaes', 'ga', 'fixed']  # existing run_upper_comparison methods

SEEDS = [42, 123, 456]

# Tier 1: main results + Pareto
# Tier 2: ablations + generalization

def build_jobs(tier, episodes, quick):
    jobs = []
    eps = 50 if quick else episodes

    if tier in (0, 1):
        # Tier 1: main (A_full + Pareto)
        for seed in SEEDS:
            jobs.append({
                'name': f'A_full_seed{seed}',
                'cmd': ['python', '-u', 'runner_v2.py',
                        '--config', 'configs_ablation/A_full.yaml',
                        '--episodes', str(eps),
                        '--seed', str(seed),
                        '--eval_pareto', '--n_eval', '5'],
                'log_dir': f'logs/A_full_seed{seed}',
                'time_est': 55 if not quick else 10,
            })

        # Tier 1: baselines (existing scripts)
        for seed in SEEDS:
            for method in BASELINES:
                jobs.append({
                    'name': f'baseline_{method}_seed{seed}',
                    'cmd': ['python', '-u', 'run_upper_comparison.py',
                            '--method', method,
                            '--episodes', str(100 if not quick else 30),
                            '--seed', str(seed)],
                    'log_dir': f'logs/upper_{method}_seed{seed}',
                    'time_est': 20 if not quick else 5,
                })

    if tier in (0, 2):
        # Tier 2: ablations
        for seed in SEEDS:
            for ablation in ABLATIONS[1:]:  # skip A_full (already in Tier 1)
                jobs.append({
                    'name': f'{ablation}_seed{seed}',
                    'cmd': ['python', '-u', 'runner_v2.py',
                            '--config', f'configs_ablation/{ablation}.yaml',
                            '--episodes', str(eps),
                            '--seed', str(seed)],
                    'log_dir': f'logs/{ablation}_seed{seed}',
                    'time_est': 55 if not quick else 10,
                })

    return jobs


# ═══════════════════════════════════════════════════════════════
#  Resource detection
# ═══════════════════════════════════════════════════════════════

def detect_resources():
    """Returns (n_workers, gpu_devices)."""
    # CPU
    n_cpu = os.cpu_count() or 1
    # Be polite on shared servers: use half the cores, min 1
    shared = os.environ.get('TRANSITDUET_SHARED', '1') == '1'
    # Each job uses ~2 threads for pytorch + env = 2 cores effectively
    # So n_workers ≈ n_cpu // 2 on shared server
    if shared:
        n_workers = max(1, n_cpu // 3)  # conservative
    else:
        n_workers = max(1, n_cpu // 2)

    # GPU
    gpu_devices = []
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],
            timeout=10).decode()
        for line in out.strip().split('\n'):
            idx, mem_free = line.split(',')
            idx, mem_free = int(idx.strip()), int(mem_free.strip())
            if mem_free >= 2000:  # need at least 2GB free
                gpu_devices.append((idx, mem_free))
    except Exception:
        pass

    return n_workers, gpu_devices


# ═══════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════

def run_job(job, gpu_id=None):
    """Run one experiment job. Returns (name, status, elapsed)."""
    t0 = time.time()
    name = job['name']
    log_dir = PROJECT_DIR / job['log_dir']

    # Skip if already completed
    if (log_dir / 'history.json').exists():
        return (name, 'skipped', 0)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'run.log'

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '2'
    env['MKL_NUM_THREADS'] = '2'
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        job['cmd'] = job['cmd'] + ['--gpu']

    try:
        with open(log_file, 'w') as f:
            f.write(f"# {' '.join(job['cmd'])}\n")
            f.flush()
            result = subprocess.run(
                job['cmd'], stdout=f, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJECT_DIR),
                timeout=3600 * 2)  # 2h timeout
        status = 'ok' if result.returncode == 0 else f'fail({result.returncode})'
    except subprocess.TimeoutExpired:
        status = 'timeout'
    except Exception as e:
        status = f'error({e})'

    elapsed = time.time() - t0
    return (name, status, elapsed)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=int, default=0,
                        help='0=all, 1=main only, 2=ablations only')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity check (50 eps)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Override auto-detected worker count (0=auto)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without running')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU-only')
    args = parser.parse_args()

    # Detect resources
    n_workers, gpu_devices = detect_resources()
    if args.workers > 0:
        n_workers = args.workers
    if args.no_gpu:
        gpu_devices = []

    jobs = build_jobs(args.tier, args.episodes, args.quick)

    print("="*70)
    print("  TransitDuet Experiment Launcher")
    print("="*70)
    print(f"  CPU workers: {n_workers}  (of {os.cpu_count()} cores)")
    print(f"  GPU devices: {[f'gpu{i}({m}MB free)' for i,m in gpu_devices] or 'none'}")
    print(f"  Tier: {args.tier}  Episodes: {args.episodes}  Quick: {args.quick}")
    print(f"  Jobs: {len(jobs)}")
    total_time = sum(j['time_est'] for j in jobs) / max(n_workers, 1)
    print(f"  Estimated wall time: {total_time:.0f} min ({total_time/60:.1f}h)")
    print(f"  Log dir: {PROJECT_DIR}/logs/")
    print("="*70)

    for j in jobs:
        status = "skip" if (PROJECT_DIR / j['log_dir'] / 'history.json').exists() else "run"
        print(f"  [{status:>4}] {j['name']:>40s}  est={j['time_est']}min")

    if args.dry_run:
        print("\nDry run — not executing.")
        return

    # Execute
    print(f"\nStarting {n_workers} parallel workers...")
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for i, job in enumerate(jobs):
            gpu_id = None
            if gpu_devices and i < n_workers:
                gpu_id = gpu_devices[i % len(gpu_devices)][0]
            fut = ex.submit(run_job, job, gpu_id)
            futures[fut] = job['name']

        completed = 0
        total = len(jobs)
        for fut in as_completed(futures):
            name, status, elapsed = fut.result()
            completed += 1
            emoji = '✓' if status == 'ok' else ('○' if status == 'skipped' else '✗')
            print(f"  [{completed:>3}/{total}] {emoji} {name:>40s}  "
                  f"{status:>10s}  {elapsed/60:.1f}min",
                  flush=True)
            results.append({'name': name, 'status': status, 'elapsed': elapsed})

    # Summary
    print("\n" + "="*70)
    ok = sum(1 for r in results if r['status'] == 'ok')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = len(results) - ok - skipped
    print(f"  Completed: {ok}/{len(results)}  skipped: {skipped}  failed: {failed}")
    print("="*70)

    # Auto-aggregate if possible
    if failed == 0 and ok > 0:
        print("\nAggregating results...")
        try:
            subprocess.run(['python', 'scripts/aggregate.py'], check=True,
                           cwd=str(PROJECT_DIR))
        except Exception as e:
            print(f"  Aggregation failed: {e}")


if __name__ == '__main__':
    main()

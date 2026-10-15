#!/usr/bin/env python3
"""Road/time-conditioned persistence, with the unchanged V53 prefix gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_conditioned_travel import road_key
from freqduet_conditional_persistence import fit_conditional_persistence, conditional_mean
from freqduet_horizon_travel import reverted_mean
from run_freqduet_v49_residual_dependence import cluster_summary
from run_freqduet_v52_clock_propagation import items
import run_freqduet_v46_travel_blocks as previous


def load_day(base, seed, scenario):
    label = f"seed{seed}_eval{scenario}"
    certified = json.loads((base / 'protocol_v6_v47_boundary_apc' / 'observed' / label / 'done.json').read_text())
    if not certified['matches_frozen_control'] or not certified['matches_v45_prefix_flows']:
        raise ValueError('conditional persistence requires certified frozen telemetry')
    legs = pd.read_csv(base / 'protocol_v6_v47_boundary_apc' / 'observed' / label /
        'completed_legs.csv', dtype={'direction': str})
    current = json.loads((base / 'protocol_v6_v50_conditioned_travel' / 'prepared' / label / 'models.json').read_text())
    horizon = json.loads((base / 'protocol_v6_v53_horizon_travel' / 'prepared' / label / 'models.json').read_text())
    rows = []
    for clock in previous.CLOCKS:
        new, old = current[str(clock)], horizon[str(clock)]
        if new['latest_completion_s'] >= clock or old['reversion']['latest_completion_s'] is not None and old['reversion']['latest_completion_s'] >= clock:
            raise ValueError('source model uses unfinished or simultaneous labels')
        for row in legs.loc[(legs.start_s > clock) & (legs.start_s <= clock + 1800)].itertuples():
            key = road_key(row.direction, row.station_id)
            rows.append(dict(road=key, origin=clock, start_s=row.start_s, end_s=row.end_s,
                actual=row.duration_s, current=new['means'][key], historical=old['historical_means'][key]))
    return label, current, horizon, pd.DataFrame(rows)


def prepare_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label, current, horizon, records = load_day(base, seed, scenario)
    models = {clock: dict(current[str(clock)], historical_means=horizon[str(clock)]['historical_means'],
        persistence=fit_conditional_persistence(records, clock)) for clock in previous.CLOCKS}
    scores = []
    for row in records.loc[records.origin.isin({item[-1] for item in items()})].itertuples():
        model = models[row.origin]['persistence']
        values = dict(v48=row.historical, v50=row.current,
            v53=float(reverted_mean(row.current, row.historical, row.start_s - row.origin,
                horizon[str(row.origin)]['reversion']['beta'])),
            conditional=conditional_mean(model, row.road, row.current, row.historical, row.origin, row.start_s))
        for mode, prediction in values.items():
            scores.append(dict(mode=mode, road=row.road, context=f"{row.origin % 3600 // 900}:{int(row.start_s // 3600 > row.origin // 3600)}",
                lead='0-900' if row.start_s <= row.origin + 900 else '900-1800',
                error=prediction - row.actual, mae=abs(prediction - row.actual)))
    out = root / 'prepared' / label
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models.json').write_text(json.dumps(models) + '\n')
    frame = pd.DataFrame(scores)
    frame.groupby(['mode', 'lead'])[['mae', 'error']].mean().to_csv(out / 'scores.csv')
    frame.groupby(['mode', 'road', 'context', 'lead'])[['mae', 'error']].mean().to_csv(out / 'road_time.csv')
    pd.DataFrame([dict(clock=c, samples=m['persistence']['samples'],
        latest_completion_s=m['persistence']['latest_completion_s'], road_groups=len(m['persistence']['roads']),
        time_groups=len(m['persistence']['contexts'])) for c, m in models.items()]).to_csv(out / 'support.csv', index=False)
    print(f"CONDITIONAL_PREPARED {label}", flush=True)


def prepare_summary(root):
    frame = pd.concat([pd.read_csv(root / 'prepared' / f'seed{s}_eval{d}' / 'scores.csv').assign(
        train_seed=s, eval_seed=d, cohort=c) for s, d, c in previous.previous.items()], ignore_index=True)
    reference = frame.loc[((frame.lead == '0-900') & (frame['mode'] == 'v50')) |
        ((frame.lead == '900-1800') & (frame['mode'] == 'v48')),
        ['train_seed', 'eval_seed', 'lead', 'mae', 'error']]
    paired = frame.loc[frame['mode'] == 'conditional'].merge(reference,
        on=['train_seed', 'eval_seed', 'lead'], suffixes=('', '_reference'), validate='one_to_one')
    paired['mae_delta'] = paired.mae - paired.mae_reference
    mean = paired.groupby(['cohort', 'lead'])[['mae', 'mae_reference']].mean()
    passed = bool((mean.mae <= mean.mae_reference).all())
    out = root / 'prepare_summary'
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / 'day_paired.csv', index=False)
    frame.groupby(['cohort', 'mode', 'lead'])[['mae', 'error']].mean().to_csv(out / 'all_modes.csv')
    result = cluster_summary(paired, ['lead'], ['mae_delta'])
    result.to_csv(out / 'paired_clusters.csv', index=False)
    (out / 'status.json').write_text(json.dumps(dict(prequential_pass=passed, policy_changed=False,
        short_reference='v50', long_reference='v48', demand_clusters=6,
        all_demand_seeds_previously_examined=True, source_commit=os.environ.get('FREQDUET_DIAGNOSTIC_COMMIT'))) + '\n')
    print(mean.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)
    print(f'CONDITIONAL_PREQUENTIAL_PASS={passed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'prepare', 'prepare-summary'))
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--data-base', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=12)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'prepare-summary':
        prepare_summary(args.out_root)
    elif args.command == 'smoke':
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] /
            'tests'), pattern=pattern) for pattern in ('test_v50*.py', 'test_v53*.py', 'test_v54*.py'))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        prepare_day((args.out_root, args.data_base, *previous.previous.items()[0]))
        out = args.out_root / 'smoke_export'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'done.json').write_text('{"conditional_persistence_smoke": true}\n')
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(prepare_day, [(args.out_root, args.data_base, *item)
                for item in previous.previous.items()[args.start:args.end]]))
        out = args.out_root / 'prepare_exports' / f'batch_{args.start}_{args.end}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'done.json').write_text('{"conditional_persistence_preparation": true}\n')

#!/usr/bin/env python3
"""Completed-prefix V54 residual calibration and unchanged short/long lead gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import pandas as pd

from freqduet_conditional_bias import attach_issued_means, fit_completed_bias, calibrated_mean, predicted_bias
from run_freqduet_v49_residual_dependence import cluster_summary
from run_freqduet_v52_clock_propagation import items
from run_freqduet_v54_conditional_persistence import load_day
import run_freqduet_v46_travel_blocks as previous


def prepare_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label, _, _, records = load_day(base, seed, scenario)
    issued = json.loads((base / 'protocol_v6_v54_conditional_persistence' / 'prepared' / label / 'models.json').read_text())
    records = attach_issued_means(records, issued)
    models = {clock: dict(issued[str(clock)], bias_calibration=fit_completed_bias(records, clock))
        for clock in previous.CLOCKS}
    scores = []
    for row in records.loc[records.origin.isin({item[-1] for item in items()})].itertuples():
        model = models[row.origin]
        correction = predicted_bias(model['bias_calibration'], row.road, row.origin, row.start_s)
        candidate = calibrated_mean(row.issued_mean, model['bias_calibration'], row.road,
            row.origin, row.start_s, model['dt_s'])
        for mode, prediction in dict(v48=row.historical, v50=row.current, v54=row.issued_mean, calibrated=candidate).items():
            scores.append(dict(mode=mode, road=row.road, origin=row.origin,
                lead='0-900' if row.start_s <= row.origin + 900 else '900-1800',
                error=prediction - row.actual, mae=abs(prediction - row.actual),
                mean_correction=correction if mode == 'calibrated' else 0.,
                physical_floor=(row.issued_mean + correction < model['dt_s']) if mode == 'calibrated' else False))
    out = root / 'prepared' / label
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models.json').write_text(json.dumps(models) + '\n')
    frame = pd.DataFrame(scores)
    frame.groupby(['mode', 'lead']).agg(mae=('mae', 'mean'), error=('error', 'mean'),
        mean_correction=('mean_correction', 'mean'), floor_predictions=('physical_floor', 'sum')).to_csv(out / 'scores.csv')
    pd.DataFrame([dict(clock=c, samples=m['bias_calibration']['samples'],
        latest_completion_s=m['bias_calibration']['latest_completion_s'], latest_issue_s=m['bias_calibration']['latest_issue_s'])
        for c, m in models.items()]).to_csv(out / 'support.csv', index=False)
    print(f'BIAS_PREPARED {label}', flush=True)


def prepare_summary(root):
    frame = pd.concat([pd.read_csv(root / 'prepared' / f'seed{s}_eval{d}' / 'scores.csv').assign(
        train_seed=s, eval_seed=d, cohort=c) for s, d, c in previous.previous.items()], ignore_index=True)
    reference = frame.loc[((frame.lead == '0-900') & (frame['mode'] == 'v50')) |
        ((frame.lead == '900-1800') & (frame['mode'] == 'v48')),
        ['train_seed', 'eval_seed', 'lead', 'mae']]
    paired = frame.loc[frame['mode'] == 'calibrated'].merge(reference,
        on=['train_seed', 'eval_seed', 'lead'], suffixes=('', '_reference'), validate='one_to_one')
    paired['mae_delta'] = paired.mae - paired.mae_reference
    old = frame.loc[frame['mode'] == 'v54', ['train_seed', 'eval_seed', 'lead', 'mae']].rename(columns={'mae': 'mae_v54'})
    paired = paired.merge(old, on=['train_seed', 'eval_seed', 'lead'], validate='one_to_one')
    paired['mae_delta_vs_v54'] = paired.mae - paired.mae_v54
    mean = paired.groupby(['cohort', 'lead'])[['mae', 'mae_reference']].mean()
    passed = bool((mean.mae <= mean.mae_reference).all())
    out = root / 'prepare_summary'
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / 'day_paired.csv', index=False)
    frame.groupby(['cohort', 'mode', 'lead'])[['mae', 'error', 'mean_correction']].mean().to_csv(out / 'all_modes.csv')
    result = cluster_summary(paired, ['lead'], ['mae_delta', 'mae_delta_vs_v54'])
    result.to_csv(out / 'paired_clusters.csv', index=False)
    floors = int(paired.floor_predictions.sum())
    (out / 'status.json').write_text(json.dumps(dict(prequential_pass=passed, policy_changed=False,
        short_reference='v50', long_reference='v48', demand_clusters=6, floor_predictions=floors,
        all_demand_seeds_previously_examined=True, source_commit=os.environ.get('FREQDUET_DIAGNOSTIC_COMMIT'))) + '\n')
    print(mean.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)
    print(f'BIAS_PREQUENTIAL_PASS={passed} physical_floor_predictions={floors}', flush=True)


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
            'tests'), pattern=pattern) for pattern in ('test_v50*.py', 'test_v53*.py', 'test_v54*.py', 'test_v55*.py'))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        prepare_day((args.out_root, args.data_base, *previous.previous.items()[0]))
        out = args.out_root / 'smoke_export'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'done.json').write_text('{"conditional_bias_smoke": true}\n')
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(prepare_day, [(args.out_root, args.data_base, *item)
                for item in previous.previous.items()[args.start:args.end]]))
        out = args.out_root / 'prepare_exports' / f'batch_{args.start}_{args.end}'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'done.json').write_text('{"conditional_bias_preparation": true}\n')

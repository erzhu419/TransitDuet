# Accelerated frozen-evaluation entry

`run.py` accepts an existing frozen actor checkpoint, a demand-day seed, a
sampler seed and an explicit policy family. It uses the qualified original
12-vehicle/seven-action physical protocol. Archived trajectories are unnecessary.

- `native_beam`: original V276 planning metric and native continuation.
- `arrival_beam`: corrected arrival-history CV2 and the original beam search.
- `arrival_current`: corrected CV2 and legal current-action enumeration.

`submit_eval.py` submits one user-selected evaluation through the scheduler with
high priority on the qualified node004 runtime. Results use a fresh directory;
only summary JSON returns locally. Actor/day/sampler/controller are required.
For example, request the launcher's arguments with:

```bash
python3 FreqDuet/optimizations/accelerated_entry/submit_eval.py --help
```

The server checks cover model parity, actor/day wiring and frozen gate behavior;
they start no traffic or training. Existing complete-day kernel receipts remain
in `v277_full_day` and `v277_current_exact`. `final_acceptance.py` consumes the
original matrix aggregation once available, compares all four same-metric
physical traces and reports the unchanged eligibility gates. Policy replacement
remains a separate decision from adopting these computational implementations.

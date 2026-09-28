# Stage-34: full-episode deployment result

**Joint gate fails:** 13/16 pre-registered adjusted root-bootstrap lower bounds
are positive; 6/8 root point gates pass. All 2560 full episodes finish on fresh
paths in tasks t101957-t101964, with unchanged Stage-33 controllers and heads.
Eight server trajectory audits and an independent root-count bootstrap match.

Remaining endpoints (positive means history improves):
- ISE vs current-repeat: +0.24064, CI [-0.03726, 0.44260].
- ISE vs lag-one: +0.09392, CI [-0.14727, 0.27840].
- Return vs lag-one: +4.22830, CI [-8.03749, 12.84261].
Both ISE/return improve versus shuffled, zero-forecast, raw views and constants;
return also improves versus current-repeat. These are bounded positive results,
not a replacement for the failed joint gate.

The original fixed50 reference is substantially better: history return 837.91
versus 912.55; ISE 2.67543 versus 1.44439. History pays 13.58984 upper calls per
episode versus 24 (43.38% fewer), including discarded previews, but incurs
85.23% higher ISE and loses 74.64 return. Fewer planning calls trades off control
quality. Wall-time means 0.906/0.911 seconds are descriptive, not a speedup test.

Method: 3081600 steps, 35029 upper calls including factual replay, 12544 previews,
zero updates/fits. Verification: 9600 steps/35029 upper calls, zero solves.
Only a 46145-byte summary is pulled; 10128126 result-JSON bytes and 152271462
raw-array bytes stay remote. Twenty-four focused tests pass.

## Limitations
This is conditional transfer of frozen models with eight-root nominal bootstrap
intervals, not independent optimizer-root confirmation or no-tradeoff evidence.
Input staging accidentally deleted the Stage-33 response caches; unchanged
replay restored every scientific metric and aggregate exactly. Recovery costs
4851100 extra steps/3440 previews/80 solves; verification adds 9600 steps/80
solves. Recovery is operational overhead, not additional scientific evidence.

Next: train plan validity and adaptive renewal frequency in the actual control
loop, with fixed50 as a primary comparator and explicit planning cost. The
150-step local response rule cannot simply replace the native update cadence.

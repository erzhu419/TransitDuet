# Trading Policy Entry

- mode: `eval`
- policy: `heuristic`
- seeds: [42, 123]
- total return mean: 0.0705
- Sharpe mean: 6.868

This is the minimal pluggable-policy entry point. The current implementation uses heuristic planner/controller classes that implement the shared policy interfaces; learned policies can replace those classes without changing the environment/tracker loop.

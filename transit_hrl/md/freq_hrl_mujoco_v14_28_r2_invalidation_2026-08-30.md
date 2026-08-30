# MuJoCo v14.28 r2 staging invalidation

Run `mujoco_v14_28_mechanism_portfolio_preflight_20260830_r2` is invalid for
scientific interpretation. Scheduler tasks `t84843`-`t84845` all executed the
same stale remote source used by r1 and failed before candidate evaluation.

The local repair was committed and pushed as `ee4cabaefd`, but scheduler cwd
staging cached readiness only by node, cwd, and TTL. Because the cwd path did
not change, dispatch skipped source rsync. The remote traceback still referenced
the deleted r1 list-comprehension at probe lines 1196-1197, proving that r2 did
not execute its preregistered source revision.

The launcher now declares `scripts/` and `freq_hrl/` as explicit launch input
directories in addition to the immutable anchor. Scheduler input staging keys
include directory file mtimes and sizes, so a source edit invalidates the cache
and forces rsync to the selected node before launch. A launcher regression test
enforces both source directories.

No r2 outcome metrics were produced or read. The frozen roots, candidates,
folds, thresholds, and validation contract remain unchanged for r3. Failed r1
and r2 task identities remain preserved.

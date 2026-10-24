# V252: read-only V251 early-link audit

Only the two accepted V249 paths;frozen V251 source `7e9faca963`.
Run synthetic checks and analysis on scheduler high/manual,node001--006.
Read each required V235/V230 payload once;reuse V250's per-query executed-dose
certificate rather than rereading large legacy records. No replay,training,
fit,new future,online selection,shadow execution,V238 or main change.

Extract observable prefixes first. Keep cycle-first linkage separate from
continuous same-plan episodes. Revisions preserve the episode;lost/switched
links close it at the LAST matching query. Put boundary H in its own ledger,
not into the new plan's internal H. Regaining a plan starts a new episode.
Only afterward label every query/episode start and cycle-first association
against FIRST next same-bus launch;never backdate the eventual matching plan.

Report cycle/episode lead times,H before/between/after first due,revisions,
rank changes,never-due/lost/unassociated/no-query/missing labels. H closes as
first-query prefix + inter-query changes + return tail + zero-query holding.
Full prefix/evaluation details stay remote;only a small summary comes back.

## Results And Next Step
Source `c491406e47`,task `t211414`,node004,high/manual,one attempt,exit 0;
8 tests passed. Only the 11,150-byte summary and log were pulled.

Both arms:262 closed cycles,260 queried,248 with a first future link;all 248
match FIRST next same-bus launch. The other 12 queried cycles have no link;
2 zero-query cycles and 12 final returns without next launch remain explicit.
All 248 links persist across sampled queries:246 see own-plan revisions
(395 changes total),and all see position-rank changes (1,194 total),without
association loss/switch. There are 142 future episodes never observed own-due.

All 106 own-due cycles share their first-link episode and plan. Mean interval
from first future link to first own-due query:1,820.26 s V107 / 1,820.30 s upper;
median 1,861 s,range 1,255--2,205 s. H at first link is zero;actual H before
first due is 14,965 / 14,940 s,and later H is 2,110 / 2,130 s. V107's full H
closes as 36,055 s within same-plan sampled intervals + 2,320 s unassociated
intervals + 1,840 s return tail = 40,215 s (upper total 40,210 s).

Early identity/timing support is present on these paths;more association
plumbing or merely earlier triggering is not the next repair. V238 already
uses future links. Next,design compensation-cost versus system supply-benefit
contrasts,without turning stable watch or accumulated H into a cut rule.
This stage defines no action selector or new-path budget;frozen items unchanged.

## Limitations
One consumed actor/day,two native arms;no query-level inference. Consecutive
query association does not prove uninterrupted physical assignment. Boundary
H,terminal tail and future-label matches do not identify removable holding.
Never observed own-due does not prove no delay outside the query times.

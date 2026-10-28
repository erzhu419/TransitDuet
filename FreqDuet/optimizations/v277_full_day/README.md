# One detached complete-day equivalence check

Task `t212103`, node004, high/manual. One start, no retry or expansion.
Reference: archived V276 p8, day2076001, actor12011, sampler27011001.
Existing code, running matrix and historical results remain unchanged.

Seven regression tests and the frozen runtime/checkpoint preflight passed.
The first supported real query matched the complete frozen128-evaluation result.
Each query is compared with the archived path before executing its action;
the first supported search additionally checks the complete old solver output.
Final acceptance requires every endpoint and saved trace record to match,
apart from callback wall time. Raw traces stay remote; only small summaries return.

Next: accept or reject this engineering candidate from the final equivalence
receipt and actual complete-day timing. No automatic main-policy replacement.

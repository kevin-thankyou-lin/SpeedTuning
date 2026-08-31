# Champion-challenger speed search v37

This prospective study uses historical speed outcomes only to pin one validated
accelerated champion and one adjacent one-phase challenger per task.  That
offline use is explicit in `CHAMPIONS.json`; no v20-v36 rollout is re-executed
or counted in the online budget.

The frozen ACT policies, four learned phases, simulator, success definition,
and common speed grid `[1, 1.5, 2, 2.5, 3]` are unchanged.  Controllers execute
as complete phase-speed schedules without a runtime risk gate.

Each task receives exactly 25 fresh online search episodes:

1. The preregistered challenger runs on five discovery seeds.
2. The champion and unchanged challenger each run on the same ten fresh paired
   seeds (`10 + 10`).

The challenger is eligible only with at least `4/5` discovery successes,
`9/10` paired successes, zero physics errors, and zero safety violations.  It
replaces a qualifying champion only when its paired success count is no lower
and its failure-aware throughput is at least `1.10x` the champion's.  If the
champion misses `9/10` but the challenger qualifies, the challenger repairs the
incumbent.  If neither reaches `9/10`, selection fails closed to native `1x`.
Otherwise ambiguous evidence retains the accelerated champion.

Only after all three searches seal does a fresh untouched 50-seed bank compare
native `1x`, the pinned champion, and the selected controller.  Identical
controllers share one executed receipt.  Search executes exactly 75 new
rollouts total; held-out evaluation executes at most 450.  No search or tuning
is permitted on held-out outcomes.

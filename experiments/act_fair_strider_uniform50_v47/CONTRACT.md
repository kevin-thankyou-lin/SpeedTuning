Fair fresh STRIDER-versus-uniform study v47
============================================

V47 compares two complete speed-search methods with the same online budget.
Each task starts both arms from the preregistered uniform `[2,2,2,2]` anchor.
Neither arm may read a historical speed schedule, ranking, search outcome, or
checkpoint. Historical rollouts are not reexecuted.

The frozen ACT policy, learned four-phase detector, phase order, environment,
action representation, controller rate, speed grid `[1,1.5,2,2.5,3]`, and
success criterion are identical across arms. The arms use disjoint randomized
search resets drawn from the same task distribution.

Each arm receives exactly two 25-rollout rounds per task. A round contains five
incumbent diagnostic resets followed by a paired ten-reset incumbent versus
challenger comparison, for 25 physical rollouts. The paired controllers see the
same ten initial states. Thus every arm consumes exactly 50 search rollouts per
task.

The uniform arm changes all four speeds by one adjacent grid rung. Its first
challenger is uniform `2.5x`. If promoted, round two tests the next upward rung;
otherwise round two tests the adjacent reliability backoff. The STRIDER arm
changes exactly one phase by one adjacent rung. It chooses the phase with the
largest preregistered predicted step saving from successful current-round
diagnostic telemetry. If neither round-one controller reaches the reliability
floor, it backs off the earliest observed failed phase by one rung.

A paired controller qualifies with at least `9/10` successes and zero physics
or safety events. A qualified challenger replaces a qualified incumbent only
when it has more successes without losing more than 5% failure-aware throughput,
or ties successes with at least 5% higher failure-aware throughput. If neither
controller qualifies, the incumbent is retained only to generate the registered
round-two repair; final selection falls back to native `1x` unless a qualified
controller exists. These are per-controller heuristic gates, not simultaneous
95% certification.

Search seeds are disjoint across methods, tasks, and final evaluation. After all
six searches seal, native `1x`, the uniform-search selection, and the STRIDER-
search selection run on the same 100 untouched randomized resets per task.
Identical selected controllers reuse one immutable final receipt rather than
duplicating rollouts. Final reporting includes paired success contingency,
successful-rollout speed, failure-aware throughput, native comparisons,
incidents, hashes, and physical versus aliased rollout accounting.

Physics and safety events are recorded as failed rollouts. Any such event in a
paired search panel prevents that controller from qualifying; final incidents
remain in the frozen held-out denominator. Final outcomes cannot alter either
search selection.

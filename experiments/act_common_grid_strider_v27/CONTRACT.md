STRIDER common-grid paired extension v27
========================================

This is a post-v26 paired extension. It does not claim that the shared final
bank is globally untouched: Tabular and Rainbow have already run on it.
STRIDER selection is nevertheless isolated from all v26 final outcomes.

The executable action grid is exactly `[1.0, 1.5, 2.0, 2.5, 3.0]`. Discovery
tests six task-independent schedules on four fresh, outcome-blind stratified
reset poses: uniform `1.5`, uniform `2.0`, and one `2.5` phase promotion from
uniform `2.0` for each of the four registered phases. The reliability-first
best adaptive candidate and uniform `1.5` are then extended to eight matched
representative poses. This costs exactly 32 search rollouts per task.

Adaptive is selected only with at least 7/8 successes, no incident, no paired
success regression against uniform `1.5`, and at least 3 percent higher
failure-aware throughput. Otherwise STRIDER selects uniform `1.5`.

After selection seals, the selected schedule is evaluated on v26's exact 50
final seeds. No v20-v26 state receipt is imported or reexecuted.

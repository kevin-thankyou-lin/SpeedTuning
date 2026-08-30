STRIDER exact-budget common-grid paired extension v28
=====================================================

This is a post-v26 paired extension. It does not claim that the shared final
bank is globally untouched: Tabular, Rainbow, and v27 STRIDER have already run
on it. V28 selection is isolated from every v26 and v27 final outcome.

The executable action grid is exactly `[1.0, 1.5, 2.0, 2.5, 3.0]`, with no
quarter-step actions. Discovery tests five task-independent schedules on three
fresh outcome-blind representative poses: uniform `1.5` and one `2.5` phase
promotion from uniform `2.0` for each of the four registered phases. Uniform
`2.0` is omitted to preserve the exact budget without dropping a phase.

The reliability-first best adaptive candidate and uniform `1.5` are then
extended on five additional matched representative poses. Cache hits for their
first three poses do not count again. The scientific search cost is therefore
`5 * 3 + 2 * 5 = 25` rollouts per task.

Adaptive is selected only with at least 7/8 successes, no incident, no paired
success regression against uniform `1.5`, and at least 3 percent higher
failure-aware throughput. Otherwise STRIDER selects uniform `1.5`.

After selection seals, the selected schedule is evaluated on v26's exact 50
final seeds. When v27 already evaluated the identical schedule on the identical
policy, environment, and seed, v28 imports the hash-verified state receipt as a
cache hit rather than executing the rollout again. New controllers run normally.
The completion receipt separates 50 scientific final samples into cache hits
and new executions and requires zero v20-v27 rollout reexecution.

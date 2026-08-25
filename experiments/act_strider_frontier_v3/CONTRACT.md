# STRIDER paired causal frontier v3 contract

This experiment corrects the v2 one-bend limitation while preserving the
frozen ACT policy, simulator, learned phase detector, controller interface,
speed grid, reliability gate, and 60-rollout search cap.

## Isolation and accounting

- Each task receives a fresh ordered 15-seed search bank and a disjoint,
  untouched 50-seed final bank. Neither bank overlaps any prior speed study.
- The search may execute at most 60 candidate rollouts. Every candidate uses
  the registered `5 -> 10 -> 15` gate: at least `4/5`, `9/10`, and `14/15`.
- Search receipts include low-dimensional causal telemetry only: detector
  phase, task reward, object positions, physics step, and nominal policy time.
- No previous schedule, outcome, ranking, or VOLT result is visible to the
  search. Failures consume their complete terminal horizon.
- A safety event rejects its candidate; a physics error terminates the lane.

## Budget-complete causal search

1. Start at uniform `2x`. Climb to uniform `2.5x` and `3x` only while the
   preceding uniform qualifies. If `2x` rejects, test uniform `1.5x` as the
   accelerated fallback.
2. Preserve the qualified uniform with greatest failure-aware throughput as an
   immutable lower bound on the observed search bank.
3. After a rejected candidate, compare each failure with the successful
   same-seed incumbent trajectory. At each detector-defined phase exit, locate
   the earliest missing phase, task-reward lag, or object-position deviation
   exceeding the preregistered `0.03 m` threshold.
4. Lower only that phase by one adjacent speed rung. If the repair exposes a
   later failure, attribute and lower only that later phase. If the same phase
   still diverges, lower the same phase one further adjacent rung.
5. Continue opening causally determined candidates while a complete 15-rollout
   gate fits within the 60-rollout cap. Thus v3 does not stop after one failed
   bend while an entire candidate gate remains available.
6. Once a repair qualifies, freeze every backed-off phase. If budget remains,
   promote one unfrozen phase by one adjacent rung, ordered by preregistered
   predicted steps saved from current-search phase workloads.
7. An adaptive schedule replaces the uniform incumbent only if it qualifies,
   has at least as many successes on the same bank, has no incidents, and
   strictly improves failure-aware throughput. Otherwise retain the uniform;
   if none qualifies, deploy native `1x`.

## Final evaluation

Seal `SELECTION.json` before opening the final bank. Evaluate native `1x`,
uniform `1.5x`, `2x`, `2.5x`, and `3x`, plus the selected STRIDER controller
when it is distinct. Each schedule-seed pair executes once. Report success rate,
failure-aware throughput change from native, successful-rollout speedup, final
empirical frontier membership, and complete rollout accounting.

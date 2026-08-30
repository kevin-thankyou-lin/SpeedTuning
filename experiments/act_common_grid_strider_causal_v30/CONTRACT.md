# Prospective common-grid causal STRIDER v30

This mechanism study restores causal repair to the common-grid comparison. It
does not reuse any v26-v29 schedule, search outcome, ranking, or final result.
The frozen ACT policy, learned four-phase detector, simulator contract, and
action grid `[1.0, 1.5, 2.0, 2.5, 3.0]` are the only inherited scientific
inputs.

## Isolation and accounting

- Every task receives 20 fresh matched search seeds and 50 disjoint final seeds.
  The final banks are registered now but remain unopened in this search-only
  workflow.
- Twenty matched native `1x` telemetry references are recorded separately from
  the adaptive candidate budget. Native must reach at least `18/20`, with zero
  physics errors and zero safety events, or the task fails closed.
- Adaptive search is capped at 100 new candidate rollouts per task. This
  deliberate mechanism-study budget can cover the complete `2x -> 2.5x -> 3x`
  uniform ladder plus two full 20-rollout causal repair rounds. Cache hits
  resume immutable schedule-seed receipts and do not increase the scientific
  count. These results must not be presented as the equal-25 comparison.
- Candidate gates are preregistered as `5 -> 10 -> 20`: `0..2/5` rejects,
  `0..8/10` rejects, and `18..20/20` qualifies. This is a selection heuristic,
  not simultaneous 95 percent certification.
- A physics error halts the lane. A safety event rejects the candidate and
  prevents selection.

## Anchor, aggressive proposal, and causal repair

1. Start at uniform `2x`. While each uniform candidate qualifies, advance the
   registered ladder through uniform `2.5x` and uniform `3x`.
2. Stop the uniform ladder at its first rejection. For each failed seed, compare
   the candidate telemetry with a successful same-seed incumbent; when the
   incumbent also failed, use the successful same-seed native reference.
3. At detector-defined phase exits, choose the earliest missing phase, reward
   lag, or object-position divergence above the frozen `0.03 m` threshold.
   If no failed rollout has a successful same-seed telemetry reference, record
   the attribution limitation and stop adapting; do not substitute a semantic
   phase guess. Also stop if matched traces exist but reveal no preregistered
   observable divergence before terminal failure.
4. Lower only the attributed phase by one adjacent common-grid rung. If the
   repair exposes a later failure, attribute that new failure and lower only the
   newly implicated phase. Never jump a phase directly to `1x`.
5. Freeze repaired phases. If a repaired controller qualifies and budget
   remains, promote one unrepaired phase by one adjacent rung, ordered by
   predicted steps saved from successful current-search phase workloads.
6. A candidate replaces its reliability lower bound only with zero incidents,
   no success regression on the matched bank, and at least 3 percent higher
   failure-aware throughput. If no accelerated controller qualifies, select
   native `1x`.

## Search-only stopping rule

Seal each task's `SELECTION.json` and `SEARCH_COMPLETE.json`, followed by one
aggregate result and checksum manifest. Do not open a final bank in this
workflow. A later benchmark must freeze these selections first and evaluate
them exactly once on the registered final seeds.

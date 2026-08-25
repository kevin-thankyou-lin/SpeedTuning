# STRIDER Frontier Search v2 Contract

This experiment evaluates a corrected reliability-first STRIDER search over a
frozen ACT policy. The controller weights, simulator, detector, task reset
distribution, and phase vocabulary are immutable.

## Search budget and banks

- Each task receives one fresh ordered 15-seed search bank.
- Search and final seeds are disjoint from each other and from prior STRIDER
  experiments.
- At most 60 new search rollouts may be executed per task.
- Candidate gates are evaluated successively at 5, 10, and 15 episodes.
- A candidate continues with at least 4/5 and 9/10 successes and qualifies
  with at least 14/15 successes.
- Any safety violation rejects a candidate. Any physics error halts the lane.
- Failures are charged their complete episode horizon when throughput is
  computed.

## Uniform incumbent and phase-conditioned bend

1. Evaluate uniform 2x.
2. If it qualifies, climb the uniform ladder through 2.5x and 3x until the
   first rejection or the ladder ends. If 2x rejects, evaluate uniform 1.5x.
3. Retain the qualified uniform with greatest failure-aware achieved
   throughput as the incumbent. All qualified schedules remain in the
   non-dominated archive.
4. Construct exactly one phase-conditioned bend:
   - after a rejected uniform, reduce the earliest implicated failure phase by
     one adjacent speed rung while leaving the other phases unchanged;
   - if all tested uniforms qualify, increase the incumbent's highest-workload
     phase by one adjacent rung.
5. The bend may replace the uniform incumbent only if it qualifies, has at
   least as many successes on the same 15 search seeds, has no incidents, and
   strictly improves failure-aware achieved throughput. Otherwise the uniform
   incumbent is retained. If no accelerated uniform qualifies, native 1x is
   the fail-closed deployment.

This guarantees that STRIDER preserves a uniform lower bound on its observed
search bank. It does not claim a deterministic held-out guarantee under finite
sampling; held-out frontier regret is reported explicitly.

## Final evaluation

After `SELECTION.json` is atomically sealed, evaluate the following on one
fresh, untouched 50-seed bank:

- native 1x;
- uniform 1.5x, 2x, 2.5x, and 3x;
- the selected STRIDER schedule, unless it is identical to a controller above.

Every physically executed schedule-seed pair has one receipt and is never
rerun. The primary metrics are success rate and failure-aware achieved
throughput improvement over native 1x. Successful-rollout speedup is secondary.
The empirical held-out Pareto frontier and whether STRIDER lies on it are
reported without changing the sealed selection.

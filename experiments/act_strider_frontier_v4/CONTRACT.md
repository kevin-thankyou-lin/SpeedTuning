# STRIDER conservative uniform-lower-bound v4 contract

This iteration addresses the v3 search-to-final reversal without reopening any
v3 seed. The frozen ACT policy, simulator, learned detector, controller
interface, speed grid, causal attribution, and 60-rollout search cap are
unchanged. The selection rule is frozen before any v4 outcome is read.

## Fresh isolation and accounting

- Each task receives a fresh ordered 20-seed search bank and a disjoint,
  untouched 50-seed final bank. All six banks are disjoint from prior studies.
- Every candidate uses the registered `5 -> 10 -> 20` gate with thresholds
  `4/5`, `9/10`, and `19/20`. The total search cap remains 60 episodes.
- Search sees only current-v4 paired telemetry. Earlier schedules and outcomes
  informed this algorithm revision but are unavailable to the v4 selector.
- A safety event rejects its candidate; a physics error terminates the lane.
  Failed episodes run to the normal task horizon and count in throughput.

## Conservative uniform lower bound

1. Start at uniform `2x`; climb the uniform ladder only while the preceding
   point reaches `19/20`. If `2x` rejects, test uniform `1.5x` as fallback.
2. Preserve the qualified uniform with greatest failure-aware throughput as the
   immutable deployment lower bound. If no uniform reaches `19/20`, deploy
   native `1x`; an adaptive repair may be recorded but cannot replace native.
3. Attribute accelerated failures with same-seed phase-exit telemetry and
   change only one phase by one adjacent speed rung per candidate.
4. An adaptive schedule may replace the uniform lower bound only if it reaches
   `19/20`, has zero incidents, and improves paired failure-aware throughput by
   at least 3%. If it slows any phase relative to the uniform incumbent, it
   must also achieve at least one additional success on the same 20 seeds.
5. Seal `SELECTION.json` before opening the final bank. Final outcomes never
   change the selected controller.

## Final benchmark

Evaluate native `1x`, uniform `1.5x`, `2x`, `2.5x`, and `3x`, plus the selected
STRIDER controller when distinct, exactly once on the fresh 50-seed bank.
Report success rate, failure-aware throughput change, successful-rollout
speedup, empirical frontier membership, incidents, hashes, and complete search
and final rollout accounting. A final-bank miss is preserved as evidence; it
is never repaired on the same bank.

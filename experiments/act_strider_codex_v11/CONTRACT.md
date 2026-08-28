# STRIDER Codex-agent causal-attribution study v11

This experiment tests whether a Codex vision agent improves phase-speed repair
for frozen ACT policies on Pick, Tea, and Insertion.

## Frozen observations and controller

- The ACT policy and four-way learned phase detector are frozen.
- Online phases use causal angle-camera RGB and robot-only proprioception; no
  simulator-oracle phase label is exposed.
- Speeds are selected from `1, 1.5, 2, 2.5, 3, 3.5, 4`.
- Tea uses the frozen center-inside-oriented-mug-volume checker in
  `experiments/act_strider_tea_release_v9/SUCCESS_CRITERION.json`.
- Pick and Insertion retain their current task-native success checkers.

## Search and causal attribution

1. Test `2x -> 2.5x -> 3x -> 3.5x`, stopping at the first rejected uniform
   candidate. If `2x` is rejected, test `1.5x` as the uniform fallback.
2. Use staged reliability gates `4/5`, `9/10`, and `19/20`.
3. Preserve the best qualified uniform controller as an immutable lower bound.
4. Match each accelerated failure to a successful same-seed incumbent rollout.
5. Diagnose at most three ordered matched failures. Codex `gpt-5.6-sol` sees a
   chronological contact sheet for each video, the task goal, and sanitized
   learned phase timelines. It receives no simulator object state, reward
   trace, contact oracle, or historical schedule result.
6. Codex returns both the first visibly failed phase and the earliest plausible
   causal phase. The causal phase must equal or precede the observed phase.
7. Independently compute the existing same-seed telemetry-divergence phase. If
   the two methods disagree, evaluate both one-rung, one-phase repairs; identical
   proposals alias one rollout set.
8. A repair replaces the uniform incumbent only if it passes the same gate,
   satisfies the v4 success-count no-regression rule, and improves failure-aware
   throughput by at least three percent.

The maximum search budget is 120 simulator-valid rollouts per task. QACC or an
equivalent MuJoCo failure remains in physical-cost accounting but is excluded
from the scientific denominator and replaced only from the registered reserve.

## Final evaluation

Seal the selection before opening the final bank. Evaluate each unique
controller among native `1x`, the uniform incumbent, Codex repair, telemetry
repair, and selected STRIDER controller on 50 simulator-valid matched poses. A
simulator error in any controller invalidates that paired pose for every method.

Report success rate, failure-aware throughput delta, successful-rollout
speedup, safety incidents, simulator-invalid attempts, full rollout accounting,
Codex/telemetry agreement, and search-versus-final repair rankings.

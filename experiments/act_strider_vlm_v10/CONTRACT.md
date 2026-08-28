# STRIDER VLM causal-attribution study v10

This experiment tests whether visual failure attribution improves phase-speed
repair for frozen ACT policies on Pick, Tea, and Insertion.

## Frozen observations and controller

- The ACT policy and four-way learned phase detector are frozen.
- Online phases use causal angle-camera RGB and robot-only proprioception; no
  simulator-oracle phase label is exposed.
- Speeds are selected from `1, 1.5, 2, 2.5, 3, 3.5, 4`.
- Tea success uses the frozen center-inside-oriented-mug-volume checker in
  `experiments/act_strider_tea_release_v9/SUCCESS_CRITERION.json`.
- Pick and Insertion retain their current task-native success checkers.

## Search and attribution

1. Test the uniform ladder `2x -> 2.5x -> 3x -> 3.5x`, stopping at the first
   rejected candidate.
2. Use staged reliability gates `4/5`, `9/10`, and `19/20`.
3. Preserve the best qualified uniform controller as the incumbent lower bound.
4. For every failed candidate with a successful same-seed incumbent reference,
   record both videos and learned phase timelines.
5. A hash-pinned local `Qwen2.5-VL-3B-Instruct` sees only the matched videos,
   task goal, and learned timelines. It returns the first visibly failed phase
   and the earliest plausible causal phase. The causal phase must be the
   observed phase or an earlier phase, never a later one.
6. Independently compute the existing same-seed telemetry-divergence phase.
7. If VLM and telemetry disagree, evaluate both one-rung, one-phase backoffs on
   the same registered search bank. Identical proposals alias one rollout set.
8. A phase-conditioned controller can replace the uniform incumbent only under
   the v4 no-regression rule: it clears the same reliability gate, satisfies the
   success-count requirement, and improves failure-aware throughput by at least
   3 percent.

The maximum search information budget is 120 simulator-valid rollouts per task.
QACC or equivalent MuJoCo failures are simulator-invalid attempts: they remain
in physical-cost accounting, do not enter success-rate denominators, and are
replaced only by the preregistered reserve bank.

## Final evaluation

Selection and its hash seal before the final bank opens. On 50 simulator-valid
matched final poses, evaluate each unique controller among native `1x`, the
uniform incumbent, the VLM repair, the telemetry repair, and STRIDER's selected
controller. A simulator error in any controller invalidates the whole matched
pair and advances all methods to the next registered reserve seed.

Report success rate, successful-rollout speedup, failure-aware throughput delta
against native, safety incidents, simulator-invalid attempts, rollout
accounting, VLM/telemetry agreement, and which repair won on search and final
banks. Search results are not final evidence.

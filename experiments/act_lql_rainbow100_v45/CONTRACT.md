LQL-Rainbow paired 100-reset study v45
======================================

V45 tests the n-step optimality inequalities from Abraham et al. (2026) as a
companion arm to V43. It uses the same frozen ACT policy, learned four-phase
observation, reward, action grid `[1, 1.5, 2, 2.5, 3]`, randomized training
resets, diagnostic resets, and held-out resets as V43. Reusing seed identities
is intentional for paired comparison; V43's baseline rollouts are not rerun.

Rainbow starts from a fresh seeded network, optimizer, replay buffer, and RNG
for every task. Each ordinary categorical Rainbow update is augmented with
two-sided squared hinge penalties over a contiguous within-episode trajectory
of eight speed decisions. Lower-bound penalties compare a logged action value
to all trajectory returns ending at least two decisions later plus a target-Q
greedy bootstrap. Upper-bound penalties compare later logged action values to
greedy target-Q values at all earlier states, including the same-state term.
Both hinge weights are fixed at one. Rewards and Q values are divided by the
fixed categorical support width before the hinge is squared; this preserves
the inequalities while placing them on a scale compatible with categorical
cross-entropy.

No success-conditioned speed ordering is imposed. In particular, success does
not label the selected speed as better than every slower speed, and terminal
failure does not assign blame to any earlier speed decision. No historical
checkpoint, rollout outcome, V43 final result, or causal repair is an input to
training.

Each task receives exactly 100 randomized training resets. The episode-100
terminal policy is always the evaluated controller. Greedy checkpoints at
episodes `10, 20, ..., 100` are evaluated on three disjoint diagnostic resets;
probe outcomes cannot alter training, choose a checkpoint, tune a controller,
or gate final evaluation.

After all three searches seal, only episode-100 LQL-Rainbow is evaluated on the
same fifty untouched V43 held-out reset seeds per task. Native, uniform, and
ordinary Rainbow receipts come from V43 and are not duplicated. Any physics
error halts the workflow; safety violations and hinge activation diagnostics
are preserved. Search, probe, and final rollout counts are sealed separately.

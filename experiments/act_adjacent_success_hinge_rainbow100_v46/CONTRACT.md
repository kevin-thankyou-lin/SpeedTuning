Adjacent-success-hinge Rainbow paired study v46
================================================

V46 isolates one proposed constraint: on a safe successful trajectory whose
task-reward milestone is non-regressive at every speed-decision endpoint, the
chosen speed's Q value should be no lower than the Q value of the exactly one
rung slower action at the same observation.

The auxiliary loss is the zero-margin L1 hinge
`mean(relu(Q(o, adjacent_slower) - Q(o, chosen)))`, divided by the fixed
categorical support width. Equality is allowed. The hinge is applied only to
chosen actions above `1x`, only from complete successful trajectories with no
physics or safety incident, and only when the decision-endpoint task-reward
trace never decreases. Failed trajectories add no ordering constraint. Speeds
more than one rung slower are never compared. The ordinary Rainbow TD loss,
reward, optimizer, exploration, observation, and action grid remain unchanged.
The paper-style lower and upper LQL trajectory-return hinges used by V45 are
disabled.

V46 uses the same frozen ACT policy, learned four-phase observation, speed grid
`[1, 1.5, 2, 2.5, 3]`, and exact training/probe/final seeds as plain Rainbow
V43 and paper-style LQL V45. This paired reuse intentionally isolates the
auxiliary loss; the older rollouts and checkpoints are not inputs to training
and are not reexecuted. Results are a paired mechanism ablation, not a fresh
certification bank.

Each task starts from a fresh seeded network, optimizer, replay buffer, and RNG,
then receives exactly 100 randomized training resets. Greedy checkpoints at
episodes `10, 20, ..., 100` are evaluated on three diagnostic resets. Probe
outcomes cannot alter training, choose a checkpoint, tune a controller, or gate
final evaluation; the episode-100 terminal policy is always evaluated.

After all three searches seal, each episode-100 policy is evaluated once on 50
paired randomized resets per task. Training, probes, and final rollouts are
accounted separately. Any physics error halts the workflow. Safety violations,
eligible/rejected trajectory counts, hinge comparison counts, activation, loss,
source identity, artifact hashes, and duplicate inventory are preserved.

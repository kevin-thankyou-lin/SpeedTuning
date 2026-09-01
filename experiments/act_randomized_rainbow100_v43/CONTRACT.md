Randomized-reset Rainbow 100-episode replication v43
=====================================================

V43 tests whether the original ACT Rainbow-50 result replicates and improves
with twice as many randomized training resets. Rainbow starts from a fresh,
seeded network, optimizer, replay buffer, and RNG for every task. No V41/V42
checkpoint, rollout, screen, held-out outcome, or selected controller is an
input. The frozen ACT policy, learned four-phase detector, reward, Rainbow
hyperparameters, decision cadence, and action grid `[1, 1.5, 2, 2.5, 3]` are
unchanged from the original benchmark.

Each task receives exactly 100 distinct randomized object resets. Immutable
resume checkpoints are written after every episode. The greedy mean network at
episodes `10, 20, ..., 100` is evaluated without learning on the same three
disjoint randomized diagnostic resets. A shared native-1x probe is evaluated on
those three seeds. Probe outcomes cannot alter training, choose a checkpoint,
tune a controller, or gate final evaluation; the episode-100 terminal policy is
always the Rainbow benchmark controller.

After all three searches seal, native 1x, uniform 2x, and episode-100 Rainbow
are each evaluated on the same fifty untouched randomized held-out resets per
task. Held-out outcomes cannot tune, repair, replace, or select a controller.
The held-out banks are disjoint from training and probes and from every V41/V42
bank. Any physics error halts the workflow; safety violations are preserved and
reported. All new training, diagnostic, and final rollouts are counted
separately, and no historical rollout is re-executed or counted as new work.

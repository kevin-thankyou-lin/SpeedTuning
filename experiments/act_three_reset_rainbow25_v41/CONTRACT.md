Three-reset phase-conditioned Rainbow exact-25 study v41
========================================================

V41 is a fresh, outcome-blind test of whether repeated exposure to three
representative object resets improves the sample efficiency of a
phase-conditioned Rainbow speed controller. It does not import any speed
schedule, selection, reward, trajectory, or rollout outcome from v20-v40.

The observable speed-controller state is the sealed learned phase detector's
four-way one-hot output. The action grid is exactly `[1, 1.5, 2, 2.5, 3]`.
The frozen ACT policy, learned phase detector, reward, Rainbow hyperparameters,
and safety monitor are unchanged from the public v26 benchmark contract.

Each task uses exactly 25 online search rollouts:

* 18 training episodes cycle round-robin over three outcome-blind,
  geometry-stratified reset poses, so each pose is visited exactly six times.
* The terminal policy is frozen after episode 18.
* Seven fresh randomized resets screen that frozen policy with no learning.

The three reset vectors and their hashes are sealed before the first policy
rollout. They span the declared independent-uniform object-position prior and
do not use policy actions, rewards, successes, historical schedules, or prior
outcomes. A search halts on any physics error. Rainbow is the selected method
only after 7/7 incident-free fresh-screen successes; otherwise selection fails
closed to native 1x. This selection rule is frozen before outcomes.

After all three exact-25 searches seal, native 1x, uniform 2x, the terminal
Rainbow policy, and the selected controller are evaluated on 50 untouched
randomized resets per task. Identical controller/seed cells reuse one
hash-verified receipt and do not count as scientific rollouts. Paper-facing
results report reliability, successful-rollout speedup, failure-aware
throughput, physics errors, and safety violations. No held-out result may tune,
repair, or replace a controller.

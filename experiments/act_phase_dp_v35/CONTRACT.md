# Finite-horizon phase-DP v35

This prospective study freezes the ACT policies, learned four-phase detector,
simulator, success criterion, and common speed grid `[1, 1.5, 2, 2.5, 3]`.
It was designed after v34, so all v35 search and final seeds are fresh. No
v20-v34 outcome is visible to the v35 estimator or selection rule.

Each task receives exactly 25 search episodes. Before outcomes, the schedules
are fixed as an `OA(25, 4, 5, 2)` orthogonal array. Every phase-speed pair is
assigned five times, and every pair of phase-speed factors occurs exactly once.
Thus the budget estimates all four phase decisions rather than evaluating a
hand-authored candidate ladder.

For each phase-speed cell, the estimator records visits, transitions to the
next registered phase (terminal task success for interaction), actual phase
steps, a Beta(1,1) posterior mean, and a one-sided 80% Wilson lower bound.
Detector backtracking after the first matching phase entry is ignored. Actions
with fewer than three visits are ineligible; if a phase has no eligible action,
it fails closed to `1x`.

Finite-horizon backward induction runs from interaction to pre-grasp. At each
phase it maximizes the recursive success lower bound, then posterior mean
success, then minimizes expected steps. This is an optimum only for the fitted
coarse four-state phase MDP and its preregistered estimator. The phase state is
not claimed to be fully Markov, and search receipts are not reliability
certificates.

After all three 25-episode searches seal, the selected schedules are frozen.
The same fresh, unopened 50-seed bank per task evaluates native `1x`, the
frozen v34 phase-only schedule, and v35 phase-DP. Identical schedules share one
controller-state cache so no final rollout is duplicated. Success, failure-aware
throughput, and paired common-success speed are reported together. Physics or
safety incidents halt search. No v20-v34 rollout is re-executed.

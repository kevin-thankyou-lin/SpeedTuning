# End-to-end schedule confirmation v36

This prospective study retains the frozen ACT policies, learned four-phase
detector, simulator, success criterion, and common speed grid
`[1, 1.5, 2, 2.5, 3]` from v26-v35. It replaces v35's marginal phase model with
complete-schedule evidence. No phase-wise transition estimate, backward
induction result, or v20-v35 speed outcome is visible to the v36 selector.
The method choice was made after inspecting v34 and v35, so this is explicitly
a prospective follow-up on fresh banks rather than an outcome-blind reanalysis.

The zero-online-rollout warm start is the pinned semantic plus SAIL-inspired
prior from v34. Its 60 offline native training rollouts are disclosed and
reused without re-execution. They rank precision-sensitive phases but contain
no v36 speed outcomes.

Each task receives exactly 25 online search episodes:

1. Five complete schedules run end-to-end on the same three discovery seeds
   (`5 x 3 = 15`). The set starts with native and the frozen prior; subsequent
   schedules make one evidence-backed phase change at a time through causal
   backoff or workload-guided promotion.
2. The best two accelerated schedules each run on five additional registered
   seeds (`2 x 5 = 10`). Each finalist therefore has eight end-to-end trials.

The selector prefers an `8/8` finalist and then higher failure-aware
throughput. A `7/8` finalist is the provisional floor; otherwise selection
fails closed to native. Zero physics and safety incidents are mandatory. No
runtime risk gate is used during search or final evaluation, so the confirmed
controller is identical to the deployed controller.

Only after all three searches seal does a fresh untouched 50-seed bank compare
native `1x` with the frozen confirmed schedule. Identical controllers share one
executed final receipt. Search uses exactly 75 new rollouts total; held-out
evaluation executes at most 300. No v20-v35 rollout is reused or re-executed.

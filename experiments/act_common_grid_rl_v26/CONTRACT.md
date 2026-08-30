Common-grid RL study v26
========================

This prospective study retrains Tabular and Rainbow from their registered
initializations for exactly 25 simulator episodes per task. Both methods use
the identical five-action speed grid `[1.0, 1.5, 2.0, 2.5, 3.0]`; quarter-step
actions and speeds above `3.0` are forbidden.

All six terminal controllers are sealed before the fresh 50-seed-per-task
final bank opens. The final bank is shared across the two RL methods and is
reserved for the capped-grid STRIDER comparison. No v20-v25 rollout receipt
is imported or re-executed.

Reliability is reported first. Successful-rollout acceleration and
failure-aware throughput are secondary. Simulator/physics errors count as
failures and safety events are reported separately.

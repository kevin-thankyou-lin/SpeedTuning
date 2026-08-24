You are the persistent implementation and experiment supervisor for the frozen
multiview ACT speed benchmark in this repository.

Read these files first and treat them as authoritative:

- `docs/ACT_MULTIVIEW_BASELINES.md`
- `experiments/act_speed_benchmark_v1/contract.json`
- `experiments/act_speed_benchmark_v1/TASKS.md`
- `/home/linke/.codex/skills/search-robot-speedups/SKILL.md`

Work on branch `codex/act-speed-benchmark-20260824`.  Preserve unrelated user
changes, never overwrite the sealed ACT roots, never duplicate a healthy
workflow, and push reviewed commits to `userfork`.  The stacked pull request is
`kevin-thankyou-lin/SpeedTuning#1`.

The ACT-specific adapter is present and its unit suite passes.  Your first
scientific gate is live uniform-1x parity using the frozen checkpoints and
accepted baseline banks.  Engineering parity rollouts are separate from the
method search budgets.  Require exact Pick 49/50, Tea 50/50, and Insertion
49/50, plus correct camera/progress/temporal-ensemble receipts, before any speed
search starts.  If parity fails, debug the adapter; do not spend search seeds.

After parity, implement and test the six method families in the frozen task
list.  Use the exact learned detector checkpoint and hash only for methods that
need phases.  Do not pass detector information to uniform, AWE, or other
methods that do not require it.  Keep `awe_offline_proxy` explicitly distinct
from full SAIL, and give `sail_inspired_adaptive` its own executable provenance;
do not claim paper-faithful SAIL unless its full stack is actually implemented.

For each task-method cell, consume exactly 50 search rollouts on the registered
search bank.  Preregister each method's internal allocation before episode one.
Freeze the selected/terminal artifact and then run exactly 50 untouched final
rollouts.  Run one shared matched native reference per task on the final states.
Count physics instability as failure and continue.  Store atomic per-state
progress, identity-gated resume state, immutable manifests, hashes, and final
completion markers.

Use independent Osmo workflows/tasks so one lane finishing cannot terminate
another.  Prefer `groot-l40-04`; inspect capacity and existing workflows before
each submission.  Monitor through completion and repair only verified stalls
without duplicating healthy work.

The final repository artifact must include a task-by-method result table with
successes/50, SR, successful first-success-step mean, matched native mean,
success-only speedup, safety violations, physics errors, and links/paths to the
manifest and per-state evidence.  Update `TASKS.md`, commit, push, and leave a
clear terminal summary.  Continue until all registered cells are complete or a
genuine external blocker is documented with evidence.


# Fresh transport-only promotion study v39

This prospective study asks one narrow question without historical schedule
advantage: does increasing only the learned detector's `transport` phase from
uniform `2x` to `2.5x` improve failure-aware throughput while retaining
reliability? The two schedules are frozen before any v39 outcome:

- uniform anchor: `[2, 2, 2, 2]`
- transport challenger: `[2, 2, 2.5, 2]`

The frozen ACT policy, simulator, learned four-phase detector, success
criterion, and speed grid `[1, 1.5, 2, 2.5, 3]` are unchanged. The detector's
current phase selects the corresponding schedule entry directly. There is no
geometry gate, terminal-approach threshold, reward input, outcome input,
future signal, or secondary speed override. No v20-v38 speed outcome is used
to initialize either schedule, and no historical rollout is re-executed.

Each task receives exactly 25 fresh online search episodes:

1. The transport challenger runs on five discovery seeds.
2. Uniform `2x` and the transport challenger each run on the same ten fresh
   paired seeds.

The challenger requires at least `4/5` discovery successes, `9/10` paired
successes, and zero physics or safety incidents. It is selected when it gains
paired successes without losing more than 5% failure-aware throughput, or ties
paired successes while improving throughput by at least 5%. Ambiguous evidence
retains uniform `2x`. Native `1x` is used only if uniform `2x` has a clear
failure (`<=7/10`) or an incident and the challenger does not qualify.

Only after all three searches seal does a fresh untouched 50-seed bank compare
native `1x`, uniform `2x`, transport-only `2.5x`, and the selected controller.
Identical selected controllers share an existing final receipt and do not add
scientific executions. Search executes exactly 75 new rollouts total; held-out
evaluation executes at most 450 unique rollouts. Held-out outcomes cannot alter
the frozen selection.

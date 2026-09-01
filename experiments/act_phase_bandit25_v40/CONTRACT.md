# Fresh phase-workload bandit study v40

This prospective study asks whether five fresh diagnostic rollouts can identify
the phase with the largest recoverable runtime and produce a better schedule
than uniform `2x`, without using any historical speed-search outcome.

The frozen ACT policy, simulator, learned four-phase detector, success
criterion, and speed grid `[1, 1.5, 2, 2.5, 3]` are unchanged. The detector's
current phase directly selects one entry in a static four-speed schedule.
There is no geometry gate, reward input, outcome input, future signal, or
secondary speed override.

Each task receives exactly 25 fresh online search episodes:

1. Run uniform `[2, 2, 2, 2]` on five diagnostic seeds.
2. For every successful, incident-free diagnostic, reconstruct each phase's
   native-equivalent workload as executed phase steps times `2`.
3. Rank each one-phase `2x -> 3x` promotion by mean predicted saved steps,
   `D_i * (1/2 - 1/3)`. Break exact ties by the frozen phase order. If no
   diagnostic succeeds incident-free, use the first phase in that order.
4. Hash-seal that one schedule before opening the paired bank.
5. Run uniform `2x` and the proposed schedule on the same ten fresh seeds.

The proposed schedule requires at least `9/10` paired successes and zero
physics or safety incidents. It is selected when it gains paired successes
without losing more than 5% failure-aware throughput, or ties successes while
improving throughput by at least 5%. Ambiguous evidence retains uniform `2x`.
Native `1x` is used only when uniform has a clear failure (`<=7/10`) or an
incident and the proposed schedule does not qualify. Any physics error halts
the study rather than launching more rollouts.

Only after all three searches seal does a fresh untouched 50-seed bank compare
native `1x`, uniform `2x`, the proposed phase-`3x` schedule, and the selected
controller. Identical selected controllers reuse the existing final receipt.
Search executes exactly 75 new rollouts total; held-out evaluation executes at
most 450 unique rollouts. Held-out outcomes cannot alter selection.

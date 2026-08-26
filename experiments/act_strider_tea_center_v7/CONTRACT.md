# STRIDER Tea center-inside success v7 contract

Success is latched only when the world-space center of the `tea_bag` geom lies
inside the inclusive oriented box site `cup_success_volume`. A tea-bag corner or
edge merely overlapping the cup volume is not sufficient. Bottom contact is not
required. The exact point-in-oriented-box implementation, XML cup volume, and
source hashes are frozen in `SUCCESS_CRITERION.json`.

Before search, a zero-rollout geometry regression must prove that a centered bag
passes, rim-only and side-only overlaps fail, and a separated bag fails. No v5
or v6 rollout outcome is visible to selection or reused as v7 evidence.

Search uses fresh, disjoint seeds `160701100--160701119`; final evaluation uses
fresh seeds `20151100--20151149`. The controller protocol remains unchanged:
frozen ACT, learned causal phase detection, a 60-rollout cap, registered
`5 -> 10 -> 20` gates (`4/5`, `9/10`, `19/20`), immutable qualified-uniform
fallback, one-phase causal repairs, and frozen selection before final opening.

Final evaluation compares native `1x`, uniform `1.5x`, `2x`, `2.5x`, and `3x`,
plus a distinct selected STRIDER schedule if needed, on the same 50 poses.
Failed-episode time remains included in achieved throughput.

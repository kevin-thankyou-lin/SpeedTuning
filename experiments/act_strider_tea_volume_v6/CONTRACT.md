# STRIDER Tea bag-volume overlap success v6 contract

Success is latched when the oriented `tea_bag` geom overlaps the explicitly
named, non-colliding `cup_success_volume`. This implements the intended rule
that any physical tea-bag volume inside the cup counts; neither the bag center
nor bottom contact is required. The exact SAT overlap implementation, XML cup
volume, and hashes are frozen in `SUCCESS_CRITERION.json`.

Before search, replay diagnostic seeds `160500100` and `160500109` at uniform
`2x`. Both must pass. Their receipts and MP4s are retained but excluded from
selection and final evidence. No v5 search or final result is reused.

Search uses fresh, disjoint seeds `160601100--160601119`; final evaluation uses
fresh seeds `20141100--20141149`. The controller protocol remains unchanged:
frozen ACT, learned causal phase detection, a 60-rollout cap, registered
`5 -> 10 -> 20` gates (`4/5`, `9/10`, `19/20`), immutable qualified-uniform
fallback, one-phase causal repairs, and frozen selection before final opening.

Final evaluation compares native `1x`, uniform `1.5x`, `2x`, `2.5x`, and `3x`,
plus a distinct selected STRIDER schedule if needed, on the same 50 poses.
Failed-episode time remains included in achieved throughput.

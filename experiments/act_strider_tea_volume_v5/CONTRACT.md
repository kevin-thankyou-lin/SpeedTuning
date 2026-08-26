# STRIDER Tea cup-volume success v5 contract

This Tea-only version corrects the task success definition. Success is latched
when the center of the `tea_bag` geom enters the explicitly named, non-colliding
`cup_success_volume`; bottom contact is no longer required. The volume spans
the cup interior from the top of its base through its rim. Exact source and XML
hashes are frozen in `SUCCESS_CRITERION.json`.

Before opening v5 search, replay the two previously inspected diagnostic seeds
`160500100` and `160500109` at uniform `2x`. Both must satisfy the new metric,
and both MP4s are retained. These two metric-regression episodes are reported
separately and are unavailable to selection; they are not v5 search or final
evidence.

The correction was motivated by inspection of a v4 trajectory that placed the
bag inside the cup but did not contact the base. Therefore no v4 search or final
outcome is reused as v5 evidence. v4 remains immutable, and v5 uses fresh,
disjoint 20-seed search and 50-seed final banks.

The controller search is otherwise unchanged from STRIDER v4: frozen ACT,
learned causal phase detection, a 60-rollout search cap, the registered
`5 -> 10 -> 20` gate (`4/5`, `9/10`, `19/20`), immutable qualified-uniform
fallback, one-phase causal repairs, and selection frozen before the final bank.

The final benchmark evaluates native `1x` and uniform `1.5x`, `2x`, `2.5x`,
and `3x`, plus a distinct selected STRIDER schedule if needed, on the same 50
fresh poses. Failed-episode time remains in achieved throughput. Results from
the old base-contact metric and this cup-volume metric must not be pooled.

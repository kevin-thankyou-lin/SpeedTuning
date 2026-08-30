# Exact-25 prospective causal STRIDER v31

This replaces the over-budget v30 mechanism diagnostic. V30 is excluded from
equal-budget claims. V31 uses exactly 25 new search episodes per task and is
blind to every v20-v30 speed-search outcome.

The frozen ACT policy, learned four-phase detector, simulator contract, and
common action grid `[1.0, 1.5, 2.0, 2.5, 3.0]` are inherited. Search uses eight
fresh seeds per task; the registered 50-seed final banks remain unopened.

## Exact accounting

- Five outcome-dependent schedule slots use the same three discovery seeds:
  `5 schedules * 3 seeds = 15` episodes.
- Two frozen finalists each receive five additional matched confirmation seeds:
  `2 finalists * 5 seeds = 10` episodes.
- The first three finalist records are cache hits from discovery and are not
  counted again. Total scientific search cost is exactly `15 + 10 = 25`.
- Native `1x` occupies the first discovery slot and therefore counts inside the
  25-episode budget. No reference or diagnostic rollout sits outside the cap.

## Outcome-dependent slots

1. Slot 1 is matched native `1x`; it must succeed `3/3` without incidents.
2. Slot 2 is the preregistered uniform `2x` anchor.
3. If the anchor is safe `3/3`, Slot 3 is uniform `3x` as an aggressive ceiling
   diagnostic. Otherwise Slot 3 backs off only the earliest causally attributed
   phase by one common-grid rung.
4. After a failed proposal, the next slot performs one-rung causal backoff using
   same-seed successful phase-exit telemetry. Repeated failure may implicate and
   back off a later phase. If telemetry shows no preregistered physical
   divergence, record the limitation and use only a registered uniform fallback.
5. After a safe proposal, use uniform `2.5x` as the registered bracket when
   appropriate; otherwise promote one unfrozen phase by one rung using current-
   search bang-for-buck. Never change multiple phases except the preregistered
   uniform diagnostics.

Every serious proposal is evaluated on all three discovery seeds immediately.
A discovery-safe controller requires `3/3` and zero incidents. The two finalists
are frozen before confirmation outcomes. Selection requires at least `7/8`, zero
incidents, then ranks successes, failure-aware throughput, and simplicity.

This is a tiny generalization study, not certification. Seal selection and
completion receipts without opening the final bank.

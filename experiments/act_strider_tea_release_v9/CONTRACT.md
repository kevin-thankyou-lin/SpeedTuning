# STRIDER Tea delayed-release v9 contract

This post-hoc development iteration tests whether slowing only the learned
`interaction` phase from `1.5x` to `1x` repairs Tea placement failures. The
proposal `[1.5, 1.5, 1.5, 1.0]` is explicitly user-informed by v8 diagnostic
seed `20161106`; it is not claimed as an independently discovered v9 proposal.

Search uses fresh seeds `160901100--160901119`. Native `1x`, uniform `1.5x`,
and delayed release are evaluated through the strict `5 -> 10 -> 20` gate
(`4/5`, `9/10`, `19/20`). The repair may replace a qualified uniform incumbent
only with at least one additional success and at least 3% greater failure-aware
throughput. If uniform is unqualified, the repair must qualify and exceed native
failure-aware throughput. Search is capped at 60 episodes.

After selection freezes, final evaluation uses untouched seeds
`20171100--20171149` for uniform `1x`, `1.5x`, `2x`, `2.5x`, `3x`, and `3.5x`,
plus the selected controller when distinct. Controllers run sequentially on one
GPU because v8 exposed a marginal same-GPU concurrency replay discrepancy.
Every actual final episode records its MP4 before its state receipt seals; no
post-hoc replay is used as final-bank media.

Success requires the world-space center of `tea_bag` to lie inside the
inclusive oriented `cup_success_volume`. Failed-episode time remains in achieved
throughput. All v9 receipts and videos are new and no v8 rollout is reused.

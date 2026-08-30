# SAIL-inspired warm-start STRIDER versus Tabular v33

This prospective study uses the frozen ACT policy, learned four-phase detector,
simulator, success criterion, and common speed grid `[1, 1.5, 2, 2.5, 3]` from
v26-v32. It does not implement or claim paper-faithful SAIL.

Before any v33 outcome is read, each task reuses the immutable, payload-hashed
offline SAIL-inspired motion-complexity artifact produced by the audited v1
benchmark from the same demonstration dataset. Its 20 nominal-time bins are
converted to four phase priors by averaging each contiguous five-bin block. The
maximum-speed profile is used and rounded to the common grid with a `1.5x`
minimum. The source artifact file, payload, dataset-array, and checkpoint hashes
are pinned separately; this reuse consumes no online rollout.

Three methods receive exactly 25 fresh online adaptation episodes per task:

1. `sail_causal`: native `1x` reference plus the SAIL-inspired prior and three
   outcome-dependent one-phase promotions or causal repairs, each on three
   matched poses (`15` episodes), followed by two frozen finalists on five
   additional poses each (`10` episodes). Causal repairs lower only the phase of
   earliest same-seed physical divergence by one rung.
2. `sail_tabular`: first-visit Monte Carlo phase-speed control initialized with
   a deterministic preference for the SAIL-inspired prior schedule. Epsilon
   decays from `0.5` to `0.05` over exactly 25 fresh sequential episodes. The
   terminal greedy four-phase schedule is then frozen.
3. `agent_causal`: an outcome-blind semantic prior starts at
   `[2, 1.5, 3, 1.5]`: moderate pre-grasp, conservative grasp/load transfer,
   aggressive transport, and conservative final interaction. It then uses the
   identical `15 + 10` one-phase causal update and finalist procedure as
   `sail_causal`.

No v20-v32 rollout is reused as adaptation evidence. Search is blind to the v32
selection outcomes and the registered final seeds. After all nine searches seal,
native `1x`, frozen v32 STRIDER, `sail_causal`, `sail_tabular`, and
`agent_causal` each run once on the exact unopened v32 50-seed final bank. This
produces 225 online search rollouts and 750 held-out final rollouts. Reliability
is reported beside failure-aware throughput and paired common-success timing.

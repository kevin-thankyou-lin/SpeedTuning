# SAIL-inspired warm-start STRIDER versus Tabular v33

This prospective study uses the frozen ACT policy, learned four-phase detector,
simulator, success criterion, and common speed grid `[1, 1.5, 2, 2.5, 3]` from
v26-v32. It does not implement or claim paper-faithful SAIL.

The originally registered demonstration artifacts are unavailable and no v33
scientific rollout opened before that was discovered. This prospective
amendment therefore trains a new SAIL-inspired phase-precision prior from
exactly 20 fresh native `1x` simulation trajectories per task (`60` offline
pretraining rollouts total). These seeds are registered in `PRIOR_BANKS.json`,
are disjoint from every earlier speed study and from v33 search/final banks, and
are never reused as online adaptation evidence.

The offline target is causal robot-motion precision: the top quartile of
normalized joint-trajectory curvature or either gripper transition. A
Beta(1,1)-smoothed Bernoulli head is fitted independently for each frozen phase.
Its phase probabilities map through preregistered thresholds to the common
speed grid with a `1.5x` minimum. Prior-training states, target threshold, model
parameters, selected schedules, payload hashes, and all `60` rollouts seal
before online search. This is newly trained *SAIL-inspired* initialization, not
paper-faithful SAIL; offline-pretraining cost is reported separately.

Three methods receive exactly 25 fresh online adaptation episodes per task:

1. `sail_causal`: native `1x` reference plus the newly trained SAIL-inspired prior and three
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
The SAIL-initialized arms share the separately charged 60-rollout prior, while
the agent-semantic arm receives no offline rollout. This is an equal-online-
budget comparison, not a total-data comparison.

# Blinded staged VLM speed-frontier search: frozen ACT Pick

This is a fresh post-hoc exploratory lane. It is disjoint from the sealed ACT
benchmark and from the earlier three-scene VLM screen.

- Frozen policy: the hash-pinned multiview ACT Pick policy from the passed ACT
  manifest. No policy training or fine-tuning is permitted.
- Observation: the frozen causal RGB/proprio phase detector.
- Phases: `pre_grasp`, `grasp_lift`, `transport`, `interaction`.
- Speeds: `1, 1.5, 2, 2.5, 3, 3.5, 4`.
- Search seeds: exactly `140200000..140200019`, in that order, for every
  schedule and its matched native control.
- Reserved final seeds: `140210000..140210099`. They remain unopened in this
  search workflow.
- Prior schedules, outcomes, and media are blinded from proposal generation.
  The only initial accelerated schedule is the preregistered interior anchor
  `[2,2,2,2]`.
- Every rollout runs the full horizon. A safety violation or physics error
  halts the lane; it cannot be relabeled as an ordinary policy failure.
- Staged gate: run 5; `0..2` successes reject and `3..5` continue. At 10,
  `0..8` reject and `9..10` continue. At 20, `0..17` reject and `18..20`
  qualify for continued search.
- Once the anchor qualifies, raise all phases by one adjacent rung until the
  first rejected uniform candidate. The first rejected uniform candidate may
  be repaired by lowering exactly one VLM-attributed phase by one rung.
- Once repaired, promote one unfailed phase at a time by one rung. Proposals
  are ordered by mean successful current-run phase workload times
  `(1/old_speed - 1/new_speed)`; ties follow the phase order above.
- A repaired phase is frozen. Exactly one registered midpoint may be used after
  the first rejected upward promotion.
- Qualified schedules rank by reliability first, matched-native speedup second,
  and schedule simplicity third. The uniform `[2,2,2,2]` anchor remains a
  mandatory comparator and fallback; an adaptive schedule cannot be selected
  over it with lower observed reliability or lower matched speed.
- Pre-final hard budget: 80 total fresh rollouts, including matched native
  controls. This supports the 20 native controls, anchor, first uniform
  expansion, and one causal repair/promotion gate. No hidden reruns are allowed.
- `18/20` means qualified for continued search, not certified or deployable.
  No final or deployment claim is allowed without the untouched final bank.

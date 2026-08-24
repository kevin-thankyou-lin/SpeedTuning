# Post-hoc Pick extension: 1.5x grasp-lift repair

This extension is registered after the bounded staged VLM search completed. It
does not alter that search's 80-rollout receipt and is not a final benchmark.

- Parent completion SHA256: `1d694a6b2ca32be65b706260fdd061be0c81bd70fc9a474304318ade7eee7ba1`.
- Parent state SHA256: `ace434d57f46b53edd1243c783541694e724e6046a8d3f63882df8d01d8745c2`.
- Parent selection SHA256: `4f7fe5eaf8f0804fb8c12c59e5a4a3e7a48824d884211f2e5da525f5df273d8c`.
- Candidate: exactly `[2.5,1.5,2.5,2.5]` for phases `pre_grasp`,
  `grasp_lift`, `transport`, `interaction`.
- Causal basis: after the one-rung grasp repair `[2.5,2,2.5,2.5]`, the
  original failed seeds `140200007` and `140200013` remained failures. This
  extension lowers only the implicated `grasp_lift` phase one adjacent rung.
- Seeds: reuse the parent's ordered `140200000..140200019` candidate bank and
  exact same-pose cached native controls. Native rollouts must be hash-verified
  and must not be rerun.
- Gate: the preregistered `5 -> 10 -> 20` thresholds from the parent contract.
- Budget: at most 20 new candidate rollouts; zero new native rollouts.
- Runtime incidents: any physics error or safety violation halts the extension.
- No-regression replacement rule: candidate must have at least the parent's
  uniform-2x `19/20` successes, zero incidents, and matched-native speedup at
  least `1.8725346968590213x`. Otherwise uniform `[2,2,2,2]` remains selected.
- The untouched final seeds `140210000..140210099` remain unopened.

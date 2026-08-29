# STRIDER representative-panel reselection diagnostic v17

This is a post-v16 diagnostic of the reset-panel design. The v16 final-bank
outcomes motivated this protocol, so v17 is not an independent replication or
a replacement final benchmark.

1. Reuse only the two finalist identities frozen in v16 `SELECTION.json` and
   the 24 cached discovery rollouts that produced them. Do not rerun discovery
   and do not read v16 final outcomes in the executable selector.
2. Define the reset prior before controller outcomes: independent uniform object
   positions over the simulator's declared reset ranges. Keep fixed height,
   orientation, and Tea scene suffix values unchanged.
3. For Pick and Tea, use a four-pose 2-D tensor Gauss-Legendre quadrature panel.
   For Insertion, use a four-pose quartile-midpoint Latin hypercube over peg x/y
   and socket x/y. Freeze a second four-pose extension before any rollout.
4. Run both frozen finalists on the first four poses. Stop only for a registered
   safety/physics event or adaptive futility at two or fewer successes. Otherwise
   continue both finalists to the eight-pose prefix.
5. At eight poses, require an accelerated controller to have at least 7/8
   successes and zero incidents. Reliability is ranked before failure-aware
   throughput. Adaptive may replace an eligible uniform comparator only without
   a success-count deficit and with at least 3% higher throughput. If uniform is
   ineligible but adaptive is eligible, choose adaptive; otherwise deploy native.
6. The effective search accounting is 24 cached discovery rollouts plus 8 or 16
   fresh confirmation rollouts: 32 with early stopping or 40 maximum. Cache hits
   do not increment physical-attempt or scientific-rollout counts.
7. Preserve v16 and all prior result roots unchanged. v17 ends after reselection;
   it does not open or reinterpret a final bank.


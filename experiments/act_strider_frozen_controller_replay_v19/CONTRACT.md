# Frozen-controller replay on the STRIDER v18 bank

This extension compares the already-selected learned-subtask, tabular-RL, and
Rainbow controllers on the exact 50 primary reset seeds used by the sealed v18
matched final bank.

1. Load the selected controller artifacts from the audited v1 benchmark. Never
   train, update, select, or otherwise mutate a controller.
2. Replay only `learned_phase_subtask`, `learned_phase_tabular_rl`, and
   `learned_phase_rainbow_rl`, for Pick, Tea, and Insertion.
3. Use exactly the ordered `final_primary` seeds in
   `act_strider_representative_final_v18/BANKS.json`. Do not consume reserve
   seeds and do not rerun v18 native, uniform-1.5x, or STRIDER controllers.
4. Preserve one immutable receipt per physical attempt. Resume only a
   contiguous identity-matched prefix; cache hits do not count as rollouts.
5. Count safety violations as policy failures. Preserve simulator/physics
   incidents and fail closed on claims of a complete paired comparison if any
   primary attempt is simulator-invalid.
6. Compare against the sealed v18 records only after all nine replay cells have
   exactly 50 receipts and valid completion hashes.

The new physical budget is exactly `3 methods x 3 tasks x 50 = 450` rollouts
when there are no simulator-invalid attempts.

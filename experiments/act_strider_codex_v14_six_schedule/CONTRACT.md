# STRIDER six-schedule diverse-panel extension v14

Extend each sealed v13 search from its existing unique schedules to exactly six
without rerunning cached rollouts or opening a final bank.

1. Verify every v13 selection, completion hash, candidate summary, state receipt,
   video hash, four-reset bank identity, and search-only status.
2. Keep the selected v13 schedule as an immutable incumbent. Freeze the phase
   causally backed off by a selected repair; when uniform remains selected,
   freeze the phase attributed to the rejected faster uniform candidate.
3. Estimate each remaining phase's native-equivalent workload from the four
   successful incumbent trajectories. Rank one-rung, one-phase promotions by
   predicted saved steps, with registered phase order as the tie-breaker.
4. Before reading new outcomes, register only enough top-ranked promotions to
   reach six total unique schedules for that task.
5. Test every new schedule on the same four diverse resets. Require `4/4`, zero
   safety incidents, and throughput no worse than the frozen incumbent.
6. Report parent cache hits and new physical rollouts separately. Keep the final
   bank closed; this extension is search evidence only.

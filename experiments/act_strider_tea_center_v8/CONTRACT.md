# STRIDER Tea center-inside parallel v8 contract

Success is latched only when the world-space center of the `tea_bag` geom lies
inside the inclusive oriented box site `cup_success_volume`. Mere rim or side
overlap is a failure; bottom contact is not required. The frozen criterion is
identical to v7, while all v8 search and final seeds are fresh.

Search uses seeds `160801100--160801119` and preserves causal chronology across
candidate schedules. If the implicated phase already runs at the minimum `1x`,
the search stops and retains its qualified uniform incumbent rather than trying
an invalid lower speed. No v7 outcome is visible to v8 selection.

Final evaluation uses seeds `20161100--20161149`. After selection is frozen,
the five uniform controllers and any distinct STRIDER controller are evaluated
by up to four independent GPU worker processes. Each controller owns a distinct
receipt directory, and cache hits never count as new rollouts. Failed-episode
time remains included in achieved throughput.

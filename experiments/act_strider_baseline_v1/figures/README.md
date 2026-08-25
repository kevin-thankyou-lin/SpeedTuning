# Reliability-throughput frontier figure

The figure plots achieved task throughput against success rate on each frozen
50-seed ACT final bank. Throughput charges successful episodes through first
success and failed episodes through their terminal horizon. Green curves join
the empirical non-dominated methods for each task; gray dashed curves connect
the independently evaluated fixed-uniform schedules. The vertical dotted line
marks 90% success rate. Internal AWE/SAIL rows remain labeled as proxies.

Paper-ready LaTeX:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/reliability_throughput_frontier.pdf}
  \caption{\textbf{Reliability--throughput frontiers on frozen ACT policies.}
  Effective task throughput includes the time spent in failed episodes and is
  reported relative to native $1\times$ execution. Each point uses the same
  50-seed final bank within a task. Green curves connect empirically
  non-dominated methods; dashed gray curves connect fixed-uniform speeds.}
  \label{fig:reliability-throughput-frontier}
\end{figure*}
```

Regenerate from the audited Markdown table:

```bash
python scripts/plot_act_reliability_throughput_frontier.py \
  --results experiments/act_strider_baseline_v1/RESULTS.md \
  --output-prefix experiments/act_strider_baseline_v1/figures/reliability_throughput_frontier
```

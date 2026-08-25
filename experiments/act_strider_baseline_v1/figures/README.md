# Reliability-throughput frontier figure

The figure plots achieved task throughput against success rate on each frozen
50-seed ACT final bank. Throughput charges successful episodes through first
success and failed episodes through their terminal horizon. Green curves join
the empirical non-dominated methods for each task; gray dashed curves connect
the independently evaluated fixed-uniform schedules. The success-rate axis is
piecewise linear: 0--25% is compressed and 90--100% is expanded so reliability
losses near 100% remain visually prominent. The vertical dotted line marks 90%
success. SAIL-inspired points are explicitly colored and annotated; the
asterisk identifies frozen-policy proxies rather than paper-faithful retraining.

Paper-ready LaTeX:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/reliability_throughput_frontier.pdf}
  \caption{\textbf{Reliability--throughput frontiers on frozen ACT policies.}
  Effective task throughput includes the time spent in failed episodes and is
  reported relative to native $1\times$ execution. Each point uses the same
  50-seed final bank within a task. Green curves connect empirically
  non-dominated methods; dashed gray curves connect fixed-uniform speeds. The
  piecewise-linear success-rate axis expands 90--100\% and compresses 0--25\%
  to emphasize small losses near native reliability. Asterisks denote
  frozen-policy proxies rather than paper-faithful retraining.}
  \label{fig:reliability-throughput-frontier}
\end{figure*}
```

Regenerate from the audited Markdown table:

```bash
python scripts/plot_act_reliability_throughput_frontier.py \
  --results experiments/act_strider_baseline_v1/RESULTS.md \
  --output-prefix experiments/act_strider_baseline_v1/figures/reliability_throughput_frontier
```

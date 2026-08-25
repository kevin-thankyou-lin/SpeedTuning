# Reliability-throughput frontier figure

The figure plots achieved task throughput against success rate on each frozen
50-seed ACT final banks. Throughput charges successful episodes through first
success and failed episodes through their terminal horizon. Filled points use
the fresh paired STRIDER/VOLT bank; hollow points use the earlier sealed
baseline bank. Green curves join only the non-dominated fresh paired-bank
methods, and gray dashed curves connect its fixed-uniform schedules. The
success-rate axis is piecewise linear: 0--80% occupies 15% of the horizontal
width and 80--100% occupies the remaining 85%, so reliability losses near 100%
remain visually prominent. Vertical dotted lines mark 80% and 90% success.
SAIL-inspired points are explicitly
colored and annotated; the asterisk identifies frozen-policy proxies rather
than paper-faithful retraining.

Paper-ready LaTeX:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/reliability_throughput_frontier.pdf}
  \caption{\textbf{Reliability--throughput frontiers on frozen ACT policies.}
  Effective task throughput includes the time spent in failed episodes and is
  reported relative to native $1\times$ execution. Each point uses the same
  50-seed final bank. Filled points share the fresh paired STRIDER/VOLT bank;
  hollow points use an earlier sealed bank and provide distribution-level
  context rather than same-seed comparisons. Green curves connect
  non-dominated methods on the fresh paired bank; dashed gray curves connect
  its fixed-uniform speeds. The explicitly nonlinear success-rate axis assigns
  85\% of its width to 80--100\% SR and compresses 0--80\% into the remaining
  15\%, emphasizing small losses near native reliability. Asterisks denote frozen-policy proxies rather than
  paper-faithful retraining.}
  \label{fig:reliability-throughput-frontier}
\end{figure*}
```

Regenerate from the audited Markdown table:

```bash
python scripts/plot_act_reliability_throughput_frontier.py \
  --results experiments/act_strider_baseline_v1/RESULTS.md \
  --output-prefix experiments/act_strider_baseline_v1/figures/reliability_throughput_frontier
```

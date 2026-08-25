#!/usr/bin/env python3
"""Plot success rate against failure-aware task throughput for ACT methods."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASKS = ("Pick", "Tea", "Insertion")
FIXED_UNIFORM_RE = re.compile(r"^Uniform (1\.5|2|2\.5|3)x$")


@dataclass(frozen=True)
class Result:
    task: str
    method: str
    successes: int
    episodes: int
    throughput_delta: float

    @property
    def success_rate_percent(self) -> float:
        return 100.0 * self.successes / self.episodes


def parse_results(path: Path) -> list[Result]:
    values: list[Result] = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 8 or fields[0] not in TASKS:
            continue
        success_match = re.fullmatch(r"(\d+)/(\d+)", fields[2])
        throughput_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", fields[5])
        if success_match is None or throughput_match is None:
            raise RuntimeError(f"malformed result row: {line}")
        values.append(Result(
            task=fields[0],
            method=fields[1],
            successes=int(success_match.group(1)),
            episodes=int(success_match.group(2)),
            throughput_delta=float(throughput_match.group(1)),
        ))
    if {value.task for value in values} != set(TASKS):
        raise RuntimeError(f"missing task rows in {path}")
    return values


def pareto_frontier(values: list[Result]) -> list[Result]:
    frontier = []
    for candidate in values:
        dominated = any(
            other.success_rate_percent >= candidate.success_rate_percent
            and other.throughput_delta >= candidate.throughput_delta
            and (
                other.success_rate_percent > candidate.success_rate_percent
                or other.throughput_delta > candidate.throughput_delta
            )
            for other in values
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda value: (value.success_rate_percent, -value.throughput_delta),
    )


def method_style(method: str) -> dict:
    if method == "Native 1x":
        return {"color": "#4d4d4d", "marker": "X", "size": 44, "zorder": 4}
    if FIXED_UNIFORM_RE.fullmatch(method):
        return {"color": "#8c8c8c", "marker": "o", "size": 30, "zorder": 4}
    if method == "Uniform sweep":
        return {"color": "#4c78a8", "marker": "D", "size": 34, "zorder": 5}
    if method == "Learned subtask":
        return {"color": "#59a14f", "marker": "s", "size": 34, "zorder": 5}
    if method == "Tabular RL":
        return {"color": "#f28e2b", "marker": "^", "size": 38, "zorder": 5}
    if method == "Rainbow RL":
        return {"color": "#e15759", "marker": "v", "size": 38, "zorder": 5}
    if method in {"AWE offline proxy", "SAIL-inspired"}:
        marker = "P" if method.startswith("AWE") else "h"
        return {"color": "#b5b5b5", "marker": marker, "size": 34, "zorder": 3}
    if method == "STRIDER":
        return {"color": "#7b2cbf", "marker": "*", "size": 105, "zorder": 7}
    raise RuntimeError(f"no plot style registered for {method}")


def annotate(ax, value: Result) -> None:
    if FIXED_UNIFORM_RE.fullmatch(value.method):
        text = value.method.removeprefix("Uniform ").replace("x", "×")
        offset = {
            ("Pick", "Uniform 2x"): (4, -11),
            ("Tea", "Uniform 1.5x"): (-20, -13),
        }.get((value.task, value.method), (4, 4))
    elif value.method == "STRIDER":
        text = "STRIDER"
        offset = {
            "Pick": (-45, -13),
            "Tea": (-39, -13),
            "Insertion": (4, -13),
        }[value.task]
    else:
        return
    ax.annotate(
        text,
        (value.success_rate_percent, value.throughput_delta),
        xytext=offset,
        textcoords="offset points",
        fontsize=6.2,
        color="#303030",
        zorder=8,
    )


def plot(results: list[Result], output_prefix: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72), sharex=True, sharey=True)

    for ax, task in zip(axes, TASKS, strict=True):
        task_values = [value for value in results if value.task == task]
        fixed = [value for value in task_values if FIXED_UNIFORM_RE.fullmatch(value.method)]
        fixed.sort(key=lambda value: float(FIXED_UNIFORM_RE.fullmatch(value.method).group(1)))
        ax.plot(
            [value.success_rate_percent for value in fixed],
            [value.throughput_delta for value in fixed],
            color="#a6a6a6",
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )

        frontier = pareto_frontier(task_values)
        ax.plot(
            [value.success_rate_percent for value in frontier],
            [value.throughput_delta for value in frontier],
            color="#1b7f5a",
            linewidth=1.35,
            zorder=3,
        )
        ax.scatter(
            [value.success_rate_percent for value in frontier],
            [value.throughput_delta for value in frontier],
            facecolors="none",
            edgecolors="#1b7f5a",
            s=76,
            linewidths=1.1,
            zorder=6,
        )

        for value in task_values:
            style = method_style(value.method)
            ax.scatter(
                value.success_rate_percent,
                value.throughput_delta,
                c=style["color"],
                marker=style["marker"],
                s=style["size"],
                edgecolors="white" if value.method != "Native 1x" else "none",
                linewidths=0.45,
                zorder=style["zorder"],
            )
            annotate(ax, value)

        ax.axvline(90, color="#d0d0d0", linewidth=0.8, linestyle=":", zorder=0)
        ax.axhline(0, color="#c7c7c7", linewidth=0.8, zorder=0)
        ax.set_title(task, fontweight="semibold", pad=4)
        ax.set_xlim(0, 103)
        ax.set_ylim(-80, 150)
        ax.set_xticks((0, 25, 50, 75, 90, 100))
        ax.set_yticks((-75, -50, 0, 50, 100, 150))
        ax.grid(True, color="#ededed", linewidth=0.55, zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Success rate (%)")

    axes[0].set_ylabel("Effective throughput change\nvs. native 1× (%)")
    legend = [
        Line2D([], [], color="#4d4d4d", marker="X", linestyle="none", markersize=5, label="Native 1×"),
        Line2D([], [], color="#8c8c8c", marker="o", linestyle="--", markersize=4, label="Fixed uniform"),
        Line2D([], [], color="#4c78a8", marker="D", linestyle="none", markersize=4, label="Uniform sweep"),
        Line2D([], [], color="#59a14f", marker="s", linestyle="none", markersize=4, label="Learned subtask"),
        Line2D([], [], color="#f28e2b", marker="^", linestyle="none", markersize=4, label="Tabular RL"),
        Line2D([], [], color="#e15759", marker="v", linestyle="none", markersize=4, label="Rainbow RL"),
        Line2D([], [], color="#b5b5b5", marker="P", linestyle="none", markersize=4, label="Internal proxies"),
        Line2D([], [], color="#7b2cbf", marker="*", linestyle="none", markersize=8, label="STRIDER"),
        Line2D([], [], color="#1b7f5a", linestyle="-", linewidth=1.4, label="Pareto frontier"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
        columnspacing=1.25,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.28, wspace=0.10)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(parse_results(args.results), args.output_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

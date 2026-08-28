#!/usr/bin/env python3
"""Run search-only STRIDER with four matched uniformly sampled reset poses."""

from __future__ import annotations

import sys

from scripts import run_act_strider_frontier_v4 as v4
from scripts import run_act_strider_vlm_v10 as base


PANEL_SIZE = 4
SEARCH_BUDGET = 32
MIN_THROUGHPUT_GAIN = 0.03


def adaptive_replaces_uniform(candidate: dict, incumbent: dict | None) -> bool:
    """Permit a 4/4 repair to replace a 4/4 uniform incumbent by throughput."""

    if incumbent is None or not candidate["qualified"] or not incumbent["qualified"]:
        return False
    candidate_summary = candidate["summary"]
    incumbent_summary = incumbent["summary"]
    if candidate_summary["safety_violations"] or candidate_summary["physics_errors"]:
        return False
    return (
        candidate_summary["episodes"] == PANEL_SIZE
        and incumbent_summary["episodes"] == PANEL_SIZE
        and candidate_summary["successes"] == PANEL_SIZE
        and incumbent_summary["successes"] == PANEL_SIZE
        and candidate_summary["achieved_throughput_per_step"]
        >= (1.0 + MIN_THROUGHPUT_GAIN)
        * incumbent_summary["achieved_throughput_per_step"]
    )


def configure() -> None:
    base.STAGES = ((PANEL_SIZE, PANEL_SIZE),)
    base.SEARCH_VALID_TARGET = PANEL_SIZE
    base.SEARCH_BUDGET = SEARCH_BUDGET
    base.CODEX_STUDY_VERSION = "v12-four-reset"
    base.CODEX_METHOD = "strider_codex_four_matched_uniform_resets"
    base.SELECTION_SCHEMA = "act-strider-codex-four-reset-selection-v12"
    v4.adaptive_replaces_uniform = adaptive_replaces_uniform


def main() -> int:
    configure()
    if "--search-only" not in sys.argv:
        sys.argv.append("--search-only")
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

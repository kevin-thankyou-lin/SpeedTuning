#!/usr/bin/env python3
"""Run search-only STRIDER on four outcome-blind diverse reset poses."""

from __future__ import annotations

import sys

from scripts import run_act_strider_codex_v12_four_reset as four_reset


def configure() -> None:
    four_reset.configure()
    four_reset.base.CODEX_STUDY_VERSION = "v13-diverse-four-reset"
    four_reset.base.CODEX_METHOD = "strider_codex_four_maximin_diverse_resets"
    four_reset.base.SELECTION_SCHEMA = (
        "act-strider-codex-diverse-four-reset-selection-v13"
    )


def main() -> int:
    configure()
    if "--search-only" not in sys.argv:
        sys.argv.append("--search-only")
    return four_reset.base.main()


if __name__ == "__main__":
    raise SystemExit(main())

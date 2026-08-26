#!/usr/bin/env python3
"""Run Tea STRIDER using tea-bag/cup interior volume overlap success."""

from scripts import run_act_strider_tea_volume_v5 as implementation


def main() -> int:
    implementation.VERSION = 6
    implementation.SUCCESS_CRITERION_SCHEMA = "tea-cup-volume-overlap-success-v1"
    implementation.METRIC_REGRESSION_SCHEMA = (
        "tea-cup-volume-overlap-regression-v1"
    )
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

from scripts import run_act_strider_codex_v13_diverse_four_reset as module


def test_v13_configuration_preserves_strict_four_reset_gate():
    original = (
        module.four_reset.base.STAGES,
        module.four_reset.base.SEARCH_VALID_TARGET,
        module.four_reset.base.SEARCH_BUDGET,
        module.four_reset.base.CODEX_STUDY_VERSION,
        module.four_reset.base.CODEX_METHOD,
        module.four_reset.base.SELECTION_SCHEMA,
        module.four_reset.v4.adaptive_replaces_uniform,
    )
    try:
        module.configure()
        assert module.four_reset.base.STAGES == ((4, 4),)
        assert module.four_reset.base.SEARCH_VALID_TARGET == 4
        assert module.four_reset.base.SEARCH_BUDGET == 32
        assert (
            module.four_reset.base.CODEX_STUDY_VERSION
            == "v13-diverse-four-reset"
        )
    finally:
        (
            module.four_reset.base.STAGES,
            module.four_reset.base.SEARCH_VALID_TARGET,
            module.four_reset.base.SEARCH_BUDGET,
            module.four_reset.base.CODEX_STUDY_VERSION,
            module.four_reset.base.CODEX_METHOD,
            module.four_reset.base.SELECTION_SCHEMA,
            module.four_reset.v4.adaptive_replaces_uniform,
        ) = original


def test_registered_primary_seeds_match_outcome_blind_panel_receipts():
    experiment = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "act_strider_codex_v13_diverse_four_reset"
    )
    banks = json.loads((experiment / "BANKS.json").read_text())
    for task_label, task in banks["tasks"].items():
        panel = json.loads((experiment / "panels" / f"{task_label}.json").read_text())
        assert panel["selection_uses_policy_outcomes"] is False
        assert panel["panel_size"] == 4
        assert task["search_primary"]["seeds"] == panel["selected_seeds"]


def test_v13_scientific_banks_are_pairwise_disjoint():
    experiment = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "act_strider_codex_v13_diverse_four_reset"
    )
    banks = json.loads((experiment / "BANKS.json").read_text())
    observed = set()
    for task in banks["tasks"].values():
        for spec in task.values():
            values = set(spec.get("seeds", ()))
            if "start" in spec:
                values = set(range(spec["start"], spec["start"] + spec["count"]))
            assert observed.isdisjoint(values)
            observed |= values

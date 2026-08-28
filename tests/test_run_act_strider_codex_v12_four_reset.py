import json
from pathlib import Path

from scripts import run_act_strider_codex_v12_four_reset as module


def report(*, throughput: float, successes: int = 4, qualified: bool = True):
    return {
        "qualified": qualified,
        "summary": {
            "episodes": 4,
            "successes": successes,
            "safety_violations": 0,
            "physics_errors": 0,
            "achieved_throughput_per_step": throughput,
        },
    }


def test_four_reset_replacement_requires_matched_perfection_and_gain():
    incumbent = report(throughput=1.0)
    assert module.adaptive_replaces_uniform(report(throughput=1.03), incumbent)
    assert not module.adaptive_replaces_uniform(report(throughput=1.029), incumbent)
    assert not module.adaptive_replaces_uniform(
        report(throughput=1.2, successes=3), incumbent
    )


def test_v12_configuration_is_search_only_four_reset_contract():
    original = (
        module.base.STAGES,
        module.base.SEARCH_VALID_TARGET,
        module.base.SEARCH_BUDGET,
        module.base.CODEX_STUDY_VERSION,
        module.base.CODEX_METHOD,
        module.base.SELECTION_SCHEMA,
        module.v4.adaptive_replaces_uniform,
    )
    try:
        module.configure()
        assert module.base.STAGES == ((4, 4),)
        assert module.base.SEARCH_VALID_TARGET == 4
        assert module.base.SEARCH_BUDGET == 32
        assert module.base.CODEX_STUDY_VERSION == "v12-four-reset"
    finally:
        (
            module.base.STAGES,
            module.base.SEARCH_VALID_TARGET,
            module.base.SEARCH_BUDGET,
            module.base.CODEX_STUDY_VERSION,
            module.base.CODEX_METHOD,
            module.base.SELECTION_SCHEMA,
            module.v4.adaptive_replaces_uniform,
        ) = original


def test_v12_banks_are_four_primary_resets_and_disjoint_from_v11():
    root = Path(__file__).resolve().parents[1] / "experiments"
    current = json.loads(
        (root / "act_strider_codex_v12_four_reset" / "BANKS.json").read_text()
    )
    prior = json.loads((root / "act_strider_codex_v11" / "BANKS.json").read_text())

    def seeds(spec):
        return set(range(spec["start"], spec["start"] + spec["count"]))

    prior_all = set()
    for task in prior["tasks"].values():
        for spec in task.values():
            prior_all |= seeds(spec)
    current_all = set()
    for task in current["tasks"].values():
        assert task["search_primary"]["count"] == 4
        for spec in task.values():
            values = seeds(spec)
            assert current_all.isdisjoint(values)
            current_all |= values
    assert current_all.isdisjoint(prior_all)
